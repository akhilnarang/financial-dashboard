"""Transaction HTML routes."""

import calendar
import json
import logging
import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Annotated, NamedTuple, cast
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from financial_dashboard.core.dates import financial_year_start
from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import (
    Account,
    Card,
    Email,
    SmsMessage,
    Transaction,
)
from financial_dashboard.db.models import Category
from financial_dashboard.exceptions import UnprocessableEntityException
from financial_dashboard.services.cashflow.scope import Scope
from financial_dashboard.services.transaction_filters import (
    SORT_COLUMNS,
    SortKey,
    TransactionFilters,
    ordered,
    transaction_totals,
)
from financial_dashboard.web.transaction_display import category_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()

PAGE_SIZE = 50


class NormalizedQueryDate(NamedTuple):
    """Parsed date filter and its canonical query representation.

    Attributes:
        parsed_date: Validated calendar date, or ``None`` for an omitted filter.
        query_value: Canonical ``YYYY-MM-DD`` string, or the preserved omitted
            representation (``None`` or the empty string).
    """

    parsed_date: date | None
    query_value: str | None


class LoadedTransaction(NamedTuple):
    """Transaction and optional records linked to its provenance.

    Attributes:
        transaction: Requested transaction row.
        email: Email attached to the transaction, when present.
        account: Account assigned to the transaction, when present.
        sms: SMS attached to the transaction, when present.
    """

    transaction: Transaction
    email: Email | None
    account: Account | None
    sms: SmsMessage | None


# Display labels for the categorization pipeline's category_method column, so a
# transaction's provenance reads as a word rather than the internal state-machine
# value. ``None`` is omitted on purpose: a row that was never run through the
# pipeline has no method to label.
CATEGORY_METHOD_LABELS: dict[str, str] = {
    "manual": "Manual",
    "rule": "Rule",
    "llm": "AI",
    "pending_llm": "Pending AI",
}


def _normalize_query_date(value: str | None, field: str) -> NormalizedQueryDate:
    """Validate and normalize one ISO date query parameter.

    Args:
        value: Query value in strict ``YYYY-MM-DD`` form. ``None`` and the empty
            string represent an omitted filter.
        field: Parameter name included in validation errors, such as
            ``"date_from"`` or ``"date_to"``.

    Returns:
        A ``NormalizedQueryDate`` containing the parsed date and canonical query
        value. Omitted filters have no parsed date and preserve their ``None`` or
        empty-string query representation. A day beyond the end of an otherwise
        valid month is clamped to that month's final day.

    Raises:
        UnprocessableEntityException: If the value is not strict ISO syntax or
            its year, month, or day cannot be safely normalized.
    """
    if value is None or value == "":
        return NormalizedQueryDate(parsed_date=None, query_value=value)
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match is None:
        raise UnprocessableEntityException(
            detail=f"{field} must use YYYY-MM-DD and be a valid calendar date",
        )
    year, month, day = (int(part) for part in match.groups())
    try:
        maximum_day = calendar.monthrange(year, month)[1]
    except (calendar.IllegalMonthError, ValueError) as exc:
        raise UnprocessableEntityException(
            detail=f"{field} must use YYYY-MM-DD and be a valid calendar date",
        ) from exc
    if day < 1:
        raise UnprocessableEntityException(
            detail=f"{field} must use YYYY-MM-DD and be a valid calendar date",
        )
    try:
        normalized = date(year, month, min(day, maximum_day))
    except ValueError as exc:
        raise UnprocessableEntityException(
            detail=f"{field} must use YYYY-MM-DD and be a valid calendar date",
        ) from exc
    return NormalizedQueryDate(
        parsed_date=normalized,
        query_value=normalized.isoformat(),
    )


def _date_display(value: str | None) -> str:
    """Render an ISO query date in the page's Indian display format.

    Args:
        value: ISO ``YYYY-MM-DD`` date string, or an omitted value.

    Returns:
        The date formatted as ``DD-MM-YYYY``. Omitted or invalid values return
        an empty string so an invalid query value is never displayed as valid.
    """
    if not value:
        return ""
    try:
        return date.fromisoformat(value).strftime("%d-%m-%Y")
    except ValueError:
        return ""


def _date_presets(today: date) -> list[dict[str, str]]:
    """Build common calendar and Indian financial-year filter ranges.

    Args:
        today: Date used as the boundary for relative ranges.

    Returns:
        Presets containing a display label and inclusive ISO ``date_from`` and
        ``date_to`` values. Indian financial years run from April through March.
    """
    current_month_start = today.replace(day=1)
    previous_month_end = current_month_start - timedelta(days=1)
    previous_month_start = previous_month_end.replace(day=1)
    next_month_start = (
        current_month_start.replace(year=current_month_start.year + 1, month=1)
        if current_month_start.month == 12
        else current_month_start.replace(month=current_month_start.month + 1)
    )
    fy_start_year = financial_year_start(today).year
    ranges = (
        ("Current month", current_month_start, next_month_start - timedelta(days=1)),
        ("Previous month", previous_month_start, previous_month_end),
        ("Last 30 days", today - timedelta(days=29), today),
        (
            "Current financial year",
            date(fy_start_year, 4, 1),
            date(fy_start_year + 1, 3, 31),
        ),
        (
            "Previous financial year",
            date(fy_start_year - 1, 4, 1),
            date(fy_start_year, 3, 31),
        ),
        ("Current calendar year", date(today.year, 1, 1), date(today.year, 12, 31)),
    )
    return [
        {"label": label, "date_from": start.isoformat(), "date_to": end.isoformat()}
        for label, start, end in ranges
    ]


def _ids(values: list[str] | None) -> tuple[int, ...]:
    """Return the ids among the query values. A value that is not a number is ignored."""
    return tuple(int(value) for value in values or () if value.strip().isdigit())


def _amount(value: str | None) -> Decimal | None:
    """Return the amount in the query value, or None when there is none or it is bad."""
    try:
        amount = Decimal(value) if value and value.strip() else None
    except InvalidOperation:
        return None
    return amount if amount is not None and amount.is_finite() else None


@router.get("/transactions", response_class=HTMLResponse)
async def transaction_list(
    request: FastAPIRequest,
    bank: Annotated[str | None, Query(description="Filter by bank name")] = None,
    account_id: Annotated[
        list[str] | None, Query(description="Filter by account ID; repeat for several")
    ] = None,
    card_id: Annotated[
        list[str] | None, Query(description="Filter by card ID; repeat for several")
    ] = None,
    direction: Annotated[
        str | None, Query(description="Filter by direction: debit or credit")
    ] = None,
    date_from: Annotated[
        str | None, Query(description="Transaction date on/after (YYYY-MM-DD)")
    ] = None,
    date_to: Annotated[
        str | None, Query(description="Transaction date on/before (YYYY-MM-DD)")
    ] = None,
    category: Annotated[
        str | None, Query(description="Filter by category slug")
    ] = None,
    counterparty: Annotated[
        str | None,
        Query(description="Filter by counterparty; blank matches no counterparty"),
    ] = None,
    card_holder: Annotated[
        str | None, Query(description="Filter by the statement's cardholder")
    ] = None,
    uncategorized: Annotated[
        str | None, Query(description="Set to 1 for rows with no usable category")
    ] = None,
    category_null: Annotated[
        str | None,
        Query(description="Set to 1 for rows carrying no category (NULL or blank)"),
    ] = None,
    internal: Annotated[
        str | None, Query(description="Set to 1 for internal-movement rows")
    ] = None,
    non_inr: Annotated[
        str | None,
        Query(
            description="1 for rows not denominated in INR, 0 for rupee rows only "
            "(INR or no currency); omit for no currency filter"
        ),
    ] = None,
    undated: Annotated[
        str | None, Query(description="Set to 1 for rows with no transaction date")
    ] = None,
    excluded: Annotated[
        str | None,
        Query(
            description="1 for rows flagged out of the cashflow, 0 for rows still in "
            "it; omit for no filter"
        ),
    ] = None,
    provisional: Annotated[
        str | None,
        Query(description="Set to 1 for screenshot rows no bank source confirmed yet"),
    ] = None,
    scope: Annotated[
        Scope | None,
        Query(
            description="Account scope: bank (bank accounts and debit cards), card "
            "(credit cards), or unaccounted (no account, or an account type neither "
            "of those names); omit for every account"
        ),
    ] = None,
    q: Annotated[
        str | None, Query(description="Text to find in the row's text fields")
    ] = None,
    amount: Annotated[str | None, Query(description="Exact amount")] = None,
    source: Annotated[str | None, Query(description="Row source, such as sms")] = None,
    review_status: Annotated[
        str | None, Query(description="Category review status")
    ] = None,
    reference: Annotated[
        str | None, Query(description="Payment reference, such as a UTR")
    ] = None,
    sort: Annotated[str, Query(description="Sort column")] = "date",
    order: Annotated[str, Query(description="Sort order: asc or desc")] = "desc",
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Render the filtered, sorted, and paginated transaction listing.

    Args:
        request: Current request used to render the template and preserve queries.
        bank: Optional exact bank-name filter.
        account_id: Optional account IDs, supplied as query strings.
        card_id: Optional card IDs, supplied as query strings.
        direction: Optional transaction direction, normally ``debit`` or ``credit``.
        date_from: Optional inclusive lower date bound in ISO ``YYYY-MM-DD`` form.
        date_to: Optional inclusive upper date bound in ISO ``YYYY-MM-DD`` form.
        category: Optional exact category-slug filter.
        counterparty: Optional exact counterparty; a blank value selects rows with
            no counterparty.
        card_holder: Optional cardholder that the card statement names.
        uncategorized: ``"1"`` selects rows outside the known category buckets.
        category_null: ``"1"`` selects rows whose category is null or blank.
        internal: ``"1"`` selects internal movement categories for the scope.
        non_inr: ``"1"`` selects foreign currencies and ``"0"`` selects INR or
            legacy rows without a currency.
        undated: ``"1"`` selects rows without a transaction date that are not
            voided.
        provisional: ``"1"`` selects screenshot rows that no bank source
            confirms yet.
        scope: Optional account perimeter used by cash-flow drill-down links.
        q: Optional text to find in the row's text fields.
        amount: Optional exact amount.
        source: Optional exact row source.
        review_status: Optional exact category review status.
        reference: Optional payment reference.
        sort: Requested sort key; invalid values fall back to transaction date.
        order: Sort direction; invalid values fall back to descending.
        page: One-indexed result page.
        session: Request-scoped asynchronous database session.

    Returns:
        A canonicalizing redirect when an overflowing date is corrected;
        otherwise, the rendered transactions page with full-filter totals and
        at most ``PAGE_SIZE`` transaction rows.
    """
    normalized_date_from = _normalize_query_date(date_from, "date_from")
    normalized_date_to = _normalize_query_date(date_to, "date_to")
    corrections: dict[str, str] = {}
    for key, original, normalized in (
        ("date_from", date_from, normalized_date_from.query_value),
        ("date_to", date_to, normalized_date_to.query_value),
    ):
        if (
            original not in (None, "")
            and normalized is not None
            and normalized != original
        ):
            corrections[key] = normalized
    if corrections:
        # multi_items keeps a repeated account_id or card_id.
        query = [
            (key, corrections.get(key, value))
            for key, value in request.query_params.multi_items()
        ]
        return RedirectResponse(
            url=f"/transactions?{urlencode(query)}",
            status_code=307,
        )
    date_from = normalized_date_from.query_value
    date_to = normalized_date_to.query_value

    page_filters = TransactionFilters(
        bank=bank or None,
        account_ids=_ids(account_id),
        card_ids=_ids(card_id),
        direction=direction or None,
        date_from=normalized_date_from.parsed_date,
        date_to=normalized_date_to.parsed_date,
        category=category or None,
        counterparty=counterparty,
        card_holder=card_holder or None,
        uncategorized=uncategorized == "1",
        category_null=category_null == "1",
        internal=internal == "1",
        non_inr={"1": True, "0": False}.get(non_inr or ""),
        undated=undated == "1",
        excluded={"1": True, "0": False}.get(excluded or ""),
        provisional=provisional == "1",
        scope=scope,
        q=q or None,
        amount=_amount(amount),
        source=source or None,
        review_status=review_status or None,
        reference=reference or None,
    )
    stmt = select(Transaction).where(*page_filters.clauses())
    totals = await transaction_totals(session, stmt)
    total_count = totals.count

    if sort not in SORT_COLUMNS:
        sort = "date"
    if order not in ("asc", "desc"):
        order = "desc"
    stmt = ordered(stmt, cast(SortKey, sort), order)

    total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
    if page < 1:
        page = 1
    if page > total_pages and total_pages > 0:
        page = total_pages
    stmt = stmt.limit(PAGE_SIZE).offset((page - 1) * PAGE_SIZE)

    result = await session.execute(stmt)
    transactions = result.scalars().all()

    bank_result = await session.execute(select(Transaction.bank).distinct())
    banks = sorted([row[0] for row in bank_result.all()])

    accounts = (
        (
            await session.execute(
                select(Account)
                .where(Account.active.is_(True))
                .order_by(Account.bank, Account.label)
            )
        )
        .scalars()
        .all()
    )

    cards = (
        (
            await session.execute(
                select(Card).where(Card.active.is_(True)).order_by(Card.card_mask)
            )
        )
        .scalars()
        .all()
    )

    # Active category slugs, label-mapped for the filter dropdown and the per-row
    # badge. Ordered by slug so the dropdown is stable and matches the vocabulary
    # the categorizer offers; a transaction whose own category has since been
    # deactivated still renders (the template falls back to a title-cased slug).
    active_category_slugs = list(
        (
            await session.execute(
                select(Category.slug)
                .where(Category.active.is_(True))
                .order_by(Category.slug)
            )
        )
        .scalars()
        .all()
    )
    category_labels = {slug: category_label(slug) for slug in active_category_slugs}

    # Build JSON for dependent dropdowns
    cards_by_account: dict[int, list] = {}
    for c in cards:
        cards_by_account.setdefault(c.account_id, []).append(
            {
                "id": c.id,
                "mask": c.card_mask,
                "label": c.label or c.card_mask,
            }
        )
    accounts_by_bank: dict[str, list] = {}
    for a in accounts:
        accounts_by_bank.setdefault(a.bank, []).append(
            {
                "id": a.id,
                "label": a.label,
                "type": a.type,
            }
        )

    # Build base query string for pagination/sort links
    form_filters = {
        "bank": bank,
        "account_id": account_id[0] if account_id else None,
        "card_id": card_id[0] if card_id else None,
        "direction": direction,
        "date_from": date_from,
        "date_to": date_to,
    }
    drill_filters = {
        "category": category,
        "counterparty": counterparty,
        "card_holder": card_holder,
        "uncategorized": uncategorized,
        "category_null": category_null,
        "internal": internal,
        "non_inr": non_inr,
        "undated": undated,
        "excluded": excluded,
        "provisional": provisional,
        # A drill param like any other, so paging and re-sorting keep it: a page-2
        # or newly-sorted link that dropped the scope would widen the listing to
        # every account while still claiming to be the figure's rows.
        "scope": scope,
    }
    filters = {**form_filters, **drill_filters}
    # The shared filters keep a blank counterparty, because it narrows the rows:
    # dropping it would widen page 2 to every row.
    query = page_filters.query()
    retained_filters = {
        k: v for k, v in query.items() if k not in {"date_from", "date_to"}
    }
    base_qs = urlencode(query, doseq=True)
    date_presets = []
    for preset in _date_presets(date.today()):
        preset_query = {
            **retained_filters,
            "date_from": preset["date_from"],
            "date_to": preset["date_to"],
            "sort": sort,
            "order": order,
        }
        date_presets.append(
            {
                **preset,
                "href": f"/transactions?{urlencode(preset_query, doseq=True)}",
                "active": (
                    date_from == preset["date_from"] and date_to == preset["date_to"]
                ),
            }
        )

    # Page window: show pages around current
    def page_window() -> list[int]:
        """Build the compact page-number window used by the paginator.

        Returns:
            Sorted unique page numbers containing the first and last pages plus
            up to two pages on either side of the current page.
        """
        pages = set()
        pages.add(1)
        pages.add(total_pages)
        for p in range(max(1, page - 2), min(total_pages, page + 2) + 1):
            pages.add(p)
        return sorted(pages)

    return templates.TemplateResponse(
        request,
        "transactions.html",
        {
            "active_page": "transactions",
            "transactions": transactions,
            "banks": banks,
            "accounts": accounts,
            "accounts_json": json.dumps(accounts_by_bank),
            "cards_json": json.dumps(cards_by_account),
            "category_labels": category_labels,
            "category_method_labels": CATEGORY_METHOD_LABELS,
            "filters": filters,
            "sort": sort,
            "order": order,
            "page": page,
            "total_count": total_count,
            "total_pages": total_pages,
            "page_size": PAGE_SIZE,
            "transaction_totals": totals,
            "date_from_display": _date_display(date_from),
            "date_to_display": _date_display(date_to),
            "date_presets": date_presets,
            "page_window": page_window(),
            "base_qs": base_qs,
        },
    )


async def _load_transaction(
    session: AsyncSession, txn_id: int
) -> LoadedTransaction | None:
    """Load a transaction and the records linked to its provenance.

    Args:
        session: Request-scoped asynchronous database session.
        txn_id: Database ID of the transaction to load.

    Returns:
        A ``LoadedTransaction`` containing the transaction and its optional
        provenance records, or ``None`` when the transaction does not exist.
    """
    result = await session.execute(
        select(Transaction, Email, Account, SmsMessage)
        .outerjoin(Email, Transaction.email_id == Email.id)
        .outerjoin(Account, Transaction.account_id == Account.id)
        .outerjoin(SmsMessage, Transaction.sms_message_id == SmsMessage.id)
        .where(Transaction.id == txn_id)
    )
    row = result.first()
    if not row:
        return None
    return LoadedTransaction(
        transaction=row[0],
        email=row[1],
        account=row[2],
        sms=row[3],
    )


async def _bank_account_picker(session: AsyncSession, bank: str) -> list[dict]:
    """Build the bounded account choices for the manual relink picker.

    Args:
        session: Request-scoped asynchronous database session.
        bank: Bank whose active accounts and cards should be offered.

    Returns:
        JSON-friendly active accounts ordered by label. Each item contains only
        the account identity fields and active card identity fields required by
        the template. Returns an empty list when the bank has no active accounts.
    """
    result = await session.execute(
        select(Account)
        .where(Account.bank == bank, Account.active.is_(True))
        .order_by(Account.label)
        .options(selectinload(Account.cards))
    )
    accounts = result.scalars().all()
    return [
        {
            "id": a.id,
            "label": a.label,
            "type": a.type,
            "cards": [
                {"id": c.id, "label": c.label, "card_mask": c.card_mask}
                for c in a.cards
                if c.active
            ],
        }
        for a in accounts
    ]


@router.get("/transactions/{txn_id}/detail", response_class=HTMLResponse)
async def transaction_detail(
    txn_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Render the transaction detail fragment used by the listing dialog.

    Args:
        txn_id: Database ID of the transaction to display.
        request: Current request used to render the template fragment.
        session: Request-scoped asynchronous database session.

    Returns:
        The rendered detail fragment, or a small 404 HTML response when the
        transaction does not exist.
    """
    loaded = await _load_transaction(session, txn_id)
    if loaded is None:
        return HTMLResponse("<p>Transaction not found.</p>", 404)
    txn = loaded.transaction
    bank_accounts = (
        await _bank_account_picker(session, txn.bank) if txn.account_id is None else []
    )
    return templates.TemplateResponse(
        request,
        "partials/transaction_detail.html",
        {
            "txn": txn,
            "email": loaded.email,
            "account": loaded.account,
            "sms": loaded.sms,
            "bank_accounts": bank_accounts,
            "category_method_labels": CATEGORY_METHOD_LABELS,
        },
    )


@router.get("/transactions/{txn_id}", response_class=HTMLResponse)
async def transaction_page(
    txn_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Render the standalone page for one transaction.

    Args:
        txn_id: Database ID of the transaction to display.
        request: Current request used to render the page template.
        session: Request-scoped asynchronous database session.

    Returns:
        The rendered transaction page, or a small 404 HTML response when the
        transaction does not exist.
    """
    loaded = await _load_transaction(session, txn_id)
    if loaded is None:
        return HTMLResponse("<p>Transaction not found.</p>", 404)
    txn = loaded.transaction
    bank_accounts = (
        await _bank_account_picker(session, txn.bank) if txn.account_id is None else []
    )
    return templates.TemplateResponse(
        request,
        "transaction_page.html",
        {
            "active_page": "transactions",
            "txn": txn,
            "email": loaded.email,
            "account": loaded.account,
            "sms": loaded.sms,
            "bank_accounts": bank_accounts,
            "category_method_labels": CATEGORY_METHOD_LABELS,
        },
    )
