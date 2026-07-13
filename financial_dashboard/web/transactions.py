"""Transaction HTML routes."""

import json
import logging
from datetime import date
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import (
    Account,
    Card,
    Email,
    SmsMessage,
    Transaction,
)
from financial_dashboard.services.cashflow.buckets import INTERNAL_SLUGS
from financial_dashboard.services.cashflow.report import (
    BLANK_CATEGORY,
    BLANK_COUNTERPARTY,
    INR_OR_NULL,
    NON_INR,
    UNCATEGORIZED,
    is_blank_counterparty,
)

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
SORT_COLUMNS = {
    "amount": Transaction.amount,
    "bank": Transaction.bank,
    "counterparty": Transaction.counterparty,
    "date": Transaction.transaction_date,
}


@router.get("/transactions", response_class=HTMLResponse)
async def transaction_list(
    request: FastAPIRequest,
    bank: Annotated[str | None, Query(description="Filter by bank name")] = None,
    account_id: Annotated[str | None, Query(description="Filter by account ID")] = None,
    card_id: Annotated[str | None, Query(description="Filter by card ID")] = None,
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
    sort: Annotated[str, Query(description="Sort column")] = "date",
    order: Annotated[str, Query(description="Sort order: asc or desc")] = "desc",
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Transaction)

    if bank:
        stmt = stmt.where(Transaction.bank == bank)
    if account_id:
        try:
            stmt = stmt.where(Transaction.account_id == int(account_id))
        except ValueError:
            pass
    if card_id:
        try:
            stmt = stmt.where(Transaction.card_id == int(card_id))
        except ValueError:
            pass
    if direction:
        stmt = stmt.where(Transaction.direction == direction)
    if date_from:
        try:
            stmt = stmt.where(
                Transaction.transaction_date >= date.fromisoformat(date_from)
            )
        except ValueError:
            pass
    if date_to:
        try:
            stmt = stmt.where(
                Transaction.transaction_date <= date.fromisoformat(date_to)
            )
        except ValueError:
            pass
    if category:
        stmt = stmt.where(Transaction.category == category)
    if counterparty is not None:
        if is_blank_counterparty(counterparty):
            # A blank counterparty is a filter, not an absent one: it selects the
            # rows that carry no counterparty at all, whether that is stored as
            # NULL, as an empty string, or as whitespace. Both the test for the
            # incoming value and the clause it produces come from the report, so
            # this lists exactly the rows the report's "(no counterparty)" line
            # counted — no spelling of blank can fall between the two.
            stmt = stmt.where(BLANK_COUNTERPARTY)
        else:
            stmt = stmt.where(Transaction.counterparty == counterparty)
    if uncategorized == "1":
        # Uncategorized means "has no category the bucket map can place": blank,
        # the 'unknown' sentinel, or a runtime slug the map does not know. The
        # clause is the one the report's uncategorized line aggregates on, imported
        # rather than restated, so the count on the tile and the rows listed here
        # cannot drift apart. No currency clause — a non-INR row with no category
        # is still uncategorized.
        stmt = stmt.where(UNCATEGORIZED)
    if category_null == "1":
        # Strictly the rows carrying no category at all — NULL or blank, which are
        # the same absence and are counted as one line by the report. This is
        # narrower than `uncategorized`, which also takes in the 'unknown' sentinel
        # and slugs the bucket map does not know: a report line that counts only
        # the category-less rows needs a filter that lists only those rows, or its
        # link would return more rows than the line it sits on says it has.
        stmt = stmt.where(BLANK_CATEGORY)
    if internal == "1":
        stmt = stmt.where(Transaction.category.in_(tuple(INTERNAL_SLUGS)))
    if non_inr == "1":
        stmt = stmt.where(NON_INR)
    elif non_inr == "0":
        # The complement, and the exact population the rupee buckets of the
        # cashflow report sum: a NULL currency is a rupee row that predates the
        # column's default. Without this value a bucket's drill-through would
        # list foreign rows its own total left out.
        stmt = stmt.where(INR_OR_NULL)
    if undated == "1":
        stmt = stmt.where(Transaction.transaction_date.is_(None))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_count = (await session.execute(count_stmt)).scalar() or 0

    sort_col = SORT_COLUMNS.get(sort, Transaction.transaction_date)
    if order not in ("asc", "desc"):
        order = "desc"
    if order == "asc":
        stmt = stmt.order_by(sort_col.asc().nullslast(), Transaction.id.asc())
    else:
        stmt = stmt.order_by(sort_col.desc().nullslast(), Transaction.id.desc())

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
        "account_id": account_id,
        "card_id": card_id,
        "direction": direction,
        "date_from": date_from,
        "date_to": date_to,
    }
    drill_filters = {
        "category": category,
        "counterparty": counterparty,
        "uncategorized": uncategorized,
        "category_null": category_null,
        "internal": internal,
        "non_inr": non_inr,
        "undated": undated,
    }
    filters = {**form_filters, **drill_filters}
    # Drill params are kept whenever they are present, empty string included: a
    # blank counterparty narrows the result set, so dropping it on truthiness
    # would silently widen page 2 to every row.
    base_qs = urlencode(
        {k: v for k, v in form_filters.items() if v}
        | {k: v for k, v in drill_filters.items() if v is not None}
    )

    # Page window: show pages around current
    def page_window():
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
            "filters": filters,
            "sort": sort,
            "order": order,
            "page": page,
            "total_count": total_count,
            "total_pages": total_pages,
            "page_size": PAGE_SIZE,
            "page_window": page_window(),
            "base_qs": base_qs,
        },
    )


async def _load_transaction(
    session: AsyncSession, txn_id: int
) -> tuple[Transaction, Email | None, Account | None, SmsMessage | None] | None:
    """Load a Transaction with its linked Email, Account, and SMS, or None."""
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
    return tuple(row)  # type: ignore[return-value]


async def _bank_account_picker(session: AsyncSession, bank: str) -> list[dict]:
    """Compact JSON-friendly account list for the manual relink picker.

    Returns active accounts of ``bank`` ordered by label, each with its
    active cards (id, label, card_mask only — no ORM internals leak
    into the template). Empty list when the bank has no active
    accounts, which the template renders as a recovery hint.
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
):
    loaded = await _load_transaction(session, txn_id)
    if loaded is None:
        return HTMLResponse("<p>Transaction not found.</p>", 404)
    txn, email, account, sms = loaded
    bank_accounts = (
        await _bank_account_picker(session, txn.bank) if txn.account_id is None else []
    )
    return templates.TemplateResponse(
        request,
        "partials/transaction_detail.html",
        {
            "txn": txn,
            "email": email,
            "account": account,
            "sms": sms,
            "bank_accounts": bank_accounts,
        },
    )


@router.get("/transactions/{txn_id}", response_class=HTMLResponse)
async def transaction_page(
    txn_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    loaded = await _load_transaction(session, txn_id)
    if loaded is None:
        return HTMLResponse("<p>Transaction not found.</p>", 404)
    txn, email, account, sms = loaded
    bank_accounts = (
        await _bank_account_picker(session, txn.bank) if txn.account_id is None else []
    )
    return templates.TemplateResponse(
        request,
        "transaction_page.html",
        {
            "active_page": "transactions",
            "txn": txn,
            "email": email,
            "account": account,
            "sms": sms,
            "bank_accounts": bank_accounts,
        },
    )
