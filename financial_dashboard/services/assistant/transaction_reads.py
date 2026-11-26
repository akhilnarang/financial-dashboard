"""Allowlisted, bounded transaction reads for the conversational assistant.

The assistant has a narrower read surface than the general transaction API.  It
can identify and explain rows, but it cannot request arbitrary columns or SQL.
The predicates below intentionally mirror the useful filters on the
``/transactions`` page: account, date, direction, category, free-text search,
and cashflow exclusion state.
"""

import datetime
import re
from decimal import Decimal, InvalidOperation
from collections.abc import Iterable
from typing import Any, Literal, NamedTuple, cast

from sqlalchemy import ColumnElement, Select, and_, func, literal, or_, select
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.references import reference_form
from financial_dashboard.db.models import (
    Account,
    Card,
    CategoryReviewDecision,
    Transaction,
)
from financial_dashboard.services.transaction_reads import (
    build_transaction_filter_clauses,
)
from financial_dashboard.services.cashflow.buckets import label_for_slug
from financial_dashboard.services.categorization.normalize import (
    redact_names,
    redact_pii,
)
from financial_dashboard.services.assistant.contracts import ShowTransactions
from financial_dashboard.services.categorization.vocabulary import get_active_slugs
from financial_dashboard.services.settings import get_redact_name_tokens
from financial_dashboard.services.transaction_filters import TransactionFilters

MAX_CONTEXT_ROWS = 20
MAX_FIELD_LENGTH = 2_000
# A shorter stored counterparty, such as "UPI", would match too many searches.
MIN_PREFIX_COUNTERPARTY = 4
MAX_SEARCH_WORDS = 8
# These words do not tell one account or card from another, such as "CC" in
# "Sample Bank Gold CC". A card that the user holds has the label "self" or
# "Primary".
GENERIC_NAME_WORDS = frozenset(
    {"account", "acct", "bank", "card", "cc", "credit", "debit", "primary", "self"}
)
_POSSESSIVE = re.compile(r"['’]s\b")
_NAME_WORD = re.compile(r"[a-z0-9+]+")

_REDACTED_REF = "[redacted-ref]"
_MASK_LABEL = re.compile(r"\[redacted-[a-z]+\]")


class AssistantTransaction(BaseModel):
    """Redacted transaction projection safe to place in a model context.

    Only these fields reach the model. Names and numbers in the text fields
    are masked before the projection is built.

    Attributes:
        review_gate_reason: Why the categoriser asked the user about the row,
            from the latest active review decision, or None.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    bank: str
    direction: str
    amount: Decimal
    currency: str | None
    transaction_date: datetime.date | None
    counterparty: str | None
    reference_number: str | None
    channel: str | None
    account_id: int | None
    card_id: int | None
    category: str | None
    category_method: str | None
    note: str | None
    exclude_from_cashflow: bool
    voided: bool
    category_confidence: float | None
    category_model: str | None
    review_status: str | None
    review_reason: str | None
    review_gate_reason: str | None


class LabelCandidate(NamedTuple):
    """An account or a card that a name can point to, with its label.

    Attributes:
        kind: ``account`` or ``card``.
        id: The account id or the card id.
        label: The label to match. A card's label also holds its account's
            label, so a name can give both.
    """

    kind: Literal["account", "card"]
    id: int
    label: str


class LabelRank(NamedTuple):
    """How closely a label fits a name. A higher rank fits better.

    Tuples compare field by field, so more covered words win first.

    Attributes:
        covered: How many words of the name the label covers.
        whole: True when the name holds every word of the label. It wins a tie.
    """

    covered: int
    whole: bool


class NamedIds(NamedTuple):
    """The accounts and cards that a name points to.

    Attributes:
        account_ids: The ids of the accounts.
        card_ids: The ids of the cards.
    """

    account_ids: list[int]
    card_ids: list[int]


class AssistantCard(BaseModel):
    """One card of an account, as the model sees it.

    Attributes:
        id: The card id, for the ``card_id`` filter.
        label: The card label, with configured names masked.
        active: False for a card that the user no longer uses.
        is_primary: True for the user's own card, False for an add-on.
    """

    id: int
    label: str | None
    active: bool
    is_primary: bool


class AssistantAccount(BaseModel):
    """One account and its cards, as the model sees them.

    Attributes:
        id: The account id, for the ``account_id`` filter.
        bank: The bank, as the rows store it.
        label: The account label, with configured names masked.
        type: The account type, such as ``credit_card``.
        active: False for an account that the user closed.
        cards: The cards of the account.
    """

    id: int
    bank: str
    label: str
    type: str
    active: bool
    cards: list[AssistantCard]


def _bounded(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:MAX_FIELD_LENGTH]


def _redacted(value: str | None, name_tokens: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    return _bounded(redact_names(redact_pii(value), name_tokens))


def _projection(
    row: Any,
    *,
    name_tokens: tuple[str, ...],
    review_gate_reason: str | None = None,
) -> AssistantTransaction:
    """Convert a SQL row using only the explicit assistant allowlist."""
    return AssistantTransaction(
        id=row.id,
        bank=row.bank,
        direction=row.direction,
        amount=row.amount,
        currency=row.currency,
        transaction_date=row.transaction_date,
        counterparty=_redacted(row.counterparty, name_tokens),
        # A payment reference names a payment, not a person, so it skips the
        # number mask. Configured identifiers are masked in it all the same.
        reference_number=None
        if row.reference_number is None
        else _bounded(
            _MASK_LABEL.sub(
                _REDACTED_REF,
                redact_names(
                    row.reference_number
                    if reference_form(row.reference_number) is not None
                    else redact_pii(row.reference_number),
                    name_tokens,
                ),
            )
        ),
        channel=_bounded(row.channel),
        account_id=row.account_id,
        card_id=row.card_id,
        category=_bounded(row.category),
        category_method=_bounded(row.category_method),
        note=_redacted(row.note, name_tokens),
        exclude_from_cashflow=bool(row.exclude_from_cashflow),
        voided=bool(row.voided),
        category_confidence=row.category_confidence,
        category_model=_bounded(row.category_model),
        review_status=_bounded(row.review_status),
        review_reason=_bounded(row.review_reason),
        review_gate_reason=_bounded(review_gate_reason),
    )


def _search_clause(search: str) -> ColumnElement[bool]:
    """Match the search text, or a counterparty that a bank cut short.

    A bank can store only the start of a name, such as "SAMPLE" for
    "Samplename". Such a counterparty matches when it is the start of the search
    text from one of its words on.

    Args:
        search: The free text that the model searches for.

    Returns:
        One clause that matches either form.
    """
    words = search.lower().split()[:MAX_SEARCH_WORDS]
    stored = func.lower(func.trim(Transaction.counterparty))
    return or_(
        *build_transaction_filter_clauses(search=search),
        *(
            and_(
                func.length(stored) >= MIN_PREFIX_COUNTERPARTY,
                func.substr(literal(" ".join(words[start:])), 1, func.length(stored))
                == stored,
            )
            for start in range(len(words))
        ),
    )


def _name_words(text: str) -> frozenset[str]:
    """Return the words that tell one account or card from another.

    Case, a possessive "'s", punctuation and generic words such as "card" do
    not count.

    Args:
        text: A name or a label.

    Returns:
        The words, in lower case.
    """
    text = _POSSESSIVE.sub("", text.lower().replace("a/c", " "))
    return frozenset(_NAME_WORD.findall(text)) - GENERIC_NAME_WORDS


def _closest_labels(
    words: frozenset[str], candidates: Iterable[LabelCandidate], *, loose: bool
) -> list[LabelCandidate]:
    """Return the candidates whose labels the name words point to most closely.

    A label points when the name holds all its words. A label of one word
    stands for a single merchant or bank name, such as "Swiggy", so only a
    loose name that is exactly that word points to it. A loose name also
    points to a label that holds all the name's words. The labels that cover
    the most words of the name win, and a label that the name holds whole
    wins a tie. So "Sample Gold" points to "Sample Gold" and not to "Sample",
    and "Pat Sample" points to the card "Pat Holder Sample" and not to the
    account "Sample".

    Args:
        words: The name words from ``_name_words``.
        candidates: The accounts and cards with their labels.
        loose: Also match a label that holds every word of the name.

    Returns:
        The closest candidates.
    """
    ranked: list[tuple[LabelRank, LabelCandidate]] = []
    for candidate in candidates:
        if not (label_words := _name_words(candidate.label)):
            continue
        if label_words <= words and (
            len(label_words) > 1 or (loose and label_words == words)
        ):
            ranked.append((LabelRank(covered=len(label_words), whole=True), candidate))
        elif loose and words <= label_words:
            ranked.append((LabelRank(covered=len(words), whole=False), candidate))
    if not ranked:
        return []
    best = max(rank for rank, _ in ranked)
    return [candidate for rank, candidate in ranked if rank == best]


async def _named_ids(session: AsyncSession, text: str, *, loose: bool) -> NamedIds:
    """Return the accounts and cards that a name from the model points to.

    The model often knows an account or a card only by the name that the user
    gives, and not by its id. Accounts and cards compete for the name. A card
    that the name points to always covers more of it than its own account,
    so the name then means the card.

    Args:
        session: The open database session.
        text: The name from the model, from a bank or search filter.
        loose: The name comes from the bank filter. It also points to an
            account or a card when the label holds every word of the name,
            such as "Gold" for "Sample Bank Gold". It also points to cards. A
            search is not loose: a merchant word such as "Swiggy" must not
            match a card named after it, and a person that the user paid must
            not match that person's add-on card.

    Returns:
        The ids of the accounts and cards that the name points to.
    """
    if not (words := _name_words(text)):
        return NamedIds(account_ids=[], card_ids=[])
    account_labels: dict[int, str] = {
        row.id: row.label
        for row in await session.execute(select(Account.id, Account.label))
    }
    candidates = [
        LabelCandidate(kind="account", id=account_id, label=label)
        for account_id, label in account_labels.items()
    ]
    if loose:
        for row in await session.execute(select(Card.id, Card.account_id, Card.label)):
            account_label = account_labels.get(row.account_id, "")
            # A card is named by its own label and its account's label, as in
            # "Pat's Sample Gold". The name must hold a word that only the
            # card's label has. If not, a card labelled "Sample Debit" would
            # stand for its "Sample Savings" account when the user names it.
            if words & (_name_words(row.label or "") - _name_words(account_label)):
                candidates.append(
                    LabelCandidate(
                        kind="card", id=row.id, label=f"{row.label} {account_label}"
                    )
                )
    closest = _closest_labels(words, candidates, loose=loose)
    return NamedIds(
        account_ids=[match.id for match in closest if match.kind == "account"],
        card_ids=[match.id for match in closest if match.kind == "card"],
    )


def _named_clause(column: ColumnElement[bool], named: NamedIds) -> ColumnElement[bool]:
    """Widen a filter to the rows of the accounts and cards that a name points to.

    Args:
        column: The filter on the row's own fields.
        named: The accounts and cards that the name points to.

    Returns:
        A clause that matches either.
    """
    return or_(
        column,
        Transaction.account_id.in_(named.account_ids),
        Transaction.card_id.in_(named.card_ids),
    )


async def _category_slug(session: AsyncSession, category: str) -> str:
    """Return the slug of the category that the model names.

    The model can give the label that the user sees, such as "Card bills",
    in place of the slug.

    Args:
        session: The open database session.
        category: A slug, a slug with spaces, or a label.

    Returns:
        The slug of the active category that fits. A name that fits no active
        category stays as it is, so the filter matches no row.
    """
    wanted = category.strip().lower()
    for slug in await get_active_slugs(session):
        if wanted in (slug, slug.replace("_", " "), label_for_slug(slug).lower()):
            return slug
    return category


async def report_filters(
    session: AsyncSession, call: ShowTransactions
) -> TransactionFilters | None:
    """Turn the filters of a transaction report into the page's filters.

    A bank filter that names accounts or cards becomes their ids, so the
    report and the page that its link opens list the same rows. A search is
    plain text here: the model names an account in the bank filter.

    Args:
        session: The open database session.
        call: The report call.

    Returns:
        The filters, or None when a date or the amount is not valid.
    """
    try:
        date_from = (
            datetime.date.fromisoformat(call.date_from) if call.date_from else None
        )
        date_to = datetime.date.fromisoformat(call.date_to) if call.date_to else None
        amount = Decimal(call.amount) if call.amount else None
    except ValueError, InvalidOperation:
        return None
    if (amount is not None and not amount.is_finite()) or (
        date_from and date_to and date_from > date_to
    ):
        return None
    account_ids = (call.account_id,) if call.account_id else ()
    card_ids = (call.card_id,) if call.card_id else ()
    bank = call.bank or None
    if bank and not account_ids and not card_ids:
        named = await _named_ids(session, bank, loose=True)
        if named.account_ids or named.card_ids:
            account_ids, card_ids = tuple(named.account_ids), tuple(named.card_ids)
            bank = None
    return TransactionFilters(
        bank=bank,
        account_ids=account_ids,
        card_ids=card_ids,
        direction=call.direction,
        date_from=date_from,
        date_to=date_to,
        category=await _category_slug(session, call.category)
        if call.category
        else None,
        excluded=call.excluded,
        q=call.search or None,
        amount=amount,
        source=call.source or None,
        review_status=call.review_status or None,
        reference=call.reference or None,
    )


async def list_accounts(session: AsyncSession) -> list[AssistantAccount]:
    """Return every account and its cards, with the ids and names.

    Args:
        session: The open database session.

    Returns:
        The accounts in id order. Configured names are masked in the labels.
    """
    name_tokens = get_redact_name_tokens()
    accounts = await session.scalars(select(Account).order_by(Account.id))
    return [
        AssistantAccount(
            id=account.id,
            bank=account.bank,
            label=_redacted(account.label, name_tokens) or "",
            type=account.type,
            active=account.active is not False,
            cards=[
                AssistantCard(
                    id=card.id,
                    label=_redacted(card.label, name_tokens),
                    active=card.active is not False,
                    is_primary=bool(card.is_primary),
                )
                for card in account.cards
            ],
        )
        for account in accounts
    ]


def _statement(**filters: object) -> Select:
    return select(
        Transaction.id,
        Transaction.bank,
        Transaction.direction,
        Transaction.amount,
        Transaction.currency,
        Transaction.transaction_date,
        Transaction.counterparty,
        Transaction.reference_number,
        Transaction.channel,
        Transaction.account_id,
        Transaction.card_id,
        Transaction.category,
        Transaction.category_method,
        Transaction.note,
        Transaction.exclude_from_cashflow,
        Transaction.voided,
        Transaction.category_confidence,
        Transaction.category_model,
        Transaction.review_status,
        Transaction.review_reason,
    ).where(*build_transaction_filter_clauses(**cast(Any, filters)))


async def get_transaction(
    session: AsyncSession,
    transaction_id: int,
) -> AssistantTransaction | None:
    """Read exactly one transaction by trusted database ID.

    Args:
        session: The open database session.
        transaction_id: The id of the row.

    Returns:
        The redacted row, or None when no row has this id.
    """
    rows = await list_transactions(session, transaction_ids=[transaction_id])
    return rows[0] if rows else None


async def list_transactions(
    session: AsyncSession,
    *,
    limit: int = MAX_CONTEXT_ROWS,
    offset: int = 0,
    account_id: int | None = None,
    card_id: int | None = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    direction: str | None = None,
    category: str | None = None,
    search: str | None = None,
    excluded: bool | None = None,
    transaction_ids: list[int] | None = None,
    amount: Decimal | None = None,
    bank: str | None = None,
    source: str | None = None,
    review_status: str | None = None,
    reference: str | None = None,
) -> list[AssistantTransaction]:
    """Return a bounded, stable page of allowlisted transaction projections.

    A bank filter also matches the accounts and cards that it names, and a
    search also matches an account whose whole label it holds. A category may
    be a label. Each row carries its review gate reason.

    Args:
        session: The open database session.
        limit: The most rows to return, at most ``MAX_CONTEXT_ROWS``.
        offset: The rows to skip.
        account_id: One account, by id.
        card_id: One card, by id.
        date_from: The first date.
        date_to: The last date.
        direction: ``debit`` or ``credit``.
        category: A category slug or label.
        search: Text to find in the row.
        excluded: True for rows out of the cashflow, False for rows in it.
        transaction_ids: Read these rows by id.
        amount: The exact amount.
        bank: A bank, or the name of an account or a card.
        source: The row source.
        review_status: The category review status.
        reference: A payment reference.

    Returns:
        The newest rows first.

    Raises:
        ValueError: The limit is not positive or the offset is negative.
    """
    if limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset cannot be negative")
    limit = min(limit, MAX_CONTEXT_ROWS)
    clauses: list[ColumnElement[bool]] = []
    if search and search.strip():
        clauses.append(
            _named_clause(
                _search_clause(search),
                await _named_ids(session, search, loose=False),
            )
        )
    if bank is not None:
        clauses.append(
            _named_clause(
                func.lower(Transaction.bank) == bank.strip().lower(),
                await _named_ids(session, bank, loose=True),
            )
        )
    if category is not None:
        category = await _category_slug(session, category)
    result = await session.execute(
        _statement(
            account_id=account_id,
            card_id=card_id,
            date_from=date_from,
            date_to=date_to,
            direction=direction,
            category=category,
            excluded=excluded,
            transaction_ids=transaction_ids,
            amount=amount,
            source=source,
            review_status=review_status,
            reference=reference,
        )
        .where(*clauses)
        .order_by(
            Transaction.transaction_date.desc().nullslast(), Transaction.id.desc()
        )
        .offset(offset)
        .limit(limit)
    )
    if not (rows := list(result)):
        return []
    # A later active decision replaces an earlier one in the dict.
    gate_reasons = {
        row.transaction_id: row.gate_reason
        for row in await session.execute(
            select(
                CategoryReviewDecision.transaction_id,
                CategoryReviewDecision.gate_reason,
            )
            .where(
                CategoryReviewDecision.transaction_id.in_([row.id for row in rows]),
                CategoryReviewDecision.status == "active",
            )
            .order_by(CategoryReviewDecision.id)
        )
    }
    name_tokens = get_redact_name_tokens()
    return [
        _projection(
            row, name_tokens=name_tokens, review_gate_reason=gate_reasons.get(row.id)
        )
        for row in rows
    ]
