"""Allowlisted, bounded transaction reads for the conversational assistant.

The assistant has a narrower read surface than the general transaction API.  It
can identify and explain rows, but it cannot request arbitrary columns or SQL.
The predicates below intentionally mirror the useful filters on the
``/transactions`` page: account, date, direction, category, free-text search,
and cashflow exclusion state.
"""

import datetime
from decimal import Decimal
from typing import Any, NamedTuple, cast

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import CategoryReviewDecision, Transaction
from financial_dashboard.services.transaction_reads import (
    build_transaction_filter_clauses,
)

MAX_CONTEXT_ROWS = 20
MAX_FIELD_LENGTH = 2_000


class AssistantTransaction(NamedTuple):
    """Redacted transaction projection safe to place in a model context."""

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
    category_confidence: float | None
    category_model: str | None
    review_status: str | None
    review_reason: str | None
    review_gate_reason: str | None


def _bounded(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:MAX_FIELD_LENGTH]


def _projection(
    row: Any, *, review_gate_reason: str | None = None
) -> AssistantTransaction:
    """Convert a SQL row using only the explicit assistant allowlist."""
    return AssistantTransaction(
        id=row.id,
        bank=row.bank,
        direction=row.direction,
        amount=row.amount,
        currency=row.currency,
        transaction_date=row.transaction_date,
        counterparty=_bounded(row.counterparty),
        reference_number=_bounded(row.reference_number),
        channel=_bounded(row.channel),
        account_id=row.account_id,
        card_id=row.card_id,
        category=_bounded(row.category),
        category_method=_bounded(row.category_method),
        note=_bounded(row.note),
        exclude_from_cashflow=bool(row.exclude_from_cashflow),
        category_confidence=row.category_confidence,
        category_model=_bounded(row.category_model),
        review_status=_bounded(row.review_status),
        review_reason=_bounded(row.review_reason),
        review_gate_reason=_bounded(review_gate_reason),
    )


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
        Transaction.category_confidence,
        Transaction.category_model,
        Transaction.review_status,
        Transaction.review_reason,
    ).where(*build_transaction_filter_clauses(**cast(Any, filters)))


async def get_transaction(
    session: AsyncSession,
    transaction_id: int,
) -> AssistantTransaction | None:
    """Read exactly one transaction by trusted database ID."""
    result = await session.execute(_statement(transaction_id=transaction_id))
    row = result.one_or_none()
    if row is None:
        return None
    gate_reason = await session.scalar(
        select(CategoryReviewDecision.gate_reason)
        .where(
            CategoryReviewDecision.transaction_id == transaction_id,
            CategoryReviewDecision.status == "active",
        )
        .order_by(CategoryReviewDecision.id.desc())
        .limit(1)
    )
    return _projection(row, review_gate_reason=gate_reason)


async def list_transactions(
    session: AsyncSession,
    *,
    limit: int = MAX_CONTEXT_ROWS,
    offset: int = 0,
    account_id: int | None = None,
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
    """Return a bounded, stable page of allowlisted transaction projections."""
    if limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset cannot be negative")
    limit = min(limit, MAX_CONTEXT_ROWS)
    result = await session.execute(
        _statement(
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            direction=direction,
            category=category,
            search=search,
            excluded=excluded,
            transaction_ids=transaction_ids,
            amount=amount,
            bank=bank,
            source=source,
            review_status=review_status,
            reference=reference,
        )
        .order_by(
            Transaction.transaction_date.desc().nullslast(), Transaction.id.desc()
        )
        .offset(offset)
        .limit(limit)
    )
    return [_projection(row) for row in result]
