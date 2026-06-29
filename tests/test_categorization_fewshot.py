from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.fewshot import get_similar_examples

pytestmark = pytest.mark.anyio


async def _add(session, **kw):
    base = dict(bank="testbank", email_type="x", amount=Decimal("10"))
    base.update(kw)
    session.add(Transaction(**base))
    await session.flush()


async def test_returns_same_direction_categorized_matches(session: AsyncSession):
    await _add(
        session,
        direction="debit",
        counterparty="ACME STORE",
        category="groceries",
        category_method="manual",
    )
    await _add(
        session,
        direction="credit",
        counterparty="ACME STORE",
        category="refund",
        category_method="manual",
    )  # wrong direction
    await _add(
        session,
        direction="debit",
        counterparty="ACME STORE",
        category=None,
        category_method=None,
    )  # uncategorized

    out = await get_similar_examples(
        session, counterparty="acmestoremumbai", direction="debit", limit=5
    )
    assert len(out) == 1
    assert out[0].category == "groceries"
    assert out[0].direction == "debit"


async def test_empty_when_no_matches(session: AsyncSession):
    out = await get_similar_examples(session, counterparty="zzz", direction="debit")
    assert out == []
