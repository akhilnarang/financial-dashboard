import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Category, Transaction

pytestmark = pytest.mark.anyio


async def test_transaction_has_categorization_columns(session: AsyncSession):
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("10"),
        category_method="rule",
        category_confidence=0.9,
        category_model="rules-v1",
        category_input_hash="abc",
        category_vocab_version=1,
        categorized_at=datetime.datetime.now(datetime.UTC),
        review_status="pending",
        review_reason="why",
        last_notified_at=None,
        notify_attempts=0,
    )
    session.add(txn)
    await session.flush()
    assert txn.id is not None


async def test_category_model_roundtrip(session: AsyncSession):
    session.add(Category(slug="groceries", active=True))
    await session.flush()
    rows = (await session.execute(select(Category.slug))).scalars().all()
    assert "groceries" in rows
