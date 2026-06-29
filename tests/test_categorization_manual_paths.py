# tests/test_categorization_manual_paths.py
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.transactions import update_transaction_category

pytestmark = pytest.mark.anyio


async def test_update_category_writes_provenance(session: AsyncSession):
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    ok, slug = await update_transaction_category(session, txn.id, "Groceries")
    assert ok is True
    assert slug == "groceries"
    assert txn.category == "groceries"
    assert txn.category_method == "manual"


async def test_update_category_missing_txn_returns_false(session: AsyncSession):
    ok, slug = await update_transaction_category(session, 9999, "Groceries")
    assert ok is False
    assert slug is None


async def test_update_category_invalid_slug_raises(session: AsyncSession):
    # txn exists but the category can't form a valid slug → ValueError (HTTP 400),
    # distinct from a missing transaction (which returns False → HTTP 404).
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    with pytest.raises(ValueError):
        await update_transaction_category(session, txn.id, "123")
