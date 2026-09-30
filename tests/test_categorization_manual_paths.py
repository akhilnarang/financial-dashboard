# tests/test_categorization_manual_paths.py
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.vocabulary import ensure_category
from financial_dashboard.services.transactions import update_transaction_category

pytestmark = pytest.mark.anyio


async def test_update_category_writes_provenance(session: AsyncSession):
    await ensure_category(session, "groceries")
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


async def test_update_category_rejects_missing_txn_and_invalid_slug(
    session: AsyncSession,
):
    # A missing transaction returns False (HTTP 404).
    # An invalid slug raises ValueError (HTTP 400).
    assert await update_transaction_category(session, 9999, "Groceries") == (
        False,
        None,
    )
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    with pytest.raises(ValueError):
        await update_transaction_category(session, txn.id, "123")
