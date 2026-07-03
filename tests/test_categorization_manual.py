# tests/test_categorization_manual.py
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.manual import assign_category_manual
from financial_dashboard.services.categorization.vocabulary import get_active_slugs

pytestmark = pytest.mark.anyio


async def test_manual_assign_existing_slug(session: AsyncSession):
    from financial_dashboard.services.categorization.vocabulary import ensure_category

    await ensure_category(session, "groceries")
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    ok, slug = await assign_category_manual(session, txn.id, "Groceries")
    assert ok is True and slug == "groceries"
    assert txn.category_method == "manual"
    assert txn.category_confidence == 1.0
    assert txn.review_status == "resolved"


async def test_manual_assign_unknown_slug_rejected_by_default(session: AsyncSession):
    """A slug not in the vocabulary is rejected (typo guard) unless create=True."""
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    ok, slug = await assign_category_manual(session, txn.id, "Goceries")
    assert ok is False and slug is None
    assert "goceries" not in await get_active_slugs(session)


async def test_manual_assign_creates_new_slug_with_opt_in(session: AsyncSession):
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    ok, slug = await assign_category_manual(session, txn.id, "Pet Care", create=True)
    assert ok is True and slug == "pet_care"
    assert "pet_care" in await get_active_slugs(session)


async def test_manual_assign_unknown_txn(session: AsyncSession):
    ok, slug = await assign_category_manual(session, 9999, "groceries")
    assert ok is False and slug is None
