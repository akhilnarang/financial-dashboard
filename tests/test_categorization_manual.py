# tests/test_categorization_manual.py
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.manual import (
    assign_category_no_commit,
    assign_category_manual,
    CategoryDirectionPolicy,
    resolve_assistant_category_slug,
)
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


async def test_manual_assign_rejects_typo_even_when_existing_category_is_close(
    session: AsyncSession,
):
    from financial_dashboard.services.categorization.vocabulary import ensure_category

    await ensure_category(session, "groceries")
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    ok, slug = await assign_category_manual(session, txn.id, "grocieis")

    assert ok is False
    assert slug is None
    assert txn.category is None


async def test_assistant_category_resolver_only_uses_active_categories(
    session: AsyncSession,
):
    from financial_dashboard.db.models import Category
    from financial_dashboard.services.categorization.vocabulary import ensure_category

    await ensure_category(session, "groceries")
    session.add(Category(slug="grocieis", active=False))
    await session.flush()

    slug, corrected = await resolve_assistant_category_slug(session, "grocieis")

    assert slug == "groceries"
    assert corrected is True


async def test_manual_assign_preserves_inactive_category_compatibility(
    session: AsyncSession,
):
    from financial_dashboard.db.models import Category

    session.add(Category(slug="historical", active=False))
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    ok, slug = await assign_category_manual(session, txn.id, "historical")

    assert ok is True
    assert slug == "historical"
    assert txn.category == "historical"


async def test_assistant_assignment_rejects_inactive_category(
    session: AsyncSession,
):
    from financial_dashboard.db.models import Category

    session.add(Category(slug="historical", active=False))
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("5")
    )
    session.add(txn)
    await session.flush()

    ok, slug = await assign_category_no_commit(
        session,
        txn.id,
        "historical",
        direction_policy=CategoryDirectionPolicy.INFERRED_STRICT,
    )

    assert ok is False
    assert slug is None
    assert txn.category is None


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
