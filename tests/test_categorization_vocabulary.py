import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.services.categorization.vocabulary import (
    SEED_CATEGORIES,
    canonicalize_slug,
    ensure_category,
    get_active_slugs,
    is_valid_slug,
)

pytestmark = pytest.mark.anyio


def test_canonicalize_and_validate():
    assert canonicalize_slug("Food & Dining") == "food_dining"
    assert is_valid_slug("groceries")
    assert not is_valid_slug("1bad")
    assert not is_valid_slug("Bad Slug")


def test_seed_has_core_slugs():
    slugs = set(SEED_CATEGORIES)
    for required in (
        "salary",
        "self_transfer",
        "credit_card_payment",
        "groceries",
        "unknown",
    ):
        assert required in slugs


async def test_ensure_category_inserts_once(session: AsyncSession):
    created = await ensure_category(session, "groceries")
    again = await ensure_category(session, "groceries")
    assert created is True
    assert again is False
    assert "groceries" in await get_active_slugs(session)
