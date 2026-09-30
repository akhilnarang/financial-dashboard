"""Tests for ``ensure_default_primary_card`` (services.accounts).

Card-type accounts get a primary card seeded from their account_number when
they have no cards yet, but the helper must be a no-op for non-card types,
accounts without an account_number, and accounts that already have any card.
"""

import pytest
from sqlalchemy import select

from financial_dashboard.db import Account, Card
from financial_dashboard.services.accounts import ensure_default_primary_card

pytestmark = pytest.mark.anyio


async def _account(session, **kwargs) -> Account:
    account = Account(bank="examplebank", label="Example", **kwargs)
    session.add(account)
    await session.flush()
    return account


async def _cards(session, account: Account) -> list[Card]:
    return list(
        (await session.scalars(select(Card).where(Card.account_id == account.id)))
    )


async def test_seeds_primary_card_for_new_credit_card_account(session):
    account = await _account(session, type="credit_card", account_number="1234")

    assert await ensure_default_primary_card(session, account) is not None
    await session.flush()

    cards = await _cards(session, account)
    assert len(cards) == 1
    assert cards[0].card_mask == "1234"
    assert cards[0].is_primary is True
    assert cards[0].label == "self"
    assert cards[0].active is True


async def test_skips_when_account_already_has_cards(session):
    account = await _account(session, type="credit_card", account_number="5678")
    session.add(
        Card(account_id=account.id, card_mask="XX5678", label="Other", is_primary=True)
    )
    await session.flush()

    assert await ensure_default_primary_card(session, account) is None
    await session.flush()

    assert [c.card_mask for c in await _cards(session, account)] == ["XX5678"]


async def test_skips_non_card_type_and_missing_account_number(session):
    bank = await _account(session, type="bank_account", account_number="000111222")
    no_number = await _account(session, type="credit_card", account_number=None)

    assert await ensure_default_primary_card(session, bank) is None
    assert await ensure_default_primary_card(session, no_number) is None
    await session.flush()

    assert await _cards(session, bank) == []
    assert await _cards(session, no_number) == []
