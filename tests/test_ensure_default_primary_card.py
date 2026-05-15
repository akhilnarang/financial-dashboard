"""Tests for ``ensure_default_primary_card`` (services.accounts).

Card-type accounts get a primary card seeded from their account_number when
they have no cards yet, but the helper must be a no-op for non-card types,
accounts without an account_number, and accounts that already have any card.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Account, Base, Card
from financial_dashboard.services.accounts import ensure_default_primary_card


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _make_account(maker, **kwargs) -> Account:
    async with maker() as session:
        account = Account(**kwargs)
        session.add(account)
        await session.commit()
        await session.refresh(account)
        return account


@pytest.mark.anyio
async def test_seeds_primary_card_for_new_credit_card_account(session_factory):
    account = await _make_account(
        session_factory,
        bank="hdfc",
        label="HDFC Swiggy CC",
        type="credit_card",
        account_number="0264",
    )
    async with session_factory() as session:
        account = await session.get(Account, account.id)
        card = await ensure_default_primary_card(session, account)
        await session.commit()

    assert card is not None
    async with session_factory() as session:
        cards = (
            (await session.execute(select(Card).where(Card.account_id == account.id)))
            .scalars()
            .all()
        )
    assert len(cards) == 1
    assert cards[0].card_mask == "0264"
    assert cards[0].is_primary is True
    assert cards[0].label == "self"
    assert cards[0].active is True


@pytest.mark.anyio
async def test_skips_when_account_already_has_cards(session_factory):
    account = await _make_account(
        session_factory,
        bank="icici",
        label="ICICI RubyX",
        type="credit_card",
        account_number="1003",
    )
    async with session_factory() as session:
        session.add(
            Card(
                account_id=account.id, card_mask="XX1003", label="Amex", is_primary=True
            )
        )
        await session.commit()

    async with session_factory() as session:
        account = await session.get(Account, account.id)
        result = await ensure_default_primary_card(session, account)
        await session.commit()

    assert result is None
    async with session_factory() as session:
        cards = (
            (await session.execute(select(Card).where(Card.account_id == account.id)))
            .scalars()
            .all()
        )
    assert len(cards) == 1
    assert cards[0].card_mask == "XX1003"


@pytest.mark.anyio
async def test_skips_bank_account_type(session_factory):
    account = await _make_account(
        session_factory,
        bank="hdfc",
        label="HDFC Savings",
        type="bank_account",
        account_number="00391000107703",
    )
    async with session_factory() as session:
        account = await session.get(Account, account.id)
        result = await ensure_default_primary_card(session, account)
        await session.commit()

    assert result is None
    async with session_factory() as session:
        cards = (
            (await session.execute(select(Card).where(Card.account_id == account.id)))
            .scalars()
            .all()
        )
    assert cards == []


@pytest.mark.anyio
async def test_skips_when_account_number_missing(session_factory):
    account = await _make_account(
        session_factory,
        bank="hdfc",
        label="HDFC CC (no number)",
        type="credit_card",
        account_number=None,
    )
    async with session_factory() as session:
        account = await session.get(Account, account.id)
        result = await ensure_default_primary_card(session, account)
        await session.commit()

    assert result is None
    async with session_factory() as session:
        cards = (
            (await session.execute(select(Card).where(Card.account_id == account.id)))
            .scalars()
            .all()
        )
    assert cards == []
