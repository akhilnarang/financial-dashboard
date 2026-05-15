"""Regression test for the MissingGreenlet error that happened when the fetcher
built Telegram notification payloads by reading ``txn.account`` / ``txn.card``
on a freshly-inserted Transaction row.

Fresh ORM instances don't have relationships populated (``lazy="joined"`` only
kicks in on query-loads, not on ``Model(...)`` construction), so the attribute
access triggered an implicit lazy load. That lazy load attempted sync IO from
async code and raised ``sqlalchemy.exc.MissingGreenlet``.

The fix is to fetch the related rows explicitly with ``await session.get(...)``
using the ``account_id`` / ``card_id`` set by the linker. These tests pin both
halves in place: (1) the old pattern really does blow up, and (2) the new
pattern produces the expected label.
"""

from decimal import Decimal

import pytest
from sqlalchemy.exc import MissingGreenlet
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Account, Base, Card, Transaction
from financial_dashboard.services.linker import build_link_context, link_transaction
from financial_dashboard.services.telegram import build_account_label


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


async def _seed_account_with_card(session: AsyncSession) -> tuple[Account, Card]:
    account = Account(
        bank="HDFC", label="HDFC Credit Card", type="credit_card", account_number=None
    )
    session.add(account)
    await session.flush()
    card = Card(account_id=account.id, card_mask="XXXX XXXX XXXX 1234", label="self")
    session.add(card)
    await session.flush()
    return account, card


def _new_transaction(email_id: int | None = None) -> Transaction:
    return Transaction(
        email_id=email_id,
        bank="HDFC",
        email_type="hdfc_cc_transaction",
        direction="debit",
        amount=Decimal("100.00"),
        card_mask="XXXX XXXX XXXX 1234",
    )


@pytest.mark.anyio
async def test_fresh_transaction_lazy_load_raises_greenlet_error(session):
    """The original bug: touching .account on a fresh row triggers a lazy load
    that isn't allowed from plain async code."""
    await _seed_account_with_card(session)
    link_ctx = await build_link_context(session)

    txn = _new_transaction()
    session.add(txn)
    await session.flush()
    link_transaction(link_ctx, txn)
    await session.flush()

    assert txn.account_id is not None
    assert txn.card_id is not None

    with pytest.raises(MissingGreenlet):
        _ = txn.account


@pytest.mark.anyio
async def test_notification_label_uses_session_get(session):
    """The fix: fetch related rows via ``await session.get(...)`` and pass
    them to ``build_account_label``."""
    account, card = await _seed_account_with_card(session)
    link_ctx = await build_link_context(session)

    txn = _new_transaction()
    session.add(txn)
    await session.flush()
    link_transaction(link_ctx, txn)
    await session.flush()

    account_obj = await session.get(Account, txn.account_id) if txn.account_id else None
    card_obj = await session.get(Card, txn.card_id) if txn.card_id else None

    assert account_obj is account
    assert card_obj is card

    label = build_account_label(account_obj, card_obj)
    assert label == f"{account.label} - {card.label}"


@pytest.mark.anyio
async def test_notification_label_handles_unlinked_transaction(session):
    """When the linker can't match anything, account_id/card_id stay None and
    build_account_label should still produce something safely."""
    link_ctx = await build_link_context(session)

    txn = Transaction(
        bank="HDFC",
        email_type="hdfc_cc_transaction",
        direction="debit",
        amount=Decimal("50.00"),
    )
    session.add(txn)
    await session.flush()
    link_transaction(link_ctx, txn)
    await session.flush()

    assert txn.account_id is None
    assert txn.card_id is None

    account_obj = await session.get(Account, txn.account_id) if txn.account_id else None
    card_obj = await session.get(Card, txn.card_id) if txn.card_id else None

    assert account_obj is None
    assert card_obj is None
    assert build_account_label(account_obj, card_obj) == ""
