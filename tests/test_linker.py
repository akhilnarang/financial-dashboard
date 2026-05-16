"""Tests for the linker's mask suffix-matching behaviour.

Covers the three cases that surfaced in production:
- Short masks (3 digits, as ICICI savings SMSes emit) should resolve
  against accounts whose ``account_number`` is the full bank account
  number.
- Bare last-4 masks (as HDFC debit-card SMSes emit) should resolve
  against cards whose stored ``card_mask`` is also a bare last-4.
- Ambiguous matches (two accounts sharing the same trailing digits)
  must not link — they should be left NULL with a warning.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Account, Base, Card, Transaction
from financial_dashboard.services.linker import build_link_context, link_transaction


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


def _txn(**overrides) -> Transaction:
    base = dict(
        bank="icici",
        email_type="t",
        direction="debit",
        amount=Decimal("10"),
        currency="INR",
    )
    base.update(overrides)
    return Transaction(**base)


@pytest.mark.anyio
async def test_short_account_mask_resolves_against_full_account_number(session):
    """A 3-digit mask (e.g. XX678) must resolve against a stored full
    account_number by suffix-matching the trailing digits."""
    acct = Account(
        bank="icici",
        type="bank_account",
        label="ICICI Savings",
        account_number="000000005678",
    )
    session.add(acct)
    await session.flush()
    ctx = await build_link_context(session)

    txn = _txn(account_mask="XX678")
    assert link_transaction(ctx, txn) is True
    assert txn.account_id == acct.id


@pytest.mark.anyio
async def test_bare_last4_card_mask_resolves(session):
    """A bare last-4 card_mask must resolve to a Card row whose stored
    card_mask is also the bare last-4."""
    acct = Account(
        bank="hdfc",
        type="bank_account",
        label="HDFC Savings",
        account_number="HDFC1",
    )
    session.add(acct)
    await session.flush()
    card = Card(account_id=acct.id, card_mask="7777", label="HDFC Debit")
    session.add(card)
    await session.flush()
    ctx = await build_link_context(session)

    txn = _txn(bank="hdfc", card_mask="7777")
    assert link_transaction(ctx, txn) is True
    assert txn.account_id == acct.id
    assert txn.card_id == card.id


@pytest.mark.anyio
async def test_ambiguous_short_mask_refuses_to_link(session):
    """If two accounts in the same bank both suffix-match the incoming
    mask digits, the linker must NOT guess. Leaves account_id NULL."""
    a1 = Account(
        bank="icici",
        type="bank_account",
        label="ICICI Savings 1",
        account_number="11115678",
    )
    a2 = Account(
        bank="icici",
        type="bank_account",
        label="ICICI Savings 2",
        account_number="22225678",
    )
    session.add_all([a1, a2])
    await session.flush()
    ctx = await build_link_context(session)

    txn = _txn(account_mask="XX678")
    assert link_transaction(ctx, txn) is False
    assert txn.account_id is None


@pytest.mark.anyio
async def test_below_minimum_digits_is_not_matched(session):
    """A mask with fewer than 3 digits is rejected even if it would
    technically suffix-match."""
    acct = Account(
        bank="icici", type="bank_account", label="X", account_number="5678"
    )
    session.add(acct)
    await session.flush()
    ctx = await build_link_context(session)

    txn = _txn(account_mask="X8")  # only 1 digit
    assert link_transaction(ctx, txn) is False
    assert txn.account_id is None


@pytest.mark.anyio
async def test_cross_bank_mask_collision_is_not_linked(session):
    """An HDFC card and an ICICI card sharing the same trailing digits
    must not be confused — bank scoping prevents the cross-bank match."""
    hdfc = Account(
        bank="hdfc", type="bank_account", label="HDFC", account_number="HDFC1"
    )
    icici_acct = Account(
        bank="icici", type="credit_card", label="ICICI CC", account_number="7777"
    )
    session.add_all([hdfc, icici_acct])
    await session.flush()
    hdfc_card = Card(account_id=hdfc.id, card_mask="7777", label="HDFC Debit")
    session.add(hdfc_card)
    await session.flush()
    ctx = await build_link_context(session)

    # ICICI transaction must NOT pick up the HDFC card.
    txn = _txn(bank="icici", card_mask="7777")
    assert link_transaction(ctx, txn) is True
    assert txn.account_id == icici_acct.id
    assert txn.card_id is None  # matched via account, not card

    # HDFC transaction must pick the HDFC card.
    txn2 = _txn(bank="hdfc", card_mask="7777")
    assert link_transaction(ctx, txn2) is True
    assert txn2.account_id == hdfc.id
    assert txn2.card_id == hdfc_card.id


@pytest.mark.anyio
async def test_bank_only_fallback_uses_email_type_for_disambiguation(session):
    """A bank with both a bank_account and a credit_card account must
    still resolve correctly for maskless SMSes — the linker uses the
    email_type's '_cc_' / '_account_' marker to pick the right one."""
    savings = Account(
        bank="slice",
        type="bank_account",
        label="Slice Savings",
        account_number="000000001111",
    )
    cc = Account(
        bank="slice",
        type="credit_card",
        label="Slice CC",
        account_number="2222",
    )
    session.add_all([savings, cc])
    await session.flush()
    ctx = await build_link_context(session)

    # A CC bill-paid SMS carries no mask but has '_cc_' in email_type.
    cc_txn = _txn(
        bank="slice",
        email_type="slice_cc_bill_paid_alert",
        direction="credit",
    )
    assert link_transaction(ctx, cc_txn) is True
    assert cc_txn.account_id == cc.id

    # A savings UPI alert carries no mask but has '_account_' in email_type.
    savings_txn = _txn(
        bank="slice",
        email_type="slice_account_upi_credit_alert",
        direction="credit",
    )
    assert link_transaction(ctx, savings_txn) is True
    assert savings_txn.account_id == savings.id


@pytest.mark.anyio
async def test_bank_only_fallback_refuses_when_two_candidates_of_same_type(session):
    """When the email_type-narrowed candidate set still has >1 accounts
    (e.g. two CCs under the same bank), the linker must not guess."""
    cc1 = Account(
        bank="hdfc", type="credit_card", label="HDFC Diners", account_number="1111"
    )
    cc2 = Account(
        bank="hdfc", type="credit_card", label="HDFC Millenia", account_number="2222"
    )
    session.add_all([cc1, cc2])
    await session.flush()
    ctx = await build_link_context(session)

    txn = _txn(bank="hdfc", email_type="hdfc_cc_payment_received_alert")
    assert link_transaction(ctx, txn) is False
    assert txn.account_id is None


@pytest.mark.anyio
async def test_bank_only_fallback_falls_back_to_unfiltered_when_email_type_uninformative(
    session,
):
    """When the email_type carries no '_cc_' / '_account_' marker, the
    linker keeps the legacy behaviour: link only when the bank has
    exactly one account total."""
    acct = Account(
        bank="x-bank", type="bank_account", label="X", account_number="0001"
    )
    session.add(acct)
    await session.flush()
    ctx = await build_link_context(session)

    txn = _txn(bank="x-bank", email_type="x_bank_misc_alert")
    assert link_transaction(ctx, txn) is True
    assert txn.account_id == acct.id
