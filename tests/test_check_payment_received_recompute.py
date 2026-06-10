"""Self-healing recompute semantics for ``check_payment_received`` and the
reusable ``recompute_cc_payment_*`` helpers.

The bug guarded against: ``check_payment_received`` used to be a blind
accumulator (``new_paid = (payment_paid_amount or 0) + amount``), so the same
payment arriving via two channels / a double-tapped Telegram button / a
force-parsed defer pair over-counted ``payment_paid_amount`` and understated
``cc_outstanding``. The rewrite RECOMPUTES the paid total from the SUM of
qualifying bill-payment credits in the cycle, making it idempotent and
self-healing on delete.

All values here are fully synthetic.
"""

import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.reminders as reminders_module
from financial_dashboard.db import (
    Account,
    Base,
    BalanceSnapshot,
    PaymentStatus,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import SnapshotCategory
from financial_dashboard.services.reminders import (
    check_payment_received,
    recompute_cc_payment_for_account,
    recompute_cc_payment_state,
)

pytestmark = pytest.mark.anyio

CYCLE_CREATED = dt.datetime(2026, 5, 10, 8, 0, tzinfo=dt.UTC)


@pytest.fixture
async def session_maker(monkeypatch):
    """In-memory aiosqlite session-maker, also installed as the global
    ``async_session`` that ``check_payment_received`` opens for itself."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(reminders_module, "async_session", maker)
    yield maker
    await engine.dispose()


async def _cc_account(session) -> Account:
    account = Account(
        bank="Test Bank",
        label="Test CC",
        type="credit_card",
        active=True,
    )
    session.add(account)
    await session.flush()
    return account


async def _upload(session, account, total_due="10000.00") -> StatementUpload:
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="cc.pdf",
        file_path="/tmp/cc.pdf",
        status="parsed",
        due_date="25/06/2026",
        total_amount_due=total_due,
        payment_status=PaymentStatus.UNPAID,
        payment_paid_amount=Decimal("0"),
        created_at=CYCLE_CREATED,
    )
    session.add(upload)
    await session.flush()
    return upload


def _credit(
    account,
    amount,
    *,
    email_type="testbank_cc_payment_alert",
    txn_date=dt.date(2026, 5, 12),
    created_at=None,
) -> Transaction:
    txn = Transaction(
        account_id=account.id,
        bank="Test Bank",
        email_type=email_type,
        direction="credit",
        amount=Decimal(str(amount)),
        transaction_date=txn_date,
    )
    if created_at is not None:
        txn.created_at = created_at
    return txn


async def test_idempotent_single_payment(session_maker):
    """Firing the check twice for one payment credit yields the single
    amount, not 2x."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account)
        s.add(_credit(account, "1000.00"))
        await s.commit()
        account_id = account.id

    assert await check_payment_received(1, account_id, Decimal("1000.00")) is False
    assert await check_payment_received(1, account_id, Decimal("1000.00")) is False

    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        assert upload.payment_paid_amount == Decimal("1000.00")
        assert upload.payment_status == PaymentStatus.PARTIALLY_PAID


async def test_two_real_rows_sum_both_but_no_inflation_on_refire(session_maker):
    """Two distinct payment rows (e.g. SMS + email, same amount) are both
    real rows in the cycle, so the recomputed sum counts BOTH (row-level
    dedup is a separate concern). Re-firing the check does NOT inflate the
    total beyond the actual row sum."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account)
        s.add(_credit(account, "1000.00"))
        s.add(_credit(account, "1000.00"))
        await s.commit()
        account_id = account.id

    await check_payment_received(1, account_id, Decimal("1000.00"))
    await check_payment_received(2, account_id, Decimal("1000.00"))
    await check_payment_received(2, account_id, Decimal("1000.00"))

    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        assert upload.payment_paid_amount == Decimal("2000.00")


async def test_refund_credit_excluded(session_maker):
    """A refund/reversal credit in the cycle is NOT summed into the paid
    amount."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account)
        s.add(_credit(account, "1000.00"))
        s.add(_credit(account, "500.00", email_type="testbank_cc_reversal"))
        s.add(_credit(account, "300.00", email_type="testbank_cc_refund_alert"))
        await s.commit()
        account_id = account.id

    await check_payment_received(1, account_id, Decimal("1000.00"))

    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        assert upload.payment_paid_amount == Decimal("1000.00")


async def test_cycle_scope_excludes_before_includes_on_or_after(session_maker):
    """A payment dated BEFORE the cycle's created_at date is excluded; one on
    the same calendar day is included; one after is included."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account)
        # Before cycle start (created_at date is 2026-05-10).
        s.add(_credit(account, "111.00", txn_date=dt.date(2026, 5, 9)))
        # Same calendar day as created_at.
        s.add(_credit(account, "1000.00", txn_date=dt.date(2026, 5, 10)))
        # After cycle start.
        s.add(_credit(account, "2000.00", txn_date=dt.date(2026, 5, 20)))
        await s.commit()
        account_id = account.id

    await check_payment_received(1, account_id, Decimal("1000.00"))

    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        assert upload.payment_paid_amount == Decimal("3000.00")


async def test_null_transaction_date_included(session_maker):
    """A payment credit with NULL transaction_date is counted (not dropped)."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account)
        s.add(_credit(account, "1000.00", txn_date=None))
        await s.commit()
        account_id = account.id

    await check_payment_received(1, account_id, Decimal("1000.00"))

    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        assert upload.payment_paid_amount == Decimal("1000.00")


async def test_null_date_row_from_prior_cycle_does_not_leak(session_maker):
    """A NULL transaction_date payment whose own created_at predates the
    cycle must NOT be summed in — otherwise one old NULL-date row would be
    counted into this cycle and every later one."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account, total_due="10000.00")
        # NULL-date payment created well before this statement's cycle start.
        s.add(
            _credit(
                account,
                "5000.00",
                txn_date=None,
                created_at=dt.datetime(2026, 4, 1, tzinfo=dt.UTC),
            )
        )
        # A real in-cycle payment.
        s.add(_credit(account, "1000.00", txn_date=dt.date(2026, 5, 12)))
        await s.commit()
        account_id = account.id

    await check_payment_received(1, account_id, Decimal("1000.00"))

    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        # Only the in-cycle 1000 counts; the stale NULL-date 5000 is excluded.
        assert upload.payment_paid_amount == Decimal("1000.00")


async def test_paid_at_preserved_on_repeat_recompute(session):
    """Recomputing a cycle that stays PAID must not overwrite the original
    payment_paid_at with a fresh 'now'. Uses the account-level helper, which
    (unlike the credit path) reconsiders already-PAID cycles."""
    account = await _cc_account(session)
    await _upload(session, account, total_due="5000.00")
    session.add(_credit(account, "5000.00"))
    await session.flush()

    paid = await recompute_cc_payment_for_account(session, account.id)
    assert paid == Decimal("5000.00")
    upload = (await session.execute(select(StatementUpload))).scalar_one()
    assert upload.payment_status == PaymentStatus.PAID
    first_paid_at = upload.payment_paid_at
    assert first_paid_at is not None

    # Re-run the recompute: still fully paid, timestamp must be unchanged.
    await recompute_cc_payment_for_account(session, account.id)
    upload = (await session.execute(select(StatementUpload))).scalar_one()
    assert upload.payment_status == PaymentStatus.PAID
    assert upload.payment_paid_at == first_paid_at


async def _enable_telegram(monkeypatch):
    """Enable telegram via the in-memory settings cache so the
    notification gate in ``check_payment_received`` passes."""
    from financial_dashboard.services import settings as settings_mod

    settings_mod._cache.update(
        {
            "telegram.enabled": "true",
            "telegram.bot_token": "synthetic-token",
            "telegram.chat_id": "123456",
            "telegram.notify_payment_received": "true",
        }
    )


async def test_notification_fires_after_commit_with_persisted_state(
    session_maker, monkeypatch
):
    """The payment-received notification must fire only AFTER commit: when the
    stub runs, the committed state is already durable in the DB."""
    await _enable_telegram(monkeypatch)

    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account, total_due="10000.00")
        s.add(_credit(account, "4000.00"))
        await s.commit()
        account_id = account.id

    observed: dict[str, object] = {}

    async def _stub(label, bank, credit, total_paid, due, chat_id):
        # Read the DB in a brand-new session: if commit already happened, the
        # recomputed paid amount is visible here.
        async with session_maker() as s:
            upload = (await s.execute(select(StatementUpload))).scalar_one()
            observed["persisted_paid"] = upload.payment_paid_amount
        observed["args"] = (label, bank, credit, total_paid, due, chat_id)

    monkeypatch.setattr(reminders_module, "_send_payment_received_notification", _stub)

    assert await check_payment_received(1, account_id, Decimal("4000.00")) is False

    assert observed["persisted_paid"] == Decimal("4000.00")
    assert observed["args"] == (
        "Test CC",
        "Test Bank",
        Decimal("4000.00"),
        Decimal("4000.00"),
        Decimal("10000.00"),
        123456,
    )


async def test_notification_not_sent_when_commit_fails(session_maker, monkeypatch):
    """If session.commit raises, the notification stub must NEVER be called —
    no false 'payment received' message on a failed commit."""
    await _enable_telegram(monkeypatch)

    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account, total_due="10000.00")
        s.add(_credit(account, "4000.00"))
        await s.commit()
        account_id = account.id

    sent = {"called": False}

    async def _stub(*args, **kwargs):
        sent["called"] = True

    monkeypatch.setattr(reminders_module, "_send_payment_received_notification", _stub)

    real_commit = AsyncSession.commit

    async def _boom(self):
        raise RuntimeError("synthetic commit failure")

    monkeypatch.setattr(AsyncSession, "commit", _boom)
    try:
        with pytest.raises(RuntimeError, match="synthetic commit failure"):
            await check_payment_received(1, account_id, Decimal("4000.00"))
    finally:
        monkeypatch.setattr(AsyncSession, "commit", real_commit)

    assert sent["called"] is False


async def test_partial_then_full(session_maker):
    """sum < due -> PARTIALLY_PAID (no paid_at); sum >= due -> PAID + paid_at."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account, total_due="10000.00")
        s.add(_credit(account, "4000.00"))
        await s.commit()
        account_id = account.id

    assert await check_payment_received(1, account_id, Decimal("4000.00")) is False
    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        assert upload.payment_status == PaymentStatus.PARTIALLY_PAID
        assert upload.payment_paid_at is None

    async with session_maker() as s:
        account = (await s.execute(select(Account))).scalar_one()
        s.add(_credit(account, "6000.00", txn_date=dt.date(2026, 5, 15)))
        await s.commit()

    assert await check_payment_received(2, account_id, Decimal("6000.00")) is True
    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        assert upload.payment_paid_amount == Decimal("10000.00")
        assert upload.payment_status == PaymentStatus.PAID
        assert upload.payment_paid_at is not None


async def test_delete_self_heal_recomputes_down(session):
    """Two CC-payment credits over-pay; deleting one recomputes the paid
    amount DOWN to the remaining sum and the cc_outstanding snapshot reflects
    it. Uses the account-level helper the delete path would call, AFTER the
    row is flushed-deleted so it no longer counts in the SUM."""
    account = await _cc_account(session)
    await _upload(session, account, total_due="10000.00")
    c1 = _credit(account, "6000.00")
    c2 = _credit(account, "6000.00")
    session.add_all([c1, c2])
    await session.flush()

    # Both present: inflated to 12000.
    paid = await recompute_cc_payment_for_account(session, account.id)
    assert paid == Decimal("12000.00")

    # Delete one and flush so the SUM no longer includes it.
    await session.delete(c2)
    await session.flush()

    paid = await recompute_cc_payment_for_account(session, account.id)
    assert paid == Decimal("6000.00")

    upload = (await session.execute(select(StatementUpload))).scalar_one()
    assert upload.payment_paid_amount == Decimal("6000.00")
    assert upload.payment_status == PaymentStatus.PARTIALLY_PAID
    # Reopening a previously-paid cycle clears the stale paid-at timestamp.
    assert upload.payment_paid_at is None

    snapshot = (
        await session.execute(
            select(BalanceSnapshot).where(
                BalanceSnapshot.category == SnapshotCategory.cc_outstanding.value
            )
        )
    ).scalar_one()
    # max(total_due - recomputed_paid, 0) = 10000 - 6000.
    assert snapshot.value == Decimal("4000.00")


async def test_snapshot_value_matches_corrected_paid(session):
    """emit_cc_snapshot (called inside the helper) writes cc_outstanding =
    max(total_due - recomputed_paid, 0) for the corrected upload."""
    account = await _cc_account(session)
    upload = await _upload(session, account, total_due="10000.00")
    session.add(_credit(account, "2500.00"))
    await session.flush()

    paid = await recompute_cc_payment_state(session, upload)
    assert paid == Decimal("2500.00")

    snapshot = (
        await session.execute(
            select(BalanceSnapshot).where(
                BalanceSnapshot.category == SnapshotCategory.cc_outstanding.value
            )
        )
    ).scalar_one()
    assert snapshot.value == Decimal("7500.00")
