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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.services.reminders as reminders_module
from financial_dashboard.db import (
    Account,
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
from tests.conftest import new_test_engine

pytestmark = pytest.mark.anyio

CYCLE_CREATED = dt.datetime(2026, 5, 10, 8, 0, tzinfo=dt.UTC)
NEXT_CYCLE_CREATED = dt.datetime(2026, 6, 10, 8, 0, tzinfo=dt.UTC)


@pytest.fixture
async def session_maker(monkeypatch):
    """In-memory aiosqlite session-maker, also installed as the global
    ``async_session`` that ``check_payment_received`` opens for itself."""
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(reminders_module, "async_session", maker)
    yield maker
    await engine.dispose()
    holder.close()


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


async def _upload(
    session, account, total_due="10000.00", created_at=CYCLE_CREATED, due="25/06/2026"
) -> StatementUpload:
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="cc.pdf",
        file_path="/tmp/cc.pdf",
        status="parsed",
        due_date=due,
        total_amount_due=total_due,
        payment_status=PaymentStatus.UNPAID,
        payment_paid_amount=Decimal("0"),
        created_at=created_at,
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
    """A payment dated before the cycle's created_at date is excluded. One on
    the same calendar day or after is included. A NULL-date payment whose own
    created_at predates the cycle must not leak into this or any later cycle."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account)
        s.add(_credit(account, "111.00", txn_date=dt.date(2026, 5, 9)))
        s.add(_credit(account, "1000.00", txn_date=dt.date(2026, 5, 10)))
        s.add(_credit(account, "2000.00", txn_date=dt.date(2026, 5, 20)))
        s.add(
            _credit(
                account,
                "5000.00",
                txn_date=None,
                created_at=dt.datetime(2026, 4, 1, tzinfo=dt.UTC),
            )
        )
        await s.commit()
        account_id = account.id

    await check_payment_received(1, account_id, Decimal("1000.00"))

    async with session_maker() as s:
        upload = (await s.execute(select(StatementUpload))).scalar_one()
        assert upload.payment_paid_amount == Decimal("3000.00")


async def test_dated_payment_split_at_due_date_boundary(session_maker):
    """A payment contributes to exactly one cycle, split at the EARLIER
    statement's DUE date (not its ingestion date). A payment ON the old bill's
    due date clears the OLD bill (the due date is inclusive); one the day after
    clears the new bill; one before clears the old bill. Never both, never
    neither."""
    async with session_maker() as s:
        account = await _cc_account(s)
        old = await _upload(s, account, due="25/06/2026", created_at=CYCLE_CREATED)
        newer = await _upload(
            s, account, due="25/07/2026", created_at=NEXT_CYCLE_CREATED
        )
        # Exactly ON old's due date (25/06) -> old cycle.
        s.add(_credit(account, "500.00", txn_date=dt.date(2026, 6, 25)))
        # Day after old's due date -> new cycle.
        s.add(_credit(account, "1000.00", txn_date=dt.date(2026, 6, 26)))
        # Before old's due date -> old cycle.
        s.add(_credit(account, "2000.00", txn_date=dt.date(2026, 5, 20)))
        await s.flush()

        old_paid = await recompute_cc_payment_state(s, old)
        newer_paid = await recompute_cc_payment_state(s, newer)
        assert old_paid == Decimal("2500.00")
        assert newer_paid == Decimal("1000.00")


async def test_early_payment_before_ingestion_counts_in_new_cycle(session_maker):
    """Regression: a bill paid AFTER the statement closed but BEFORE the
    dashboard ingested it. The payment predates the new statement's created_at,
    so the old created_at-anchored window dropped it and the statement showed
    unpaid. The due-date anchor counts it in the new cycle, and does not sweep
    it into the already-paid old cycle."""
    async with session_maker() as s:
        account = await _cc_account(s)
        prev = await _upload(
            s,
            account,
            total_due="10000.00",
            due="04/08/2026",
            created_at=dt.datetime(2026, 7, 18, tzinfo=dt.UTC),
        )
        latest = await _upload(
            s,
            account,
            total_due="750.00",
            due="04/09/2026",
            created_at=dt.datetime(2026, 8, 18, tzinfo=dt.UTC),
        )
        # Paid 8/16 — after prev's due (8/4), before latest's ingestion (8/18).
        s.add(_credit(account, "750.00", txn_date=dt.date(2026, 8, 16)))
        await s.flush()

        latest_paid = await recompute_cc_payment_state(s, latest)
        prev_paid = await recompute_cc_payment_state(s, prev)
        assert latest_paid == Decimal("750.00")
        assert prev_paid == Decimal("0.00")
        assert latest.payment_status == PaymentStatus.PAID


async def test_dueless_upload_between_cycles_does_not_strand_payment(session_maker):
    """A password-required / parse-error upload has no due date and must NOT act
    as a cycle boundary. An early payment for the latest bill is still counted,
    not stranded by the malformed upload sitting between the two real cycles."""
    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(
            s,
            account,
            total_due="10000.00",
            due="04/08/2026",
            created_at=dt.datetime(2026, 7, 18, tzinfo=dt.UTC),
        )
        # Malformed upload: no due date, ingested between the two real cycles.
        await _upload(
            s,
            account,
            total_due="0.00",
            due=None,
            created_at=dt.datetime(2026, 8, 10, tzinfo=dt.UTC),
        )
        latest = await _upload(
            s,
            account,
            total_due="750.00",
            due="04/09/2026",
            created_at=dt.datetime(2026, 8, 18, tzinfo=dt.UTC),
        )
        s.add(_credit(account, "750.00", txn_date=dt.date(2026, 8, 16)))
        await s.flush()

        assert await recompute_cc_payment_state(s, latest) == Decimal("750.00")


async def test_late_ingested_statement_no_double_count(session_maker):
    """When a statement's due date precedes its own ingestion (a late poll), the
    boundary with the next statement is still one shared value, so a payment is
    counted in exactly one cycle, never both."""
    async with session_maker() as s:
        account = await _cc_account(s)
        s1 = await _upload(
            s,
            account,
            total_due="5000.00",
            due="04/08/2026",
            created_at=dt.datetime(2026, 8, 18, tzinfo=dt.UTC),
        )
        s2 = await _upload(
            s,
            account,
            total_due="5000.00",
            due="04/09/2026",
            created_at=dt.datetime(2026, 9, 18, tzinfo=dt.UTC),
        )
        s.add(_credit(account, "5000.00", txn_date=dt.date(2026, 8, 20)))
        await s.flush()

        p1 = await recompute_cc_payment_state(s, s1)
        p2 = await recompute_cc_payment_state(s, s2)
        assert p1 + p2 == Decimal("5000.00")  # counted once, not in both


async def test_null_date_payment_placed_by_created_at_against_due_boundary(
    session_maker,
):
    """A date-less payment is placed by its created_at against the earlier
    bill's due date. It counts in exactly one cycle, never both."""
    async with session_maker() as s:
        account = await _cc_account(s)
        old = await _upload(s, account, due="25/06/2026", created_at=CYCLE_CREATED)
        newer = await _upload(
            s, account, due="25/07/2026", created_at=NEXT_CYCLE_CREATED
        )
        s.add(
            _credit(
                account,
                "1500.00",
                txn_date=None,
                created_at=dt.datetime(2026, 5, 12, tzinfo=dt.UTC),
            )
        )
        s.add(
            _credit(
                account,
                "700.00",
                txn_date=None,
                created_at=dt.datetime(2026, 6, 27, tzinfo=dt.UTC),
            )
        )
        await s.flush()

        assert await recompute_cc_payment_state(s, old) == Decimal("1500.00")
        assert await recompute_cc_payment_state(s, newer) == Decimal("700.00")


async def test_recompute_unpaid_when_zero_flag(session_maker):
    """``unpaid_when_zero=True`` (ingestion / reparse) makes a zero paid sum read
    as UNPAID; the default (delete / credit-arrival) keeps PARTIALLY_PAID."""
    async with session_maker() as s:
        account = await _cc_account(s)
        upload = await _upload(s, account, total_due="5000.00")
        await s.flush()

        assert await recompute_cc_payment_state(
            s, upload, unpaid_when_zero=True
        ) == Decimal("0.00")
        assert upload.payment_status == PaymentStatus.UNPAID

        assert await recompute_cc_payment_state(s, upload) == Decimal("0.00")
        assert upload.payment_status == PaymentStatus.PARTIALLY_PAID


async def test_resync_preserves_manually_paid_statement(session_maker):
    """The reparse self-heal must NOT revert a manually 'Mark as Paid' statement.
    Such a statement is PAID with no backing credit; a blind recompute would
    zero it. resync leaves any PAID statement untouched."""
    from financial_dashboard.services.reminders import resync_tracked_cc_payment_state

    async with session_maker() as s:
        account = await _cc_account(s)
        upload = await _upload(s, account, total_due="5000.00")
        # Manual mark: PAID, full amount, but no payment credit exists.
        upload.payment_status = PaymentStatus.PAID
        upload.payment_paid_amount = Decimal("5000.00")
        upload.payment_paid_at = dt.datetime(2026, 5, 1, tzinfo=dt.UTC)
        await s.flush()

        did = await resync_tracked_cc_payment_state(s, upload)
        assert did is False
        assert upload.payment_status == PaymentStatus.PAID
        assert upload.payment_paid_amount == Decimal("5000.00")
        assert upload.payment_paid_at is not None


async def test_resync_heals_active_prepaid_statement(session_maker):
    """resync recomputes an ACTIVE (unpaid) statement, so a reparse marks paid a
    bill whose credit already exists — the prod #450 remediation path."""
    from financial_dashboard.services.reminders import resync_tracked_cc_payment_state

    async with session_maker() as s:
        account = await _cc_account(s)
        upload = await _upload(s, account, total_due="750.00")  # UNPAID
        s.add(_credit(account, "750.00", txn_date=dt.date(2026, 5, 12)))
        await s.flush()

        did = await resync_tracked_cc_payment_state(s, upload)
        assert did is True
        assert upload.payment_status == PaymentStatus.PAID
        assert upload.payment_paid_amount == Decimal("750.00")


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


async def test_partial_then_full_notifies_after_commit(session_maker, monkeypatch):
    """sum < due -> PARTIALLY_PAID (no paid_at); sum >= due -> PAID + paid_at.
    The payment-received notification fires only after commit, so the stub
    reads the persisted paid amount in a fresh session."""
    await _enable_telegram(monkeypatch)

    async with session_maker() as s:
        account = await _cc_account(s)
        await _upload(s, account, total_due="10000.00")
        s.add(_credit(account, "4000.00"))
        await s.commit()
        account_id = account.id

    observed: dict[str, object] = {}

    async def _stub(label, bank, credit, total_paid, due, chat_id):
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
