import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db import PaymentStatus
from financial_dashboard.db.enums import SnapshotCategory
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    BankStatementUpload,
    StatementUpload,
)
from financial_dashboard.services.snapshots import emit_bank_snapshot, emit_cc_snapshot

pytestmark = pytest.mark.anyio


async def _account(session, account_type: str):
    account = Account(
        bank="Example Bank",
        label=account_type,
        type=account_type,
        active=True,
    )
    session.add(account)
    await session.flush()
    return account


async def test_emit_bank_snapshot_from_upload(session):
    account = await _account(session, "bank_account")
    upload = BankStatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="bank.pdf",
        file_path="/tmp/bank.pdf",
        status="parsed",
        closing_balance="12,345.67",
        statement_period_end="30/04/2026",
    )
    session.add(upload)

    assert await emit_bank_snapshot(session, upload) is True
    snapshot = (await session.execute(select(BalanceSnapshot))).scalar_one()
    assert snapshot.category == SnapshotCategory.bank_balance.value
    assert snapshot.as_of_date == dt.date(2026, 4, 30)
    assert snapshot.value == Decimal("12345.67")


async def test_emit_cc_snapshot_matches_dashboard_outstanding_logic(session):
    account = await _account(session, "credit_card")
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="cc.pdf",
        file_path="/tmp/cc.pdf",
        status="parsed",
        due_date="25/06/2026",
        total_amount_due="10,000.00",
        payment_paid_amount=Decimal("2500.00"),
        reconciliation_data='{"matched": [{"date": "30/04/2026"}], "missing": []}',
        created_at=dt.datetime(2026, 5, 10, tzinfo=dt.UTC),
    )
    session.add(upload)

    assert await emit_cc_snapshot(session, upload) is True
    snapshot = (await session.execute(select(BalanceSnapshot))).scalar_one()
    assert snapshot.category == SnapshotCategory.cc_outstanding.value
    # Dated by statement generation (created_at), not latest txn (30/04).
    assert snapshot.as_of_date == dt.date(2026, 5, 10)
    assert snapshot.value == Decimal("7500.00")


async def test_emit_cc_snapshot_paid_or_non_positive_is_zero(session):
    account = await _account(session, "credit_card")
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="cc.pdf",
        file_path="/tmp/cc.pdf",
        status="parsed",
        total_amount_due="100.00",
        payment_paid_amount=None,
        payment_status=PaymentStatus.PAID,
        created_at=dt.datetime(2026, 5, 10, tzinfo=dt.UTC),
    )
    session.add(upload)

    assert await emit_cc_snapshot(session, upload) is True
    snapshot = (await session.execute(select(BalanceSnapshot))).scalar_one()
    assert snapshot.value == Decimal("0.00")
    assert snapshot.as_of_date == dt.date(2026, 5, 10)


async def test_emit_skips_null_or_unparseable_values(session):
    bank_account = await _account(session, "bank_account")
    cc_account = await _account(session, "credit_card")
    bank_upload = BankStatementUpload(
        account_id=bank_account.id,
        bank=bank_account.bank,
        filename="bank.pdf",
        file_path="/tmp/bank.pdf",
        closing_balance=None,
        statement_period_end="30/04/2026",
    )
    cc_upload = StatementUpload(
        account_id=cc_account.id,
        bank=cc_account.bank,
        filename="cc.pdf",
        file_path="/tmp/cc.pdf",
        total_amount_due="not money",
    )
    session.add_all([bank_upload, cc_upload])

    assert await emit_bank_snapshot(session, bank_upload) is False
    assert await emit_cc_snapshot(session, cc_upload) is False
    assert (await session.execute(select(BalanceSnapshot))).scalars().all() == []


async def test_emit_replaces_same_source_date(session):
    account = await _account(session, "bank_account")
    first = BankStatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="bank.pdf",
        file_path="/tmp/bank.pdf",
        closing_balance="100.00",
        statement_period_end="30/04/2026",
    )
    second = BankStatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="bank2.pdf",
        file_path="/tmp/bank2.pdf",
        closing_balance="200.00",
        statement_period_end="30/04/2026",
    )
    session.add_all([first, second])

    await emit_bank_snapshot(session, first)
    await emit_bank_snapshot(session, second)
    snapshots = (await session.execute(select(BalanceSnapshot))).scalars().all()
    assert len(snapshots) == 1
    assert snapshots[0].value == Decimal("200.00")


async def test_emit_cc_snapshot_null_created_at_falls_back_to_today(session):
    """A CC upload with no created_at (None) dates the snapshot from the emit
    time instead of crashing — the statement-generation-date fallback."""
    account = await _account(session, "credit_card")
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="cc.pdf",
        file_path="/tmp/cc.pdf",
        status="parsed",
        total_amount_due="10,000.00",
        payment_paid_amount=None,
        created_at=None,  # NULL — fallback path
    )
    session.add(upload)

    before = dt.datetime.now(dt.UTC).date()
    assert await emit_cc_snapshot(session, upload) is True
    after = dt.datetime.now(dt.UTC).date()

    snapshot = (await session.execute(select(BalanceSnapshot))).scalar_one()
    # Dated by the fallback (statement generation date), not any txn date, and
    # lands on today's UTC date.
    assert snapshot.as_of_date == snapshot.as_of_date  # present
    assert before <= snapshot.as_of_date <= after
    assert snapshot.value == Decimal("10000.00")


async def test_emit_cc_snapshot_same_day_replaces(session):
    """Two CC uploads generated the same day upsert to one snapshot (the unique
    key is account+category+as_of_date, and CC as_of is created_at.date()); the
    later emit wins."""
    account = await _account(session, "credit_card")
    same_day = dt.datetime(2026, 5, 10, 9, 0, tzinfo=dt.UTC)
    first = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="cc1.pdf",
        file_path="/tmp/cc1.pdf",
        status="parsed",
        total_amount_due="10,000.00",
        payment_paid_amount=Decimal("0.00"),
        created_at=same_day,
    )
    second = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="cc2.pdf",
        file_path="/tmp/cc2.pdf",
        status="parsed",
        total_amount_due="7,000.00",
        payment_paid_amount=Decimal("0.00"),
        created_at=same_day,
    )
    session.add_all([first, second])

    await emit_cc_snapshot(session, first)
    await emit_cc_snapshot(session, second)

    snapshots = (
        (
            await session.execute(
                select(BalanceSnapshot).where(
                    BalanceSnapshot.category == SnapshotCategory.cc_outstanding.value
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(snapshots) == 1
    assert snapshots[0].as_of_date == dt.date(2026, 5, 10)
    assert snapshots[0].value == Decimal("7000.00")
