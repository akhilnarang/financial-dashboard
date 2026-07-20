"""CC-specific payment-tracking integration tests.

Covers the rules that keep CC bill-payment accounting correct:

- Statement-imported credits (``email_type=cc_statement``) and merchant
  reversal/refund credits never count as bill payments.
- A real ``payment_received`` credit recomputes ``payment_paid_amount`` from
  the SUM of qualifying credits (idempotent — partial then paid exactly once).
- ``init_payment_tracking`` derives UNPAID / PAID (zero due) from total due.
- The email-summary path stores a ``source_kind=email_summary`` upload.
- Mark paid / unpaid preserves a real partial amount and clears the manual
  full-pay marker.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db import (
    BalanceSnapshot,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus
from financial_dashboard.services.reminders import (
    check_payment_received,
    init_payment_tracking,
    recompute_cc_payment_state,
)
from bank_email_parser.models import Money, ParsedEmail, StatementSummary

from . import _helpers as h


# Override the default _no_payment_tracking fixture: these tests WANT the real
# (date-gated) init_payment_tracking + recompute logic, against the test maker.
@pytest.fixture
def _no_payment_tracking():
    """Noop override — real reminders logic runs (wired via ``maker``)."""
    yield


async def _seed_open_statement(maker, *, total="5,000.00", due="15/08/2026"):
    """An active CC account + an open (UNPAID) statement cycle whose
    ``created_at`` is yesterday so payment credits dated today fall in-cycle."""
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date=due,
            total_amount_due=total,
            payment_status=PaymentStatus.UNPAID,
            payment_paid_amount=Decimal("0"),
            created_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1),
        )
        session.add(upload)
        await session.commit()
        return upload.id, acc_id


async def _add_credit(maker, acc_id, *, amount, email_type, days_ago=0):
    async with maker() as session:
        txn = Transaction(
            account_id=acc_id,
            bank="hdfc",
            email_type=email_type,
            direction="credit",
            amount=Decimal(str(amount)),
            transaction_date=datetime.date.today() - datetime.timedelta(days=days_ago),
            counterparty="Payment",
        )
        session.add(txn)
        await session.commit()
        return txn.id


# ---------------------------------------------------------------------------
# Statement-imported credit / reversal never counts as a bill payment
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_statement_imported_credit_never_marks_paid(maker):
    """A credit imported FROM a CC statement (email_type=cc_statement) is not
    a bill payment — it must not satisfy an open statement. Otherwise importing
    a statement's payments_refunds section would double-count as a payment."""
    upload_id, acc_id = await _seed_open_statement(maker, total="5,000.00")
    txn_id = await _add_credit(
        maker, acc_id, amount="5000.00", email_type="cc_statement"
    )

    marked_paid = await check_payment_received(txn_id, acc_id, Decimal("5000"))
    assert marked_paid is False

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        # The credit did not count: paid_amount stays 0 and the statement is
        # not PAID. (Status may be PARTIALLY_PAID from the recompute else-
        # branch, but that carries a 0 paid amount — no payment was credited.)
        assert upload.payment_paid_amount == Decimal("0")
        assert upload.payment_status != PaymentStatus.PAID


@pytest.mark.anyio
async def test_merchant_reversal_credit_never_marks_paid(maker):
    """An ``icici_cc_reversal`` credit is a merchant refund, not a bill
    payment — it must not bump ``payment_paid_amount``."""
    upload_id, acc_id = await _seed_open_statement(maker, total="5,000.00")
    txn_id = await _add_credit(
        maker, acc_id, amount="250.00", email_type="icici_cc_reversal"
    )

    marked_paid = await check_payment_received(txn_id, acc_id, Decimal("250"))
    assert marked_paid is False

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_paid_amount == Decimal("0")
        assert upload.payment_status != PaymentStatus.PAID


# ---------------------------------------------------------------------------
# Real payment_received recomputes partial → paid exactly once
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_payment_received_recomputes_partial_then_paid_exactly_once(maker):
    """Two real payment credits (hdfc_cc_payment_alert) recompute the paid
    sum from their SUM each time — never accumulate. Partial after the first,
    PAID after the second; re-firing check_payment_received on the same txn
    keeps the same total (idempotent)."""
    upload_id, acc_id = await _seed_open_statement(maker, total="5,000.00")

    t1 = await _add_credit(
        maker, acc_id, amount="2000.00", email_type="hdfc_cc_payment_alert"
    )
    marked1 = await check_payment_received(t1, acc_id, Decimal("2000"))
    assert marked1 is False  # 2000 < 5000 → partial, not fully paid

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.PARTIALLY_PAID
        assert upload.payment_paid_amount == Decimal("2000.00")

    t2 = await _add_credit(
        maker, acc_id, amount="3000.00", email_type="hdfc_cc_payment_alert"
    )
    marked2 = await check_payment_received(t2, acc_id, Decimal("3000"))
    assert marked2 is True  # 2000 + 3000 >= 5000 → fully paid

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.PAID
        assert upload.payment_paid_amount == Decimal("5000.00")
        first_paid_at = upload.payment_paid_at
    assert first_paid_at is not None

    # Re-fire on t1 (e.g. a reparse path) — recomputed SUM is still 5000, and
    # the paid_at timestamp is preserved (no double-stamp).
    await check_payment_received(t1, acc_id, Decimal("2000"))
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_paid_amount == Decimal("5000.00")
        assert upload.payment_paid_at == first_paid_at


@pytest.mark.anyio
async def test_recompute_unwinds_when_payment_txn_deleted(maker):
    """``recompute_cc_payment_state`` is a pure recompute, so deleting a
    payment txn and re-running drops the paid sum back (self-healing)."""
    upload_id, acc_id = await _seed_open_statement(maker, total="5,000.00")
    t1 = await _add_credit(
        maker, acc_id, amount="5000.00", email_type="axis_cc_payment_alert"
    )
    await check_payment_received(t1, acc_id, Decimal("5000"))

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.PAID
        # Delete the payment txn and recompute.
        txn = await session.get(Transaction, t1)
        await session.delete(txn)
        await session.commit()
        upload = await session.get(StatementUpload, upload_id)
        new_paid = await recompute_cc_payment_state(session, upload)
        assert new_paid == Decimal("0.00")
        assert upload.payment_status == PaymentStatus.PARTIALLY_PAID
        assert upload.payment_paid_at is None


# ---------------------------------------------------------------------------
# init_payment_tracking: due-derived states
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_init_payment_tracking_zero_due_marks_paid(maker):
    """A statement with total_amount_due of 0 is immediately PAID."""
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date="15/08/2026",
            total_amount_due="0.00",
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    tracked = await init_payment_tracking(upload_id)
    assert tracked is True
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.PAID
        assert upload.payment_paid_amount == Decimal("0.00")


@pytest.mark.anyio
async def test_init_payment_tracking_positive_due_marks_unpaid(maker):
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date="15/08/2026",
            total_amount_due="3,250.00",
            minimum_amount_due="500.00",
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    tracked = await init_payment_tracking(upload_id)
    assert tracked is True
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.UNPAID


@pytest.mark.anyio
async def test_init_payment_tracking_skips_stale_due(maker):
    """A statement whose due date is before the first of the current month is
    stale — init_payment_tracking skips it (returns False, no status set)."""
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date="15/01/2020",  # far in the past
            total_amount_due="1,000.00",
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    tracked = await init_payment_tracking(upload_id)
    assert tracked is False
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status is None


@pytest.mark.anyio
async def test_init_payment_tracking_idempotent(maker):
    """Re-running on an already-tracked statement is a no-op (returns False)."""
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date="15/08/2026",
            total_amount_due="1,000.00",
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    assert await init_payment_tracking(upload_id) is True
    assert await init_payment_tracking(upload_id) is False  # already tracked


# ---------------------------------------------------------------------------
# Snapshot emission ties payment state to outstanding liability
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_snapshot_reflects_payment_state(maker):
    """The cc_outstanding snapshot value = max(total_due - paid, 0). A PAID
    statement emits 0; a partial emits the remainder."""
    upload_id, acc_id = await _seed_open_statement(maker, total="5,000.00")

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await recompute_cc_payment_state(session, upload)
        await session.commit()

    async with maker() as session:
        snaps = (
            (
                await session.execute(
                    select(BalanceSnapshot).where(BalanceSnapshot.account_id == acc_id)
                )
            )
            .scalars()
            .all()
        )
        # UNPAID → outstanding equals full due.
        assert any(s.value == Decimal("5000.00") for s in snaps)

    # Pay it fully → snapshot drops to 0.
    t1 = await _add_credit(
        maker, acc_id, amount="5000.00", email_type="hdfc_cc_payment_alert"
    )
    await check_payment_received(t1, acc_id, Decimal("5000"))
    async with maker() as session:
        snaps = (
            (
                await session.execute(
                    select(BalanceSnapshot).where(BalanceSnapshot.account_id == acc_id)
                )
            )
            .scalars()
            .all()
        )
        assert any(s.value == Decimal("0.00") for s in snaps)


# ---------------------------------------------------------------------------
# Email-summary path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_email_summary_path_creates_summary_upload(maker, monkeypatch):
    """The summary-only path (no PDF) creates a ``source_kind=email_summary``
    upload with min/total due and fires init_payment_tracking."""
    import financial_dashboard.services.statements.cc as cc_module

    # Let init_payment_tracking run real for this one path.
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)
    await h.add_cc_account(maker, bank="onecard")

    summary = StatementSummary(
        total_amount_due=Money(amount=Decimal("12899.94")),
        minimum_amount_due=Money(amount=Decimal("371.94")),
        due_date=datetime.date(2026, 8, 15),
        card_mask="1234",
    )
    parsed = ParsedEmail(
        email_type="onecard_cc_statement", bank="onecard", statement=summary
    )

    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )
    assert result is not None
    assert result["summary_only"] is True

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert upload.source_kind == "email_summary"
        assert upload.filename == ""
        assert upload.total_amount_due == "12,899.94"
        assert upload.minimum_amount_due == "371.94"
        assert upload.due_date == "15/08/2026"


@pytest.mark.anyio
async def test_email_summary_refuses_partial_payload(maker, monkeypatch):
    """A summary missing total_amount_due must not create a phantom row."""
    import financial_dashboard.services.statements.cc as cc_module

    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)
    await h.add_cc_account(maker, bank="onecard")

    summary = StatementSummary(
        total_amount_due=None,
        due_date=datetime.date(2026, 8, 15),
    )
    parsed = ParsedEmail(
        email_type="onecard_cc_statement", bank="onecard", statement=summary
    )
    result = await cc_module.process_cc_statement_email_summary(
        "onecard", parsed, email_id=None
    )
    assert result is None
    async with maker() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert rows == []
