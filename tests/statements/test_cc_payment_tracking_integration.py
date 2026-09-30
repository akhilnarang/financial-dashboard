"""CC-specific payment-tracking integration tests.

Covers the rules that keep CC bill-payment accounting correct:

- Statement-imported credits (``email_type=cc_statement``) never count as
  bill payments.
- ``init_payment_tracking`` derives UNPAID / PAID (zero due) from total due.
"""

import datetime
from decimal import Decimal

import pytest

from financial_dashboard.db import StatementUpload, Transaction
from financial_dashboard.db.enums import PaymentStatus
from financial_dashboard.services.reminders import (
    check_payment_received,
    init_payment_tracking,
)

from . import _helpers as h


def _days_from_today(days: int) -> datetime.date:
    return datetime.date.today() + datetime.timedelta(days=days)


def _due_this_month(day: int = 15) -> str:
    """A due date in the current month, in the statement's DD/MM/YYYY form.

    init_payment_tracking does not track a statement whose due date is
    before the first of the current month. A fixed date passes that gate
    only until the month ends, so the tests derive the date from today.
    """
    return datetime.date.today().replace(day=day).strftime("%d/%m/%Y")


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
# Statement-imported credit never counts as a bill payment
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
            due_date=_due_this_month(),
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
async def test_init_payment_tracking_positive_due_marks_unpaid_once(maker):
    """A positive due marks UNPAID. A second run on a tracked statement is a no-op."""
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date=_due_this_month(),
            total_amount_due="3,250.00",
            minimum_amount_due="500.00",
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    assert await init_payment_tracking(upload_id) is True
    assert await init_payment_tracking(upload_id) is False
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.UNPAID


@pytest.mark.anyio
async def test_init_payment_tracking_detects_prepaid_bill(maker):
    """A bill paid BEFORE its statement is ingested (the payment predates the
    new statement's created_at) is detected at ingestion. init recomputes from
    the cycle's qualifying credits and marks the statement PAID, not UNPAID, so
    no false reminder fires."""
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        prior = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date=_days_from_today(-25).strftime("%d/%m/%Y"),
            total_amount_due="10,000.00",
            payment_status=PaymentStatus.PAID,
            created_at=datetime.datetime.combine(
                _days_from_today(-42), datetime.time(), tzinfo=datetime.UTC
            ),
        )
        session.add(prior)
        # Paid after prior's due, before the new statement exists.
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="hdfc_cc_payment_alert",
                direction="credit",
                amount=Decimal("750.00"),
                transaction_date=_days_from_today(-13),
                counterparty="Payment received",
            )
        )
        await session.commit()

    async with maker() as session:
        latest = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date=_days_from_today(6).strftime("%d/%m/%Y"),
            total_amount_due="750.00",
            created_at=datetime.datetime.combine(
                _days_from_today(-11), datetime.time(), tzinfo=datetime.UTC
            ),
        )
        session.add(latest)
        await session.commit()
        latest_id = latest.id

    tracked = await init_payment_tracking(latest_id)
    assert tracked is True
    async with maker() as session:
        latest = await session.get(StatementUpload, latest_id)
        assert latest.payment_status == PaymentStatus.PAID
        assert latest.payment_paid_amount == Decimal("750.00")
        assert latest.payment_paid_at is not None


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
