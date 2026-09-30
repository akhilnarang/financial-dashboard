"""Payments panel + manual-settle integration tests.

Covers the statement page's provisional-payment feature:

- A provisional HDFC bill-payment SMS (notify-only, no ledger row) appears in
  the page's pending set, derived live by re-parsing.
- "Settle" promotes that provisional SMS into a real credit transaction and
  recomputes the cycle's paid amount exactly once.
- Once settled the SMS links to its transaction and drops off the pending set.
- A settled real credit shows in the settled set, not pending.

Card mask 1234 matches the default test CC account (account_number ...1234).
Amounts are synthetic.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db import (
    SmsMessage,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus
from financial_dashboard.services.linker import build_link_context
from financial_dashboard.services.sms_pipeline import process_sms_row
from financial_dashboard.services.statement_payments import build_payments_view

from . import _helpers as h


# The provisional "received but not settled" HDFC template: no reference,
# carries available limit, day-granularity date.
_PROVISIONAL_BODY = (
    "DEAR HDFCBANK CARDMEMBER, PAYMENT OF Rs. 30000.00 RECEIVED TOWARDS "
    "YOUR CREDIT CARD ENDING WITH 1234 ON {day}-8-2026. "
    "YOUR AVAILABLE LIMIT IS RS. 100000.00"
)


@pytest.fixture
def _no_payment_tracking():
    """Noop override — the real reminders recompute logic runs."""
    yield


async def _seed_open_statement(maker, *, total="50,000.00", due="25/08/2026"):
    acc_id = await h.add_cc_account(maker, cards=["XXXX1234"])
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
            created_at=datetime.datetime(2026, 8, 6, 6, 20, tzinfo=datetime.UTC),
        )
        session.add(upload)
        await session.commit()
        return upload.id, acc_id


async def _add_provisional_sms(maker, *, day=7, hour=6, minute=0):
    async with maker() as session:
        sms = SmsMessage(
            bank="hdfc",
            sender="AD-HDFCBK",
            body=_PROVISIONAL_BODY.format(day=day),
            received_at=datetime.datetime(
                2026, 8, day, hour, minute, tzinfo=datetime.UTC
            ),
            status="parsed",
        )
        session.add(sms)
        await session.commit()
        return sms.id


async def _get_upload(maker, upload_id):
    async with maker() as session:
        return await session.get(StatementUpload, upload_id)


@pytest.mark.anyio
async def test_settle_promotes_provisional_to_real_credit(maker):
    """Settling a provisional SMS creates one real credit and moves paid.
    A repeated settle creates no second credit."""
    upload_id, acc_id = await _seed_open_statement(maker)
    sms_id = await _add_provisional_sms(maker, day=7)

    # Before settle the SMS shows as pending, not settled.
    async with maker() as session:
        view = await build_payments_view(session, await _get_upload(maker, upload_id))
    assert view.settled == []
    assert [(p.amount, p.card_mask[-4:]) for p in view.pending] == [
        ("30,000.00", "1234")
    ]

    async def _settle():
        async with maker() as session, session.begin():
            sms = await session.get(SmsMessage, sms_id)
            link_ctx = await build_link_context(session)
            return await process_sms_row(
                session, sms, link_ctx, settle_provisional=True
            )

    outcome = await _settle()
    assert outcome.transaction_id is not None
    assert outcome.pending_payment_check is not None
    # A second settle reuses the same credit.
    assert (await _settle()).transaction_id == outcome.transaction_id

    # Fire the recompute hook (web handler does this post-commit).
    from financial_dashboard.services.reminders import check_payment_received

    await check_payment_received(*outcome.pending_payment_check)

    # Exactly one credit row exists, linked to the SMS.
    async with maker() as session:
        credits = (
            (
                await session.execute(
                    select(Transaction).where(
                        Transaction.account_id == acc_id,
                        Transaction.direction == "credit",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(credits) == 1
        sms = await session.get(SmsMessage, sms_id)
        assert sms.transaction_id == credits[0].id

    # Paid amount moved by the settled amount.
    upload = await _get_upload(maker, upload_id)
    assert Decimal(str(upload.payment_paid_amount)) == Decimal("30000.00")

    # The settled payment now shows in settled, and pending is empty.
    async with maker() as session:
        view = await build_payments_view(session, upload)
    assert len(view.settled) == 1
    assert view.settled[0].amount == "30,000.00"
    assert view.pending == []


async def _add_settled_credit(maker, acc_id, *, amount, day=8):
    """A real bank-settled CC-payment credit already in the ledger."""
    async with maker() as session:
        txn = Transaction(
            account_id=acc_id,
            bank="hdfc",
            email_type="hdfc_cc_payment_received_alert",
            direction="credit",
            amount=Decimal(str(amount)),
            transaction_date=datetime.date(2026, 8, day),
            counterparty="Payment",
            reference_number="REF123",
        )
        session.add(txn)
        await session.commit()
        return txn.id


@pytest.mark.anyio
async def test_two_provisionals_one_settled_keeps_both_pending(maker):
    """An amount-only settled row does not hide either provisional."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7, minute=7)
    await _add_provisional_sms(maker, day=7, minute=11)
    await _add_settled_credit(maker, acc_id, amount="30000.00", day=8)

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.settled) == 1
    assert len(view.pending) == 2


@pytest.mark.anyio
async def test_confident_settled_match_hides_provisional(maker):
    """Equal authoritative balance and identity hide the settled provisional."""
    upload_id, acc_id = await _seed_open_statement(maker)
    await _add_provisional_sms(maker, day=7, hour=6, minute=0)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="hdfc_cc_payment_received_alert",
                direction="credit",
                amount=Decimal("30000.00"),
                currency="INR",
                transaction_date=datetime.date(2026, 8, 7),
                transaction_time=datetime.time(11, 30),
                card_mask="1234",
                balance=Decimal("100000.00"),
                source="email",
            )
        )
        await session.commit()

    upload = await _get_upload(maker, upload_id)
    async with maker() as session:
        view = await build_payments_view(session, upload)

    assert len(view.settled) == 1
    assert view.pending == []


async def _seed_second_statement(maker, acc_id, *, created_at, total="50,000.00"):
    """A later statement for the SAME account, opening the next cycle."""

    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc2.pdf",
            file_path="/tmp/cc2.pdf",
            status="imported",
            due_date="25/09/2026",
            total_amount_due=total,
            payment_status=PaymentStatus.UNPAID,
            payment_paid_amount=Decimal("0"),
            created_at=created_at,
        )
        session.add(upload)
        await session.commit()
        return upload.id


async def _add_null_date_credit(maker, acc_id, *, amount, created_at):
    """A date-less real CC-payment credit with a controlled created_at."""
    async with maker() as session:
        txn = Transaction(
            account_id=acc_id,
            bank="hdfc",
            email_type="hdfc_cc_payment_received_alert",
            direction="credit",
            amount=Decimal(str(amount)),
            transaction_date=None,
            counterparty="Payment",
        )
        txn.created_at = created_at
        session.add(txn)
        await session.commit()
        return txn.id


@pytest.mark.anyio
async def test_null_date_credit_bounded_to_its_cycle_by_created_at(maker):
    """A date-less settled credit created after the next statement shows in the
    NEXT cycle's settled list only, never the old one. The old cycle bounds
    date-less rows by ``created_at`` (< the next statement's ``created_at``), so
    it does not double-count."""
    old_id, acc_id = await _seed_open_statement(maker)  # created 2026-08-06
    newer_id = await _seed_second_statement(
        maker,
        acc_id,
        created_at=datetime.datetime(2026, 9, 6, 6, 20, tzinfo=datetime.UTC),
    )
    await _add_null_date_credit(
        maker,
        acc_id,
        amount="30000.00",
        created_at=datetime.datetime(2026, 9, 10, 6, 0, tzinfo=datetime.UTC),
    )

    old_upload = await _get_upload(maker, old_id)
    newer_upload = await _get_upload(maker, newer_id)
    async with maker() as session:
        old_view = await build_payments_view(session, old_upload)
        newer_view = await build_payments_view(session, newer_upload)

    assert old_view.settled == []
    assert len(newer_view.settled) == 1
    assert newer_view.settled[0].amount == "30,000.00"


@pytest.mark.anyio
async def test_settle_rejects_other_card_and_non_payment_sms(maker):
    """Settle rejects a provisional for another card and a plain spend SMS."""
    from financial_dashboard.services.statement_payments import (
        is_settleable_provisional,
    )

    upload_id, _ = await _seed_open_statement(maker)  # card ...1234
    upload = await _get_upload(maker, upload_id)

    # A provisional for a DIFFERENT card (9999) on the same bank.
    async with maker() as session:
        other = SmsMessage(
            bank="hdfc",
            sender="AD-HDFCBK",
            body=(
                "DEAR HDFCBANK CARDMEMBER, PAYMENT OF Rs. 30000.00 RECEIVED "
                "TOWARDS YOUR CREDIT CARD ENDING WITH 9999 ON 7-8-2026. "
                "YOUR AVAILABLE LIMIT IS RS. 100000.00"
            ),
            received_at=datetime.datetime(2026, 8, 7, 9, 0, tzinfo=datetime.UTC),
            status="parsed",
        )
        session.add(other)
        await session.commit()

        spend = SmsMessage(
            bank="hdfc",
            sender="VK-HDFCBK",
            body=(
                "Spent Rs.500 From HDFC Bank Card x1234 At Zomato On "
                "2026-08-07:14:23:00 Bal Rs.1000"
            ),
            received_at=datetime.datetime(2026, 8, 7, 9, 0, tzinfo=datetime.UTC),
            status="parsed",
        )
        session.add(spend)
        await session.commit()
        ids = [other.id, spend.id]

    async with maker() as session:
        for sms_id in ids:
            sms = await session.get(SmsMessage, sms_id)
            assert await is_settleable_provisional(session, sms, upload) is False
