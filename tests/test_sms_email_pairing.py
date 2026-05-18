"""Cross-channel matrix tests: SMS-first-then-email and vice versa."""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Base, SmsMessage
from financial_dashboard.services.linker import build_link_context
from financial_dashboard.services.sms_pipeline import process_sms_row
from financial_dashboard.services.txn_merge import merge_transaction


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


@pytest.mark.anyio
async def test_sms_first_then_email_enriches(session):
    """SMS arrives first, creates the Transaction. Email arrives with
    a different (richer) counterparty — overwrites and produces a diff."""
    # 1. SMS arrives.
    sms = SmsMessage(
        bank="indusind",
        sender="VK-INDUSB",
        body=(
            "A/C *XX1234 credited by Rs 5000.00 from 9999999999@bank."
            " RRN:000000000001. Avl Bal:10000.00. Not you? Call 18602677777"
            " - IndusInd bank"
        ),
        received_at=datetime.datetime(2026, 5, 2, 9, 0, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()
    link_ctx = await build_link_context(session)

    # Use begin_nested() because the prior session.flush() autobegan an
    # outer transaction.
    async with session.begin_nested():
        sms_outcome = await process_sms_row(session, sms, link_ctx)
    assert sms_outcome.status == "parsed"
    txn_id = sms_outcome.transaction_id

    # 2. Email arrives with same ref but richer counterparty.
    async with session.begin_nested():
        outcome, row, diff = await merge_transaction(
            session,
            "email",
            {
                "bank": "indusind",
                "email_type": "indusind_account_transaction_alert",
                "direction": "credit",
                "amount": Decimal("5000.00"),
                "currency": "INR",
                "reference_number": "000000000001",
                "counterparty": "John Doe via PhonePe",
            },
            email_id=42,
        )
    assert outcome == "enriched"
    assert row.id == txn_id
    assert row.source == "sms+email"
    assert row.notified_channel == "sms"  # primary stays SMS
    # SMS counterparty was a VPA, email's is richer → email wins.
    assert "PhonePe" in row.counterparty


@pytest.mark.anyio
async def test_email_first_then_sms_does_not_overwrite(session):
    """Email arrives first. SMS arrives with poorer-quality counterparty
    — does NOT overwrite. Diff is empty, no enrichment notification."""
    # 1. Email arrives first.
    async with session.begin():
        outcome, row, _ = await merge_transaction(
            session,
            "email",
            {
                "bank": "hdfc",
                "email_type": "hdfc_dc_transaction_alert",
                "direction": "debit",
                "amount": Decimal("500"),
                "currency": "INR",
                "reference_number": "IMPS:E1",
                "counterparty": "Phone Pe Private Limited",
                "transaction_date": datetime.date(2026, 5, 2),
                "transaction_time": datetime.time(14, 23),
            },
            email_id=1,
        )
    txn_id = row.id

    # 2. SMS arrives second.
    async with session.begin():
        outcome2, row2, diff = await merge_transaction(
            session,
            "sms",
            {
                "bank": "hdfc",
                "email_type": "hdfc_dc_transaction_alert",
                "direction": "debit",
                "amount": Decimal("500"),
                "currency": "INR",
                "reference_number": "IMPS:E1",
                "counterparty": "PZCREDIT0000000",
                "transaction_date": datetime.date(2026, 5, 2),
                "transaction_time": datetime.time(14, 23),
            },
            sms_message_id=99,
        )
    assert outcome2 == "enriched"
    assert row2.id == txn_id
    assert row2.counterparty == "Phone Pe Private Limited"  # unchanged
    assert row2.source == "sms+email"
    assert row2.sms_message_id == 99
    assert diff.changed_fields == []  # silent enrichment


@pytest.mark.anyio
async def test_no_pairing_when_amounts_differ_creates_two_rows(session):
    """₹500 SMS and ₹501 email are kept as two separate rows."""
    async with session.begin():
        await merge_transaction(
            session,
            "sms",
            {
                "bank": "hdfc",
                "email_type": "hdfc_dc_transaction_alert",
                "direction": "debit",
                "amount": Decimal("500"),
                "currency": "INR",
                "reference_number": None,
                "transaction_date": datetime.date(2026, 5, 2),
                "transaction_time": datetime.time(14, 23),
                "counterparty": "Zomato",
            },
        )
    async with session.begin():
        outcome, _, _ = await merge_transaction(
            session,
            "email",
            {
                "bank": "hdfc",
                "email_type": "hdfc_dc_transaction_alert",
                "direction": "debit",
                "amount": Decimal("501"),
                "currency": "INR",
                "reference_number": None,
                "transaction_date": datetime.date(2026, 5, 2),
                "transaction_time": datetime.time(14, 23),
                "counterparty": "Zomato",
            },
        )
    assert outcome == "created"
