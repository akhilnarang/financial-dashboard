"""Unit + integration tests for services/sms_pipeline.py."""

import datetime
from decimal import Decimal

import pytest  # noqa: F401
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bank_sms_parser.models import Money, ParsedSms, SmsTransactionAlert

from financial_dashboard.db import Base, SmsMessage
from financial_dashboard.services.sms_pipeline import parsed_sms_to_txn_data


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


def _sms_row(**overrides):
    base = dict(
        id=1,
        bank="hdfc",
        sender="VK-HDFCBK",
        body="Spent Rs.500...",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
    )
    base.update(overrides)
    return SmsMessage(**base)


def test_parsed_sms_to_txn_data_happy_path():
    parsed = ParsedSms(
        email_type="hdfc_dc_transaction_alert",
        bank="hdfc",
        transaction=SmsTransactionAlert(
            direction="debit",
            amount=Money(amount=Decimal("500"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 2),
            transaction_time=datetime.time(14, 23, 0),
            counterparty="Zomato",
            card_mask="x1234",
            channel=None,
            reference_number=None,
        ),
    )
    sms_row = _sms_row()
    data = parsed_sms_to_txn_data(parsed, sms_row)
    assert data["bank"] == "hdfc"
    assert data["email_type"] == "hdfc_dc_transaction_alert"
    assert data["direction"] == "debit"
    assert data["amount"] == Decimal("500")
    assert data["currency"] == "INR"
    assert data["transaction_date"] == datetime.date(2026, 5, 2)
    assert data["transaction_time"] == datetime.time(14, 23, 0)
    assert data["counterparty"] == "Zomato"
    assert data["card_mask"] == "x1234"


def test_parsed_sms_to_txn_data_returns_none_for_non_transaction():
    parsed = ParsedSms(
        email_type="onecard_cc_statement_ready",
        bank="onecard",
        transaction=None,
    )
    sms_row = _sms_row(bank="onecard")
    assert parsed_sms_to_txn_data(parsed, sms_row) is None


def test_parsed_sms_to_txn_data_date_fallback_to_received_at_ist():
    parsed = ParsedSms(
        email_type="indusind_account_transaction_alert",
        bank="indusind",
        transaction=SmsTransactionAlert(
            direction="credit",
            amount=Money(amount=Decimal("100"), currency="INR"),
            transaction_date=None,
            transaction_time=None,
        ),
    )
    # 21:00 UTC = 02:30 IST the next day — exercises the spec's
    # "convert to IST before extracting .date()/.time()" rule.
    sms_row = _sms_row(
        bank="indusind",
        received_at=datetime.datetime(2026, 5, 2, 21, 0, 0, tzinfo=datetime.UTC),
    )
    data = parsed_sms_to_txn_data(parsed, sms_row)
    assert data["transaction_date"] == datetime.date(2026, 5, 3)
    assert data["transaction_time"] == datetime.time(2, 30, 0)


from bank_sms_parser.exceptions import ParseError, UnsupportedSmsTypeError  # noqa: E402,F401

from financial_dashboard.services.sms_pipeline import process_sms_row  # noqa: E402


@pytest.mark.anyio
async def test_process_sms_row_happy_path_creates_transaction(
    session, monkeypatch
):
    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="Spent Rs.500 From HDFC Bank Card x1234 At Zomato On 2026-05-02:14:23:00 Bal Rs.1000",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context
    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "parsed"
    assert outcome.transaction_id is not None
    assert outcome.primary_notification is not None
    assert sms.status == "parsed"
    assert sms.transaction_id == outcome.transaction_id
    assert sms.parsed_at is not None


@pytest.mark.anyio
async def test_process_sms_row_parse_error_marks_row_error(session):
    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="OTP for your Rs.3000 transaction is 123456. Do not share.",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context
    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "error"
    assert outcome.transaction_id is None
    assert sms.status == "error"
    assert sms.parse_error is not None


@pytest.mark.anyio
async def test_process_sms_row_unsupported_bank_marks_skipped(session):
    sms = SmsMessage(
        bank="unknown",
        sender="XX-UNKNOWN",
        body="some body",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context
    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "skipped"
    assert sms.status == "skipped"


@pytest.mark.anyio
async def test_process_sms_row_declined_bypasses_merge(session, monkeypatch):
    """If a parser ever emits direction='declined', it goes to the
    declined-notification path, not merge_transaction."""
    from bank_sms_parser.models import Money, SmsTransactionAlert
    parsed = ParsedSms(
        email_type="hdfc_cc_transaction_declined",
        bank="hdfc",
        transaction=SmsTransactionAlert(
            direction="declined",
            amount=Money(amount=Decimal("500"), currency="INR"),
        ),
    )

    def _fake_parse(*args, **kwargs):
        return parsed

    monkeypatch.setattr("financial_dashboard.services.sms_pipeline.parse_sms", _fake_parse)

    sms = SmsMessage(
        bank="hdfc", sender="VK-HDFCBK",
        body="<declined body>",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context
    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "parsed"
    assert outcome.transaction_id is None  # no Transaction inserted
    assert outcome.primary_notification is not None
    assert outcome.primary_notification.get("_declined") is True


@pytest.mark.anyio
async def test_process_sms_row_cc_payment_received_calls_check_payment_received(
    session, monkeypatch
):
    """When the parsed email_type ends with _cc_payment_received_alert and
    account_id resolves, the outcome carries a pending_payment_check
    tuple that the caller will dispatch."""
    from financial_dashboard.db import Account
    from bank_sms_parser.models import Money, SmsTransactionAlert

    # Set up an Axis CC account so the linker resolves.
    acct = Account(bank="axis", type="credit_card", label="Axis CC")
    session.add(acct)
    await session.flush()

    parsed = ParsedSms(
        email_type="axis_cc_payment_received_alert",
        bank="axis",
        transaction=SmsTransactionAlert(
            direction="credit",
            amount=Money(amount=Decimal("15000"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 2),
            card_mask="XX0000",
        ),
    )

    def _fake_parse(*args, **kwargs):
        return parsed
    monkeypatch.setattr("financial_dashboard.services.sms_pipeline.parse_sms", _fake_parse)

    sms = SmsMessage(
        bank="axis", sender="VK-AXISBK",
        body="<axis payment received body>",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context
    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "parsed"
    assert outcome.pending_payment_check is not None
    assert outcome.pending_payment_check[1] == acct.id  # account_id
    assert outcome.pending_payment_check[2] == Decimal("15000")  # amount
