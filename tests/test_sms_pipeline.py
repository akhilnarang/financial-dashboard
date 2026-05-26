"""Unit + integration tests for services/sms_pipeline.py."""

import datetime
from decimal import Decimal

import pytest  # noqa: F401
from sqlalchemy import select
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
async def test_process_sms_row_happy_path_creates_transaction(session, monkeypatch):
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
async def test_process_sms_row_clears_stale_parse_error_on_success(session):
    """A row that previously failed (status=error, parse_error set) and is
    reparsed successfully must clear the stale parse_error — otherwise the
    resolved row still shows its old error alongside parsed/enriched."""
    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="Spent Rs.500 From HDFC Bank Card x1234 At Zomato On 2026-05-02:14:23:00 Bal Rs.1000",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
        status="error",
        parse_error="No parser for bank 'hdfc' could handle this SMS.",
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context

    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "parsed"
    assert sms.status == "parsed"
    assert sms.parse_error is None


@pytest.mark.anyio
async def test_process_sms_row_naive_received_at_still_parses(session):
    """Regression: SQLite returns naive datetimes for DateTime columns.
    bank-sms-parser rejects naive received_at when it falls back to it
    for transaction_date, so the pipeline must re-attach UTC before
    calling parse_sms. The IndusInd UPI parser is the canary — it
    always consults received_at when the body lacks a date."""
    sms = SmsMessage(
        bank="indusind",
        sender="AD-INDUSB-S",
        body=(
            "A/C *XX1234 credited by Rs 5000.00 from 9999999999@bank."
            " RRN:000000000001. Avl Bal:10000.00."
            " Not you? Call 18602677777 - IndusInd bank"
        ),
        # Naive datetime — what SQLAlchemy hands back after a session.refresh().
        received_at=datetime.datetime(2026, 5, 2, 9, 0, 0),
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context

    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "parsed", f"got error: {sms.parse_error}"
    assert outcome.transaction_id is not None


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

    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms", _fake_parse
    )

    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
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

    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms", _fake_parse
    )

    sms = SmsMessage(
        bank="axis",
        sender="VK-AXISBK",
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


@pytest.mark.anyio
async def test_process_sms_row_cc_payment_amount_match_resolves_account(
    session, monkeypatch
):
    """Maskless CC bill-payment SMS with multiple CC candidates: the
    amount-based matcher resolves to the single CC whose open statement
    total_amount_due equals the payment amount. The resolved account_id
    must be persisted on the Transaction row AND surfaced in the
    pending_payment_check tuple."""
    from financial_dashboard.db import Account, PaymentStatus, StatementUpload
    from bank_sms_parser.models import Money, SmsTransactionAlert

    a = Account(bank="indusind", type="credit_card", label="IndusInd A")
    b = Account(bank="indusind", type="credit_card", label="IndusInd B")
    c = Account(bank="indusind", type="credit_card", label="IndusInd C")
    session.add_all([a, b, c])
    await session.flush()
    session.add_all(
        [
            StatementUpload(
                account_id=a.id,
                bank="indusind",
                filename="x.pdf",
                file_path="/tmp/x.pdf",
                status="imported",
                due_date="20/05/2026",
                total_amount_due="1,616.00",
                payment_status=PaymentStatus.UNPAID,
            ),
            StatementUpload(
                account_id=b.id,
                bank="indusind",
                filename="x.pdf",
                file_path="/tmp/x.pdf",
                status="imported",
                due_date="20/05/2026",
                total_amount_due="133.00",
                payment_status=PaymentStatus.UNPAID,
            ),
            StatementUpload(
                account_id=c.id,
                bank="indusind",
                filename="x.pdf",
                file_path="/tmp/x.pdf",
                status="imported",
                due_date="20/05/2026",
                total_amount_due="4,661.00",
                payment_status=PaymentStatus.UNPAID,
            ),
        ]
    )
    await session.flush()

    parsed = ParsedSms(
        email_type="indusind_cc_payment_received_alert",
        bank="indusind",
        transaction=SmsTransactionAlert(
            direction="credit",
            amount=Money(amount=Decimal("133"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 17),
        ),
    )

    def _fake_parse(*args, **kwargs):
        return parsed

    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms", _fake_parse
    )

    sms = SmsMessage(
        bank="indusind",
        sender="VK-INDBNK",
        body="<indusind payment received body>",
        received_at=datetime.datetime(2026, 5, 17, 13, 12, 35, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context

    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.pending_payment_check is not None
    assert outcome.pending_payment_check[1] == b.id  # resolved by amount
    assert outcome.pending_disambiguation is None
    # The Transaction row itself must carry the resolved account_id, so a
    # subsequent notification render shows the correct account label.
    from financial_dashboard.db import Transaction

    txn = (
        await session.execute(
            select(Transaction).where(Transaction.id == outcome.transaction_id)
        )
    ).scalar_one()
    assert txn.account_id == b.id


@pytest.mark.anyio
async def test_process_sms_row_cc_payment_no_amount_match_emits_prompt(
    session, monkeypatch
):
    """Maskless CC bill-payment SMS with multiple CC candidates and no
    matching statement total — the outcome carries pending_disambiguation
    (so the caller fires the Telegram inline-keyboard prompt) and the
    Transaction row stays unlinked."""
    from financial_dashboard.db import Account, PaymentStatus, StatementUpload
    from bank_sms_parser.models import Money, SmsTransactionAlert

    a = Account(bank="indusind", type="credit_card", label="IndusInd A")
    b = Account(bank="indusind", type="credit_card", label="IndusInd B")
    session.add_all([a, b])
    await session.flush()
    session.add_all(
        [
            StatementUpload(
                account_id=a.id,
                bank="indusind",
                filename="x.pdf",
                file_path="/tmp/x.pdf",
                status="imported",
                due_date="20/05/2026",
                total_amount_due="500.00",
                payment_status=PaymentStatus.UNPAID,
            ),
            StatementUpload(
                account_id=b.id,
                bank="indusind",
                filename="x.pdf",
                file_path="/tmp/x.pdf",
                status="imported",
                due_date="20/05/2026",
                total_amount_due="999.00",
                payment_status=PaymentStatus.UNPAID,
            ),
        ]
    )
    await session.flush()

    parsed = ParsedSms(
        email_type="indusind_cc_payment_received_alert",
        bank="indusind",
        transaction=SmsTransactionAlert(
            direction="credit",
            amount=Money(amount=Decimal("133"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 17),
        ),
    )

    def _fake_parse(*args, **kwargs):
        return parsed

    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms", _fake_parse
    )

    sms = SmsMessage(
        bank="indusind",
        sender="VK-INDBNK",
        body="<indusind payment received body>",
        received_at=datetime.datetime(2026, 5, 17, 13, 12, 35, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    from financial_dashboard.services.linker import build_link_context

    link_ctx = await build_link_context(session)

    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.pending_disambiguation is not None
    assert outcome.pending_payment_check is None
    assert set(outcome.pending_disambiguation["candidate_account_ids"]) == {a.id, b.id}


@pytest.mark.anyio
async def test_process_sms_row_hdfc_provisional_payment_is_notify_only(
    session, monkeypatch
):
    """HDFC payment-received SMS with NO reference number → notify only:
    no Transaction row, transaction_id None, _provisional flag set, and
    neither payment-check nor disambiguation hooks fire."""
    from financial_dashboard.services.sms_pipeline import process_sms_row
    from financial_dashboard.services.linker import build_link_context
    from financial_dashboard.db import Transaction

    parsed = ParsedSms(
        email_type="hdfc_cc_payment_received_alert",
        bank="hdfc",
        transaction=SmsTransactionAlert(
            direction="credit",
            amount=Money(amount=Decimal("50000"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 17),
            reference_number=None,  # provisional variant
            card_mask="0000",
        ),
    )
    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms",
        lambda *a, **k: parsed,
    )

    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="DEAR HDFCBANK CARDMEMBER, PAYMENT OF Rs. 50000.00 RECEIVED ...",
        received_at=datetime.datetime(2026, 5, 17, 17, 15, 0, tzinfo=datetime.UTC),
        parse_error="stale error from a previous run",
    )
    session.add(sms)
    await session.flush()

    link_ctx = await build_link_context(session)
    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "parsed"
    assert outcome.transaction_id is None
    assert outcome.primary_notification is not None
    assert outcome.primary_notification.get("_provisional") is True
    assert outcome.pending_payment_check is None
    assert outcome.pending_disambiguation is None

    # No Transaction row created.
    count = len((await session.execute(select(Transaction))).scalars().all())
    assert count == 0

    # SMS row marked handled, stale error cleared.
    assert sms.status == "parsed"
    assert sms.transaction_id is None
    assert sms.parse_error is None


@pytest.mark.anyio
async def test_process_sms_row_hdfc_settlement_creates_row(session, monkeypatch):
    """HDFC payment-received SMS WITH a reference number → normal credit row."""
    from financial_dashboard.services.sms_pipeline import process_sms_row
    from financial_dashboard.services.linker import build_link_context
    from financial_dashboard.db import Transaction

    parsed = ParsedSms(
        email_type="hdfc_cc_payment_received_alert",
        bank="hdfc",
        transaction=SmsTransactionAlert(
            direction="credit",
            amount=Money(amount=Decimal("50000"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 17),
            reference_number="000000000Ref0001",  # settlement variant
            card_mask="0000",
        ),
    )
    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms",
        lambda *a, **k: parsed,
    )

    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="HDFC Bank Cardmember, Online Payment of Rs.50000 vide Ref# ...",
        received_at=datetime.datetime(2026, 5, 18, 14, 53, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    link_ctx = await build_link_context(session)
    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.status == "parsed"
    assert outcome.transaction_id is not None
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1
    assert rows[0].reference_number == "000000000Ref0001"


@pytest.mark.anyio
async def test_process_sms_row_hdfc_payment_order_independent(session, monkeypatch):
    """Whichever order the two variants arrive, exactly one row results."""
    from financial_dashboard.services.sms_pipeline import process_sms_row
    from financial_dashboard.services.linker import build_link_context
    from financial_dashboard.db import Transaction

    def _provisional():
        return ParsedSms(
            email_type="hdfc_cc_payment_received_alert",
            bank="hdfc",
            transaction=SmsTransactionAlert(
                direction="credit",
                amount=Money(amount=Decimal("50000"), currency="INR"),
                transaction_date=datetime.date(2026, 5, 17),
                reference_number=None,
                card_mask="0000",
            ),
        )

    def _settlement():
        return ParsedSms(
            email_type="hdfc_cc_payment_received_alert",
            bank="hdfc",
            transaction=SmsTransactionAlert(
                direction="credit",
                amount=Money(amount=Decimal("50000"), currency="INR"),
                transaction_date=datetime.date(2026, 5, 18),
                reference_number="000000000Ref0001",
                card_mask="0000",
            ),
        )

    link_ctx = await build_link_context(session)

    async def _run(parsed, body, received):
        monkeypatch.setattr(
            "financial_dashboard.services.sms_pipeline.parse_sms",
            lambda *a, **k: parsed,
        )
        sms = SmsMessage(
            bank="hdfc", sender="VK-HDFCBK", body=body, received_at=received
        )
        session.add(sms)
        await session.flush()
        async with session.begin_nested():
            await process_sms_row(session, sms, link_ctx)

    # provisional first, then settlement
    await _run(
        _provisional(),
        "provisional body",
        datetime.datetime(2026, 5, 17, 17, 15, 0, tzinfo=datetime.UTC),
    )
    await _run(
        _settlement(),
        "settlement body",
        datetime.datetime(2026, 5, 18, 14, 53, 0, tzinfo=datetime.UTC),
    )

    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1
    assert rows[0].reference_number == "000000000Ref0001"


@pytest.mark.anyio
async def test_process_sms_row_non_hdfc_payment_no_ref_still_creates_row(
    session, monkeypatch
):
    """The provisional gate is HDFC-specific: a no-ref payment-received SMS
    from another bank (e.g. axis) must still create a transaction row."""
    from financial_dashboard.services.sms_pipeline import process_sms_row
    from financial_dashboard.services.linker import build_link_context
    from financial_dashboard.db import Transaction

    parsed = ParsedSms(
        email_type="axis_cc_payment_received_alert",
        bank="axis",
        transaction=SmsTransactionAlert(
            direction="credit",
            amount=Money(amount=Decimal("15000"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 17),
            reference_number=None,  # no ref, but NOT hdfc
            card_mask="XX0000",
        ),
    )
    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms",
        lambda *a, **k: parsed,
    )

    sms = SmsMessage(
        bank="axis",
        sender="VK-AXISBK",
        body="axis payment received body",
        received_at=datetime.datetime(2026, 5, 17, 17, 15, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    link_ctx = await build_link_context(session)
    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    assert outcome.transaction_id is not None
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1


@pytest.mark.anyio
async def test_process_sms_row_hdfc_payment_settlement_first_then_provisional(
    session, monkeypatch
):
    """Reverse order: settlement (with ref) arrives first and creates the
    row; the later provisional (no ref) is notify-only and adds no row."""
    from financial_dashboard.services.sms_pipeline import process_sms_row
    from financial_dashboard.services.linker import build_link_context
    from financial_dashboard.db import Transaction

    def _settlement():
        return ParsedSms(
            email_type="hdfc_cc_payment_received_alert",
            bank="hdfc",
            transaction=SmsTransactionAlert(
                direction="credit",
                amount=Money(amount=Decimal("50000"), currency="INR"),
                transaction_date=datetime.date(2026, 5, 18),
                reference_number="000000000Ref0001",
                card_mask="0000",
            ),
        )

    def _provisional():
        return ParsedSms(
            email_type="hdfc_cc_payment_received_alert",
            bank="hdfc",
            transaction=SmsTransactionAlert(
                direction="credit",
                amount=Money(amount=Decimal("50000"), currency="INR"),
                transaction_date=datetime.date(2026, 5, 17),
                reference_number=None,
                card_mask="0000",
            ),
        )

    link_ctx = await build_link_context(session)

    async def _run(parsed, body, received):
        monkeypatch.setattr(
            "financial_dashboard.services.sms_pipeline.parse_sms",
            lambda *a, **k: parsed,
        )
        sms = SmsMessage(
            bank="hdfc", sender="VK-HDFCBK", body=body, received_at=received
        )
        session.add(sms)
        await session.flush()
        async with session.begin_nested():
            return await process_sms_row(session, sms, link_ctx)

    # settlement first
    await _run(
        _settlement(),
        "settlement body",
        datetime.datetime(2026, 5, 18, 14, 53, 0, tzinfo=datetime.UTC),
    )
    # then provisional — must not add a second row
    outcome = await _run(
        _provisional(),
        "provisional body",
        datetime.datetime(2026, 5, 17, 17, 15, 0, tzinfo=datetime.UTC),
    )

    assert outcome.transaction_id is None
    assert outcome.primary_notification.get("_provisional") is True
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1
    assert rows[0].reference_number == "000000000Ref0001"


@pytest.mark.anyio
async def test_process_sms_row_hdfc_payment_received_non_credit_not_gated(
    session, monkeypatch
):
    """Defense-in-depth: a hdfc_cc_payment_received_alert with no ref but a
    non-credit direction must NOT be swallowed by the provisional gate
    (which runs before the declined pre-gate). A declined-direction shape
    takes the declined path, not notify-only-provisional."""
    from financial_dashboard.services.sms_pipeline import process_sms_row
    from financial_dashboard.services.linker import build_link_context

    parsed = ParsedSms(
        email_type="hdfc_cc_payment_received_alert",
        bank="hdfc",
        transaction=SmsTransactionAlert(
            direction="declined",  # not credit
            amount=Money(amount=Decimal("50000"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 17),
            reference_number=None,
            card_mask="0000",
        ),
    )
    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms",
        lambda *a, **k: parsed,
    )

    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="<hypothetical non-credit payment-received body>",
        received_at=datetime.datetime(2026, 5, 17, 17, 15, 0, tzinfo=datetime.UTC),
    )
    session.add(sms)
    await session.flush()

    link_ctx = await build_link_context(session)
    async with session.begin_nested():
        outcome = await process_sms_row(session, sms, link_ctx)

    # Routed to the declined path, NOT the provisional notify-only path.
    assert outcome.primary_notification is not None
    assert outcome.primary_notification.get("_provisional") is None
    assert outcome.primary_notification.get("_declined") is True
