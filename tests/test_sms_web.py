"""Tests for /sms web routes."""

import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.core.deps import get_session
from financial_dashboard.db import Base, SmsMessage
from financial_dashboard.web import router as web_app_router


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


def _build_app(session_factory):
    app = FastAPI()
    app.dependency_overrides[get_session] = session_factory
    app.include_router(web_app_router)
    return app


async def _client(session):
    async def _override():
        yield session

    return AsyncClient(
        transport=ASGITransport(app=_build_app(_override)), base_url="http://test"
    )


@pytest.mark.anyio
async def test_reparse_single_pending_sms_returns_200(session):
    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="Spent Rs.500 From HDFC Bank Card x1234 At Zomato On 2026-05-02:14:23:00 Bal Rs.1000",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
        status="pending",
    )
    session.add(sms)
    await session.commit()

    async with await _client(session) as client:
        resp = await client.post(f"/sms/{sms.id}/reparse")
    assert resp.status_code == 200
    data = resp.json()
    assert data["new_status"] == "parsed"
    assert data["txn_id"] is not None


@pytest.mark.anyio
async def test_reparse_single_otp_sms_returns_422(session):
    sms = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="OTP for your transaction is 123456. Valid 5 mins.",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
        status="pending",
    )
    session.add(sms)
    await session.commit()

    async with await _client(session) as client:
        resp = await client.post(f"/sms/{sms.id}/reparse")
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_reparse_single_not_found_returns_404(session):
    async with await _client(session) as client:
        resp = await client.post("/sms/99999/reparse")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_reparse_force_new_creates_row_for_deferred_sms(session):
    """A [dup-defer] SMS reparsed with ?force_new=true must create a real
    transaction (manual confirmation that it's a genuine second charge),
    bypassing the matcher that would DEFER it again."""
    from financial_dashboard.db import Transaction

    # A pre-existing balance-less ICICI CC row the incoming SMS collides with.
    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=__import__("decimal").Decimal("5000.00"),
        currency="INR",
        transaction_date=datetime.date(2026, 6, 7),
        transaction_time=datetime.time(21, 36, 0),
        counterparty="TESTMERCHANT",
        card_mask="XX1234",
        balance=None,
        source="email",
        email_id=999,
    )
    session.add(existing)
    sms = SmsMessage(
        bank="icici",
        sender="AD-ICICIT-S",
        body=(
            "Rs 5,000.00 spent on ICICI Bank Card XX1234 on 07-Jun-26 at "
            "TESTMERCHANT. Avl Lmt: Rs 1,00,000.00. To dispute, call "
            "18002662/SMS BLOCK 1234 to 9215676766."
        ),
        received_at=datetime.datetime(2026, 6, 7, 16, 6, 0, tzinfo=datetime.UTC),
        status="skipped",
        parse_error="[dup-defer] possible duplicate",
    )
    session.add(sms)
    await session.commit()

    async with await _client(session) as client:
        # Without force_new it defers again (no row).
        resp = await client.post(f"/sms/{sms.id}/reparse")
        assert resp.status_code == 200
        assert resp.json()["new_status"] == "skipped"
        assert resp.json()["txn_id"] is None

        # With force_new it creates a real transaction.
        resp = await client.post(f"/sms/{sms.id}/reparse?force_new=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["new_status"] == "parsed"
    assert data["txn_id"] is not None


@pytest.mark.anyio
async def test_reparse_all_pending_and_error(session):
    good = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="Spent Rs.500 From HDFC Bank Card x1234 At Zomato On 2026-05-02:14:23:00 Bal Rs.1000",
        received_at=datetime.datetime(2026, 5, 2, 8, 53, 0, tzinfo=datetime.UTC),
        status="pending",
    )
    bad = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="OTP for your transaction is 123456",
        received_at=datetime.datetime(2026, 5, 2, 8, 54, 0, tzinfo=datetime.UTC),
        status="error",
        parse_error="prior parse error",
    )
    skipped = SmsMessage(
        bank="hdfc",
        sender="VK-HDFCBK",
        body="(already skipped row that must NOT be reprocessed)",
        received_at=datetime.datetime(2026, 5, 2, 8, 55, 0, tzinfo=datetime.UTC),
        status="skipped",
    )
    session.add_all([good, bad, skipped])
    await session.commit()

    async with await _client(session) as client:
        resp = await client.post("/sms/reparse-all-failed")
    assert resp.status_code == 200
    data = resp.json()
    # good → processed; bad → still_error; skipped is not in the target set.
    assert data["processed"] >= 1
    assert data["still_error"] >= 1
    # The "skipped" row stays untouched.
    await session.refresh(skipped)
    assert skipped.status == "skipped"


@pytest.mark.anyio
async def test_reparse_maskless_multi_card_dispatches_disambiguation_prompt(
    session, monkeypatch
):
    """A maskless CC bill-payment SMS with multiple CC candidates (and no
    statement amount-match) reaches the notification boundary: the web reparse
    route dispatches ``send_disambiguation_prompt`` exactly once, post-commit,
    with the candidate payload — never silently guessing an account."""
    from bank_sms_parser.models import Money, ParsedSms, SmsTransactionAlert

    from financial_dashboard.db import Account

    # Two IndusInd CC accounts → linker can't pick, amount-match finds nothing.
    a = Account(bank="indusind", type="credit_card", label="IndusInd A")
    b = Account(bank="indusind", type="credit_card", label="IndusInd B")
    session.add_all([a, b])
    await session.flush()

    parsed = ParsedSms(
        email_type="indusind_cc_payment_received_alert",
        bank="indusind",
        transaction=SmsTransactionAlert(
            direction="credit",
            amount=Money(amount=Decimal("133"), currency="INR"),
            transaction_date=datetime.date(2026, 5, 17),
            # maskless — no card_mask / account_mask
        ),
    )
    monkeypatch.setattr(
        "financial_dashboard.services.sms_pipeline.parse_sms",
        lambda *a, **k: parsed,
    )

    sms = SmsMessage(
        bank="indusind",
        sender="VK-INDBNK",
        body="<maskless multi-card payment body>",
        received_at=datetime.datetime(2026, 5, 17, 13, 12, 35, tzinfo=datetime.UTC),
        status="error",
    )
    session.add(sms)
    await session.commit()

    prompt_mock = AsyncMock()
    with patch(
        "financial_dashboard.services.telegram.send_disambiguation_prompt",
        prompt_mock,
    ):
        async with await _client(session) as client:
            resp = await client.post(f"/sms/{sms.id}/reparse")

    assert resp.status_code == 200
    # Fired exactly once at the notification boundary, carrying both candidates.
    prompt_mock.assert_awaited_once()
    payload = prompt_mock.await_args.args[0]
    assert set(payload["candidate_account_ids"]) == {a.id, b.id}
    assert payload["txn_id"] is not None
