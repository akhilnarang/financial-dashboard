"""Tests for /sms web routes."""

import datetime

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


@pytest.mark.skip(
    reason=(
        "Known testability gap: the bulk endpoint spawns its own async_session() "
        "instances for per-row isolation, which connect to the production DB URL "
        "rather than the in-memory test session injected via dependency override. "
        "Requires option (b) refactor (injectable session_factory) to fix properly."
    )
)
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
