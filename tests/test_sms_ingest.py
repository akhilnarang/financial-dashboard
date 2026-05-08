"""Tests for SMS ingest: schema validation, service behavior, and endpoint."""

import datetime

import pytest
from pydantic import ValidationError

from bank_email_fetcher.schemas.sms import SmsIngestRequest


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSmsIngestRequestSchema:
    def _valid_payload(self) -> dict:
        return {
            "bank": "HDFC",
            "sender": "VK-HDFCBK",
            "body": "Sent Rs.500 from A/c XX1234 to ...",
            "received_at": "2026-05-02T14:23:11+05:30",
        }

    def test_valid_payload_parses(self):
        req = SmsIngestRequest.model_validate(self._valid_payload())
        assert req.bank == "HDFC"
        assert req.sender == "VK-HDFCBK"
        assert req.body == "Sent Rs.500 from A/c XX1234 to ..."
        # Stored as UTC-aware datetime
        assert req.received_at.tzinfo is not None
        assert req.received_at.utcoffset() == datetime.timedelta(0)

    def test_strips_whitespace_on_text_fields(self):
        payload = self._valid_payload()
        payload["bank"] = "  HDFC  "
        payload["sender"] = "\tVK-HDFCBK\n"
        payload["body"] = "  hello  "
        req = SmsIngestRequest.model_validate(payload)
        assert req.bank == "HDFC"
        assert req.sender == "VK-HDFCBK"
        assert req.body == "hello"

    @pytest.mark.parametrize("field", ["bank", "sender", "body"])
    def test_rejects_empty_string(self, field):
        payload = self._valid_payload()
        payload[field] = ""
        with pytest.raises(ValidationError):
            SmsIngestRequest.model_validate(payload)

    @pytest.mark.parametrize("field", ["bank", "sender", "body"])
    def test_rejects_whitespace_only(self, field):
        payload = self._valid_payload()
        payload[field] = "   "
        with pytest.raises(ValidationError):
            SmsIngestRequest.model_validate(payload)

    @pytest.mark.parametrize("field", ["bank", "sender", "body", "received_at"])
    def test_rejects_missing_field(self, field):
        payload = self._valid_payload()
        payload.pop(field)
        with pytest.raises(ValidationError):
            SmsIngestRequest.model_validate(payload)

    def test_rejects_naive_datetime(self):
        payload = self._valid_payload()
        payload["received_at"] = "2026-05-02T14:23:11"  # no tz
        with pytest.raises(ValidationError):
            SmsIngestRequest.model_validate(payload)

    def test_normalizes_to_utc(self):
        payload = self._valid_payload()
        # 14:23:11+05:30 == 08:53:11 UTC
        req = SmsIngestRequest.model_validate(payload)
        assert req.received_at == datetime.datetime(
            2026, 5, 2, 8, 53, 11, tzinfo=datetime.UTC
        )


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bank_email_fetcher.db import Base, SmsMessage
from bank_email_fetcher.services.sms import ingest_sms


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


def _payload() -> SmsIngestRequest:
    return SmsIngestRequest.model_validate(
        {
            "bank": "HDFC",
            "sender": "VK-HDFCBK",
            "body": "Sent Rs.500 from A/c XX1234 to ...",
            "received_at": "2026-05-02T14:23:11+05:30",
        }
    )


@pytest.mark.anyio
class TestIngestSmsService:
    async def test_happy_path_stores_row(self, session):
        row, stored = await ingest_sms(session, _payload())

        assert stored is True
        assert row.id is not None
        assert row.bank == "HDFC"
        assert row.sender == "VK-HDFCBK"
        assert row.body == "Sent Rs.500 from A/c XX1234 to ..."

        # received_at stored as UTC (08:53:11 == 14:23:11+05:30)
        assert row.received_at.replace(tzinfo=datetime.UTC) == datetime.datetime(
            2026, 5, 2, 8, 53, 11, tzinfo=datetime.UTC
        )

        # Confirm a single row was actually committed
        result = await session.execute(select(SmsMessage))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].id == row.id

    async def test_dedup_returns_existing_row(self, session):
        row1, stored1 = await ingest_sms(session, _payload())
        assert stored1 is True

        row2, stored2 = await ingest_sms(session, _payload())
        assert stored2 is False
        assert row2.id == row1.id

        result = await session.execute(select(SmsMessage))
        rows = result.scalars().all()
        assert len(rows) == 1

    async def test_dedup_ignores_bank_difference(self, session):
        """Same (sender, received_at, body) but different bank label is still a duplicate.

        Per the spec: dedup key omits `bank`. The existing row's bank is NOT updated.
        """
        row1, stored1 = await ingest_sms(session, _payload())
        assert stored1 is True
        assert row1.bank == "HDFC"

        # Repost with a corrected bank label
        repost = SmsIngestRequest.model_validate(
            {
                "bank": "ICICI",  # different
                "sender": "VK-HDFCBK",
                "body": "Sent Rs.500 from A/c XX1234 to ...",
                "received_at": "2026-05-02T14:23:11+05:30",
            }
        )
        row2, stored2 = await ingest_sms(session, repost)
        assert stored2 is False
        assert row2.id == row1.id
        assert row2.bank == "HDFC"  # unchanged

    async def test_different_sms_does_not_dedup(self, session):
        row1, stored1 = await ingest_sms(session, _payload())
        assert stored1 is True

        other = SmsIngestRequest.model_validate(
            {
                "bank": "HDFC",
                "sender": "VK-HDFCBK",
                "body": "A different message body",
                "received_at": "2026-05-02T14:23:11+05:30",
            }
        )
        row2, stored2 = await ingest_sms(session, other)
        assert stored2 is True
        assert row2.id != row1.id

        result = await session.execute(select(SmsMessage))
        rows = result.scalars().all()
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Endpoint integration tests
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from bank_email_fetcher.api import router as api_router
from bank_email_fetcher.core.deps import get_session


def _build_test_app(session_dep):
    """Mount the real api_router with `get_session` overridden to a test session."""
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_session] = session_dep
    return app


def _valid_request_json() -> dict:
    return {
        "bank": "HDFC",
        "sender": "VK-HDFCBK",
        "body": "Sent Rs.500 from A/c XX1234 to ...",
        "received_at": "2026-05-02T14:23:11+05:30",
    }


@pytest.mark.anyio
class TestSmsEndpoint:
    async def _client(self, session):
        async def _override():
            yield session

        app = _build_test_app(_override)
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def test_post_new_returns_201_empty_body(self, session):
        async with await self._client(session) as client:
            r = await client.post("/api/sms", json=_valid_request_json())
            assert r.status_code == 201
            assert r.content == b""

        result = await session.execute(select(SmsMessage))
        assert len(result.scalars().all()) == 1

    async def test_post_duplicate_returns_204_empty_body(self, session):
        async with await self._client(session) as client:
            r1 = await client.post("/api/sms", json=_valid_request_json())
            assert r1.status_code == 201

            r2 = await client.post("/api/sms", json=_valid_request_json())
            assert r2.status_code == 204
            assert r2.content == b""

        result = await session.execute(select(SmsMessage))
        assert len(result.scalars().all()) == 1

    @pytest.mark.parametrize(
        "mutation",
        [
            {"bank": ""},
            {"sender": "   "},
            {"body": ""},
            {"received_at": "2026-05-02T14:23:11"},  # naive
            {"received_at": "not-a-date"},
        ],
    )
    async def test_post_invalid_returns_422(self, session, mutation):
        payload = _valid_request_json()
        payload.update(mutation)
        async with await self._client(session) as client:
            r = await client.post("/api/sms", json=payload)
            assert r.status_code == 422

        result = await session.execute(select(SmsMessage))
        assert result.scalars().all() == []

    async def test_post_missing_field_returns_422(self, session):
        payload = _valid_request_json()
        del payload["sender"]
        async with await self._client(session) as client:
            r = await client.post("/api/sms", json=payload)
            assert r.status_code == 422

        result = await session.execute(select(SmsMessage))
        assert result.scalars().all() == []
