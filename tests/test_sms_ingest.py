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
    return SmsIngestRequest.model_validate({
        "bank": "HDFC",
        "sender": "VK-HDFCBK",
        "body": "Sent Rs.500 from A/c XX1234 to ...",
        "received_at": "2026-05-02T14:23:11+05:30",
    })


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
