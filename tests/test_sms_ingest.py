"""Tests for SMS ingest: service behavior and endpoint."""

import datetime

import pytest
from sqlalchemy import select

from financial_dashboard.db import SmsMessage
from financial_dashboard.schemas.sms import SmsIngestRequest
from financial_dashboard.services.sms import ingest_sms


def _valid_request_json() -> dict:
    return {
        "bank": "HDFC",
        "sender": "VK-HDFCBK",
        "body": "Sent Rs.500 from A/c XX1234 to ...",
        "received_at": "2026-05-02T14:23:11+05:30",
    }


def _payload() -> SmsIngestRequest:
    return SmsIngestRequest.model_validate(_valid_request_json())


@pytest.mark.anyio
class TestIngestSmsService:
    async def test_happy_path_stores_row(self, session):
        payload = _valid_request_json()
        payload["bank"] = "  HDFC  "
        payload["sender"] = "\tVK-HDFCBK\n"
        row, stored = await ingest_sms(
            session, SmsIngestRequest.model_validate(payload)
        )

        assert stored is True
        assert row.id is not None
        assert row.bank == "HDFC"
        assert row.sender == "VK-HDFCBK"
        assert row.body == "Sent Rs.500 from A/c XX1234 to ..."

        # received_at stored as UTC (08:53:11 == 14:23:11+05:30)
        assert row.received_at.replace(tzinfo=datetime.UTC) == datetime.datetime(
            2026, 5, 2, 8, 53, 11, tzinfo=datetime.UTC
        )

        result = await session.execute(select(SmsMessage))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].id == row.id

    async def test_dedup_ignores_bank_difference(self, session):
        """Same (sender, received_at, body) but different bank label is still a duplicate.

        Per the spec: dedup key omits `bank`. The existing row's bank is NOT updated.
        """
        row1, stored1 = await ingest_sms(session, _payload())
        assert stored1 is True
        assert row1.bank == "HDFC"

        repost = _valid_request_json()
        repost["bank"] = "ICICI"
        row2, stored2 = await ingest_sms(
            session, SmsIngestRequest.model_validate(repost)
        )
        assert stored2 is False
        assert row2.id == row1.id
        assert row2.bank == "HDFC"  # unchanged

    async def test_different_sms_does_not_dedup(self, session):
        row1, stored1 = await ingest_sms(session, _payload())
        assert stored1 is True

        other = _valid_request_json()
        other["body"] = "A different message body"
        row2, stored2 = await ingest_sms(
            session, SmsIngestRequest.model_validate(other)
        )
        assert stored2 is True
        assert row2.id != row1.id

        result = await session.execute(select(SmsMessage))
        rows = result.scalars().all()
        assert len(rows) == 2


@pytest.mark.anyio
class TestSmsEndpoint:
    async def test_post_new_returns_201_empty_body(self, client, session):
        r = await client.post("/api/sms", json=_valid_request_json())
        assert r.status_code == 201
        assert r.content == b""

        result = await session.execute(select(SmsMessage))
        assert len(result.scalars().all()) == 1

    async def test_post_duplicate_returns_204_empty_body(self, client, session):
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
            {"sender": "   "},
            {"received_at": "2026-05-02T14:23:11"},  # naive
        ],
    )
    async def test_post_invalid_returns_422(self, client, session, mutation):
        payload = _valid_request_json()
        payload.update(mutation)
        r = await client.post("/api/sms", json=payload)
        assert r.status_code == 422

        result = await session.execute(select(SmsMessage))
        assert result.scalars().all() == []
