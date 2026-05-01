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
