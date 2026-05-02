"""Schema for SMS ingest endpoint."""

import datetime

from pydantic import BaseModel, field_validator


class SmsIngestRequest(BaseModel):
    bank: str
    sender: str
    body: str
    received_at: datetime.datetime

    @field_validator("bank", "sender", "body")
    @classmethod
    def _strip_and_require_nonempty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("received_at")
    @classmethod
    def _require_aware_and_normalize_utc(
        cls, v: datetime.datetime
    ) -> datetime.datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("received_at must be timezone-aware (ISO8601 with offset)")
        return v.astimezone(datetime.UTC)
