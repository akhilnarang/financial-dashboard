"""Schema for SMS ingest endpoint."""

import datetime

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


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


class SmsTransactionLink(BaseModel):
    id: int
    email_type: str
    direction: str
    source: str | None


class SmsRead(BaseModel):
    id: int
    bank: str
    bank_truncated: bool
    sender: str
    sender_truncated: bool
    received_at: datetime.datetime
    created_at: datetime.datetime
    status: str
    parse_error: str | None
    parse_error_truncated: bool
    parsed_at: datetime.datetime | None
    transaction: SmsTransactionLink | None


class SmsListResponse(BaseModel):
    items: Annotated[list[SmsRead], Field(max_length=100)]
    returned_count: Annotated[int, Field(ge=0, le=100)]
    total_count: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=100)]
    offset: Annotated[int, Field(ge=0)]


class SmsDetailResponse(SmsRead):
    body: str
    body_truncated: bool
    attached_transaction_ids: Annotated[list[int], Field(max_length=100)]
    attached_transactions_truncated: bool


class SmsBatchRequest(BaseModel):
    ids: Annotated[list[int], Field(min_length=1, max_length=100)]

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 9_223_372_036_854_775_807 for value in values):
            raise ValueError("ids must fit positive database integers")
        if len(set(values)) != len(values):
            raise ValueError("ids must be unique")
        return values


class SmsBatchResponse(BaseModel):
    items: Annotated[list[SmsRead], Field(max_length=100)]
    missing_ids: Annotated[list[int], Field(max_length=100)]
