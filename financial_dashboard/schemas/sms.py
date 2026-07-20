"""Schema for SMS ingest endpoint."""

import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from financial_dashboard.schemas.common import DatabaseIdBatch


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
    ids: DatabaseIdBatch


class SmsBatchResponse(BaseModel):
    items: Annotated[list[SmsRead], Field(max_length=100)]
    missing_ids: Annotated[list[int], Field(max_length=100)]


class SmsParsedTransactionPreview(BaseModel):
    bank: Annotated[str, Field(max_length=64)]
    email_type: Annotated[str, Field(max_length=128)]
    direction: Annotated[str, Field(max_length=16)]
    amount: Decimal
    currency: Annotated[str | None, Field(max_length=8)]
    transaction_date: datetime.date | None
    transaction_time: datetime.time | None
    counterparty: Annotated[str | None, Field(max_length=1_000)]
    card_mask: Annotated[str | None, Field(max_length=8)]
    account_mask: Annotated[str | None, Field(max_length=8)]
    reference_number: Annotated[str | None, Field(max_length=256)]
    channel: Annotated[str | None, Field(max_length=64)]
    balance: Decimal | None


class SmsParserPreview(BaseModel):
    disposition: Literal["transaction", "non_transaction", "skipped", "error"]
    email_type: Annotated[str | None, Field(max_length=128)]
    ledger_role: Annotated[str | None, Field(max_length=32)]
    error: Annotated[str | None, Field(max_length=1_000)]
    transaction: SmsParsedTransactionPreview | None


class SmsMergePreview(BaseModel):
    action: Literal[
        "none",
        "notify_only",
        "declined",
        "match",
        "insert",
        "defer",
    ]
    target_transaction_id: int | None
    match_kind: Annotated[str | None, Field(max_length=32)]
    changed_fields: Annotated[list[str], Field(max_length=32)]
    identity_conflicts: Annotated[list[str], Field(max_length=8)]


class SmsParsePreviewResponse(BaseModel):
    sms_id: int
    current_status: Annotated[str, Field(max_length=32)]
    current_transaction_id: int | None
    parser: SmsParserPreview
    merge: SmsMergePreview
