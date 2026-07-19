"""Email endpoint request and response schemas."""

import datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator

from financial_dashboard.schemas.common import DatabaseId, DatabaseIdBatch


class ReparseEmailResponse(BaseModel):
    message: str
    new_status: Literal["parsed", "skipped"]
    txn_id: int | None = None


class ReparseAllFailedResponse(BaseModel):
    succeeded: int
    skipped: int
    failed: int


PreviewToken = Annotated[
    str,
    Field(
        min_length=67,
        max_length=67,
        pattern=r"^v1\.[0-9a-f]{64}$",
    ),
]


class DuplicateResolutionRequest(BaseModel):
    """Preview by default; applying requires the token returned by a preview."""

    transaction_id: DatabaseId
    apply: bool = False
    preview_token: PreviewToken | None = None

    @model_validator(mode="after")
    def validate_apply_token(self) -> Self:
        if self.apply and not self.preview_token:
            raise ValueError("preview_token is required when apply=true")
        if not self.apply and self.preview_token is not None:
            raise ValueError("preview_token is only accepted when apply=true")
        return self


class TransactionEnrichmentState(BaseModel):
    transaction_date: datetime.date | None
    transaction_time: datetime.time | None
    counterparty: str | None
    card_mask: str | None
    account_mask: str | None
    reference_number: str | None
    channel: str | None
    balance: Decimal | None
    raw_description: str | None
    email_id: int | None
    source: str | None


EnrichmentValue = str | datetime.date | datetime.time | Decimal | None


class EnrichmentFieldChange(BaseModel):
    before: EnrichmentValue
    after: EnrichmentValue


class DuplicateEnrichmentDiff(BaseModel):
    changed_fields: list[str]
    filled: list[str]
    overwritten: list[str]
    changes: dict[str, EnrichmentFieldChange]


class DuplicateResolutionResponse(BaseModel):
    mode: Literal["preview", "applied"]
    email_id: int
    transaction_id: int
    email_status: Literal["skipped", "parsed"]
    preview_token: str
    before: TransactionEnrichmentState
    after: TransactionEnrichmentState
    diff: DuplicateEnrichmentDiff


class EmailRuleSummary(BaseModel):
    id: int
    bank: str
    email_kind: str | None


class EmailTransactionLink(BaseModel):
    id: int
    email_type: str
    direction: str
    source: str | None


class EmailStatementLink(BaseModel):
    id: int
    kind: Literal["cc", "bank", "cas"]
    status: str


class EmailRead(BaseModel):
    id: int
    provider: str
    source_id: int | None
    sender: str | None
    sender_truncated: bool
    subject: str | None
    subject_truncated: bool
    received_at: datetime.datetime | None
    fetched_at: datetime.datetime | None
    status: str | None
    error: str | None
    error_truncated: bool
    rule: EmailRuleSummary | None
    transactions: Annotated[list[EmailTransactionLink], Field(max_length=10)]
    transactions_truncated: bool
    statements: Annotated[list[EmailStatementLink], Field(max_length=10)]
    statements_truncated: bool


class EmailListResponse(BaseModel):
    items: Annotated[list[EmailRead], Field(max_length=100)]
    returned_count: Annotated[int, Field(ge=0, le=100)]
    total_count: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=100)]
    offset: Annotated[int, Field(ge=0)]


class EmailDetailResponse(EmailRead):
    message_id: str
    message_id_truncated: bool
    remote_id: str | None
    remote_id_truncated: bool


class EmailRawResponse(BaseModel):
    email_id: int
    content_type: Literal["text/plain", "text/html"]
    body: Annotated[str, Field(max_length=100_000)]
    body_truncated: bool
    raw_byte_size: Annotated[int, Field(ge=0)]


class EmailBatchRequest(BaseModel):
    ids: DatabaseIdBatch


class EmailBatchResponse(BaseModel):
    items: Annotated[list[EmailRead], Field(max_length=100)]
    missing_ids: Annotated[list[int], Field(max_length=100)]
