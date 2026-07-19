import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class TransactionNoteUpdate(BaseModel):
    note: str = ""


class TransactionNoteResponse(BaseModel):
    ok: bool
    note: str | None = None


class TransactionCategoryUpdate(BaseModel):
    category: str = ""


class TransactionCategoryResponse(BaseModel):
    ok: bool
    category: str | None = None


class TransactionRelinkUpdate(BaseModel):
    """Request body for manual relink. Either field may be null to clear
    the corresponding link. If only ``card_id`` is given, the service
    derives ``account_id`` from the card's owning account."""

    account_id: int | None = None
    card_id: int | None = None


class TransactionRelinkResponse(BaseModel):
    ok: bool
    account_id: int | None = None
    card_id: int | None = None
    account_label: str | None = None
    card_label: str | None = None
    statement_marked_paid: bool = False


class TransactionRead(BaseModel):
    id: int
    bank: str
    email_type: str
    direction: str
    amount: Decimal
    currency: str | None
    transaction_date: datetime.date | None
    transaction_time: datetime.time | None
    counterparty: str | None
    card_mask: str | None
    account_mask: str | None
    reference_number: str | None
    channel: str | None
    balance: Decimal | None
    account_id: int | None
    card_id: int | None
    email_id: int | None
    sms_message_id: int | None
    statement_upload_id: int | None
    bank_statement_upload_id: int | None
    source: str | None
    category: str | None
    category_method: str | None
    review_status: str | None
    created_at: datetime.datetime | None
    enriched_at: datetime.datetime | None


class TransactionListResponse(BaseModel):
    items: Annotated[list[TransactionRead], Field(max_length=100)]
    returned_count: Annotated[int, Field(ge=0, le=100)]
    total_count: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=100)]
    offset: Annotated[int, Field(ge=0)]


class TransactionAccountLink(BaseModel):
    id: int
    bank: str
    label: str
    type: str


class TransactionCardLink(BaseModel):
    id: int
    label: str | None
    card_mask: str | None
    is_primary: bool


class TransactionSourceLink(BaseModel):
    id: int
    status: str | None
    timestamp: datetime.datetime | None


class TransactionStatementLink(BaseModel):
    id: int
    kind: str
    status: str
    account_id: int


class TransactionDetailResponse(TransactionRead):
    raw_description: str | None
    raw_description_truncated: bool
    note: str | None
    note_truncated: bool
    category_confidence: float | None
    category_model: str | None
    category_input_hash: str | None
    category_vocab_version: int | None
    categorized_at: datetime.datetime | None
    review_reason: str | None
    last_notified_at: datetime.datetime | None
    notify_attempts: int | None
    notified_channel: str | None
    account: TransactionAccountLink | None
    card: TransactionCardLink | None
    email: TransactionSourceLink | None
    sms: TransactionSourceLink | None
    statement: TransactionStatementLink | None
    may_affect_cc_payment_state: bool


class TransactionBatchRequest(BaseModel):
    ids: Annotated[list[int], Field(min_length=1, max_length=100)]

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 9_223_372_036_854_775_807 for value in values):
            raise ValueError("ids must fit positive database integers")
        if len(set(values)) != len(values):
            raise ValueError("ids must be unique")
        return values


class TransactionBatchResponse(BaseModel):
    items: Annotated[list[TransactionRead], Field(max_length=100)]
    missing_ids: Annotated[list[int], Field(max_length=100)]
