import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class StatementAccountLink(BaseModel):
    id: int
    bank: str
    label: str
    type: str


class StatementReadBase(BaseModel):
    id: int
    account_id: int
    email_id: int | None
    bank: str
    filename: str
    filename_truncated: bool
    status: str
    parsed_transaction_count: int | None
    matched_count: int | None
    missing_count: int | None
    imported_count: int | None
    error: str | None
    error_truncated: bool
    created_at: datetime.datetime | None
    account: StatementAccountLink | None


class CcStatementRead(StatementReadBase):
    source_kind: str
    card_mask: str | None
    statement_name: str | None
    statement_name_truncated: bool
    due_date: str | None
    total_amount_due: str | None
    minimum_amount_due: str | None
    payment_status: str | None
    payment_paid_at: datetime.datetime | None
    payment_paid_amount: Decimal | None
    payment_last_reminded_at: datetime.datetime | None


class BankStatementRead(StatementReadBase):
    account_mask: str | None
    account_holder_name: str | None
    account_holder_name_truncated: bool
    opening_balance: str | None
    closing_balance: str | None
    statement_period_start: str | None
    statement_period_end: str | None


class CcStatementListResponse(BaseModel):
    items: Annotated[list[CcStatementRead], Field(max_length=100)]
    returned_count: Annotated[int, Field(ge=0, le=100)]
    total_count: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=100)]
    offset: Annotated[int, Field(ge=0)]


class BankStatementListResponse(BaseModel):
    items: Annotated[list[BankStatementRead], Field(max_length=100)]
    returned_count: Annotated[int, Field(ge=0, le=100)]
    total_count: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=100)]
    offset: Annotated[int, Field(ge=0)]


class StatementReconciliationSummary(BaseModel):
    status: Literal["absent", "parsed", "too_large", "malformed"]
    matched_transaction_ids: Annotated[list[int], Field(max_length=100)]
    matched_transaction_ids_truncated: bool
    imported_transaction_ids: Annotated[list[int], Field(max_length=100)]
    imported_transaction_ids_truncated: bool
    ambiguous_entry_count: int | None
    import_error_entry_count: int | None


class CcStatementDetailResponse(CcStatementRead):
    reconciliation: StatementReconciliationSummary


class BankStatementDetailResponse(BankStatementRead):
    reconciliation: StatementReconciliationSummary


class StatementBatchRequest(BaseModel):
    ids: Annotated[list[int], Field(min_length=1, max_length=100)]

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 9_223_372_036_854_775_807 for value in values):
            raise ValueError("ids must fit positive database integers")
        if len(set(values)) != len(values):
            raise ValueError("ids must be unique")
        return values


class CcStatementBatchResponse(BaseModel):
    items: Annotated[list[CcStatementRead], Field(max_length=100)]
    missing_ids: Annotated[list[int], Field(max_length=100)]


class BankStatementBatchResponse(BaseModel):
    items: Annotated[list[BankStatementRead], Field(max_length=100)]
    missing_ids: Annotated[list[int], Field(max_length=100)]
