"""Email endpoint request and response schemas."""

import datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator


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

    transaction_id: int
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
