import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field


class AccountCardRead(BaseModel):
    id: int
    label: str | None
    is_primary: bool
    active: bool
    card_mask: str | None


class AccountRead(BaseModel):
    id: int
    bank: str
    label: str
    type: str
    active: bool
    account_mask: str | None
    cards: Annotated[list[AccountCardRead], Field(max_length=50)]
    cards_truncated: bool


class AccountListResponse(BaseModel):
    items: Annotated[list[AccountRead], Field(max_length=100)]
    returned_count: Annotated[int, Field(ge=0, le=100)]
    total_count: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=100)]
    offset: Annotated[int, Field(ge=0)]


class AccountBalanceSnapshotRead(BaseModel):
    snapshot_id: int
    category: str
    as_of_date: datetime.date
    value: Decimal
    currency: str


class AccountDetailResponse(AccountRead):
    transaction_count: Annotated[int, Field(ge=0)]
    cc_statement_count: Annotated[int, Field(ge=0)]
    bank_statement_count: Annotated[int, Field(ge=0)]
    latest_balance_snapshots: Annotated[
        list[AccountBalanceSnapshotRead], Field(max_length=50)
    ]
