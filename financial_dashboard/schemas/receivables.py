"""Request and response bodies for the receivables ledger."""

import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from financial_dashboard.db.enums import ReceivableRole, ReceivableStatus
from financial_dashboard.schemas.common import DatabaseId

Amount = Annotated[Decimal, Field(gt=0, max_digits=12, decimal_places=2)]
Person = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
]
Note = Annotated[str, Field(max_length=2000)]
LinkIds = Annotated[list[DatabaseId], Field(max_length=100)]


class ReceivableCreate(BaseModel):
    person: Person
    amount: Amount
    note: Note | None = None
    charge_transaction_ids: LinkIds = []
    payment_transaction_ids: LinkIds = []


class ReceivableUpdate(BaseModel):
    """Change only the fields the body names. A null note clears the note."""

    amount: Amount | None = None
    note: Note | None = None
    status: ReceivableStatus | None = None


class ReceivableLinkCreate(BaseModel):
    transaction_id: DatabaseId
    role: ReceivableRole


class ReceivableRead(BaseModel):
    id: int
    person: str
    amount: Decimal
    note: str | None
    status: ReceivableStatus
    created_at: datetime.datetime
    closed_at: datetime.datetime | None
    paid: Decimal
    outstanding: Decimal
    charge_transaction_ids: list[int]
    payment_transaction_ids: list[int]


class ReceivableListResponse(BaseModel):
    items: list[ReceivableRead]
