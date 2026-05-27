import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class ManualItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    kind: str
    category: str
    active: bool
    notes: str | None = None


class ManualValueCreate(BaseModel):
    value: Decimal
    as_of_date: datetime.date
