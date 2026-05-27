import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class CasUploadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    portfolio_key: str
    depository_source: str
    investor_name: str | None = None
    statement_date: datetime.date
    grand_total: Decimal
    portfolio_ok: bool
    portfolio_delta: Decimal | None = None
