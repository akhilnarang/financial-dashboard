import datetime
from decimal import Decimal

from pydantic import BaseModel


class SourceBalance(BaseModel):
    label: str
    category: str
    kind: str
    value: Decimal
    as_of_date: datetime.date
    stale: bool
    unreconciled: bool = False


class HoldingBreakdown(BaseModel):
    asset_class: str
    label: str
    value: Decimal


class CategoryGroup(BaseModel):
    category: str
    kind: str
    total: Decimal
    rows: list[SourceBalance]


class TrendPoint(BaseModel):
    month: str
    value: Decimal


class NetWorthSummary(BaseModel):
    net_worth: Decimal
    total_assets: Decimal
    total_liabilities: Decimal
    groups: list[CategoryGroup]
    investment_breakdown: list[HoldingBreakdown]
    has_stale: bool
    as_of: datetime.date
