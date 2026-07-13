"""Cashflow JSON endpoints.

Thin passthrough over ``services.cashflow.report``: the service owns every
number and the date-range normalization, so these routes only hand the query
string to it and return what it builds.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import DEFAULT_TREND_MONTHS
from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.cashflow import CashflowSummary, TrendPoint
from financial_dashboard.services.cashflow.report import (
    MAX_TREND_MONTHS,
    cashflow_summary,
    cashflow_trend,
    resolve_range,
)

router = APIRouter()


@router.get("/cashflow/summary", response_model=CashflowSummary)
async def get_summary(
    date_from: str | None = None,
    date_to: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> CashflowSummary:
    """Every cashflow figure for one inclusive ``transaction_date`` range.

    Both bounds are ISO ``YYYY-MM-DD`` and both are optional: a missing or
    unparseable bound is defaulted rather than rejected, so the response always
    describes a concrete range and echoes back the one it actually used. The
    range it echoes is authoritative — a caller that typed a bad bound should
    read its range from the response, not from what it sent.
    """
    start, end = resolve_range(date_from, date_to)
    return await cashflow_summary(session, start, end)


@router.get("/cashflow/trend", response_model=list[TrendPoint])
async def get_trend(
    months: int = DEFAULT_TREND_MONTHS,
    session: AsyncSession = Depends(get_session),
) -> list[TrendPoint]:
    """Month-by-month income, expense and net-invested over a trailing window.

    The window ends with the current (partial) month and is pre-seeded, so a
    month with no transactions comes back as a zero point rather than as a gap:
    the series always has exactly ``months`` entries, oldest first.

    ``months`` is clamped to 1..``MAX_TREND_MONTHS`` rather than rejected, so an
    out-of-range widget value still renders a chart instead of a 422.
    """
    months = max(1, min(months, MAX_TREND_MONTHS))
    return await cashflow_trend(session, months)
