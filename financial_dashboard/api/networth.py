"""Net-worth JSON endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.networth import NetWorthSummary, TrendPoint
from financial_dashboard.services.networth import current_networth, monthly_trend

router = APIRouter()


@router.get("/networth/summary", response_model=NetWorthSummary)
async def get_summary(session: AsyncSession = Depends(get_session)) -> NetWorthSummary:
    return await current_networth(session)


@router.get("/networth/trend", response_model=list[TrendPoint])
async def get_trend(session: AsyncSession = Depends(get_session)) -> list[TrendPoint]:
    return await monthly_trend(session)
