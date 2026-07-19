from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.system import (
    ForeignKeyCheckResponse,
    SystemHealthResponse,
    SystemInfoResponse,
)
from financial_dashboard.services.system import (
    get_system_foreign_key_check,
    get_system_health,
    get_system_info,
)

router = APIRouter()


@router.get("/system/info")
async def system_info(
    session: AsyncSession = Depends(get_session),
) -> SystemInfoResponse:
    return await get_system_info(session)


@router.get("/system/health")
async def system_health(
    session: AsyncSession = Depends(get_session),
) -> SystemHealthResponse:
    return await get_system_health(session)


@router.get("/system/foreign-key-check")
async def system_foreign_key_check(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    session: AsyncSession = Depends(get_session),
) -> ForeignKeyCheckResponse:
    return await get_system_foreign_key_check(session, limit=limit)
