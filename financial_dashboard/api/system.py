from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas import system as system_schemas
from financial_dashboard.services.system import (
    get_system_foreign_key_check,
    get_system_health,
    get_system_info,
)
from financial_dashboard.services.system_backups import (
    create_system_backup,
    list_system_backups,
)

router = APIRouter()


@router.get("/system/info")
async def system_info(
    session: AsyncSession = Depends(get_session),
) -> system_schemas.SystemInfoResponse:
    return await get_system_info(session)


@router.get("/system/health")
async def system_health(
    session: AsyncSession = Depends(get_session),
) -> system_schemas.SystemHealthResponse:
    return await get_system_health(session)


@router.get("/system/foreign-key-check")
async def system_foreign_key_check(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    session: AsyncSession = Depends(get_session),
) -> system_schemas.ForeignKeyCheckResponse:
    return await get_system_foreign_key_check(session, limit=limit)


@router.get("/system/backups")
async def system_backups(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    session: AsyncSession = Depends(get_session),
) -> system_schemas.SystemBackupListResponse:
    return await list_system_backups(session, limit=limit)


@router.post("/system/backups")
async def system_backup_create(
    session: AsyncSession = Depends(get_session),
) -> system_schemas.SystemBackupCreateResponse:
    return await create_system_backup(session)
