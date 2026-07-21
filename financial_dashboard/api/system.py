"""Deployment and SQLite operational diagnostics."""

from typing import Annotated

from fastapi import APIRouter, Query

from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.schemas import system as system_schemas
from financial_dashboard.services.database import (
    get_system_foreign_key_check,
    get_system_health,
)
from financial_dashboard.services.system import get_system_info
from financial_dashboard.services.system_backups import (
    create_system_backup,
    list_system_backups,
)

router = APIRouter()


@router.get("/system/info")
async def system_info(session: AsyncSessionDep) -> system_schemas.SystemInfoResponse:
    """Return deployed revision, runtime, parser, and schema metadata."""
    return await get_system_info(session)


@router.get("/system/health")
async def system_health(
    session: AsyncSessionDep,
) -> system_schemas.SystemHealthResponse:
    """Return connectivity and bounded SQLite health diagnostics."""
    return await get_system_health(session)


@router.get("/system/foreign-key-check")
async def system_foreign_key_check(
    session: AsyncSessionDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> system_schemas.ForeignKeyCheckResponse:
    """Return a bounded list of SQLite foreign-key violations."""
    return await get_system_foreign_key_check(session, limit=limit)


@router.get("/system/backups")
async def system_backups(
    session: AsyncSessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> system_schemas.SystemBackupListResponse:
    """List verified backup metadata without exposing filesystem paths."""
    return await list_system_backups(session, limit=limit)


@router.post("/system/backups")
async def system_backup_create(
    session: AsyncSessionDep,
) -> system_schemas.SystemBackupCreateResponse:
    """Create and verify an online SQLite backup."""
    return await create_system_backup(session)
