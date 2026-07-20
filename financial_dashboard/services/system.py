"""Application/runtime metadata for system information endpoints."""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Setting
from financial_dashboard.schemas import system as system_schemas
from financial_dashboard.services import system_metadata

SCHEMA_VERSION_SETTING = "migrations.schema_version"


async def _schema_state(session: AsyncSession) -> system_schemas.SchemaState:
    """Read the schema version and ordered migration markers."""
    schema_version_row = await session.get(Setting, SCHEMA_VERSION_SETTING)
    schema_version = (
        schema_version_row.value
        if schema_version_row is not None and schema_version_row.value
        else None
    )
    applied_markers = (
        (
            await session.execute(
                select(Setting.key)
                .where(
                    Setting.key.like("migrations.%"),
                    Setting.key != SCHEMA_VERSION_SETTING,
                )
                .order_by(Setting.key)
            )
        )
        .scalars()
        .all()
    )
    return system_schemas.SchemaState(
        schema_version=schema_version,
        applied_migration_markers=list(applied_markers),
    )


async def get_system_info(session: AsyncSession) -> system_schemas.SystemInfoResponse:
    """Return deployment, runtime, parser-package, and schema metadata."""
    runtime = await asyncio.to_thread(system_metadata.collect_runtime_metadata)
    return system_schemas.SystemInfoResponse(
        package_name=system_metadata.APP_DISTRIBUTION,
        package_version=runtime.package_version,
        app_revision=runtime.app_revision.value,
        app_revision_source=runtime.app_revision.source,
        runtime=runtime.runtime,
        schema_state=await _schema_state(session),
        parser_packages=runtime.parser_packages,
    )
