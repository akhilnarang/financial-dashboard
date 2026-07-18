import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Setting
from financial_dashboard.schemas import system as system_schemas
from financial_dashboard.services import system_metadata

SCHEMA_VERSION_SETTING = "migrations.schema_version"


async def _schema_state(session: AsyncSession) -> system_schemas.SchemaState:
    schema_version_row = await session.get(Setting, SCHEMA_VERSION_SETTING)
    schema_version = None
    if schema_version_row is not None and schema_version_row.value:
        schema_version = schema_version_row.value

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
    runtime_metadata = await asyncio.to_thread(system_metadata.collect_runtime_metadata)
    return system_schemas.SystemInfoResponse(
        package_name=system_metadata.APP_DISTRIBUTION,
        package_version=runtime_metadata.package_version,
        app_revision=runtime_metadata.app_revision.value,
        app_revision_source=runtime_metadata.app_revision.source,
        runtime=runtime_metadata.runtime,
        schema_state=await _schema_state(session),
        parser_packages=runtime_metadata.parser_packages,
    )
