from typing import Annotated, Literal, NamedTuple

from pydantic import BaseModel, Field


class AppRevisionResult(NamedTuple):
    value: str | None
    source: str


class DistributionMetadata(NamedTuple):
    version: str | None
    vcs_commit_id: str | None


class RuntimeInfo(BaseModel):
    implementation: str
    python_version: str


class SchemaState(BaseModel):
    schema_version: str | None = None
    applied_migration_markers: list[str]


class ParserPackageInfo(BaseModel):
    package: str
    version: str | None = None
    vcs_commit_id: str | None = None


class SystemRuntimeMetadata(NamedTuple):
    package_version: str | None
    app_revision: AppRevisionResult
    runtime: RuntimeInfo
    parser_packages: list[ParserPackageInfo]


class SystemInfoResponse(BaseModel):
    package_name: str
    package_version: str | None = None
    app_revision: str | None = None
    app_revision_source: str
    runtime: RuntimeInfo
    schema_state: SchemaState
    parser_packages: list[ParserPackageInfo]


HealthStatus = Literal["ok", "degraded", "unavailable"]
DatabaseBackend = Literal["sqlite", "other"]
ForeignKeyCheckStatus = Literal["ok", "violations", "unavailable", "unsupported"]
SQLiteSchemaName = Annotated[str, Field(min_length=1, max_length=256)]
SQLiteJournalMode = Literal[
    "delete", "truncate", "persist", "memory", "wal", "off", "unknown"
]
SQLiteSynchronousMode = Literal["off", "normal", "full", "extra", "unknown"]
SQLiteQuickCheck = Literal["ok", "failed", "unavailable"]
SQLiteQuickCheckSource = Literal["live", "cache", "unavailable"]


class DiagnosticScalarResult(NamedTuple):
    value: object | None
    succeeded: bool


class SQLiteForeignKeyCheckRow(NamedTuple):
    child_table: object
    child_row_id: object
    parent_table: object
    fk_constraint_index: object


class QuickCheckDiagnosticResult(NamedTuple):
    quick_check: SQLiteQuickCheck
    source: SQLiteQuickCheckSource


class SQLiteHealthDiagnostics(BaseModel):
    journal_mode: SQLiteJournalMode | None = None
    foreign_keys_enabled: bool | None = None
    busy_timeout_ms: int | None = None
    synchronous_mode: SQLiteSynchronousMode | None = None
    quick_check: SQLiteQuickCheck
    quick_check_source: SQLiteQuickCheckSource
    diagnostics_complete: bool


class DatabaseHealth(BaseModel):
    backend: DatabaseBackend
    connected: bool
    sqlite: SQLiteHealthDiagnostics | None = None


class SystemHealthResponse(BaseModel):
    status: HealthStatus
    database: DatabaseHealth


class ForeignKeyViolation(BaseModel):
    child_table: SQLiteSchemaName
    child_row_id: int | None
    parent_table: SQLiteSchemaName
    fk_constraint_index: Annotated[int, Field(ge=0)]


class ForeignKeyCheckResponse(BaseModel):
    status: ForeignKeyCheckStatus
    backend: DatabaseBackend
    returned_count: Annotated[int, Field(ge=0, le=500)]
    limit: Annotated[int, Field(ge=1, le=500)]
    truncated: bool
    violations: Annotated[list[ForeignKeyViolation], Field(max_length=500)]
