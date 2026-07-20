from datetime import UTC, datetime
from typing import Annotated, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
DatabaseBackend = Literal["sqlite"]
ForeignKeyCheckStatus = Literal["ok", "violations", "unavailable"]
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


BackupId = Annotated[
    str,
    Field(
        min_length=39,
        max_length=39,
        pattern=r"^\d{8}T\d{12}Z-[0-9a-f]{16}$",
    ),
]
BackupSha256 = Annotated[
    str,
    Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
BackupFileStatus = Literal["verified", "missing", "size_mismatch"]
BackupListStatus = Literal["ok", "busy", "unsupported", "unavailable"]
BackupCreateStatus = Literal["created", "busy", "unsupported", "unavailable"]


class SystemBackupMetadata(BaseModel):
    backup_id: BackupId
    created_at: datetime
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: BackupSha256
    status: BackupFileStatus


class SystemBackupListResponse(BaseModel):
    status: BackupListStatus
    backend: DatabaseBackend
    returned_count: Annotated[int, Field(ge=0, le=100)]
    limit: Annotated[int, Field(ge=1, le=100)]
    truncated: bool
    backups: Annotated[list[SystemBackupMetadata], Field(max_length=100)]


class SystemBackupCreateResponse(BaseModel):
    status: BackupCreateStatus
    backend: DatabaseBackend
    backup: SystemBackupMetadata | None


class SystemBackupManifest(BaseModel):
    """Private, versioned sidecar format; never returned directly by the API."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    backup_id: BackupId
    created_at: datetime
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: BackupSha256
    quick_check: Literal["ok"]

    @field_validator("created_at")
    @classmethod
    def require_utc_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        if value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("created_at must be UTC")
        return value.astimezone(UTC)


class SystemBackupListResult(NamedTuple):
    backups: list[SystemBackupMetadata]
    truncated: bool


class SystemBackupHash(NamedTuple):
    sha256: str
    size_bytes: int


class SystemBackupWorkerOutcome(NamedTuple):
    backup: SystemBackupMetadata | None
    error: Exception | None


class SystemBackupListWorkerOutcome(NamedTuple):
    result: SystemBackupListResult | None
    error: Exception | None
