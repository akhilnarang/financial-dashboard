from typing import NamedTuple

from pydantic import BaseModel


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
