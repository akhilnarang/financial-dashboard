import json
import os
import platform
import re
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path

from financial_dashboard.schemas import system as system_schemas

APP_DISTRIBUTION = "financial-dashboard"
PARSER_DISTRIBUTIONS = (
    "bank-email-parser",
    "bank-sms-parser",
    "bank-statement-parser",
    "cas-parser",
    "cc-parser",
)
REVISION_ENV_KEYS = (
    "FINANCIAL_DASHBOARD_REVISION",
    "APP_REVISION",
    "GIT_COMMIT",
    "COMMIT_SHA",
)
_SAFE_REVISION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,127}$")


def _normalize_revision(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    if _SAFE_REVISION_RE.fullmatch(normalized) is None:
        return None
    return normalized


def _get_distribution(
    distribution_name: str,
) -> importlib_metadata.Distribution | None:
    try:
        return importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _distribution_commit_id(
    distribution: importlib_metadata.Distribution,
) -> str | None:
    payload = distribution.read_text("direct_url.json")
    if payload is None:
        return None
    try:
        parsed_payload = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed_payload, dict):
        return None
    vcs_info = parsed_payload.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit_id = vcs_info.get("commit_id")
    if not isinstance(commit_id, str):
        return None
    return _normalize_revision(commit_id)


def _distribution_metadata(
    distribution_name: str,
) -> system_schemas.DistributionMetadata:
    distribution = _get_distribution(distribution_name)
    if distribution is None:
        return system_schemas.DistributionMetadata(version=None, vcs_commit_id=None)
    return system_schemas.DistributionMetadata(
        version=distribution.version,
        vcs_commit_id=_distribution_commit_id(distribution),
    )


def _source_checkout_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_head_revision() -> str | None:
    repo_root = _source_checkout_root()
    if not (repo_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return _normalize_revision(result.stdout)


def _resolve_app_revision(
    distribution_revision: str | None,
) -> system_schemas.AppRevisionResult:
    for env_key in REVISION_ENV_KEYS:
        env_value = os.environ.get(env_key)
        if env_value is None:
            continue
        revision = _normalize_revision(env_value)
        if revision is not None:
            return system_schemas.AppRevisionResult(
                value=revision, source="environment"
            )

    if distribution_revision is not None:
        return system_schemas.AppRevisionResult(
            value=distribution_revision, source="distribution"
        )

    git_revision = _git_head_revision()
    if git_revision is not None:
        return system_schemas.AppRevisionResult(value=git_revision, source="git")

    return system_schemas.AppRevisionResult(value=None, source="unknown")


def _runtime_info() -> system_schemas.RuntimeInfo:
    return system_schemas.RuntimeInfo(
        implementation=platform.python_implementation(),
        python_version=platform.python_version(),
    )


def _parser_packages() -> list[system_schemas.ParserPackageInfo]:
    packages: list[system_schemas.ParserPackageInfo] = []
    for distribution_name in PARSER_DISTRIBUTIONS:
        distribution_metadata = _distribution_metadata(distribution_name)
        packages.append(
            system_schemas.ParserPackageInfo(
                package=distribution_name,
                version=distribution_metadata.version,
                vcs_commit_id=distribution_metadata.vcs_commit_id,
            )
        )
    return packages


def collect_runtime_metadata() -> system_schemas.SystemRuntimeMetadata:
    """Collect blocking process/package metadata for thread-pool execution."""
    app_distribution_metadata = _distribution_metadata(APP_DISTRIBUTION)
    return system_schemas.SystemRuntimeMetadata(
        package_version=app_distribution_metadata.version,
        app_revision=_resolve_app_revision(app_distribution_metadata.vcs_commit_id),
        runtime=_runtime_info(),
        parser_packages=_parser_packages(),
    )
