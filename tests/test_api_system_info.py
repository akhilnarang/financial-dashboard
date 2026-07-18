import json
from importlib import metadata as importlib_metadata

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.config import Settings
from financial_dashboard.core import security
from financial_dashboard.db.models import Setting
from financial_dashboard.schemas import system as system_schemas
from financial_dashboard.services import system as system_service
from financial_dashboard.services import system_metadata

pytestmark = pytest.mark.anyio


class _FakeDistribution:
    def __init__(self, version: str, direct_url_payload: dict | None = None):
        self.version = version
        self._direct_url_payload = direct_url_payload

    def read_text(self, filename: str) -> str | None:
        if filename != "direct_url.json" or self._direct_url_payload is None:
            return None
        return json.dumps(self._direct_url_payload)


def _make_settings(
    username: str = "",
    password: str = "",
    skip_cidrs: str = "",
) -> Settings:
    return Settings(
        auth_username=username,
        auth_password=SecretStr(password),
        auth_skip_cidrs=skip_cidrs,
    )


@pytest.fixture(autouse=True)
def _reset_trusted_cache():
    security._get_trusted_networks.cache_clear()
    try:
        yield
    finally:
        security._get_trusted_networks.cache_clear()


async def test_system_info_success_shape_and_redaction(client, session, monkeypatch):
    monkeypatch.setattr(security, "settings", _make_settings())

    session.add(Setting(key="migrations.zeta", value="1"))
    session.add(Setting(key="migrations.alpha", value="1"))
    session.add(
        Setting(key=system_service.SCHEMA_VERSION_SETTING, value="synthetic-schema-v1")
    )
    await session.commit()

    monkeypatch.setenv("FINANCIAL_DASHBOARD_REVISION", "build-2026-07-18")

    fake_distributions: dict[str, _FakeDistribution] = {
        "financial-dashboard": _FakeDistribution(
            "9.9.9",
            {
                "url": "file:///very/secret/local/checkout",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                },
            },
        ),
        "bank-email-parser": _FakeDistribution(
            "1.0.0",
            {
                "url": "ssh://git@example.com/private/repo.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "1111111111111111111111111111111111111111",
                },
            },
        ),
        "bank-sms-parser": _FakeDistribution("2.0.0"),
        "bank-statement-parser": _FakeDistribution(
            "3.0.0",
            {
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "2222222222222222222222222222222222222222",
                }
            },
        ),
        "cas-parser": _FakeDistribution(
            "4.0.0",
            {
                "url": "file:///tmp/do-not-expose",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "3333333333333333333333333333333333333333",
                },
            },
        ),
        "cc-parser": _FakeDistribution("5.0.0"),
    }

    def _distribution(distribution_name: str):
        distribution = fake_distributions.get(distribution_name)
        if distribution is None:
            raise importlib_metadata.PackageNotFoundError(distribution_name)
        return distribution

    monkeypatch.setattr(
        system_metadata.importlib_metadata, "distribution", _distribution
    )

    response = await client.get("/api/system/info")
    assert response.status_code == 200

    body = response.json()
    assert body["package_name"] == "financial-dashboard"
    assert body["package_version"] == "9.9.9"
    assert body["app_revision"] == "build-2026-07-18"
    assert body["app_revision_source"] == "environment"

    assert body["schema_state"] == {
        "schema_version": "synthetic-schema-v1",
        "applied_migration_markers": ["migrations.alpha", "migrations.zeta"],
    }

    parser_packages = body["parser_packages"]
    assert [pkg["package"] for pkg in parser_packages] == list(
        system_metadata.PARSER_DISTRIBUTIONS
    )
    assert parser_packages[0]["version"] == "1.0.0"
    assert (
        parser_packages[0]["vcs_commit_id"]
        == "1111111111111111111111111111111111111111"
    )
    assert parser_packages[1]["vcs_commit_id"] is None
    assert (
        parser_packages[2]["vcs_commit_id"]
        == "2222222222222222222222222222222222222222"
    )

    assert body["runtime"]["implementation"]
    assert body["runtime"]["python_version"]

    assert "file:///" not in response.text
    assert "/very/secret/local/checkout" not in response.text
    assert "git@example.com" not in response.text
    assert "tmp/do-not-expose" not in response.text


async def test_get_system_info_offloads_runtime_metadata_to_thread(
    session: AsyncSession, monkeypatch
):
    expected_runtime_metadata = system_schemas.SystemRuntimeMetadata(
        package_version="7.7.7",
        app_revision=system_schemas.AppRevisionResult(
            value="1234567890abcdef1234567890abcdef12345678",
            source="git",
        ),
        runtime=system_schemas.RuntimeInfo(
            implementation="CPython",
            python_version="3.14.0",
        ),
        parser_packages=[
            system_schemas.ParserPackageInfo(
                package="bank-email-parser",
                version="1.2.3",
                vcs_commit_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
        ],
    )

    def _fake_runtime_metadata() -> system_schemas.SystemRuntimeMetadata:
        return expected_runtime_metadata

    to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def _fake_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(
        system_metadata, "collect_runtime_metadata", _fake_runtime_metadata
    )
    monkeypatch.setattr(system_service.asyncio, "to_thread", _fake_to_thread)

    body = await system_service.get_system_info(session)

    assert len(to_thread_calls) == 1
    to_thread_func, to_thread_args, to_thread_kwargs = to_thread_calls[0]
    assert to_thread_func is _fake_runtime_metadata
    assert to_thread_args == ()
    assert to_thread_kwargs == {}

    assert body.package_name == "financial-dashboard"
    assert body.package_version == "7.7.7"
    assert body.app_revision == "1234567890abcdef1234567890abcdef12345678"
    assert body.app_revision_source == "git"
    assert body.runtime.model_dump() == {
        "implementation": "CPython",
        "python_version": "3.14.0",
    }
    assert [pkg.model_dump() for pkg in body.parser_packages] == [
        {
            "package": "bank-email-parser",
            "version": "1.2.3",
            "vcs_commit_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        }
    ]
    assert body.schema_state.model_dump() == {
        "schema_version": None,
        "applied_migration_markers": [],
    }


async def test_system_info_falls_back_to_unknown_metadata(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(security, "settings", _make_settings())

    for env_key in system_metadata.REVISION_ENV_KEYS:
        monkeypatch.delenv(env_key, raising=False)

    def _missing_distribution(distribution_name: str):
        raise importlib_metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(
        system_metadata.importlib_metadata,
        "distribution",
        _missing_distribution,
    )
    monkeypatch.setattr(system_metadata, "_source_checkout_root", lambda: tmp_path)

    def _unexpected_subprocess(*args, **kwargs):
        raise AssertionError("git must not run outside a source checkout")

    monkeypatch.setattr(system_metadata.subprocess, "run", _unexpected_subprocess)

    response = await client.get("/api/system/info")
    assert response.status_code == 200

    body = response.json()
    assert body["package_name"] == "financial-dashboard"
    assert body["package_version"] is None
    assert body["app_revision"] is None
    assert body["app_revision_source"] == "unknown"
    assert body["schema_state"] == {
        "schema_version": None,
        "applied_migration_markers": [],
    }

    parser_packages = body["parser_packages"]
    assert [pkg["package"] for pkg in parser_packages] == list(
        system_metadata.PARSER_DISTRIBUTIONS
    )
    for package in parser_packages:
        assert package["version"] is None
        assert package["vcs_commit_id"] is None
