"""Deployment-level opt-in coverage for the optional Paisa extension."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from financial_dashboard.config import Settings, settings
from financial_dashboard.db.init_db import init_db
from financial_dashboard.main import create_app
from financial_dashboard.services.extensions import bootstrap_extensions
from financial_dashboard.services.settings import SETTINGS_REGISTRY


def test_paisa_deployment_flag_defaults_off_and_parses_environment(monkeypatch):
    monkeypatch.delenv("PAISA_ENABLED", raising=False)
    assert Settings(_env_file=None).paisa_enabled is False

    monkeypatch.setenv("PAISA_ENABLED", "true")
    assert Settings(_env_file=None).paisa_enabled is True


def test_disabled_bootstrap_has_no_paisa_manifest_runtime_or_settings():
    original_registry = dict(SETTINGS_REGISTRY)
    try:
        bootstrap_extensions(session_factory=async_sessionmaker(), paisa_enabled=True)
        assert "paisa.mode" in SETTINGS_REGISTRY

        manager = bootstrap_extensions(
            session_factory=async_sessionmaker(), paisa_enabled=False
        )

        assert manager.all() == ()
        assert "paisa" not in manager
        assert not any(key.startswith("paisa.") for key in SETTINGS_REGISTRY)
    finally:
        SETTINGS_REGISTRY.clear()
        SETTINGS_REGISTRY.update(original_registry)


def test_disabled_app_does_not_mount_paisa_routes(monkeypatch):
    monkeypatch.setattr(settings, "paisa_enabled", False)
    app = create_app()
    paths = set(app.openapi()["paths"])

    assert "/api/extensions" in paths
    assert "/extensions" in paths
    assert not any(path.startswith("/api/extensions/paisa") for path in paths)
    assert not any(path.startswith("/extensions/paisa") for path in paths)


def test_enabled_app_mounts_paisa_routes(monkeypatch):
    monkeypatch.setattr(settings, "paisa_enabled", True)
    app = create_app()
    paths = set(app.openapi()["paths"])

    assert "/api/extensions/paisa/status" in paths
    assert "/extensions/paisa" in paths


@pytest.mark.anyio
async def test_disabling_existing_database_removes_paisa_state_and_triggers(
    tmp_path, monkeypatch
):
    from financial_dashboard.services import settings as settings_service
    from financial_dashboard.services.categorization import merchant_rules

    async def _noop():
        return None

    monkeypatch.setattr(settings_service, "load_all_settings", _noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", _noop)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/gate.db")
    try:
        await init_db(engine, paisa_enabled=True)
        async with engine.connect() as conn:
            state_count = await conn.scalar(
                text(
                    "SELECT COUNT(*) FROM extension_sync_state "
                    "WHERE extension_id = 'paisa'"
                )
            )
            trigger_count = await conn.scalar(
                text(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type = 'trigger' AND name LIKE 'ext_sync_dirty_%'"
                )
            )
        assert state_count == 1
        assert trigger_count == 21

        await init_db(engine, paisa_enabled=False)
        async with engine.connect() as conn:
            state_count = await conn.scalar(
                text(
                    "SELECT COUNT(*) FROM extension_sync_state "
                    "WHERE extension_id = 'paisa'"
                )
            )
            trigger_count = await conn.scalar(
                text(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type = 'trigger' AND name LIKE 'ext_sync_dirty_%'"
                )
            )
        assert state_count == 0
        assert trigger_count == 0
    finally:
        await engine.dispose()
