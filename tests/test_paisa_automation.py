"""Paisa automatic-sync runtime tests (transaction-driven coordinator model).

The runtime now owns exactly one :class:`PaisaCoordinator` background task
(started in ``startup``, stopped in ``shutdown``) and ``after_fetch_cycle`` is a
backward-compatible wake hint. The deep coordinator behavior is exercised in
``test_paisa_coordinator.py``; this file owns the runtime lifecycle contract:

* startup launches exactly one coordinator task; shutdown cancels/awaits it.
* startup is inert w.r.t. Paisa when mode is not project / auto is off (no I/O).
* after_fetch_cycle only wakes the coordinator (no work of its own).
* the runtime exposes its coordinator for manual single-flight delegation.
"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db.models import Base
from financial_dashboard.services.paisa.automation import PaisaAutomationRuntime

pytestmark = pytest.mark.anyio


def _set_paisa_settings(**overrides):
    """Populate the settings cache for the paisa.* keys the runtime reads."""
    defaults = {
        "paisa.mode": "disabled",
        "paisa.auto_sync_enabled": "false",
        "paisa.auto_sync_min_interval_minutes": "1",
        "paisa.notify_sync_failures": "false",
    }
    for key, value in overrides.items():
        defaults[key if key.startswith("paisa.") else f"paisa.{key}"] = value
    settings_mod._cache.update(defaults)


@pytest.fixture
async def factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _count_automatic_runs(factory) -> int:
    from financial_dashboard.services.paisa.audit import (
        OPERATION_AUTOMATIC,
        recent_runs,
    )

    async with factory() as s:
        rows = await recent_runs(s, extension_id="paisa", operation=OPERATION_AUTOMATIC)
        return len(rows)


# --------------------------------------------------------------------------- #
# Lifecycle: one coordinator task, inert until eligible
# --------------------------------------------------------------------------- #


async def test_startup_launches_one_coordinator_task(factory):
    _set_paisa_settings(mode="disabled")  # inert: no I/O
    rt = PaisaAutomationRuntime(session_factory=factory)
    assert rt.coordinator is None
    await rt.startup()
    coord = rt.coordinator
    assert coord is not None
    assert coord._task is not None
    assert not coord._task.done()
    await rt.shutdown()
    assert coord._task is None or coord._task.done()


async def test_startup_is_inert_when_disabled(factory, monkeypatch):
    _set_paisa_settings(mode="disabled", auto_sync_enabled="true")
    # If startup accidentally did I/O, ensure_sync_state/load_config would hit;
    # patch the coordinator's tick to assert it never runs real work.
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.startup()
    # Let the loop tick once with a tiny poll interval.
    coord = rt.coordinator
    coord._poll_interval = 0.01
    await asyncio.sleep(0.05)
    # No automatic audit rows in disabled mode.
    assert await _count_automatic_runs(factory) == 0
    await rt.shutdown()


async def test_startup_is_inert_when_auto_off(factory):
    _set_paisa_settings(mode="project", auto_sync_enabled="false")
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.startup()
    coord = rt.coordinator
    coord._poll_interval = 0.01
    await asyncio.sleep(0.05)
    assert await _count_automatic_runs(factory) == 0
    await rt.shutdown()


async def test_shutdown_cancels_and_awaids_task(factory):
    _set_paisa_settings(mode="disabled")
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.startup()
    task = rt.coordinator._task
    await rt.shutdown()
    assert task.done()


async def test_shutdown_without_startup_is_safe(factory):
    rt = PaisaAutomationRuntime(session_factory=factory)
    # Must not raise.
    await rt.shutdown()


async def test_double_startup_starts_one_task(factory):
    _set_paisa_settings(mode="disabled")
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.startup()
    task1 = rt.coordinator._task
    await rt.startup()  # idempotent: no second task
    assert rt.coordinator._task is task1
    await rt.shutdown()


# --------------------------------------------------------------------------- #
# after_fetch_cycle is a wake hint only
# --------------------------------------------------------------------------- #


async def test_after_fetch_cycle_only_wakes(factory, monkeypatch):
    _set_paisa_settings(mode="disabled")
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.startup()
    coord = rt.coordinator
    woken = []
    monkeypatch.setattr(coord, "wake", lambda: woken.append(True))
    await rt.after_fetch_cycle()
    assert woken == [True]
    # No audit row from the wake hint itself.
    assert await _count_automatic_runs(factory) == 0
    await rt.shutdown()


async def test_after_fetch_cycle_safe_before_startup(factory):
    rt = PaisaAutomationRuntime(session_factory=factory)
    # No coordinator yet — must not raise.
    await rt.after_fetch_cycle()


async def test_after_fetch_cycle_failure_is_isolated(factory, monkeypatch):
    _set_paisa_settings(mode="disabled")
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.startup()
    monkeypatch.setattr(
        rt.coordinator, "wake", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    # Must not raise.
    await rt.after_fetch_cycle()
    await rt.shutdown()


# --------------------------------------------------------------------------- #
# Coordinator is exposed for manual single-flight
# --------------------------------------------------------------------------- #


async def test_runtime_exposes_coordinator_for_manual_ops(factory):
    _set_paisa_settings(mode="disabled")
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.startup()
    assert rt.coordinator is not None
    assert rt.coordinator.session_factory is factory
    await rt.shutdown()
