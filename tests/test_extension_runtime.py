"""Extension runtime + manager lifecycle tests.

Covers: Paisa manifest metadata, runtime registration validation,
startup/shutdown with per-extension failure isolation, after-fetch-cycle
isolation, and FetchService integration (one callback per fetch cycle, ordered
after native steps, loop survives a failing hook).
"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_dashboard.extensions import (
    EXTENSION_CONTRACT_VERSION,
    ExtensionManifest,
    PAISA_EXTENSION,
    register_builtin_extensions,
)
from financial_dashboard.extensions.base import Capability
from financial_dashboard.extensions.registry import ExtensionRegistry
from financial_dashboard.services.extensions import (
    ExtensionManager,
    bootstrap_extensions,
)
from financial_dashboard.services.fetch import FetchService

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _register_builtins_module():
    """Ensure PAISA_EXTENSION's settings are in the global registry for the
    module (idempotent)."""
    reg = ExtensionRegistry()
    register_builtin_extensions(reg)


# --------------------------------------------------------------------------- #
# Manifest maturity metadata
# --------------------------------------------------------------------------- #


def test_paisa_manifest_exposes_navigation_routes_health_and_automation():
    assert PAISA_EXTENSION.contract_version == EXTENSION_CONTRACT_VERSION
    nav = PAISA_EXTENSION.navigation
    assert [(item.label, item.path) for item in nav] == [("Paisa", "/extensions/paisa")]
    assert "/api/extensions/paisa" in PAISA_EXTENSION.route_prefixes
    assert "/extensions/paisa" in PAISA_EXTENSION.route_prefixes
    assert PAISA_EXTENSION.health is not None
    assert PAISA_EXTENSION.health.status_path == "/api/extensions/paisa/status"
    assert Capability.AUTOMATION in PAISA_EXTENSION.capabilities


# --------------------------------------------------------------------------- #
# Fake runtime helpers
# --------------------------------------------------------------------------- #


class FakeRuntime:
    """Records every lifecycle call and can be made to raise on demand."""

    def __init__(self, ext_id: str) -> None:
        self.extension_id = ext_id
        self.calls: list[str] = []
        self.startup_raises = False
        self.shutdown_raises = False
        self.cycle_raises = False

    async def startup(self) -> None:
        self.calls.append("startup")
        if self.startup_raises:
            raise RuntimeError(f"{self.extension_id} startup boom")

    async def shutdown(self) -> None:
        self.calls.append("shutdown")
        if self.shutdown_raises:
            raise RuntimeError(f"{self.extension_id} shutdown boom")

    async def after_fetch_cycle(self) -> None:
        self.calls.append("after_fetch_cycle")
        if self.cycle_raises:
            raise RuntimeError(f"{self.extension_id} cycle boom")


def _manager_with(*ext_ids: str) -> tuple[ExtensionManager, dict[str, FakeRuntime]]:
    reg = ExtensionRegistry()
    runtimes: dict[str, FakeRuntime] = {}
    for eid in ext_ids:
        reg.register(ExtensionManifest(id=eid, display_name=eid.upper()))
    manager = ExtensionManager(reg)
    for eid in ext_ids:
        rt = FakeRuntime(eid)
        runtimes[eid] = rt
        manager.register_runtime(eid, rt)
    return manager, runtimes


# --------------------------------------------------------------------------- #
# Runtime registration validation
# --------------------------------------------------------------------------- #


def test_register_runtime_rejects_invalid_registrations():
    with pytest.raises(ValueError, match="unknown extension"):
        ExtensionManager().register_runtime("ghost", FakeRuntime("ghost"))

    reg = ExtensionRegistry()
    reg.register(ExtensionManifest(id="a", display_name="A"))
    with pytest.raises(ValueError, match="does not match"):
        ExtensionManager(reg).register_runtime("a", FakeRuntime("b"))

    manager, runtimes = _manager_with("a")
    with pytest.raises(ValueError, match="already registered"):
        manager.register_runtime("a", FakeRuntime("a"))

    assert manager.get_runtime("a") is runtimes["a"]


# --------------------------------------------------------------------------- #
# Lifecycle ordering + failure isolation
# --------------------------------------------------------------------------- #


async def test_startup_shutdown_run_in_registration_order():
    manager, runtimes = _manager_with("a", "b", "c")
    await manager.startup_all()
    await manager.shutdown_all()
    order = [r.extension_id for r in manager.runtimes()]
    assert order == ["a", "b", "c"]
    for rt in runtimes.values():
        assert "startup" in rt.calls
        assert "shutdown" in rt.calls
    running = {s.id: s.running for s in manager.status()}
    assert running == {"a": False, "b": False, "c": False}


async def test_startup_failure_isolated():
    manager, runtimes = _manager_with("a", "b", "c")
    runtimes["b"].startup_raises = True
    await manager.startup_all()
    # a and c still started; b did not.
    assert "startup" in runtimes["a"].calls
    assert "startup" in runtimes["c"].calls
    assert "startup" in runtimes["b"].calls  # it was attempted
    running = {s.id: s.running for s in manager.status()}
    assert running["a"] is True
    assert running["b"] is False
    assert running["c"] is True


async def test_shutdown_failure_isolated():
    manager, runtimes = _manager_with("a", "b", "c")
    await manager.startup_all()
    runtimes["b"].shutdown_raises = True
    await manager.shutdown_all()
    # Every shutdown was attempted despite b raising.
    for rt in runtimes.values():
        assert "shutdown" in rt.calls


async def test_after_fetch_cycle_failure_isolated():
    manager, runtimes = _manager_with("a", "b", "c")
    runtimes["b"].cycle_raises = True
    # Should not raise.
    await manager.after_fetch_cycle_all()
    for rt in runtimes.values():
        assert "after_fetch_cycle" in rt.calls


async def test_shutdown_without_startup_is_safe():
    manager, runtimes = _manager_with("a")
    # Never started — shutdown must still be callable and not raise.
    await manager.shutdown_all()
    assert "shutdown" in runtimes["a"].calls


# --------------------------------------------------------------------------- #
# FetchService integration: one callback per cycle, ordered, isolated
# --------------------------------------------------------------------------- #


class _RecordingManager:
    """Stand-in ExtensionManager that records call order and can raise."""

    def __init__(self) -> None:
        self.cycle_calls = 0

    async def after_fetch_cycle_all(self) -> None:
        self.cycle_calls += 1


async def _run_one_poll_iteration(extension_manager, monkeypatch):
    """Drive exactly one _poll_loop iteration, then cancel at the sleep.

    Each native step appends to ``order`` so the test can assert the extension
    hook runs AFTER polling/reminders/categorization.
    """
    order: list[str] = []

    async def fake_poll_all(*a, **kw):
        order.append("poll")

    async def fake_reminders():
        order.append("reminders")
        return 0

    async def fake_categorization():
        order.append("categorization")

    class _ExtWrap:
        def __init__(self, inner):
            self._inner = inner

        async def after_fetch_cycle_all(self):
            order.append("after_fetch_cycle")
            await self._inner.after_fetch_cycle_all()

    monkeypatch.setattr(
        "financial_dashboard.services.fetch.fetch_orchestrator.poll_all", fake_poll_all
    )
    monkeypatch.setattr(
        "financial_dashboard.services.fetch.check_and_send_reminders", fake_reminders
    )
    monkeypatch.setattr(
        "financial_dashboard.services.fetch.run_categorization_cycle",
        fake_categorization,
    )
    # A huge interval so the sleep is the cancellation point.
    monkeypatch.setattr(
        "financial_dashboard.services.fetch.get_setting_int", lambda *a, **kw: 999999
    )

    svc = FetchService(extension_manager=_ExtWrap(extension_manager))  # type: ignore[arg-type]

    async def _cancel_on_sleep(*a, **kw):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        "financial_dashboard.services.fetch.asyncio.sleep", _cancel_on_sleep
    )
    with pytest.raises(asyncio.CancelledError):
        await svc._poll_loop()
    return order


async def test_after_fetch_cycle_called_once_per_cycle_and_ordered(monkeypatch):
    mgr = _RecordingManager()
    order = await _run_one_poll_iteration(mgr, monkeypatch)
    assert mgr.cycle_calls == 1
    assert order == ["poll", "reminders", "categorization", "after_fetch_cycle"]


async def test_fetch_loop_survives_failing_extension_hook(monkeypatch):
    # Make the wrapper raise to prove the loop's own isolation (separate from
    # the manager's per-extension isolation).
    class _Raising:
        async def after_fetch_cycle_all(self):
            raise RuntimeError("hook boom")

    monkeypatch.setattr(
        "financial_dashboard.services.fetch.fetch_orchestrator.poll_all",
        lambda *a, **kw: _noop_coro(),
    )

    async def fake_reminders():
        return 0

    async def fake_categorization():
        return None

    monkeypatch.setattr(
        "financial_dashboard.services.fetch.check_and_send_reminders", fake_reminders
    )
    monkeypatch.setattr(
        "financial_dashboard.services.fetch.run_categorization_cycle",
        fake_categorization,
    )
    monkeypatch.setattr(
        "financial_dashboard.services.fetch.get_setting_int", lambda *a, **kw: 1
    )

    svc = FetchService(extension_manager=_Raising())  # type: ignore[arg-type]

    iterations = {"n": 0}

    async def _cancel_after_two(*a, **kw):
        iterations["n"] += 1
        if iterations["n"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(
        "financial_dashboard.services.fetch.asyncio.sleep", _cancel_after_two
    )
    # Must not raise despite the failing hook — loop ran twice then cancelled.
    with pytest.raises(asyncio.CancelledError):
        await svc._poll_loop()
    assert iterations["n"] == 2


async def _noop_coro():
    return None


# --------------------------------------------------------------------------- #
# Bootstrap wires the Paisa runtime
# --------------------------------------------------------------------------- #


async def test_bootstrap_manager_attaches_paisa_and_lifecycle_is_safe():
    # startup/shutdown of the real Paisa runtime must be no-ops (no network,
    # no auto-sync kick) and not raise.
    session_factory = async_sessionmaker()
    manager = bootstrap_extensions(session_factory=session_factory)
    snap = {s.id: s for s in manager.status()}
    assert snap["paisa"].has_runtime is True
    await manager.startup_all()
    snap = {s.id: s for s in manager.status()}
    assert snap["paisa"].running is True
    assert manager.get_runtime("paisa").coordinator.session_factory is session_factory
    await manager.shutdown_all()
    snap = {s.id: s for s in manager.status()}
    assert snap["paisa"].running is False
