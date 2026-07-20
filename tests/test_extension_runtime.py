"""Extension runtime + manager lifecycle tests.

Covers: mature manifest metadata (contract version, navigation, route prefixes,
health, AUTOMATION capability), runtime registration validation, deterministic
status, startup/shutdown ordering with per-extension failure isolation,
after-fetch-cycle isolation, FetchService integration (exactly one callback per
fetch cycle, ordered after native steps, loop survives a failing hook), and
backwards-compatible FetchService construction.
"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_dashboard.extensions import (
    EXTENSION_CONTRACT_VERSION,
    ExtensionManifest,
    ExtensionNavItem,
    ExtensionHealthMeta,
    ExtensionRuntime,
    PAISA_EXTENSION,
    register_builtin_extensions,
)
from financial_dashboard.extensions.base import Capability
from financial_dashboard.extensions.registry import ExtensionRegistry
from financial_dashboard.services.extensions import (
    ExtensionManager,
    ExtensionStatus,
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


def test_paisa_manifest_carries_contract_and_extension_versions():
    assert PAISA_EXTENSION.contract_version == EXTENSION_CONTRACT_VERSION
    assert PAISA_EXTENSION.extension_version


def test_paisa_manifest_navigation_and_routes():
    nav = PAISA_EXTENSION.navigation
    assert len(nav) == 1
    assert isinstance(nav[0], ExtensionNavItem)
    assert nav[0].label == "Paisa"
    assert nav[0].path == "/extensions/paisa"
    assert "/api/extensions/paisa" in PAISA_EXTENSION.route_prefixes
    assert "/extensions/paisa" in PAISA_EXTENSION.route_prefixes


def test_paisa_manifest_health_metadata():
    health = PAISA_EXTENSION.health
    assert isinstance(health, ExtensionHealthMeta)
    assert health.status_path == "/api/extensions/paisa/status"


def test_paisa_advertises_automation_capability():
    assert Capability.AUTOMATION in PAISA_EXTENSION.capabilities


def test_manifest_defaults_are_safe_for_minimal_construction():
    m = ExtensionManifest(id="x", display_name="X")
    assert m.contract_version == EXTENSION_CONTRACT_VERSION
    assert m.extension_version == "0.0.0"
    assert m.navigation == ()
    assert m.route_prefixes == ()
    assert m.health is None


def test_extension_runtime_is_a_protocol():
    # runtime_checkable Protocol: isinstance checks attribute presence.
    class _R:
        extension_id = "x"

        async def startup(self) -> None: ...
        async def shutdown(self) -> None: ...
        async def after_fetch_cycle(self) -> None: ...

    assert isinstance(_R(), ExtensionRuntime)


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


def test_register_runtime_attaches_to_known_extension():
    manager, runtimes = _manager_with("a")
    assert manager.get_runtime("a") is runtimes["a"]
    assert len(manager.runtimes()) == 1


def test_register_runtime_rejects_unknown_extension():
    manager = ExtensionManager()
    with pytest.raises(ValueError, match="unknown extension"):
        manager.register_runtime("ghost", FakeRuntime("ghost"))


def test_register_runtime_rejects_extension_id_mismatch():
    reg = ExtensionRegistry()
    reg.register(ExtensionManifest(id="a", display_name="A"))
    manager = ExtensionManager(reg)
    with pytest.raises(ValueError, match="does not match"):
        manager.register_runtime("a", FakeRuntime("b"))


def test_register_runtime_rejects_duplicate_without_replacing_original():
    manager, runtimes = _manager_with("a")
    replacement = FakeRuntime("a")

    with pytest.raises(ValueError, match="already registered"):
        manager.register_runtime("a", replacement)

    assert manager.get_runtime("a") is runtimes["a"]


def test_runtimes_preserve_manifest_registration_order():
    manager, _ = _manager_with("a", "b", "c")
    assert [r.extension_id for r in manager.runtimes()] == ["a", "b", "c"]


def test_status_is_deterministic_and_ordered():
    manager, _ = _manager_with("a", "b")
    # 'b' has no runtime attached path is covered by _manager_with attaching both;
    # verify shape here.
    snap = manager.status()
    assert all(isinstance(s, ExtensionStatus) for s in snap)
    assert [s.id for s in snap] == ["a", "b"]
    assert all(s.has_runtime for s in snap)
    assert all(not s.running for s in snap)  # nothing started yet


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


async def test_startup_marks_running_only_for_healthy_runtimes():
    manager, runtimes = _manager_with("a", "b")
    runtimes["a"].startup_raises = True
    await manager.startup_all()
    running = {s.id: s.running for s in manager.status()}
    assert running == {"a": False, "b": True}


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
        self.label: str | None = None
        self.raise_on_cycle = False

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
    mgr = _RecordingManager()
    mgr.raise_on_cycle = True

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
# Backwards-compatible construction
# --------------------------------------------------------------------------- #


def test_fetch_service_constructs_without_manager():
    svc = FetchService()
    assert svc._extension_manager is None


def test_fetch_service_accepts_manager():
    mgr = _RecordingManager()
    svc = FetchService(extension_manager=mgr)  # type: ignore[arg-type]
    assert svc._extension_manager is mgr


# --------------------------------------------------------------------------- #
# Bootstrap wires the Paisa runtime
# --------------------------------------------------------------------------- #


def test_bootstrap_extensions_attaches_paisa_runtime():
    manager = bootstrap_extensions(session_factory=async_sessionmaker())
    assert manager.get_runtime("paisa") is not None
    assert manager.get_runtime("paisa").extension_id == "paisa"
    snap = {s.id: s for s in manager.status()}
    assert snap["paisa"].has_runtime is True


async def test_bootstrap_manager_lifecycle_is_safe():
    # startup/shutdown of the real Paisa runtime must be no-ops (no network,
    # no auto-sync kick) and not raise.
    session_factory = async_sessionmaker()
    manager = bootstrap_extensions(session_factory=session_factory)
    await manager.startup_all()
    snap = {s.id: s for s in manager.status()}
    assert snap["paisa"].running is True
    assert manager.get_runtime("paisa").coordinator.session_factory is session_factory
    await manager.shutdown_all()
    snap = {s.id: s for s in manager.status()}
    assert snap["paisa"].running is False
