"""Extension registry, settings contribution, and manager lifecycle tests."""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_dashboard.extensions import (
    ExtensionManifest,
    register_builtin_extensions,
)
from financial_dashboard.extensions.base import ExtensionRegistrationError
from financial_dashboard.extensions.registry import ExtensionRegistry
from financial_dashboard.services.extensions import (
    ExtensionManager,
    bootstrap_extensions,
)
from financial_dashboard.services.fetch import FetchService
from financial_dashboard.services.settings import (
    SETTINGS_REGISTRY,
    SettingDef,
    parse_form_updates,
    register_setting,
)

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _register_builtins_module():
    """Put the Paisa settings in the global registry (idempotent)."""
    register_builtin_extensions(ExtensionRegistry())


def test_registration_rejects_collisions_and_is_idempotent(monkeypatch):
    reg = ExtensionRegistry()
    reg.register(ExtensionManifest(id="dup", display_name="First"))
    with pytest.raises(ExtensionRegistrationError, match="dup"):
        reg.register(ExtensionManifest(id="dup", display_name="Second"))

    with pytest.raises(ValueError, match="telegram.chat_id"):
        register_setting("telegram.chat_id", SETTINGS_REGISTRY["telegram.chat_id"])

    # The same definitions register again without error.
    again = ExtensionRegistry()
    register_builtin_extensions(again)
    assert "paisa" in again

    # A different definition for a present key must raise.
    conflicting = SettingDef(
        default="not-the-real-default",
        data_type="str",
        category="Paisa",
        label="Conflict",
    )
    monkeypatch.setitem(SETTINGS_REGISTRY, "paisa.mode", conflicting)
    with pytest.raises(ExtensionRegistrationError, match="paisa.mode"):
        register_builtin_extensions(ExtensionRegistry())


def test_parse_form_updates_omits_internal_paisa_settings():
    # A settings form that omits the internal keys must not overwrite them.
    updates, errors = parse_form_updates({})
    assert errors == []
    for key in (
        "paisa.generated_path",
        "paisa.selected_account_ids",
        "paisa.account_mappings",
        "paisa.category_mappings",
    ):
        assert key not in updates, key


class FakeRuntime:
    """Records every lifecycle call and can be made to raise on demand."""

    def __init__(self, ext_id: str, log: list[str] | None = None) -> None:
        self.extension_id = ext_id
        self.calls: list[str] = []
        self.log = log if log is not None else []
        self.raises = False

    async def _record(self, name: str) -> None:
        self.calls.append(name)
        self.log.append(f"{self.extension_id}.{name}")
        if self.raises:
            raise RuntimeError(f"{self.extension_id} {name} boom")

    async def startup(self) -> None:
        await self._record("startup")

    async def shutdown(self) -> None:
        await self._record("shutdown")

    async def after_fetch_cycle(self) -> None:
        await self._record("after_fetch_cycle")


def _manager_with(*ext_ids: str) -> tuple[ExtensionManager, dict[str, FakeRuntime]]:
    reg = ExtensionRegistry()
    runtimes: dict[str, FakeRuntime] = {}
    log: list[str] = []
    for eid in ext_ids:
        reg.register(ExtensionManifest(id=eid, display_name=eid.upper()))
    manager = ExtensionManager(reg)
    for eid in ext_ids:
        rt = FakeRuntime(eid, log)
        runtimes[eid] = rt
        manager.register_runtime(eid, rt)
    return manager, runtimes


async def test_register_runtime_rejects_invalid_registrations():
    with pytest.raises(ValueError, match="unknown extension"):
        ExtensionManager().register_runtime("ghost", FakeRuntime("ghost"))

    reg = ExtensionRegistry()
    reg.register(ExtensionManifest(id="a", display_name="A"))
    with pytest.raises(ValueError, match="does not match"):
        ExtensionManager(reg).register_runtime("a", FakeRuntime("b"))

    manager, runtimes = _manager_with("a")
    duplicate = FakeRuntime("a")
    with pytest.raises(ValueError, match="already registered"):
        manager.register_runtime("a", duplicate)

    await manager.startup_all()
    assert runtimes["a"].calls == ["startup"]
    assert duplicate.calls == []


async def test_startup_shutdown_run_in_registration_order():
    manager, runtimes = _manager_with("a", "b", "c")
    await manager.startup_all()
    await manager.shutdown_all()
    assert runtimes["a"].log == [
        "a.startup",
        "b.startup",
        "c.startup",
        "a.shutdown",
        "b.shutdown",
        "c.shutdown",
    ]


async def test_lifecycle_failures_are_isolated_per_extension():
    manager, runtimes = _manager_with("a", "b", "c")
    runtimes["b"].raises = True

    # Shutdown before startup is safe.
    await manager.shutdown_all()
    await manager.startup_all()
    await manager.after_fetch_cycle_all()
    await manager.shutdown_all()
    for rt in runtimes.values():
        assert rt.calls == ["shutdown", "startup", "after_fetch_cycle", "shutdown"]


async def test_fetch_loop_runs_hook_once_per_cycle_after_native_steps(monkeypatch):
    order: list[str] = []

    async def fake_poll_all(*a, **kw):
        order.append("poll")

    async def fake_reminders():
        order.append("reminders")
        return 0

    async def fake_categorization():
        order.append("categorization")

    class _RaisingManager:
        async def after_fetch_cycle_all(self):
            order.append("after_fetch_cycle")
            raise RuntimeError("hook boom")

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
    monkeypatch.setattr(
        "financial_dashboard.services.fetch.get_setting_int", lambda *a, **kw: 1
    )
    sleeps = 0

    async def _cancel_on_second_sleep(*a, **kw):
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(
        "financial_dashboard.services.fetch.asyncio.sleep", _cancel_on_second_sleep
    )

    svc = FetchService(extension_manager=_RaisingManager())  # type: ignore[arg-type]
    # A failing hook must not stop the loop.
    with pytest.raises(asyncio.CancelledError):
        await svc._poll_loop()

    assert order == ["poll", "reminders", "categorization", "after_fetch_cycle"] * 2


async def test_bootstrap_manager_attaches_paisa_and_lifecycle_is_safe():
    # The real Paisa runtime starts and stops without network or auto-sync.
    manager = bootstrap_extensions(session_factory=async_sessionmaker())
    assert "paisa" in manager
    with pytest.raises(ValueError, match="already registered"):
        manager.register_runtime("paisa", FakeRuntime("paisa"))
    await manager.startup_all()
    await manager.shutdown_all()
