"""Runtime extension manager, attached to app.state during lifespan.

Holds the per-app :class:`ExtensionRegistry` (manifests) AND the optional
:class:`~financial_dashboard.extensions.base.ExtensionRuntime` instances
attached to those extensions. The manager is the sole owner of extension
lifecycle:

* :meth:`ExtensionManager.startup_all` / :meth:`ExtensionManager.shutdown_all`
  run once each in the application lifespan.
* :meth:`ExtensionManager.after_fetch_cycle_all` runs at most once per
  FetchService poll cycle, after native polling/reminders/categorization.

Lifecycle hooks are isolated per extension: a raise in one extension's hook is
logged and swallowed, never propagated — the fetch loop and other extensions
keep running. The manager spawns no background tasks of its own; extensions that
need periodic work ride the existing FetchService loop via
``after_fetch_cycle``.
"""

import logging

from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_dashboard.extensions import (
    ExtensionManifest,
    ExtensionRegistry,
    ExtensionRuntime,
    register_builtin_extensions,
)

logger = logging.getLogger(__name__)


class ExtensionManager:
    """Holds the per-app ExtensionRegistry plus attached extension runtimes."""

    def __init__(self, registry: ExtensionRegistry | None = None) -> None:
        self._registry: ExtensionRegistry = (
            registry if registry is not None else ExtensionRegistry()
        )
        self._runtimes: dict[str, ExtensionRuntime] = {}

    # ------------------------------------------------------------------
    # Manifest accessors (backwards-compatible)
    # ------------------------------------------------------------------

    @property
    def registry(self) -> ExtensionRegistry:
        return self._registry

    def all(self) -> tuple[ExtensionManifest, ...]:
        return self._registry.all()

    def __contains__(self, ext_id: object) -> bool:
        return ext_id in self._registry

    # ------------------------------------------------------------------
    # Runtime registration
    # ------------------------------------------------------------------

    def register_runtime(self, ext_id: str, runtime: ExtensionRuntime) -> None:
        """Attach a runtime to an already-registered extension.

        ``ext_id`` must be a registered manifest id and ``runtime.extension_id``
        must match it, so a runtime can never be attached to the wrong manifest.
        A second runtime for the same id is rejected: silently replacing one
        would lose lifecycle ownership of the original runtime.
        """
        if ext_id not in self._registry:
            raise ValueError(f"Cannot attach runtime: unknown extension {ext_id!r}")
        if ext_id in self._runtimes:
            raise ValueError(
                f"Cannot attach runtime: runtime already registered for {ext_id!r}"
            )
        # ``extension_id`` is part of the ExtensionRuntime protocol. Direct
        # attribute access — the protocol guarantees it; let AttributeError
        # surface for a non-conforming object.
        if runtime.extension_id != ext_id:
            raise ValueError(
                f"Runtime extension_id {runtime.extension_id!r} does not match "
                f"target extension {ext_id!r}"
            )
        self._runtimes[ext_id] = runtime

    # ------------------------------------------------------------------
    # Lifecycle (isolated per extension)
    # ------------------------------------------------------------------

    async def startup_all(self) -> None:
        """Run ``startup`` on every attached runtime, in registration order.

        Failures are isolated: a raise in one extension is logged and the rest
        still start.
        """
        for manifest in self._registry:
            runtime = self._runtimes.get(manifest.id)
            if runtime is None:
                continue
            try:
                await runtime.startup()
            except Exception:
                logger.exception("Extension %r startup failed; continuing", manifest.id)

    async def shutdown_all(self) -> None:
        """Run ``shutdown`` on every attached runtime, best-effort.

        Shutdown is always attempted for every runtime regardless of startup
        outcome (a runtime should tolerate shutdown-without-startup). Isolation
        is per-extension: one slow/failing shutdown does not block the others.
        """
        for manifest in self._registry:
            runtime = self._runtimes.get(manifest.id)
            if runtime is None:
                continue
            try:
                await runtime.shutdown()
            except Exception:
                logger.exception(
                    "Extension %r shutdown failed; continuing", manifest.id
                )

    async def after_fetch_cycle_all(self) -> None:
        """Invoke ``after_fetch_cycle`` once per attached runtime per cycle.

        This is the single hook the FetchService calls after native
        polling/reminders/categorization. Each extension is coalesced to one
        call here (no per-extension loops), and a failure in one extension's
        hook never affects another or the fetch loop itself.
        """
        for manifest in self._registry:
            runtime = self._runtimes.get(manifest.id)
            if runtime is None:
                continue
            try:
                await runtime.after_fetch_cycle()
            except Exception:
                logger.exception(
                    "Extension %r after_fetch_cycle failed; continuing", manifest.id
                )


def bootstrap_extensions(
    *, session_factory: async_sessionmaker, paisa_enabled: bool = True
) -> ExtensionManager:
    """Build a manager, register enabled manifests+settings, and attach runtimes.

    Contributed settings land in the process-wide SETTINGS_REGISTRY (idempotent
    across restarts), so this must run before ``load_all_settings()``. Runtime
    construction is side-effect-free (no settings reads or DB I/O) and receives
    the application-owned session factory explicitly. Runtimes defer all work
    to their lifecycle hooks, so attaching them here is safe even though
    settings are not loaded yet.
    """
    manager = ExtensionManager()
    register_builtin_extensions(manager.registry, paisa_enabled=paisa_enabled)
    _register_builtin_runtimes(manager, session_factory=session_factory)
    return manager


def _register_builtin_runtimes(
    manager: ExtensionManager, *, session_factory: async_sessionmaker
) -> None:
    """Attach the first-party runtimes. No dynamic discovery.

    Function-local import keeps the extension-framework layer free of a
    load-time dependency on the Paisa integration, and avoids any import cycle
    with ``services.paisa`` (which imports settings/db at its own top level).
    A builtin failing to import is a first-party build error and propagates.
    """
    if "paisa" in manager:
        from financial_dashboard.services.paisa.automation import PaisaAutomationRuntime

        manager.register_runtime(
            "paisa", PaisaAutomationRuntime(session_factory=session_factory)
        )


__all__ = [
    "ExtensionManager",
    "bootstrap_extensions",
]
