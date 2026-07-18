"""Automatic Paisa sync runtime — the :class:`ExtensionRuntime` for the Paisa
extension.

The runtime owns exactly one :class:`~financial_dashboard.services.paisa.coordinator.PaisaCoordinator`
background task (the transaction-driven, bulk-safe worker). There is no
module-global poll loop: :meth:`PaisaAutomationRuntime.startup` starts the
coordinator, :meth:`PaisaAutomationRuntime.shutdown` cancels and awaits it, and
:meth:`PaisaAutomationRuntime.after_fetch_cycle` is a backward-compatible wake
hint (it only signals the coordinator's wake event; it does no work itself).

Design rules (enforced here, not elsewhere):

* **One worker per app.** Exactly one coordinator task; the manager starts it
  once in the lifespan and stops it once. No module-global loop.
* **Startup/shutdown are inert w.r.t. Paisa.** Neither starts Paisa, opens the
  network, nor enables auto-sync. Auto-sync is opt-in via
  ``paisa.auto_sync_enabled`` AND requires ``project`` mode (the coordinator
  checks both every tick).
* **Failure isolation.** The coordinator catches its own tick errors; the
  runtime's lifecycle hooks never propagate an optional-extension failure.
* **Manual delegation.** Manual generate/sync go through the same persisted
  lease as the coordinator (single-flight). The runtime exposes the
  coordinator (or its session factory) so the surface can claim that lease.
"""

import logging

from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_dashboard.services.paisa.coordinator import PaisaCoordinator

logger = logging.getLogger(__name__)

#: Kept for backward-compatible imports from existing tests/callers. The
#: automatic debounce is now owned by the coordinator's persisted-state
#: eligibility; this constant is the historical default for the legacy
#: min-interval setting and is no longer the event debounce.
DEFAULT_MIN_INTERVAL_MINUTES = 1


class PaisaAutomationRuntime:
    """Implements :class:`~financial_dashboard.extensions.base.ExtensionRuntime`.

    Construct with no args in production (it resolves the shared
    ``async_session`` lazily so importing this module is side-effect-free).
    Tests inject an ``async_sessionmaker`` and optional coordinator seams to
    keep the worker deterministic and fast.
    """

    extension_id = "paisa"

    def __init__(
        self,
        session_factory: async_sessionmaker | None = None,
        *,
        coordinator: PaisaCoordinator | None = None,
    ) -> None:
        self._session_factory: async_sessionmaker | None = session_factory
        # When provided, use this coordinator verbatim (tests inject one with
        # fake seams). Otherwise build one at startup from the resolved factory.
        self._coordinator: PaisaCoordinator | None = coordinator

    def _resolve_session_factory(self) -> async_sessionmaker:
        # Lazy import: keeps module import free of a DB/engine touch and breaks
        # the import cycle with financial_dashboard.db at construction time.
        if self._session_factory is None:
            from financial_dashboard.db import async_session

            self._session_factory = async_session
        return self._session_factory

    @property
    def coordinator(self) -> PaisaCoordinator | None:
        """The started coordinator (None before startup / after shutdown)."""
        return self._coordinator

    # ------------------------------------------------------------------
    # ExtensionRuntime protocol
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Start exactly one coordinator task.

        Inert w.r.t. Paisa itself: the coordinator only does I/O when
        ``paisa.mode=project`` AND ``paisa.auto_sync_enabled=true``, both
        checked every tick. Construction never starts Paisa or enables
        auto-sync.
        """
        if self._coordinator is None:
            self._coordinator = PaisaCoordinator(
                session_factory=self._resolve_session_factory()
            )
        try:
            await self._coordinator.start()
        except Exception:
            # A second startup (coordinator already running) or a transient
            # error must never break the lifespan. Log and move on; the next
            # tick / wake will retry.
            logger.exception("Paisa coordinator startup failed; continuing")

    async def shutdown(self) -> None:
        """Cancel and await the coordinator task, then drop the reference."""
        coord = self._coordinator
        if coord is None:
            return
        try:
            await coord.stop()
        except Exception:
            logger.exception("Paisa coordinator shutdown failed; continuing")
        # Keep the coordinator object reusable for a later startup only if it
        # was the one we constructed; an injected coordinator is owned by the
        # test.

    async def after_fetch_cycle(self) -> None:
        """Backward-compatible wake hint.

        The fetch loop no longer drives Paisa directly; the coordinator polls
        persisted state. This hook only signals the coordinator's wake event so
        a fetch cycle that dirtied data reconciles sooner than the 2s poll. It
        never does work itself and never raises.
        """
        coord = self._coordinator
        if coord is None:
            return
        try:
            coord.wake()
        except Exception:
            logger.debug("coordinator wake hint failed", exc_info=True)


__all__ = [
    "DEFAULT_MIN_INTERVAL_MINUTES",
    "PaisaAutomationRuntime",
]
