"""Commit-aware wake optimization for the Paisa coordinator.

This is a pure latency optimization, NOT a correctness mechanism. Correctness
comes from the coordinator's persisted-state polling (every 2s): a coordinator
always observes a committed change within one poll interval whether or not a
wake signal fires. The wake just lets it react to a commit sooner.

Design:

* A coordinator registers a **zero-arg signal callable** here (e.g. a lambda
  that schedules ``event.set()`` on its loop). The callable must not await or
  block — it runs in the SQLAlchemy commit path.
* :mod:`financial_dashboard.db` registers a single ``after_commit`` listener on
  the sync ``Session`` class (which backs ``AsyncSession``), so it fires for
  every committed session app-wide. That listener calls :func:`_fire_commit_wake`.
* Signals are fire-and-forget: a failure in one is logged and swallowed so a
  commit never fails. Duplicate/nested wake hints coalesce (an ``asyncio.Event``
  set twice is still just set), and a wake that arrives between polls or during
  a tick is folded into the next tick — over-signaling is harmless.
* Register/unregister is clean (used by the coordinator's start/stop) so no
  stale signal lingers after shutdown.

Because this only *adds* a wake and never *gates* the coordinator, a
multi-process deployment (or an external change to the DB) is still found by
the 2s poll — the wake is best-effort within one process.
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Registered zero-arg signal callables. A coordinator adds one at startup and
#: removes it at shutdown. The list is mutated only under the GIL (list
#: append/remove are atomic in CPython), and the commit listener iterates a
#: snapshot copy, so a signal registering/unregistering during fire cannot
#: corrupt the iteration.
_wake_signals: list[Callable[[], None]] = []


def register_commit_wake(signal: Callable[[], None]) -> None:
    """Register a zero-arg signal callable fired on every observed commit.

    Idempotent: registering the same callable twice is a no-op so a coordinator
    that restarts in the same process does not double-fire.
    """
    if signal not in _wake_signals:
        _wake_signals.append(signal)


def unregister_commit_wake(signal: Callable[[], None]) -> None:
    """Remove a previously registered signal. Missing is a no-op."""
    try:
        _wake_signals.remove(signal)
    except ValueError:
        pass


def _fire_commit_wake() -> None:
    """Fire every registered signal. Swallows errors so a commit never fails."""
    for sig in list(_wake_signals):
        try:
            sig()
        except Exception:
            logger.debug("commit wake signal failed", exc_info=True)


def _reset_for_test() -> None:
    """Clear all signals. Test-only isolation helper."""
    _wake_signals.clear()


__all__ = [
    "register_commit_wake",
    "unregister_commit_wake",
]
