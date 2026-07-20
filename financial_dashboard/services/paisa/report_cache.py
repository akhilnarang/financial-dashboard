"""Server-side TTL cache for curated Paisa report reads.

The Paisa REST API may be configured with a ``TokenAuthMiddleware`` that
authenticates the ``X-Auth`` header; that middleware *also* rate-limits at
**6 requests per minute** (burst 3). The limiter only gates requests when
upstream ``user_accounts`` auth is configured, but when it is in play a
dashboard surface that re-fetched every report on every page load would blow
that budget in seconds. This module provides a small, bounded, per-application
cache that:

* Stores the most recent normalized result per report kind for a configurable
  TTL (``paisa.report_cache_ttl_seconds``). A TTL of ``0`` disables caching
  (every read hits upstream) but still coalesces concurrent in-flight reads.
* **Coalesces concurrent requests** — N readers asking for the same report at
  once share a single upstream call (a per-key ``asyncio.Lock``), so a burst of
  page opens cannot amplify into a burst of upstream calls.
* Is **scoped to the application**, not a module global: it lives on
  ``app.state.paisa_report_cache`` (see :func:`get_report_cache`), so it is
  created once per app and dies with it. There is no unbounded module-level
  table.
* Is **bounded** in entry count; a hard cap evicts oldest entries so a long
  run can never grow it without limit.

The cache never holds credentials (the :class:`~financial_dashboard.integrations.paisa.PaisaClient`
is built transiently per fetch and closed). It stores only the typed,
normalized report NamedTuples, which contain no secrets and no raw journal text.

When the Paisa extension mode is ``disabled`` the surface never reaches the
cache at all, so disabled guarantees zero upstream calls.
"""

import asyncio
import logging
import time
from collections.abc import Hashable
from typing import Any, Awaitable, Callable, NamedTuple

logger = logging.getLogger(__name__)

#: Hard cap on cached report/connection identities. Connection settings are
#: part of the key, so this bound also prevents repeated config switches from
#: growing the cache without limit.
DEFAULT_MAX_ENTRIES = 64


class CacheRead(NamedTuple):
    """The outcome of one :meth:`PaisaReportCache.read` call.

    Returning the hit flag *with* the value (rather than asking the caller to
    snapshot a shared counter) makes ``hit`` atomic per key: it is decided
    inside the per-key lock, so it cannot be raced by a concurrent read of a
    *different* key that happens to bump a global counter between a caller's
    before/after snapshots. Per the repo style, this is a ``NamedTuple`` so
    positional unpacking (``value, hit = cache.read(...)``) and named access
    (``result.hit``) both work.
    """

    value: Any
    hit: bool


class _CachedEntry:
    __slots__ = ("value", "fetched_monotonic")

    def __init__(self, value: Any, fetched_monotonic: float) -> None:
        self.value = value
        self.fetched_monotonic = fetched_monotonic


class _InFlight:
    __slots__ = ("task", "retain")

    def __init__(self, task: asyncio.Task[Any], retain: bool) -> None:
        self.task = task
        # If any same-key reader has TTL=0, that generation must not leave a
        # completed entry behind. The task still remains shareable while it is
        # running.
        self.retain = retain


class PaisaReportCache:
    """Per-application TTL cache with concurrent-request coalescing.

    Construct one per app (typically lazily on first use via
    :func:`get_report_cache`). Tests construct one directly and may read
    ``upstream_calls`` to assert call counts.
    """

    def __init__(self, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._entries: dict[Hashable, _CachedEntry] = {}
        self._inflight: dict[Hashable, _InFlight] = {}
        self._guard = asyncio.Lock()
        self._max = max(1, max_entries)
        #: Number of upstream fetch calls actually made (test/diagnostics hook).
        self.upstream_calls: int = 0

    async def _fetch_once(
        self,
        key: Hashable,
        fetch: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Run one upstream fetch and remove its in-flight marker on finish."""
        self.upstream_calls += 1
        try:
            value = await fetch()
        except BaseException:
            current_task = asyncio.current_task()
            async with self._guard:
                current = self._inflight.get(key)
                if current is not None and current.task is current_task:
                    self._inflight.pop(key, None)
            raise

        current_task = asyncio.current_task()
        async with self._guard:
            current = self._inflight.get(key)
            if current is not None and current.task is current_task:
                # Publish a completed cacheable value before dropping the
                # in-flight marker. There is no gap where a later reader could
                # see neither and start a duplicate fetch.
                if current.retain:
                    self._entries[key] = _CachedEntry(value, time.monotonic())
                    self._evict_if_needed()
                # Removal happens before the task becomes done, so a later
                # caller can never reuse a completed TTL=0 result.
                self._inflight.pop(key, None)
        return value

    async def read(
        self,
        key: Hashable,
        ttl_seconds: int,
        fetch: Callable[[], Awaitable[Any]],
    ) -> CacheRead:
        """Return a cached value for ``key`` if fresh, else call ``fetch`` once.

        Concurrent readers of the same ``key`` share one in-flight ``fetch``.
        ``ttl_seconds <= 0`` disables completed-result retention but not that
        in-flight sharing. ``fetch`` may raise; the exception is not cached and
        propagates to every waiter that joined that call.

        The returned :class:`CacheRead` carries the value plus a ``hit`` flag
        that is ``True`` only when the value was served from a fresh cache
        entry or joined a cacheable in-flight generation (no additional
        upstream call). TTL=0 readers always report ``hit=False`` because no
        completed cache value is retained. The decision is per key and cannot
        be flipped by a concurrent different-key request.
        """
        ttl = max(0, int(ttl_seconds))
        async with self._guard:
            entry = self._entries.get(key)
            now = time.monotonic()
            if ttl > 0 and entry is not None and (now - entry.fetched_monotonic) < ttl:
                return CacheRead(value=entry.value, hit=True)

            if ttl <= 0:
                # A runtime switch to TTL=0 must not leave an older completed
                # entry available if TTL is later re-enabled.
                self._entries.pop(key, None)

            flight = self._inflight.get(key)
            joined = flight is not None
            if flight is None:
                task = asyncio.create_task(self._fetch_once(key, fetch))
                flight = _InFlight(task, retain=ttl > 0)
                self._inflight[key] = flight
            elif ttl <= 0:
                flight.retain = False

        # Shield the shared operation from one disconnected/cancelled waiter;
        # other concurrent readers must still receive the one upstream result.
        value = await asyncio.shield(flight.task)

        return CacheRead(value=value, hit=joined and ttl > 0 and flight.retain)

    def _evict_if_needed(self) -> None:
        # Only invoked when an entry was just added, so the dict is non-empty.
        if len(self._entries) <= self._max:
            return
        # Evict the entry with the oldest fetch time (LRU-ish by freshness).
        oldest_key: Hashable | None = None
        oldest_time = float("inf")
        for k, e in self._entries.items():
            if e.fetched_monotonic < oldest_time:
                oldest_time = e.fetched_monotonic
                oldest_key = k
        if oldest_key is not None:
            self._entries.pop(oldest_key, None)

    def invalidate(self, key: Hashable | None = None) -> None:
        """Drop one key, or the whole cache when ``key`` is None."""
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    @property
    def size(self) -> int:
        return len(self._entries)


def get_report_cache(app_state: Any) -> PaisaReportCache:
    """Return the per-app Paisa report cache, creating it on first use.

    Stored on ``app.state.paisa_report_cache`` so it is scoped to the
    application lifetime (created once, dies with the app) — never a module
    global. This is the documented legitimate ``getattr(app.state, ...)`` form.
    """
    cache = getattr(app_state, "paisa_report_cache", None)
    if cache is None:
        cache = PaisaReportCache()
        app_state.paisa_report_cache = cache
    return cache


__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "CacheRead",
    "PaisaReportCache",
    "get_report_cache",
]
