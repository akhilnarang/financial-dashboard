"""Transaction-driven, bulk-safe Paisa reconcile coordinator.

The coordinator is the automatic worker that keeps the generated Paisa include
file (and the remote Paisa instance) aligned with the dashboard's core data. It
is the sole driver of automatic Paisa I/O — there is no module-global poll
loop; one :class:`PaisaCoordinator` task lives on the app (started/stopped by
:class:`~financial_dashboard.services.paisa.automation.PaisaAutomationRuntime`).

Operating model (see AGENTS.md for the full rationale):

* **Eligibility** is read off the persisted singleton
  (:mod:`financial_dashboard.services.paisa.sync_state`): only ``project`` mode
  + ``paisa.auto_sync_enabled`` performs any I/O. ``disabled``/``connect``/
  auto-off accumulate dirty state with no I/O. Fixed quiet debounce 5s, max
  dirty latency 30s, 6h periodic reload. ``paisa.auto_sync_min_interval_minutes``
  (default 1) is a hard floor between *remote* reloads/retries only — not the
  event debounce or max latency. Settings changes reset the retry backoff via
  the SQLite trigger.
* **Single flight / lease**: a reconcile claims the persisted singleton lease
  atomically (one of N coordinators wins), heartbeats it throughout the whole
  attempt, and releases it on finish. A 90s TTL + fencing token guard every
  state mutation; a 120s overall timeout bounds one attempt. State transactions
  are short (claim, capture, each completion step) — the projection read
  transaction is closed before any file/network I/O.
* **One pass**: preflight (mode/readiness/remote capability/backend/readonly)
  before any file write; generate+publish exactly once; checkpoint
  ``last_published_hash``; skip the remote POST only when the body hash equals
  ``last_remote_hash`` AND the run is not force-driven; otherwise reuse one
  client for POST + diagnosis. An accepted POST advances ``applied`` through
  the captured target ``R`` and stamps ``last_remote_hash`` even when diagnosis
  is fatal/unknown; the healthy hash is recorded only on a healthy diagnosis.
  A pre-POST/ambiguous failure leaves the row dirty and arms the deterministic
  1/2/5/10/15-min backoff. A six-hour force reload calls remote even when clean.
* **Crash safety**: the lease-token guard means a stale (expired/reclaimed)
  run's completions raise :class:`LeaseStaleError` and are abandoned — they are
  never committed. A commit during a run leaves ``desired > applied`` so the
  next tick reconciles again (coalesced).

All seams (clock, sleep, client factory, orchestrator stages, the attempt
itself) are injectable so tests are deterministic and never touch the network
or sleep for real.
"""

import asyncio
import datetime
import logging
from typing import Any, Awaitable, Callable, NamedTuple

from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_dashboard.db.models import ExtensionRun, utc_now
from financial_dashboard.integrations.paisa import PaisaClient
from financial_dashboard.services.paisa.audit import (
    OPERATION_AUTOMATIC,
    STATUS_FAILURE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    complete_run,
    last_run,
    sanitize_error,
    start_run,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig, load_config
from financial_dashboard.services.paisa.orchestrator import (
    GenerateResult,
    PreflightReport,
    RemoteSyncReport,
    _build_client,
    generate,
    preflight,
    sync_remote,
)
from financial_dashboard.services.paisa.sync_state import (
    DEFAULT_LEASE_TTL_SECONDS,
    DIAGNOSIS_ACCEPTED,
    DIAGNOSIS_FATAL,
    DIAGNOSIS_HEALTHY,
    DIAGNOSIS_UNKNOWN,
    EXTENSION_PAISA,
    LeaseStaleError,
    capture_target,
    claim_lease,
    ensure_sync_state,
    heartbeat_lease,
    read_sync_state,
    record_accepted_post,
    record_diagnosis,
    record_hash_noop,
    record_pre_post_failure,
    record_published_hash,
    reconcile_eligibility,
    release_lease,
    should_skip_remote_post,
)
from financial_dashboard.services.settings import (
    get_setting_bool,
    get_setting_int,
)

logger = logging.getLogger(__name__)

#: Fixed timings (non-tunable per AGENTS.md).
POLL_INTERVAL_SECONDS = 2.0
HEARTBEAT_INTERVAL_SECONDS = 20.0
ATTEMPT_TIMEOUT_SECONDS = 120.0
LEASE_TTL_SECONDS = DEFAULT_LEASE_TTL_SECONDS  # 90s

#: Manual single-flight wait: how long a manual generate/sync polls for the
#: lease before returning ``busy`` instead of overlapping.
MANUAL_LEASE_WAIT_SECONDS = 30.0
MANUAL_LEASE_POLL_SECONDS = 0.5

#: Outcomes that represent a real failure worth notifying about (mirrors the
#: automatic runtime's classification).
FAILURE_OUTCOMES = frozenset(
    {
        "unreachable",
        "readonly",
        "unsupported_backend",
        "publish_failed",
        "sync_rejected",
        "diagnosis_failed",
        "generate_failed",
        "error",
    }
)

#: Outcomes that are mode/readiness guards, not real attempts.
GUARD_OUTCOMES = frozenset({"disabled", "connect_only", "not_configured"})

#: Default owner tag for the automatic coordinator's lease claims.
DEFAULT_OWNER = "paisa-coordinator"

#: Trigger token recorded on automatic audit rows.
TRIGGER_TRANSACTION_COORDINATOR = "transaction_coordinator"

#: Type aliases for injectable seams.
Clock = Callable[[], datetime.datetime]
Sleep = Callable[[float], Awaitable[None]]
ClientFactory = Callable[[PaisaProjectionConfig], PaisaClient]
PreflightFn = Callable[..., Awaitable[PreflightReport]]
GenerateFn = Callable[..., Awaitable[GenerateResult]]
SyncRemoteFn = Callable[..., Awaitable[RemoteSyncReport]]


class AttemptResult(NamedTuple):
    """Outcome of one reconcile attempt, audit-ready.

    ``attempted`` is False for ticks that did no work (not eligible, lost the
    lease, no row) so the caller writes no audit row — a 2s no-op never spams
    the audit table.
    """

    attempted: bool
    status: str
    outcome: str
    output_hash: str | None
    emitted_count: int | None
    details: Any
    error: str | None
    notify_failure: bool


NO_OP = AttemptResult(
    attempted=False,
    status="",
    outcome="",
    output_hash=None,
    emitted_count=None,
    details=None,
    error=None,
    notify_failure=False,
)


class ManualClaim(NamedTuple):
    """Result of attempting to claim the lease for a manual operation."""

    claimed: bool
    token: str | None
    reason: str  # "claimed" | "busy" | "no_row"


# --------------------------------------------------------------------------- #
# Failure fingerprint (dedupe identical notifications across restarts)
# --------------------------------------------------------------------------- #


def _failure_fingerprint(outcome: str, error: str | None) -> str:
    """Stable short hash of a failure's identity (outcome + sanitized error).

    Reused from the automatic runtime contract: identical failures dedupe, a
    changed outcome/error notifies again. No credentials (error is already
    sanitized).
    """
    import hashlib

    norm_outcome = (outcome or "").strip()
    norm_error = " ".join((error or "").split())
    digest = hashlib.sha256(f"{norm_outcome}\n{norm_error}".encode("utf-8"))
    return digest.hexdigest()[:16]


def _notify_fp_from_run(run: ExtensionRun | None) -> str | None:
    """Read the persisted notification fingerprint from a prior run's details."""
    import json

    if run is None or not run.details:
        return None
    try:
        decoded = json.loads(run.details)
    except ValueError, TypeError:
        return None
    if isinstance(decoded, dict):
        fp = decoded.get("notify_fp")
        if isinstance(fp, str) and fp:
            return fp
    return None


def _guard_outcome_token(reason: str | None) -> str | None:
    """Classify a refusal reason as a mode/readiness guard token, else None."""
    text = (reason or "").lower()
    if not text:
        return None
    if "disabled" in text:
        return "disabled"
    if "connect_only" in text:
        return "connect_only"
    if (
        "not_configured" in text
        or "generated_path" in text
        or "cutover" in text
        or "selected account" in text
    ):
        return "not_configured"
    return None


# --------------------------------------------------------------------------- #
# Remote-floor helper
# --------------------------------------------------------------------------- #


def _remote_floor_satisfied(
    snapshot, *, now: datetime.datetime, min_interval_minutes: int
) -> bool:
    """Whether enough time has passed since the last remote attempt.

    The floor gates *remote* reloads/retries only — not the event debounce or
    max latency. It never blocks force/periodic reloads (those bypass this
    check at the caller).
    """
    last = snapshot.last_remote_attempt_at
    if last is None:
        return True
    return now - last >= datetime.timedelta(minutes=max(1, min_interval_minutes))


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #


class PaisaCoordinator:
    """One automatic reconcile worker, owned by the runtime.

    Construct with the shared ``session_factory`` (and, in tests, injectable
    clock/sleep/client/orchestrator seams). :meth:`start` launches exactly one
    background task; :meth:`stop` cancels and awaits it; :meth:`wake` is the
    commit-driven fast-path hint (also called by the runtime's
    ``after_fetch_cycle`` for backward compatibility).
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        owner: str = DEFAULT_OWNER,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
        attempt_timeout: float = ATTEMPT_TIMEOUT_SECONDS,
        lease_ttl: int = LEASE_TTL_SECONDS,
        now: Clock | None = None,
        sleep: Sleep | None = None,
        client_factory: ClientFactory | None = None,
        preflight_fn: PreflightFn | None = None,
        generate_fn: GenerateFn | None = None,
        sync_remote_fn: SyncRemoteFn | None = None,
        min_interval_minutes_fn: Callable[[], int] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._owner = owner
        self._poll_interval = poll_interval
        self._heartbeat_interval = heartbeat_interval
        self._attempt_timeout = attempt_timeout
        self._lease_ttl = lease_ttl
        self._now: Clock = now or utc_now
        self._sleep: Sleep = sleep or asyncio.sleep
        self._client_factory = client_factory
        self._preflight: PreflightFn = preflight_fn or preflight
        self._generate: GenerateFn = generate_fn or generate
        self._sync_remote: SyncRemoteFn = sync_remote_fn or sync_remote
        self._min_interval_minutes_fn = min_interval_minutes_fn or (
            lambda: max(1, get_setting_int("paisa.auto_sync_min_interval_minutes", 1))
        )

        self._event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stopped = False
        self._wake_signal: Callable[[], None] | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def session_factory(self) -> async_sessionmaker:
        return self._session_factory

    def wake(self) -> None:
        """Fast-path wake hint. Coalesced; never blocks."""
        self._event.set()

    async def start(self) -> None:
        """Launch exactly one coordinator task and register the commit wake."""
        if self._task is not None and not self._task.done():
            return
        self._stopped = False
        # Register a commit wake that schedules event.set() on this loop. The
        # signal runs in the SQLAlchemy commit path (sync, loop thread), so it
        # must not block; call_soon_threadsafe is non-blocking and safe from
        # any thread.
        loop = asyncio.get_running_loop()
        event = self._event

        def _signal() -> None:
            loop.call_soon_threadsafe(event.set)

        self._wake_signal = _signal
        from financial_dashboard.services.paisa.wakeup import (
            register_commit_wake,
        )

        register_commit_wake(self._wake_signal)
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel and await the coordinator task; unregister the wake signal."""
        self._stopped = True
        if self._wake_signal is not None:
            from financial_dashboard.services.paisa.wakeup import (
                unregister_commit_wake,
            )

            unregister_commit_wake(self._wake_signal)
            self._wake_signal = None
        self.wake()  # release a pending wait so the task observes _stopped
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("coordinator task ended with error")
            self._task = None

    async def _run(self) -> None:
        """Main loop: tick, then wait for the poll interval OR a wake."""
        while not self._stopped:
            # Clear before the tick so a wake during the tick re-arms the event
            # and triggers an immediate follow-up tick (no lost wake).
            self._event.clear()
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("coordinator tick failed")
            if self._stopped:
                break
            try:
                await asyncio.wait_for(self._event.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise

    # ------------------------------------------------------------------ #
    # One tick
    # ------------------------------------------------------------------ #

    async def _tick(self) -> None:
        """Read eligibility, and if due+eligible, run one reconciled attempt."""
        config = load_config()
        # Eligibility: only project mode + auto enabled does any I/O.
        if not config.can_project or not get_setting_bool(
            "paisa.auto_sync_enabled", False
        ):
            return

        now = self._now()
        async with self._session_factory() as session:
            snapshot = await read_sync_state(session)
            # ``init_db`` seeds this singleton on every boot.  Keep the hot tick
            # read-only when it exists: SQLite's ``INSERT ... DO NOTHING`` still
            # requests the writer lock, which made a harmless eligibility poll
            # block behind a large uncommitted statement import.  Seed only for
            # the recovery/test case where an administrator removed the row.
            if snapshot is None:
                snapshot = await ensure_sync_state(session, now=now)
                await session.commit()
            else:
                # A SELECT-only transaction must never call commit: the global
                # after_commit listener would wake this coordinator from its
                # own poll and turn a clean, caught-up row into a tight loop.
                await session.rollback()
        if snapshot is None:
            return

        eligibility = reconcile_eligibility(snapshot, now=now)
        if not eligibility.due:
            return

        # Remote floor: throttle remote reloads/retries for dirty/max-latency
        # reasons. force_reload and periodic reload bypass it (explicit reload
        # intent; periodic is 6h anyway).
        min_minutes = self._min_interval_minutes_fn()
        if eligibility.reason in ("dirty", "max_latency") and not snapshot.force_reload:
            if not _remote_floor_satisfied(
                snapshot, now=now, min_interval_minutes=min_minutes
            ):
                return

        force_remote = snapshot.force_reload or eligibility.reason == "periodic_reload"
        await self._attempt(config, snapshot, force_remote=force_remote)

    # ------------------------------------------------------------------ #
    # One reconcile attempt (audited, lease-guarded)
    # ------------------------------------------------------------------ #

    async def _attempt(
        self,
        config: PaisaProjectionConfig,
        snapshot,
        *,
        force_remote: bool,
    ) -> None:
        """Claim the lease, run one staged reconcile, write an audit row.

        No audit row is written when the attempt did no real work (lost the
        lease) so a 2s no-op never spams the audit table.
        """
        now = self._now()
        # Claim the lease in a short transaction.
        async with self._session_factory() as session:
            claim = await claim_lease(
                session, owner=self._owner, ttl_seconds=self._lease_ttl, now=now
            )
            await session.commit()
        if not claim.claimed or claim.token is None:
            return  # another coordinator owns it; back off silently

        token = claim.token
        # The lease protects every potentially slow stage, not only the remote
        # POST: preflight and projection/publication may also exceed the TTL.
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(token))
        result = AttemptResult(
            attempted=True,
            status=STATUS_FAILURE,
            outcome="error",
            output_hash=None,
            emitted_count=None,
            details=None,
            error=None,
            notify_failure=False,
        )
        previous_fp: str | None = None
        try:
            # Prior automatic run, for notify dedupe.
            async with self._session_factory() as session:
                previous = await last_run(
                    session,
                    extension_id=EXTENSION_PAISA,
                    operation=OPERATION_AUTOMATIC,
                )
                previous_fp = _notify_fp_from_run(previous)
                await session.rollback()

            result = await asyncio.wait_for(
                self._run_attempt_stages(config, token, force_remote=force_remote),
                timeout=self._attempt_timeout,
            )
        except asyncio.TimeoutError:
            result = AttemptResult(
                attempted=True,
                status=STATUS_FAILURE,
                outcome="error",
                output_hash=None,
                emitted_count=None,
                details={"reason": "attempt_timeout"},
                error=sanitize_error("attempt timed out"),
                notify_failure=True,
            )
        except LeaseStaleError as exc:
            # Our lease expired/was reclaimed mid-run. Abandon: do not commit
            # any state writes (they would be rejected by the token guard
            # anyway). Leave the row dirty for the next tick.
            logger.info("coordinator lease went stale: %s", exc)
            result = AttemptResult(
                attempted=True,
                status=STATUS_SKIPPED,
                outcome="stale_lease",
                output_hash=None,
                emitted_count=None,
                details={"reason": "stale_lease"},
                error=None,
                notify_failure=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("coordinator attempt raised")
            result = AttemptResult(
                attempted=True,
                status=STATUS_FAILURE,
                outcome="error",
                output_hash=None,
                emitted_count=None,
                details={"exception": type(exc).__name__},
                error=sanitize_error(f"{type(exc).__name__}: {exc}"),
                notify_failure=True,
            )
        finally:
            # Stop renewal before releasing so this attempt can never renew a
            # lease after its terminal success/error/cancellation cleanup.
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError, Exception:
                pass

            # Always release our lease so a peer can pick up immediately.
            try:
                async with self._session_factory() as session:
                    await release_lease(session, token=token, now=self._now())
                    await session.commit()
            except Exception:
                logger.debug("lease release failed", exc_info=True)

        if not result.attempted:
            return

        # Persist the audit row + dedupe the notification. The notify_fp is
        # stored in details so dedupe survives restarts.
        fp = _failure_fingerprint(result.outcome, result.error)
        should_notify = result.notify_failure and fp != previous_fp
        details = dict(result.details) if result.details else {}
        notify_sent = False
        if result.notify_failure:
            details["notify_fp"] = fp
        async with self._session_factory() as session:
            run = await start_run(
                session,
                extension_id=EXTENSION_PAISA,
                operation=OPERATION_AUTOMATIC,
                trigger=TRIGGER_TRANSACTION_COORDINATOR,
            )
            run_id = run.id
            await complete_run(
                session,
                run,
                status=result.status,
                outcome=result.outcome,
                output_hash=result.output_hash,
                emitted_count=result.emitted_count,
                details=details or None,
                error=result.error,
            )
            await session.commit()

        if should_notify:
            notify_sent = await self._maybe_notify_failure(result.outcome, result.error)
        # Persist the truthful notify_sent flag (reflects an actually attempted
        # eligible/configured notification, not merely dedupe intent) on the
        # audit row we just wrote. Stamped by the captured ``run_id`` (not
        # "most recent") so it can never land on the wrong row even if a second
        # automatic run somehow starts between the write and the stamp.
        if result.notify_failure:
            await self._stamp_notify_sent(run_id, notify_sent)

    async def _stamp_notify_sent(self, run_id: int, notify_sent: bool) -> None:
        """Record the truthful notify_sent flag on a specific automatic run.

        ``notify_sent`` is determined only AFTER the audit row is written (it
        reflects an actually attempted notification, not dedupe intent), so the
        flag is stamped in a short follow-up transaction keyed by ``run_id``
        (captured at ``start_run`` time) so it always lands on the correct row
        even under single-flight relaxation. Uses the same JSON encoding
        discipline as the audit helpers (sort_keys, default=str).
        """
        import json

        from sqlalchemy import select

        from financial_dashboard.db.models import ExtensionRun

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(ExtensionRun).where(ExtensionRun.id == run_id)
                )
            ).scalar_one_or_none()
            if row is not None:
                try:
                    decoded = json.loads(row.details) if row.details else {}
                    if not isinstance(decoded, dict):
                        decoded = {}
                except ValueError, TypeError:
                    decoded = {}
                decoded["notify_sent"] = bool(notify_sent)
                row.details = json.dumps(decoded, default=str, sort_keys=True)
                await session.commit()

    async def _run_attempt_stages(
        self,
        config: PaisaProjectionConfig,
        token: str,
        *,
        force_remote: bool,
    ) -> AttemptResult:
        """preflight → generate once → publish checkpoint → (skip|POST+diagnosis).

        Each state mutation is a short transaction guarded by ``token``. The
        projection read transaction is closed (session ends) before any file or
        network I/O. A stale token raises :class:`LeaseStaleError`, which the
        caller abandons.
        """
        now = self._now()
        owns_client = self._client_factory is None
        client = (
            self._client_factory(config)
            if self._client_factory is not None
            else _build_client(config)
        )
        try:
            # 1. Preflight (mode/readiness/remote capability/backend/readonly)
            #    BEFORE any file write. Session-free.
            pre = await self._preflight(config, client=client)
            if not pre.ok:
                return await self._record_preflight_outcome(token, pre, now=now)

            # 2. Capture target revision R in a fresh state session.
            async with self._session_factory() as state_sess:
                target = await capture_target(state_sess)
                await state_sess.commit()
            R = target.target_revision

            # 3. Generate + publish exactly once from a fresh coherent read
            #    session. Close that session before any file/network I/O.
            async with self._session_factory() as read_sess:
                generated = await self._generate(read_sess, config)
            # The publisher ran inside generate (synchronous file write), but
            # the read session is now closed — no open transaction during I/O.
            if (
                not generated.ok
                or generated.report is None
                or generated.publish is None
            ):
                return await self._record_generate_outcome(token, generated, now=now)

            body_hash = generated.publish.body_hash
            emitted = generated.report.emitted_count
            skipped = len(generated.report.skipped)

            # 4. Checkpoint the published hash (restart resumability).
            async with self._session_factory() as session:
                await record_published_hash(
                    session, body_hash=body_hash, token=token, now=now
                )
                await session.commit()

            # 5. Decide remote: hash-noop (skip) vs POST+diagnosis.
            async with self._session_factory() as session:
                snap = await read_sync_state(session)
                await session.rollback()

            if snap is None:
                # The singleton vanished mid-run (admin reset); abandon. The
                # next tick re-seeds via ensure_sync_state.
                return AttemptResult(
                    attempted=True,
                    status=STATUS_SKIPPED,
                    outcome="no_state",
                    output_hash=body_hash,
                    emitted_count=emitted,
                    details={"reason": "sync state row missing mid-run"},
                    error=None,
                    notify_failure=False,
                )

            if not force_remote and should_skip_remote_post(body_hash, snap):
                async with self._session_factory() as session:
                    await record_hash_noop(
                        session, target_revision=R, token=token, now=now
                    )
                    await session.commit()
                return AttemptResult(
                    attempted=True,
                    status=STATUS_SUCCESS,
                    outcome="skipped_unchanged",
                    output_hash=body_hash,
                    emitted_count=emitted,
                    details={
                        "reason": "generated content matches remote; reload skipped"
                    },
                    error=None,
                    notify_failure=False,
                )

            # 6. POST + diagnosis. The attempt-wide heartbeat started directly
            #    after the lease claim and remains active through this stage.
            remote = await self._sync_remote(generated.report, config, client=client)

            return await self._record_remote_outcome(
                token,
                remote,
                body_hash=body_hash,
                emitted=emitted,
                skipped=skipped,
                target_revision=R,
                now=now,
            )
        finally:
            if owns_client:
                try:
                    await client.aclose()
                except Exception:
                    logger.debug("client close failed", exc_info=True)

    async def _heartbeat_loop(self, token: str) -> None:
        """Extend the attempt's held lease until outer cleanup cancels it."""
        try:
            while True:
                await self._sleep(self._heartbeat_interval)
                try:
                    async with self._session_factory() as session:
                        await heartbeat_lease(
                            session,
                            token=token,
                            ttl_seconds=self._lease_ttl,
                            now=self._now(),
                        )
                        await session.commit()
                except Exception:
                    logger.debug("lease heartbeat failed", exc_info=True)
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------ #
    # Outcome recording (token-guarded; raise LeaseStaleError if stale)
    # ------------------------------------------------------------------ #

    async def _record_preflight_outcome(
        self, token: str, pre: PreflightReport, *, now: datetime.datetime
    ) -> AttemptResult:
        outcome = pre.outcome or "error"
        reason = pre.reason
        guard = _guard_outcome_token(reason) or _guard_outcome_token(outcome)
        if guard is not None:
            # Mode/readiness guard: skipped, no remote attempt, no backoff.
            return AttemptResult(
                attempted=True,
                status=STATUS_SKIPPED,
                outcome=guard,
                output_hash=None,
                emitted_count=None,
                details={"reason": reason, "stage": "preflight"},
                error=None,
                notify_failure=False,
            )
        # A real preflight failure (readonly/unsupported_backend/unreachable)
        # happened before any POST: leave dirty, arm backoff.
        async with self._session_factory() as session:
            await record_pre_post_failure(session, token=token, now=now)
            await session.commit()
        return AttemptResult(
            attempted=True,
            status=STATUS_FAILURE,
            outcome=str(outcome),
            output_hash=None,
            emitted_count=None,
            details={"reason": reason, "stage": "preflight"},
            error=sanitize_error(reason),
            notify_failure=str(outcome) in FAILURE_OUTCOMES,
        )

    async def _record_generate_outcome(
        self, token: str, generated: GenerateResult, *, now: datetime.datetime
    ) -> AttemptResult:
        reason = generated.reason
        guard = _guard_outcome_token(reason)
        if guard is not None:
            return AttemptResult(
                attempted=True,
                status=STATUS_SKIPPED,
                outcome=guard,
                output_hash=None,
                emitted_count=None,
                details={"reason": reason, "stage": "generate"},
                error=None,
                notify_failure=False,
            )
        # generate/publish failure before POST: leave dirty, arm backoff.
        async with self._session_factory() as session:
            await record_pre_post_failure(session, token=token, now=now)
            await session.commit()
        return AttemptResult(
            attempted=True,
            status=STATUS_FAILURE,
            outcome="generate_failed",
            output_hash=None,
            emitted_count=None,
            details={"reason": reason, "stage": "generate"},
            error=sanitize_error(reason),
            notify_failure=True,
        )

    async def _record_remote_outcome(
        self,
        token: str,
        remote: RemoteSyncReport,
        *,
        body_hash: str,
        emitted: int | None,
        skipped: int | None,
        target_revision: int,
        now: datetime.datetime,
    ) -> AttemptResult:
        details: dict[str, Any] = {
            "outcome": remote.outcome,
            "diagnosis_ok": remote.diagnosis_ok,
        }
        if remote.diagnosis_expected is not None:
            details["diagnosis_expected"] = remote.diagnosis_expected
        if remote.diagnosis_accepted is not None:
            details["diagnosis_accepted"] = remote.diagnosis_accepted
        if remote.diagnosis_fatal is not None:
            details["diagnosis_fatal"] = remote.diagnosis_fatal

        if remote.post_accepted:
            # Advance applied through R and stamp the remote hash, even when
            # diagnosis is fatal/unknown: the content for R was delivered.
            async with self._session_factory() as session:
                await record_accepted_post(
                    session,
                    target_revision=target_revision,
                    remote_hash=body_hash,
                    token=token,
                    now=now,
                )
                await session.commit()
            # Diagnosis quality (token-guarded).
            diag_state, healthy_hash = _classify_diagnosis(remote, body_hash)
            async with self._session_factory() as session:
                await record_diagnosis(
                    session,
                    state=diag_state,
                    token=token,
                    healthy_hash=healthy_hash,
                    now=now,
                )
                await session.commit()
            ok = remote.ok and diag_state in (DIAGNOSIS_HEALTHY, DIAGNOSIS_ACCEPTED)
            return AttemptResult(
                attempted=True,
                status=STATUS_SUCCESS if ok else STATUS_FAILURE,
                outcome=str(remote.outcome),
                output_hash=body_hash,
                emitted_count=emitted,
                details=details,
                error=None if ok else sanitize_error(remote.reason),
                notify_failure=(not ok) and (remote.outcome in FAILURE_OUTCOMES),
            )

        # Pre-POST/ambiguous failure: the row stays dirty; arm backoff.
        async with self._session_factory() as session:
            await record_pre_post_failure(session, token=token, now=now)
            await session.commit()
        return AttemptResult(
            attempted=True,
            status=STATUS_FAILURE,
            outcome=str(remote.outcome),
            output_hash=body_hash,
            emitted_count=emitted,
            details=details,
            error=sanitize_error(remote.reason),
            notify_failure=remote.outcome in FAILURE_OUTCOMES,
        )

    # ------------------------------------------------------------------ #
    # Notification
    # ------------------------------------------------------------------ #

    async def _maybe_notify_failure(self, outcome: str, reason: str | None) -> bool:
        """Send a Telegram message on a real sync failure, if opted in.

        Returns True only when a notification was actually *attempted*
        (configured + send called). Best-effort and fully isolated. Reused
        verbatim from the automatic runtime contract.
        """
        if not get_setting_bool("paisa.notify_sync_failures", False):
            return False
        import html

        try:
            from financial_dashboard.services import telegram as telegram_service
            from financial_dashboard.services.settings import (
                get_telegram_chat_id,
                is_telegram_configured,
            )

            if not is_telegram_configured() or telegram_service.tg_app is None:
                return False
            text = (
                "\u26a0\ufe0f <b>Paisa auto-sync failed</b>\n"
                f"outcome: {html.escape(outcome)}"
                + (f"\n{html.escape(reason)}" if reason else "")
            )
            await telegram_service._send_with_retry(
                telegram_service.tg_app,
                chat_id=get_telegram_chat_id(),
                text=text,
            )
            return True
        except Exception:
            logger.warning("Paisa sync-failure notification failed", exc_info=True)
            return False


def _classify_diagnosis(
    remote: RemoteSyncReport, body_hash: str
) -> tuple[str, str | None]:
    """Map a RemoteSyncReport to (diagnosis_state, healthy_hash).

    * healthy/accepted diagnosis → (HEALTHY|ACCEPTED, body_hash)
    * fatal → (FATAL, None) — arms retry backoff, no healthy hash
    * unknown (diagnosis did not run) → (UNKNOWN, None)
    """
    if remote.diagnosis_ok is True:
        # accepted contra-expense postings are still a healthy remote.
        state = (
            DIAGNOSIS_ACCEPTED
            if (remote.diagnosis_accepted or 0) > 0
            else DIAGNOSIS_HEALTHY
        )
        return state, body_hash
    if remote.diagnosis_ok is False:
        return DIAGNOSIS_FATAL, None
    return DIAGNOSIS_UNKNOWN, None


# --------------------------------------------------------------------------- #
# Manual single-flight lease acquisition (shared with the surface)
# --------------------------------------------------------------------------- #


async def claim_manual_lease(
    session,
    *,
    owner: str,
    wait_seconds: float = MANUAL_LEASE_WAIT_SECONDS,
    poll_seconds: float = MANUAL_LEASE_POLL_SECONDS,
    ttl_seconds: int = LEASE_TTL_SECONDS,
    now: datetime.datetime | None = None,
    sleep: Sleep | None = None,
) -> ManualClaim:
    """Claim the singleton lease for a manual operation, waiting up to
    ``wait_seconds`` if it is held.

    Uses the caller-owned ``session`` (the request session for manual routes).
    Commits between poll attempts so a held lease does not pin a long open
    transaction. Returns ``claimed=True`` with the fencing token, or
    ``claimed=False`` with ``reason="busy"`` when the wait elapses.

    The wait deadline is measured in real wall-clock time (independent of the
    optional ``now`` stamp used for lease stamping), so a held lease always
    resolves to ``busy`` after ``wait_seconds`` regardless of the injected
    clock.

    **Premature-commit note**: the ``session.commit()`` calls inside this
    function commit *everything* pending in the caller's session, including an
    audit ``running`` row that ``_audited`` started before calling this. That is
    intentional and safe: the committed ``running`` row means a crash during the
    manual operation leaves an observable ``running`` row (not a silently lost
    one), and the subsequent ``complete_run`` + commit finalizes it. The commit
    does not corrupt the caller's transaction — it just ends it early, starting
    a fresh transaction for the remaining work. A caller that must not have its
    prior writes committed by this function should call it before starting any
    other writes on the session.
    """
    sleep_fn = sleep or asyncio.sleep
    stamp = now  # datetime | None — used only for lease stamping
    deadline = utc_now() + datetime.timedelta(seconds=wait_seconds)
    await ensure_sync_state(session, now=stamp)
    await session.commit()
    while True:
        claim = await claim_lease(
            session, owner=owner, ttl_seconds=ttl_seconds, now=stamp
        )
        await session.commit()
        if claim.claimed and claim.token is not None:
            return ManualClaim(claimed=True, token=claim.token, reason="claimed")
        if utc_now() >= deadline:
            return ManualClaim(claimed=False, token=None, reason="busy")
        await sleep_fn(poll_seconds)


__all__ = [
    "ATTEMPT_TIMEOUT_SECONDS",
    "AttemptResult",
    "ClientFactory",
    "DEFAULT_OWNER",
    "FAILURE_OUTCOMES",
    "HEARTBEAT_INTERVAL_SECONDS",
    "LEASE_TTL_SECONDS",
    "MANUAL_LEASE_POLL_SECONDS",
    "MANUAL_LEASE_WAIT_SECONDS",
    "ManualClaim",
    "NO_OP",
    "POLL_INTERVAL_SECONDS",
    "PaisaCoordinator",
    "TRIGGER_TRANSACTION_COORDINATOR",
    "claim_manual_lease",
]
