"""Database service primitives for the per-extension sync-state singleton.

This module owns every read and mutation of the
:class:`~financial_dashboard.db.models.ExtensionSyncState` row (the
``extension_sync_state`` table). The row is the *current* coordinator state for
an extension — exactly one row per extension, never growing with operation count
(the per-operation audit log lives in ``extension_runs``, written by
:mod:`financial_dashboard.services.paisa.audit`).

The table's ``desired_revision`` / ``first_dirty_at`` / ``last_dirty_at`` are
maintained by SQLite ``AFTER INSERT/UPDATE/DELETE`` triggers
:mod:`financial_dashboard.db.init_db` installs on the core projection source
tables; this module owns *every other* column. ``desired_revision`` is never
written here — it moves only with the core data, transactionally.

Conventions (mirroring :mod:`financial_dashboard.services.paisa.audit`):

* **Caller owns the commit.** Every function only ``execute``\\ s its statement
  inside the session's open transaction and returns. Nothing commits. A request
  handler commits its request session; the coordinator commits the session it
  opens.
* **Caller-owned ``AsyncSession``.** Each function takes ``session`` as its
  first argument. The caller decides transaction scope. The lease is durable
  across transactions: a coordinator may commit the claim, do its network work,
  then complete in a fresh transaction guarded by the persisted lease token.
* **No implicit ORM identity-map reliance.** Every read and mutation is a raw
  ``text()`` statement with an explicit ``RETURNING`` column list, so the
  snapshot returned is always the post-write row the database committed within
  this transaction — never a stale identity-map cached object. Datetimes read
  back from SQLite (which stores them naive) are normalized to tz-aware UTC.

Lease / fencing model:

* :func:`claim_lease` is an atomic ``UPDATE ... WHERE lease_expires_at IS NULL OR
  lease_expires_at < :now RETURNING ...`` — exactly one of two concurrent
  coordinators wins; the other's ``WHERE`` matches no row. The claim mints (or
  accepts) a fencing ``token`` and an ``expires_at``.
* The completion-path mutations (:func:`record_published_hash`,
  :func:`record_hash_noop`, :func:`record_accepted_post`, :func:`record_diagnosis`,
  :func:`record_pre_post_failure`) all carry ``WHERE lease_token = :token``. If
  the lease was lost (expired and reclaimed, or released) between claim and
  completion, the token no longer matches, no row updates, and the call raises
  :class:`LeaseStaleError` — the stale run's results are abandoned and must not
  be committed. This is the "lease-token guarded completion" contract.
* :func:`heartbeat_lease` extends a held lease by token; :func:`release_lease`
  clears it.

Reconcile completion semantics (the subtle part):

* ``applied_revision`` is advanced through the *captured target* ``R`` (the
  ``desired_revision`` observed at :func:`capture_target` time), clamped to the
  *current* ``desired_revision`` evaluated at UPDATE time — so
  ``applied`` can never exceed ``desired`` (the table's CHECK constraint mirrors
  this).
* If a core write landed *during* the run (a concurrent commit bumped
  ``desired_revision`` past ``R``), the UPDATE evaluates the larger
  ``desired_revision`` and (a) clamps ``applied`` to ``R`` and (b) **preserves**
  ``first_dirty_at`` / ``last_dirty_at`` and ``force_reload`` — the row stays
  dirty so the coordinator reconciles again for the new revision. Dirty
  timestamps and ``force_reload`` are cleared *only* when the run fully caught
  up (``applied >= desired``).
* A hash-noop (the generated include hash equals ``last_remote_hash``: the
  remote already holds that exact file) advances ``applied`` through ``R`` with
  no remote round-trip and no ``last_remote_attempt_at`` change, so the six-hour
  periodic reload timer keeps ticking.
* A fatal diagnosis after an accepted POST does **not** re-dirty or re-force:
  the content for ``R`` was delivered, so ``applied``/dirty/force were already
  settled by the accepted POST; the diagnosis only records ``diagnosis_state``
  and arms the retry backoff. Combined with "not dirty / not force" that makes
  the row not-due → no reload loop; the next attempt waits for a drift event,
  the six-hour periodic reload, or the backoff window on the next drift.

Backoff is the fixed, non-tunable 1/2/5/10/15-minute schedule. It is injectable
on the failure/recording functions purely for deterministic testing; the runtime
always passes the default.
"""

import datetime
import logging
import secrets
from typing import Any, NamedTuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import utc_now

logger = logging.getLogger(__name__)

#: The single extension id this table is used for today. The table and these
#: helpers are generic over the id, but only Paisa exists.
EXTENSION_PAISA = "paisa"

#: Fixed, non-tunable retry backoff schedule in minutes (per AGENTS.md).
#: ``failure_count`` 1→1m, 2→2m, 3→5m, 4→10m, ≥5→15m.
BACKOFF_MINUTES: tuple[int, ...] = (1, 2, 5, 10, 15)

#: Quiet debounce window: once data dirties, wait this long with no new dirty
#: event before reconciling, so a burst of writes coalesces into one reload.
DEFAULT_QUIET_DEBOUNCE = datetime.timedelta(seconds=5)

#: Maximum dirty latency: never wait longer than this from ``first_dirty_at``
#: before reconciling, even if writes keep landing.
DEFAULT_MAX_DIRTY_LATENCY = datetime.timedelta(seconds=30)

#: Periodic full-reload interval: force a reload when this long has elapsed
#: since the last remote attempt, even with no observable drift.
DEFAULT_FORCE_RELOAD_INTERVAL = datetime.timedelta(hours=6)

#: Default lease TTL. Generous enough for a full generate + POST + diagnosis on
#: a large journal, short enough that a crashed coordinator's lease is reclaimed
#: promptly. The coordinator heartbeats if a run runs longer.
DEFAULT_LEASE_TTL_SECONDS = 90

# Diagnosis-state tokens. Free-form on the column, pinned here so query filters
# and the completion policy are stable.
DIAGNOSIS_HEALTHY = "healthy"
DIAGNOSIS_ACCEPTED = "accepted"
DIAGNOSIS_FATAL = "fatal"
DIAGNOSIS_UNKNOWN = "unknown"
_DIAGNOSIS_STATES = frozenset(
    {DIAGNOSIS_HEALTHY, DIAGNOSIS_ACCEPTED, DIAGNOSIS_FATAL, DIAGNOSIS_UNKNOWN}
)

#: The ordered column list every ``RETURNING`` clause emits, so the positional
#: mapper below stays in lockstep with the SQL. Adding a model column means
#: extending both this string and :func:`_row_to_snapshot`.
_RETURNING_SQL = (
    "extension_id, desired_revision, applied_revision, "
    "first_dirty_at, last_dirty_at, last_published_hash, "
    "last_remote_hash, last_healthy_hash, last_remote_attempt_at, "
    "next_attempt_at, failure_count, diagnosis_state, force_reload, "
    "lease_owner, lease_token, lease_expires_at"
)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


class SyncStateSnapshot(NamedTuple):
    """A read view of the singleton sync-state row, with tz-aware UTC datetimes.

    Every datetime field is normalized to tz-aware UTC; SQLite stores them
    naive, so reads are run through :func:`to_utc`. ``None`` passes through.
    """

    extension_id: str
    desired_revision: int
    applied_revision: int
    first_dirty_at: datetime.datetime | None
    last_dirty_at: datetime.datetime | None
    last_published_hash: str | None
    last_remote_hash: str | None
    last_healthy_hash: str | None
    last_remote_attempt_at: datetime.datetime | None
    next_attempt_at: datetime.datetime | None
    failure_count: int
    diagnosis_state: str | None
    force_reload: bool
    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: datetime.datetime | None


class ReconcileEligibility(NamedTuple):
    """Whether a coordinator should reconcile now, and why.

    ``eligible_at`` is when the row next becomes due (for the poller's sleep),
    or ``None`` when it is due now or not due awaiting an external event (a
    drift, an operator force). It is tz-aware UTC.
    """

    due: bool
    reason: str
    eligible_at: datetime.datetime | None


class LeaseClaim(NamedTuple):
    """Result of attempting to acquire the singleton lease."""

    claimed: bool
    owner: str
    token: str
    expires_at: datetime.datetime | None
    reason: str
    snapshot: SyncStateSnapshot | None


class LeaseHeartbeat(NamedTuple):
    """Result of extending a held lease."""

    renewed: bool
    token: str
    expires_at: datetime.datetime | None
    reason: str


class LeaseRelease(NamedTuple):
    """Result of releasing a held lease."""

    released: bool
    reason: str


class ReconcileTarget(NamedTuple):
    """The captured ``desired_revision`` a reconcile targets, plus the snapshot."""

    target_revision: int
    snapshot: SyncStateSnapshot


class AppliedUpdate(NamedTuple):
    """Result of advancing ``applied_revision`` (accepted POST or hash-noop).

    ``caught_up`` is ``applied_revision >= desired_revision`` after the write;
    ``cleared_dirty`` is whether the dirty window + ``force_reload`` were
    cleared (only when caught up).
    """

    target_revision: int
    applied_revision: int
    desired_revision: int
    caught_up: bool
    cleared_dirty: bool
    remote_hash: str | None
    snapshot: SyncStateSnapshot


class HashUpdate(NamedTuple):
    """Result of recording a freshly published include-file hash."""

    last_published_hash: str
    snapshot: SyncStateSnapshot


class DiagnosisUpdate(NamedTuple):
    """Result of recording a diagnosis outcome."""

    state: str
    last_healthy_hash: str | None
    failure_count: int
    next_attempt_at: datetime.datetime | None
    snapshot: SyncStateSnapshot


class FailureUpdate(NamedTuple):
    """Result of recording a pre-POST remote failure (with retry backoff)."""

    failure_count: int
    next_attempt_at: datetime.datetime
    backoff_seconds: int
    last_remote_attempt_at: datetime.datetime
    snapshot: SyncStateSnapshot


class ForceReloadUpdate(NamedTuple):
    """Result of an operator/admin force-reload toggle."""

    force_reload: bool
    snapshot: SyncStateSnapshot


class ResetUpdate(NamedTuple):
    """Result of an admin reset to the dirty seed state."""

    snapshot: SyncStateSnapshot


class LeaseStaleError(Exception):
    """Raised when a token-guarded completion runs against a stale lease.

    The lease was lost — expired and reclaimed by another coordinator, or
    released — between :func:`claim_lease` and the guarded call, so the
    in-flight results belong to a run that no longer owns the singleton. The
    caller must abandon them and **not commit** the open transaction (it should
    roll back). Carries the extension id and the stale token for diagnostics.
    """

    def __init__(
        self, extension_id: str, *, token: str, reason: str = "stale_token"
    ) -> None:
        self.extension_id = extension_id
        self.token = token
        self.reason = reason
        super().__init__(
            f"lease for {extension_id!r} is stale (token mismatch: {reason})"
        )


# --------------------------------------------------------------------------- #
# Datetime helpers
# --------------------------------------------------------------------------- #


def to_utc(value: datetime.datetime | str | None) -> datetime.datetime | None:
    """Normalize a datetime (or SQLite TEXT read) to tz-aware UTC.

    Raw ``text()`` reads bypass SQLAlchemy's column typing, so datetimes come
    back as ISO strings; an ORM-typed read comes back as a ``datetime``. Both
    are handled: a ``str`` is parsed via ``fromisoformat`` (accepting the
    space-separated SQLite form), a naive ``datetime`` is assumed UTC, and an
    aware one is converted to UTC. ``None`` passes through.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.datetime.fromisoformat(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC)


def _now_utc(now: datetime.datetime | None = None) -> datetime.datetime:
    """Resolve a caller-supplied ``now`` (or wall-clock) to tz-aware UTC."""
    dt = now if now is not None else utc_now()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.UTC)
    return dt.astimezone(datetime.UTC)


def _store(dt: datetime.datetime) -> datetime.datetime:
    """Strip a tz-aware UTC datetime to naive UTC for SQLite storage.

    Naive ISO storage is consistent with the triggers' ``CURRENT_TIMESTAMP``
    and with the existing direct-SQL tests, so cross-format comparisons and
    round-trips stay well-behaved.
    """
    return dt.astimezone(datetime.UTC).replace(tzinfo=None)


def _row_to_snapshot(row: Any) -> SyncStateSnapshot:
    """Map a ``RETURNING``/``SELECT`` row (positional, in ``_RETURNING_SQL``
    order) to a tz-normalized snapshot."""
    return SyncStateSnapshot(
        extension_id=row[0],
        desired_revision=int(row[1]),
        applied_revision=int(row[2]),
        first_dirty_at=to_utc(row[3]),
        last_dirty_at=to_utc(row[4]),
        last_published_hash=row[5],
        last_remote_hash=row[6],
        last_healthy_hash=row[7],
        last_remote_attempt_at=to_utc(row[8]),
        next_attempt_at=to_utc(row[9]),
        failure_count=int(row[10]),
        diagnosis_state=row[11],
        force_reload=bool(row[12]),
        lease_owner=row[13],
        lease_token=row[14],
        lease_expires_at=to_utc(row[15]),
    )


# --------------------------------------------------------------------------- #
# Backoff (pure, injectable, deterministic)
# --------------------------------------------------------------------------- #


def backoff_delta(
    failure_count: int,
    backoff_minutes: tuple[int, ...] = BACKOFF_MINUTES,
) -> datetime.timedelta:
    """The retry delay for a given (1-based) consecutive failure count.

    ``failure_count`` 1→1m, 2→2m, 3→5m, 4→10m, ≥5→15m, clamped to the last step.
    Counts ≤ 0 are treated as 1. ``backoff_minutes`` is injectable so tests are
    deterministic; the runtime always passes the default fixed schedule.
    """
    if len(backoff_minutes) == 0:
        return datetime.timedelta(0)
    idx = max(0, failure_count - 1)
    idx = min(idx, len(backoff_minutes) - 1)
    return datetime.timedelta(minutes=backoff_minutes[idx])


def new_lease_token() -> str:
    """Mint an opaque, URL-safe fencing token for a lease."""
    return secrets.token_urlsafe(16)


# --------------------------------------------------------------------------- #
# Read / ensure
# --------------------------------------------------------------------------- #


async def read_sync_state(
    session: AsyncSession,
    *,
    extension_id: str = EXTENSION_PAISA,
) -> SyncStateSnapshot | None:
    """Read the singleton row, or ``None`` if it does not exist.

    Uses a raw ``SELECT`` (not ``session.get``) so the result never reflects a
    stale ORM identity-map entry left by a prior bulk ``UPDATE`` in this
    session. Datetimes are normalized to tz-aware UTC.
    """
    row = (
        await session.execute(
            text(
                f"SELECT {_RETURNING_SQL} FROM extension_sync_state "
                "WHERE extension_id = :eid"
            ),
            {"eid": extension_id},
        )
    ).first()
    return _row_to_snapshot(row) if row is not None else None


async def ensure_sync_state(
    session: AsyncSession,
    *,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> SyncStateSnapshot:
    """Idempotently seed the singleton row dirty and return it.

    ``init_db`` already seeds the Paisa singleton on every boot, so this is a
    safety net for tests / fresh schemas. The seed mirrors ``init_db`` exactly
    — ``desired_revision=1``, ``applied_revision=0``, dirty, ``force_reload`` —
    so the next enabled coordinator reconciles once. ``ON CONFLICT DO NOTHING``
    leaves a coordinator-owned row authoritative; this never overwrites one.
    """
    stored = _store(_now_utc(now))
    await session.execute(
        text(
            "INSERT INTO extension_sync_state "
            "(extension_id, desired_revision, applied_revision, "
            " first_dirty_at, last_dirty_at, force_reload, failure_count, "
            " created_at, updated_at) "
            "VALUES (:eid, 1, 0, :now, :now, 1, 0, :now, :now) "
            "ON CONFLICT(extension_id) DO NOTHING"
        ),
        {"eid": extension_id, "now": stored},
    )
    snapshot = await read_sync_state(session, extension_id=extension_id)
    assert snapshot is not None, f"ensure_sync_state failed to seed {extension_id!r}"
    return snapshot


async def _fetch_guarded(
    session: AsyncSession, extension_id: str, token: str
) -> SyncStateSnapshot:
    """Read the row only if ``token`` currently holds its lease, else raise.

    The read+mutate completion primitives use this to capture the pre-write
    snapshot (e.g. current ``failure_count``) under the same token guard the
    subsequent mutation applies, so a stale-token run fails fast.
    """
    row = (
        await session.execute(
            text(
                f"SELECT {_RETURNING_SQL} FROM extension_sync_state "
                "WHERE extension_id = :eid AND lease_token = :token"
            ),
            {"eid": extension_id, "token": token},
        )
    ).first()
    if row is None:
        raise LeaseStaleError(extension_id, token=token)
    return _row_to_snapshot(row)


# --------------------------------------------------------------------------- #
# Dirty / debounce eligibility (pure)
# --------------------------------------------------------------------------- #


def reconcile_eligibility(
    snapshot: SyncStateSnapshot,
    *,
    now: datetime.datetime | None = None,
    quiet_debounce: datetime.timedelta = DEFAULT_QUIET_DEBOUNCE,
    max_dirty_latency: datetime.timedelta = DEFAULT_MAX_DIRTY_LATENCY,
    force_reload_interval: datetime.timedelta = DEFAULT_FORCE_RELOAD_INTERVAL,
) -> ReconcileEligibility:
    """Decide whether a coordinator should reconcile now.

    Policy (fixed, non-tunable except for deterministic tests):

    1. **Backoff gate** — if a prior failure scheduled ``next_attempt_at`` and
       it has not arrived, not due (wait for the retry window).
    2. **One-shot force** — ``force_reload`` (set by migration or an operator)
       means reconcile now.
    3. **Periodic reload** — ``last_remote_attempt_at`` older than
       ``force_reload_interval`` means force a full reload.
    4. **Dirty** — ``desired_revision > applied_revision`` reconciles once the
       quiet debounce has elapsed since ``last_dirty_at``, or once
       ``max_dirty_latency`` has elapsed since ``first_dirty_at`` (whichever
       comes first).
    5. **Caught up** — otherwise not due (waiting on an external event).
    """
    now_utc = _now_utc(now)

    next_attempt = to_utc(snapshot.next_attempt_at)
    if next_attempt is not None and now_utc < next_attempt:
        return ReconcileEligibility(False, "backoff", next_attempt)

    if snapshot.force_reload:
        return ReconcileEligibility(True, "force_reload", now_utc)

    last_attempt = to_utc(snapshot.last_remote_attempt_at)
    if last_attempt is not None and now_utc - last_attempt >= force_reload_interval:
        return ReconcileEligibility(True, "periodic_reload", now_utc)

    if snapshot.desired_revision <= snapshot.applied_revision:
        return ReconcileEligibility(False, "caught_up", None)

    first_dirty = to_utc(snapshot.first_dirty_at)
    last_dirty = to_utc(snapshot.last_dirty_at)
    if first_dirty is not None and now_utc - first_dirty >= max_dirty_latency:
        return ReconcileEligibility(True, "max_latency", now_utc)
    if last_dirty is not None and now_utc - last_dirty < quiet_debounce:
        return ReconcileEligibility(
            False, "debounce", to_utc(last_dirty + quiet_debounce)
        )
    return ReconcileEligibility(True, "dirty", now_utc)


def should_skip_remote_post(
    published_hash: str | None, snapshot: SyncStateSnapshot
) -> bool:
    """Whether the remote already holds the freshly generated include bytes.

    True when the just-published hash equals ``last_remote_hash`` — Paisa
    already loaded that exact file, so a ``/api/sync`` POST would be a redundant
    reload. ``None`` published hash or a never-attempted remote (``None``
    ``last_remote_hash``) never skips: the first delivery must always POST.
    """
    if published_hash is None or snapshot.last_remote_hash is None:
        return False
    return snapshot.last_remote_hash == published_hash


# --------------------------------------------------------------------------- #
# Lease: claim / heartbeat / release
# --------------------------------------------------------------------------- #


async def claim_lease(
    session: AsyncSession,
    *,
    owner: str,
    token: str | None = None,
    extension_id: str = EXTENSION_PAISA,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    now: datetime.datetime | None = None,
) -> LeaseClaim:
    """Atomically acquire the singleton lease.

    The ``UPDATE ... WHERE lease_expires_at IS NULL OR lease_expires_at < :now
    RETURNING ...`` runs as one statement, so of two concurrent coordinators
    only the winner's ``WHERE`` matches a row (the loser's ``WHERE`` excludes
    the now-valid lease). The winner's lease records ``owner``, a fencing
    ``token`` (minted if not supplied), and ``expires_at = now + ttl``.

    Returns a :class:`LeaseClaim` whose ``claimed`` is True on success; on
    failure ``reason`` is ``"no_row"`` (the singleton is missing — call
    :func:`ensure_sync_state`) or ``"held"`` (a live lease blocks). The loser
    does **not** raise — it learns it lost and backs off.
    """
    now_utc = _now_utc(now)
    lease_token = token or new_lease_token()
    expires_at = now_utc + datetime.timedelta(seconds=ttl_seconds)
    stored_now = _store(now_utc)
    stored_expires = _store(expires_at)
    row = (
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET lease_owner = :owner, lease_token = :token, "
                "    lease_expires_at = :expires, updated_at = :now "
                "WHERE extension_id = :eid "
                "  AND (lease_expires_at IS NULL OR lease_expires_at < :now) "
                f"RETURNING {_RETURNING_SQL}"
            ),
            {
                "eid": extension_id,
                "owner": owner,
                "token": lease_token,
                "expires": stored_expires,
                "now": stored_now,
            },
        )
    ).first()
    if row is not None:
        return LeaseClaim(
            claimed=True,
            owner=owner,
            token=lease_token,
            expires_at=expires_at,
            reason="claimed",
            snapshot=_row_to_snapshot(row),
        )
    existing = await read_sync_state(session, extension_id=extension_id)
    if existing is None:
        return LeaseClaim(
            claimed=False,
            owner=owner,
            token=lease_token,
            expires_at=None,
            reason="no_row",
            snapshot=None,
        )
    return LeaseClaim(
        claimed=False,
        owner=owner,
        token=lease_token,
        expires_at=existing.lease_expires_at,
        reason="held",
        snapshot=existing,
    )


async def heartbeat_lease(
    session: AsyncSession,
    *,
    token: str,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> LeaseHeartbeat:
    """Extend a held lease by ``ttl`` (token-guarded).

    Only the current token holder may renew; a stale token renews nothing. A
    long-running reconcile (large journal generate + POST + diagnosis) calls
    this before its lease would expire.
    """
    now_utc = _now_utc(now)
    expires_at = now_utc + datetime.timedelta(seconds=ttl_seconds)
    row = (
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET lease_expires_at = :expires, updated_at = :now "
                "WHERE extension_id = :eid AND lease_token = :token "
                "RETURNING extension_id"
            ),
            {
                "eid": extension_id,
                "token": token,
                "expires": _store(expires_at),
                "now": _store(now_utc),
            },
        )
    ).first()
    if row is None:
        return LeaseHeartbeat(False, token, None, "stale_token")
    return LeaseHeartbeat(True, token, expires_at, "renewed")


async def release_lease(
    session: AsyncSession,
    *,
    token: str,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> LeaseRelease:
    """Clear a held lease (token-guarded).

    A coordinator releases on a clean finish so a peer can pick up immediately
    rather than waiting for expiry. A stale token releases nothing.
    """
    row = (
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL, "
                "    updated_at = :now "
                "WHERE extension_id = :eid AND lease_token = :token "
                "RETURNING extension_id"
            ),
            {"eid": extension_id, "token": token, "now": _store(_now_utc(now))},
        )
    ).first()
    if row is None:
        return LeaseRelease(False, "stale_token")
    return LeaseRelease(True, "released")


# --------------------------------------------------------------------------- #
# Capture target revision
# --------------------------------------------------------------------------- #


async def capture_target(
    session: AsyncSession,
    *,
    extension_id: str = EXTENSION_PAISA,
) -> ReconcileTarget:
    """Read ``desired_revision`` as the reconcile target ``R``.

    The coordinator reconciles *up to* ``R``. A concurrent core write may bump
    ``desired_revision`` past ``R`` during the run; the completion primitives
    re-evaluate ``desired_revision`` at UPDATE time and clamp/preserve
    accordingly, so capturing ``R`` never races the bump.
    """
    snapshot = await read_sync_state(session, extension_id=extension_id)
    if snapshot is None:
        snapshot = await ensure_sync_state(session, extension_id=extension_id)
    return ReconcileTarget(snapshot.desired_revision, snapshot)


# --------------------------------------------------------------------------- #
# Completion primitives (lease-token guarded)
# --------------------------------------------------------------------------- #


async def record_published_hash(
    session: AsyncSession,
    *,
    body_hash: str,
    token: str,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> HashUpdate:
    """Checkpoint the hash of the include file just written to disk.

    A mid-run checkpoint: after the publisher writes the generated bytes, the
    coordinator records their hash so a restart can resume (the file is already
    on disk with this hash). This does not advance ``applied`` or clear dirty —
    the remote has not been told yet. Token-guarded.
    """
    row = (
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET last_published_hash = :hash, updated_at = :now "
                "WHERE extension_id = :eid AND lease_token = :token "
                f"RETURNING {_RETURNING_SQL}"
            ),
            {
                "eid": extension_id,
                "token": token,
                "hash": body_hash,
                "now": _store(_now_utc(now)),
            },
        )
    ).first()
    if row is None:
        raise LeaseStaleError(extension_id, token=token)
    snapshot = _row_to_snapshot(row)
    return HashUpdate(body_hash, snapshot)


async def record_hash_noop(
    session: AsyncSession,
    *,
    target_revision: int,
    token: str,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> AppliedUpdate:
    """Advance ``applied`` through ``R`` when the remote already has the bytes.

    The generated include hash equals ``last_remote_hash`` (see
    :func:`should_skip_remote_post`), so Paisa already holds that exact file and
    a ``/api/sync`` POST would be a redundant reload. Advance ``applied`` to
    ``min(R, desired)`` and clear the dirty window + ``force_reload`` iff the
    run caught up (``applied >= desired``). No remote call is made, so
    ``last_remote_attempt_at`` is untouched — the six-hour periodic-reload
    timer keeps ticking. Token-guarded.
    """
    return await _apply_caught_up(
        session,
        target_revision=target_revision,
        token=token,
        extension_id=extension_id,
        now=now,
        set_remote_hash=False,
    )


async def record_guard_caught_up(
    session: AsyncSession,
    *,
    target_revision: int,
    token: str,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> AppliedUpdate:
    """Consume the current revision when a preflight guard blocks the run.

    A guard outcome (mode not ``project``, ``connect``-only, or not configured —
    e.g. no accounts selected) means there is nothing to reconcile *until the
    operator changes config*. That config change re-bumps ``desired_revision``,
    so it is safe — and necessary — to advance ``applied`` to the captured
    target here. Without it the row stays ``desired > applied`` (perpetually
    dirty); because the lease and audit commits each fire the ``after_commit``
    coordinator wake, the loop would re-tick immediately and rewrite a SKIPPED
    audit row every cycle. No remote call was made, so remote/hash fields are
    untouched — identical to a hash-noop completion but for the guard case.
    Token-guarded.
    """
    return await _apply_caught_up(
        session,
        target_revision=target_revision,
        token=token,
        extension_id=extension_id,
        now=now,
        set_remote_hash=False,
    )


async def record_accepted_post(
    session: AsyncSession,
    *,
    target_revision: int,
    remote_hash: str,
    token: str,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> AppliedUpdate:
    """Advance ``applied`` through ``R`` after the remote accepted the POST.

    Paisa's ``/api/sync`` accepted the journal (HTTP success), so the remote
    now reflects the captured revision. Advance ``applied`` to
    ``min(R, desired)``, record ``last_remote_hash`` and
    ``last_remote_attempt_at``, and clear the retry backoff (the network path
    succeeded). Clear the dirty window + ``force_reload`` iff caught up. The
    *quality* of what landed is diagnosed separately via
    :func:`record_diagnosis`. Token-guarded.
    """
    return await _apply_caught_up(
        session,
        target_revision=target_revision,
        token=token,
        extension_id=extension_id,
        now=now,
        set_remote_hash=True,
        remote_hash=remote_hash,
    )


async def _apply_caught_up(
    session: AsyncSession,
    *,
    target_revision: int,
    token: str,
    extension_id: str,
    now: datetime.datetime | None,
    set_remote_hash: bool,
    remote_hash: str | None = None,
) -> AppliedUpdate:
    """Shared conditional UPDATE for accepted-POST and hash-noop completion.

    ``applied_revision`` is clamped to ``min(target, desired_revision)`` (the
    CHECK constraint keeps ``applied <= desired``); the dirty window +
    ``force_reload`` are cleared iff ``target >= desired_revision`` (caught up).
    A concurrent bump past ``target`` leaves them set, so the row stays dirty
    and the coordinator reconciles again. Re-evaluates ``desired_revision`` at
    UPDATE time, so it sees a bump that committed during the run.
    """
    stored_now = _store(_now_utc(now))
    if set_remote_hash:
        set_clause = (
            "last_remote_hash = :remote_hash, "
            "last_remote_attempt_at = :now, "
            "failure_count = 0, "
            "next_attempt_at = NULL, "
        )
        params: dict[str, object] = {
            "eid": extension_id,
            "token": token,
            "target": target_revision,
            "remote_hash": remote_hash,
            "now": stored_now,
        }
    else:
        set_clause = ""
        params = {
            "eid": extension_id,
            "token": token,
            "target": target_revision,
            "now": stored_now,
        }
    row = (
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET applied_revision = "
                "    CASE WHEN :target < desired_revision THEN :target "
                "         ELSE desired_revision END, " + set_clause + "force_reload = "
                "    CASE WHEN :target >= desired_revision THEN 0 "
                "         ELSE force_reload END, "
                "first_dirty_at = "
                "    CASE WHEN :target >= desired_revision THEN NULL "
                "         ELSE first_dirty_at END, "
                "last_dirty_at = "
                "    CASE WHEN :target >= desired_revision THEN NULL "
                "         ELSE last_dirty_at END, "
                "updated_at = :now "
                "WHERE extension_id = :eid AND lease_token = :token "
                f"RETURNING {_RETURNING_SQL}"
            ),
            params,
        )
    ).first()
    if row is None:
        raise LeaseStaleError(extension_id, token=token)
    snapshot = _row_to_snapshot(row)
    caught_up = snapshot.applied_revision >= snapshot.desired_revision
    return AppliedUpdate(
        target_revision=target_revision,
        applied_revision=snapshot.applied_revision,
        desired_revision=snapshot.desired_revision,
        caught_up=caught_up,
        cleared_dirty=caught_up,
        remote_hash=snapshot.last_remote_hash,
        snapshot=snapshot,
    )


async def record_diagnosis(
    session: AsyncSession,
    *,
    state: str,
    token: str,
    healthy_hash: str | None = None,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
    backoff_minutes: tuple[int, ...] = BACKOFF_MINUTES,
) -> DiagnosisUpdate:
    """Record a post-POST diagnosis outcome (token-guarded).

    * ``healthy`` / ``accepted`` — the remote passed the diagnosis (with or
      without accepted contra-expense postings): set ``last_healthy_hash`` (if
      provided) and clear the retry backoff (the run is known-good).
    * ``fatal`` — at least one danger stayed fatal: set ``diagnosis_state`` and
      arm the retry backoff (``failure_count + 1``, ``next_attempt_at`` raised).
      ``applied`` / dirty / ``force_reload`` are *not* touched here — the
      accepted POST already settled delivery for ``R``; the diagnosis only
      records quality, so a fatal result does not re-dirty or re-force (no
      reload loop).
    * ``unknown`` — diagnosis did not run or could not be classified: record
      the state and leave failure/backoff/healthy fields unchanged.
    """
    if state not in _DIAGNOSIS_STATES:
        raise ValueError(f"unknown diagnosis state: {state!r}")
    now_utc = _now_utc(now)
    stored_now = _store(now_utc)

    if state == DIAGNOSIS_FATAL:
        pre = await _fetch_guarded(session, extension_id, token)
        new_failure = pre.failure_count + 1
        delay = backoff_delta(new_failure, backoff_minutes)
        next_at = now_utc + delay
        row = (
            await session.execute(
                text(
                    "UPDATE extension_sync_state "
                    "SET diagnosis_state = :state, "
                    "    failure_count = :fc, next_attempt_at = :next, "
                    "    updated_at = :now "
                    "WHERE extension_id = :eid AND lease_token = :token "
                    f"RETURNING {_RETURNING_SQL}"
                ),
                {
                    "eid": extension_id,
                    "token": token,
                    "state": state,
                    "fc": new_failure,
                    "next": _store(next_at),
                    "now": stored_now,
                },
            )
        ).first()
        if row is None:
            raise LeaseStaleError(extension_id, token=token)
        snapshot = _row_to_snapshot(row)
        return DiagnosisUpdate(
            state=state,
            last_healthy_hash=snapshot.last_healthy_hash,
            failure_count=snapshot.failure_count,
            next_attempt_at=to_utc(snapshot.next_attempt_at),
            snapshot=snapshot,
        )

    if state in (DIAGNOSIS_HEALTHY, DIAGNOSIS_ACCEPTED):
        row = (
            await session.execute(
                text(
                    "UPDATE extension_sync_state "
                    "SET diagnosis_state = :state, "
                    "    last_healthy_hash = COALESCE(:hh, last_healthy_hash), "
                    "    failure_count = 0, next_attempt_at = NULL, "
                    "    updated_at = :now "
                    "WHERE extension_id = :eid AND lease_token = :token "
                    f"RETURNING {_RETURNING_SQL}"
                ),
                {
                    "eid": extension_id,
                    "token": token,
                    "state": state,
                    "hh": healthy_hash,
                    "now": stored_now,
                },
            )
        ).first()
    else:  # DIAGNOSIS_UNKNOWN
        row = (
            await session.execute(
                text(
                    "UPDATE extension_sync_state "
                    "SET diagnosis_state = :state, updated_at = :now "
                    "WHERE extension_id = :eid AND lease_token = :token "
                    f"RETURNING {_RETURNING_SQL}"
                ),
                {
                    "eid": extension_id,
                    "token": token,
                    "state": state,
                    "now": stored_now,
                },
            )
        ).first()
    if row is None:
        raise LeaseStaleError(extension_id, token=token)
    snapshot = _row_to_snapshot(row)
    return DiagnosisUpdate(
        state=state,
        last_healthy_hash=snapshot.last_healthy_hash,
        failure_count=snapshot.failure_count,
        next_attempt_at=to_utc(snapshot.next_attempt_at),
        snapshot=snapshot,
    )


async def record_pre_post_failure(
    session: AsyncSession,
    *,
    token: str,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
    backoff_minutes: tuple[int, ...] = BACKOFF_MINUTES,
) -> FailureUpdate:
    """Record a remote failure that happened before a successful POST.

    Probe unreachable, generate error, network failure — the reconcile did not
    deliver ``R``, so ``applied`` / dirty / ``force_reload`` are left untouched
    (the row stays due). Increments ``failure_count``, raises
    ``next_attempt_at`` by the deterministic backoff, and stamps
    ``last_remote_attempt_at``. Token-guarded. ``backoff_minutes`` is injectable
    for deterministic tests.
    """
    now_utc = _now_utc(now)
    stored_now = _store(now_utc)
    pre = await _fetch_guarded(session, extension_id, token)
    new_failure = pre.failure_count + 1
    delay = backoff_delta(new_failure, backoff_minutes)
    next_at = now_utc + delay
    row = (
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET failure_count = :fc, next_attempt_at = :next, "
                "    last_remote_attempt_at = :now, updated_at = :now "
                "WHERE extension_id = :eid AND lease_token = :token "
                f"RETURNING {_RETURNING_SQL}"
            ),
            {
                "eid": extension_id,
                "token": token,
                "fc": new_failure,
                "next": _store(next_at),
                "now": stored_now,
            },
        )
    ).first()
    if row is None:
        raise LeaseStaleError(extension_id, token=token)
    snapshot = _row_to_snapshot(row)
    return FailureUpdate(
        failure_count=snapshot.failure_count,
        next_attempt_at=to_utc(snapshot.next_attempt_at) or next_at,
        backoff_seconds=int(delay.total_seconds()),
        last_remote_attempt_at=to_utc(snapshot.last_remote_attempt_at) or now_utc,
        snapshot=snapshot,
    )


# --------------------------------------------------------------------------- #
# Operator / admin primitives (no lease token)
# --------------------------------------------------------------------------- #


async def set_force_reload(
    session: AsyncSession,
    *,
    enabled: bool = True,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> ForceReloadUpdate:
    """Toggle the one-shot force-reload flag (operator-facing).

    Setting it makes the next enabled coordinator reconcile immediately
    regardless of drift (subject to the backoff gate). Clearing it cancels a
    pending force. Not token-guarded: this is an operator/admin action that may
    run with no active reconcile.
    """
    stored_now = _store(_now_utc(now))
    row = (
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET force_reload = :flag, updated_at = :now "
                "WHERE extension_id = :eid "
                f"RETURNING {_RETURNING_SQL}"
            ),
            {"eid": extension_id, "flag": 1 if enabled else 0, "now": stored_now},
        )
    ).first()
    if row is None:
        row = (
            await session.execute(
                text(
                    "INSERT INTO extension_sync_state "
                    "(extension_id, desired_revision, applied_revision, "
                    " first_dirty_at, last_dirty_at, force_reload, failure_count, "
                    " created_at, updated_at) "
                    "VALUES (:eid, 1, 0, :now, :now, :flag, 0, :now, :now) "
                    "ON CONFLICT(extension_id) DO UPDATE "
                    "SET force_reload = :flag, updated_at = :now "
                    f"RETURNING {_RETURNING_SQL}"
                ),
                {
                    "eid": extension_id,
                    "flag": 1 if enabled else 0,
                    "now": stored_now,
                },
            )
        ).first()
    assert row is not None
    snapshot = _row_to_snapshot(row)
    return ForceReloadUpdate(snapshot.force_reload, snapshot)


async def reset_sync_state(
    session: AsyncSession,
    *,
    extension_id: str = EXTENSION_PAISA,
    now: datetime.datetime | None = None,
) -> ResetUpdate:
    """Reset the row to the dirty seed state (admin recovery).

    Clears all hashes / backoff / diagnosis / lease and rewinds
    ``desired_revision=1``, ``applied_revision=0``, dirty, ``force_reload`` so
    the next enabled coordinator reconciles from scratch. Not token-guarded.
    """
    stored_now = _store(_now_utc(now))
    row = (
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET desired_revision = 1, applied_revision = 0, "
                "    first_dirty_at = :now, last_dirty_at = :now, "
                "    last_published_hash = NULL, last_remote_hash = NULL, "
                "    last_healthy_hash = NULL, last_remote_attempt_at = NULL, "
                "    next_attempt_at = NULL, failure_count = 0, "
                "    diagnosis_state = NULL, force_reload = 1, "
                "    lease_owner = NULL, lease_token = NULL, "
                "    lease_expires_at = NULL, updated_at = :now "
                "WHERE extension_id = :eid "
                f"RETURNING {_RETURNING_SQL}"
            ),
            {"eid": extension_id, "now": stored_now},
        )
    ).first()
    if row is None:
        # No row yet — ensure seeds it dirty with force_reload, then read back.
        await ensure_sync_state(session, extension_id=extension_id, now=now)
        snapshot = await read_sync_state(session, extension_id=extension_id)
        assert snapshot is not None
        return ResetUpdate(snapshot)
    return ResetUpdate(_row_to_snapshot(row))


__all__ = [
    "BACKOFF_MINUTES",
    "DEFAULT_FORCE_RELOAD_INTERVAL",
    "DEFAULT_LEASE_TTL_SECONDS",
    "DEFAULT_MAX_DIRTY_LATENCY",
    "DEFAULT_QUIET_DEBOUNCE",
    "DIAGNOSIS_ACCEPTED",
    "DIAGNOSIS_FATAL",
    "DIAGNOSIS_HEALTHY",
    "DIAGNOSIS_UNKNOWN",
    "EXTENSION_PAISA",
    "AppliedUpdate",
    "DiagnosisUpdate",
    "FailureUpdate",
    "ForceReloadUpdate",
    "HashUpdate",
    "LeaseClaim",
    "LeaseHeartbeat",
    "LeaseRelease",
    "LeaseStaleError",
    "ReconcileEligibility",
    "ReconcileTarget",
    "ResetUpdate",
    "SyncStateSnapshot",
    "backoff_delta",
    "capture_target",
    "claim_lease",
    "ensure_sync_state",
    "heartbeat_lease",
    "new_lease_token",
    "read_sync_state",
    "record_accepted_post",
    "record_diagnosis",
    "record_hash_noop",
    "record_pre_post_failure",
    "record_published_hash",
    "reconcile_eligibility",
    "release_lease",
    "reset_sync_state",
    "set_force_reload",
    "should_skip_remote_post",
    "to_utc",
]
