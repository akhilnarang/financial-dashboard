"""Database primitive tests for the Paisa sync-state service layer.

Exercises :mod:`financial_dashboard.services.paisa.sync_state` — the typed
read/mutation primitives over the singleton ``extension_sync_state`` row —
across every behavior the coordinator runtime will rely on:

* read / ensure (idempotent dirty seed, None on missing);
* tz-aware UTC normalization of naive SQLite TEXT reads;
* dirty / debounce eligibility (force_reload, periodic reload, quiet debounce,
  max dirty latency, backoff gate);
* atomic singleton lease claim — concurrent claims, expired reclaim, stale
  fencing token rejection;
* heartbeat / release;
* captured-target reconcile completion: accepted POST and hash-noop both clamp
  ``applied`` to ``min(R, desired)`` and preserve dirty when a concurrent bump
  advanced ``desired`` past ``R`` during the run (the "commit during run" case);
* remote-vs-local hash comparison and the hash-noop applied path;
* accepted POST then fatal diagnosis does **not** re-dirty or re-force → no
  reload loop;
* pre-POST failure backoff (deterministic 1/2/5/10/15-minute schedule, capped);
* operator force-reload toggle and admin reset.

All primitives are caller-owns-commit: no function here commits, and these
tests commit explicitly at each transaction boundary that the runtime would.
Multi-transaction scenarios (concurrent claim, commit-during-run) use a WAL
file DB so two connections observe each other's committed writes.
"""

import asyncio
import datetime

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db.models import Base
from financial_dashboard.services.paisa.sync_state import (
    BACKOFF_MINUTES,
    DEFAULT_LEASE_TTL_SECONDS,
    DIAGNOSIS_FATAL,
    DIAGNOSIS_HEALTHY,
    DIAGNOSIS_UNKNOWN,
    LeaseStaleError,
    backoff_delta,
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
    reset_sync_state,
    set_force_reload,
    should_skip_remote_post,
    to_utc,
)

pytestmark = pytest.mark.anyio

_TZ = datetime.UTC
T0 = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_TZ)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def engine(tmp_path):
    """A WAL file DB with the schema created. WAL + busy_timeout lets two
    connections observe each other's committed writes (concurrent claim /
    commit-during-run) without ``database is locked`` errors."""
    db_path = tmp_path / "sync_state.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    @event.listens_for(engine.sync_engine, "connect")
    def _pragmas(dbapi_connection, _record):
        cur = dbapi_connection.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=10000")
        finally:
            cur.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


def maker(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# --------------------------------------------------------------------------- #
# Datetime normalization
# --------------------------------------------------------------------------- #


def test_to_utc_none_passes_through():
    assert to_utc(None) is None


def test_to_utc_parses_sqlite_text_and_assumes_utc():
    parsed = to_utc("2026-01-01 12:00:00")
    assert parsed == datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ)
    iso = to_utc("2026-01-01T12:00:00+00:00")
    assert iso == datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ)


def test_to_utc_naive_assumed_utc_aware_converted():
    naive = datetime.datetime(2026, 1, 1, 12, 0)
    assert to_utc(naive).tzinfo is not None
    aware_utc = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ)
    assert to_utc(aware_utc) == aware_utc


# --------------------------------------------------------------------------- #
# read / ensure
# --------------------------------------------------------------------------- #


async def test_read_returns_none_when_row_missing(engine):
    async with maker(engine)() as session:
        assert await read_sync_state(session) is None


async def test_ensure_seeds_dirty_singleton_with_force_reload(engine):
    async with maker(engine)() as session:
        snap = await ensure_sync_state(session, now=T0)
        await session.commit()
    assert snap.extension_id == "paisa"
    assert snap.desired_revision == 1
    assert snap.applied_revision == 0
    assert snap.force_reload is True
    assert snap.failure_count == 0
    assert snap.first_dirty_at == T0
    assert snap.last_dirty_at == T0
    # All datetimes normalized to tz-aware UTC.
    assert snap.first_dirty_at.tzinfo is not None


async def test_ensure_is_idempotent_preserves_coordinator_row(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        # Coordinator reconciles successfully.
        claim = await claim_lease(session, owner="c1", token="T", now=T0)
        assert claim.claimed
        await record_accepted_post(
            session, target_revision=1, remote_hash="H", token="T", now=T0
        )
        await session.commit()

    async with maker(engine)() as session:
        # Re-ensure must NOT overwrite the reconciled row.
        snap = await ensure_sync_state(session, now=T0 + datetime.timedelta(hours=1))
        await session.commit()
    assert snap.applied_revision == 1
    assert snap.last_remote_hash == "H"
    assert snap.force_reload is False  # accepted_post cleared it
    assert snap.first_dirty_at is None  # accepted_post cleared it


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #


async def test_eligibility_force_reload_then_caught_up(engine):
    async with maker(engine)() as session:
        snap = await ensure_sync_state(session, now=T0)
        # Fresh seed is dirty + force_reload → due immediately via force_reload.
        elig = reconcile_eligibility(snap, now=T0)
        assert elig.due
        assert elig.reason == "force_reload"

        # Reconcile fully.
        claim = await claim_lease(session, owner="c", token="T", now=T0)
        assert claim.claimed
        await record_accepted_post(
            session, target_revision=1, remote_hash="H", token="T", now=T0
        )
        await session.commit()
        snap2 = claim.snapshot
        # Force a fresh read to confirm.
        snap2 = await read_sync_state(session)

    elig_caught = reconcile_eligibility(snap2, now=T0 + datetime.timedelta(seconds=10))
    assert not elig_caught.due
    assert elig_caught.reason == "caught_up"


async def test_eligibility_periodic_reload_after_interval(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        # Simulate a healthy, caught-up row whose last remote attempt is old.
        await session.execute(
            text(
                "UPDATE extension_sync_state SET force_reload = 0, "
                "applied_revision = desired_revision, "
                "first_dirty_at = NULL, last_dirty_at = NULL, "
                "last_remote_attempt_at = :lra, failure_count = 0, "
                "next_attempt_at = NULL WHERE extension_id = 'paisa'"
            ),
            {"lra": T0 - datetime.timedelta(hours=7)},
        )
        await session.commit()
        snap = await read_sync_state(session)
    elig = reconcile_eligibility(snap, now=T0)
    assert elig.due
    assert elig.reason == "periodic_reload"


async def test_eligibility_debounce_then_max_latency(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await session.execute(
            text(
                "UPDATE extension_sync_state SET force_reload = 0, "
                "desired_revision = 5, applied_revision = 3, "
                "first_dirty_at = :first, last_dirty_at = :last, "
                "next_attempt_at = NULL WHERE extension_id = 'paisa'"
            ),
            {
                "first": T0,
                "last": T0 + datetime.timedelta(seconds=2),
            },
        )
        await session.commit()
        snap = await read_sync_state(session)

    # 3s in: still within the 5s quiet debounce since last_dirty (2s+3s<... no):
    # last_dirty = T0+2s, now = T0+3s → elapsed 1s < 5s → debounced.
    elig = reconcile_eligibility(snap, now=T0 + datetime.timedelta(seconds=3))
    assert not elig.due
    assert elig.reason == "debounce"
    assert elig.eligible_at == T0 + datetime.timedelta(seconds=7)  # last+5s

    # Past quiet debounce, under max latency → due via "dirty".
    elig2 = reconcile_eligibility(snap, now=T0 + datetime.timedelta(seconds=8))
    assert elig2.due
    assert elig2.reason == "dirty"

    # Past max dirty latency (30s from first_dirty) → due via "max_latency".
    elig3 = reconcile_eligibility(snap, now=T0 + datetime.timedelta(seconds=31))
    assert elig3.due
    assert elig3.reason == "max_latency"


async def test_eligibility_backoff_gate_blocks_force_reload(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await session.execute(
            text(
                "UPDATE extension_sync_state SET force_reload = 1, "
                "next_attempt_at = :na WHERE extension_id = 'paisa'"
            ),
            {"na": T0 + datetime.timedelta(minutes=1)},
        )
        await session.commit()
        snap = await read_sync_state(session)
    # force_reload set but a retry is scheduled in the future → backoff wins.
    elig = reconcile_eligibility(snap, now=T0)
    assert not elig.due
    assert elig.reason == "backoff"
    assert elig.eligible_at == T0 + datetime.timedelta(minutes=1)


# --------------------------------------------------------------------------- #
# Lease: concurrent claim, expired reclaim, stale-token fencing
# --------------------------------------------------------------------------- #


async def _claim_and_commit(engine, owner, token, ttl, now):
    async with maker(engine)() as session:
        claim = await claim_lease(
            session, owner=owner, token=token, ttl_seconds=ttl, now=now
        )
        await session.commit()
        return claim


async def test_concurrent_claim_only_one_wins(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await session.commit()

    claims = await asyncio.gather(
        _claim_and_commit(engine, "coordinator-a", "TA", DEFAULT_LEASE_TTL_SECONDS, T0),
        _claim_and_commit(engine, "coordinator-b", "TB", DEFAULT_LEASE_TTL_SECONDS, T0),
    )
    winners = [c for c in claims if c.claimed]
    losers = [c for c in claims if not c.claimed]
    assert len(winners) == 1
    assert len(losers) == 1
    assert winners[0].reason == "claimed"
    assert winners[0].expires_at == T0 + datetime.timedelta(
        seconds=DEFAULT_LEASE_TTL_SECONDS
    )
    assert losers[0].reason == "held"
    # The loser's reported expiry reflects the winner's lease.
    assert losers[0].expires_at == winners[0].expires_at


async def test_claim_missing_row_reports_no_row(engine):
    async with maker(engine)() as session:
        claim = await claim_lease(session, owner="c", token="T", now=T0)
    assert not claim.claimed
    assert claim.reason == "no_row"
    assert claim.snapshot is None


async def test_expired_lease_is_reclaimable(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await session.commit()
    first = await _claim_and_commit(engine, "a", "TA", ttl=60, now=T0)
    assert first.claimed
    # Past the TTL → a different owner may reclaim.
    second = await _claim_and_commit(
        engine, "b", "TB", ttl=60, now=T0 + datetime.timedelta(seconds=61)
    )
    assert second.claimed
    assert second.token == "TB"
    assert second.owner == "b"


async def test_stale_token_fenced_on_completion(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await session.commit()
    await _claim_and_commit(engine, "a", "TA", ttl=60, now=T0)
    # Lease expires, a second coordinator reclaims with a fresh token.
    await _claim_and_commit(
        engine, "b", "TB", ttl=60, now=T0 + datetime.timedelta(seconds=61)
    )

    async with maker(engine)() as session:
        with pytest.raises(LeaseStaleError) as exc_info:
            await record_accepted_post(
                session, target_revision=1, remote_hash="H", token="TA", now=T0
            )
        assert exc_info.value.extension_id == "paisa"
        assert exc_info.value.token == "TA"


async def test_stale_token_fenced_on_every_completion_primitive(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await session.commit()
    await _claim_and_commit(engine, "a", "TA", ttl=60, now=T0)
    await _claim_and_commit(
        engine, "b", "TB", ttl=60, now=T0 + datetime.timedelta(seconds=61)
    )

    async with maker(engine)() as session:
        for body in (
            lambda: record_published_hash(session, body_hash="H", token="TA"),
            lambda: record_hash_noop(session, target_revision=1, token="TA"),
            lambda: record_diagnosis(session, state=DIAGNOSIS_HEALTHY, token="TA"),
            lambda: record_pre_post_failure(session, token="TA"),
        ):
            with pytest.raises(LeaseStaleError):
                await body()


# --------------------------------------------------------------------------- #
# Heartbeat / release
# --------------------------------------------------------------------------- #


async def test_heartbeat_extends_and_release_clears(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        claim = await claim_lease(session, owner="c", token="T", ttl_seconds=60, now=T0)
        assert claim.claimed

        hb = await heartbeat_lease(
            session, token="T", ttl_seconds=60, now=T0 + datetime.timedelta(seconds=30)
        )
        assert hb.renewed
        assert hb.expires_at == T0 + datetime.timedelta(seconds=90)
        await session.commit()

    async with maker(engine)() as session:
        snap = await read_sync_state(session)
        assert snap.lease_expires_at == T0 + datetime.timedelta(seconds=90)
        rel = await release_lease(session, token="T")
        assert rel.released
        await session.commit()
        snap2 = await read_sync_state(session)
    assert snap2.lease_owner is None
    assert snap2.lease_token is None
    assert snap2.lease_expires_at is None


async def test_heartbeat_and_release_reject_stale_token(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await claim_lease(session, owner="c", token="T", ttl_seconds=60, now=T0)
        hb = await heartbeat_lease(session, token="WRONG", ttl_seconds=60, now=T0)
        assert not hb.renewed
        assert hb.reason == "stale_token"
        rel = await release_lease(session, token="WRONG")
        assert not rel.released
        assert rel.reason == "stale_token"


# --------------------------------------------------------------------------- #
# Target capture
# --------------------------------------------------------------------------- #


async def test_capture_target_reads_current_desired(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await session.execute(
            text(
                "UPDATE extension_sync_state SET desired_revision = 7 "
                "WHERE extension_id = 'paisa'"
            )
        )
        await session.commit()
        target = await capture_target(session)
    assert target.target_revision == 7
    assert target.snapshot.desired_revision == 7


async def test_capture_target_ensures_row_when_missing(engine):
    async with maker(engine)() as session:
        target = await capture_target(session)
    assert target.target_revision == 1  # ensure seed default
    assert target.snapshot.applied_revision == 0


# --------------------------------------------------------------------------- #
# Commit during run: dirty preserved when desired advances past R
# --------------------------------------------------------------------------- #


async def test_commit_during_run_preserves_dirty(engine):
    # 1. Coordinator claims and captures R before the bump.
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        claim = await claim_lease(
            session, owner="c", token="T", ttl_seconds=300, now=T0
        )
        assert claim.claimed
        target = await capture_target(session)
        assert target.target_revision == 1
        await session.commit()

    # 2. A core write commits during the run, bumping desired_revision past R.
    async with maker(engine)() as session:
        await session.execute(
            text(
                "UPDATE extension_sync_state SET desired_revision = 2 "
                "WHERE extension_id = 'paisa'"
            )
        )
        await session.commit()

    # 3. The run completes against R=1 in a fresh transaction; the UPDATE
    #    re-evaluates desired_revision (now 2) and clamps/preserves.
    async with maker(engine)() as session:
        update = await record_accepted_post(
            session, target_revision=1, remote_hash="H1", token="T", now=T0
        )
        await session.commit()

    assert update.target_revision == 1
    assert update.applied_revision == 1
    assert update.desired_revision == 2
    assert not update.caught_up
    assert not update.cleared_dirty

    async with maker(engine)() as session:
        snap = await read_sync_state(session)
    assert snap.desired_revision == 2
    assert snap.applied_revision == 1
    # Dirty window + force_reload preserved: still due for revision 2.
    assert snap.force_reload is True
    assert snap.first_dirty_at is not None
    assert snap.last_dirty_at is not None
    elig = reconcile_eligibility(snap, now=T0 + datetime.timedelta(seconds=31))
    assert elig.due  # reconciles again for the new revision


# --------------------------------------------------------------------------- #
# Remote vs local hash + hash-noop applied path
# --------------------------------------------------------------------------- #


async def test_should_skip_remote_post_logic():
    from financial_dashboard.services.paisa.sync_state import SyncStateSnapshot

    base = SyncStateSnapshot(
        extension_id="paisa",
        desired_revision=1,
        applied_revision=1,
        first_dirty_at=None,
        last_dirty_at=None,
        last_published_hash=None,
        last_remote_hash="Hr",
        last_healthy_hash=None,
        last_remote_attempt_at=None,
        next_attempt_at=None,
        failure_count=0,
        diagnosis_state=None,
        force_reload=False,
        lease_owner=None,
        lease_token=None,
        lease_expires_at=None,
    )
    assert should_skip_remote_post("Hr", base) is True
    assert should_skip_remote_post("Hl", base) is False
    # Never-attempted remote (None) and None published hash never skip.
    assert should_skip_remote_post("Hr", base._replace(last_remote_hash=None)) is False
    assert should_skip_remote_post(None, base) is False


async def test_hash_noop_advances_applied_without_remote_call(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        claim = await claim_lease(
            session, owner="c", token="T", ttl_seconds=300, now=T0
        )
        assert claim.claimed
        # Simulate a prior successful POST of 'remote' at applied=1, then a
        # core write that dirties desired to 2 (unchanged bytes by assumption).
        await session.execute(
            text(
                "UPDATE extension_sync_state SET last_remote_hash = 'remote', "
                "applied_revision = 1, desired_revision = 2, force_reload = 0, "
                "first_dirty_at = :d, last_dirty_at = :d, "
                "last_remote_attempt_at = :lra WHERE extension_id = 'paisa'"
            ),
            {"d": T0, "lra": T0 - datetime.timedelta(minutes=1)},
        )
        await session.commit()
        snap_before = await read_sync_state(session)

    # Different generated hash → must POST.
    assert should_skip_remote_post("local", snap_before) is False
    # Same generated hash → skip the POST, advance applied via hash-noop.
    assert should_skip_remote_post("remote", snap_before) is True

    async with maker(engine)() as session:
        pub = await record_published_hash(
            session, body_hash="remote", token="T", now=T0
        )
        assert pub.last_published_hash == "remote"
        update = await record_hash_noop(session, target_revision=2, token="T", now=T0)
        await session.commit()

    assert update.applied_revision == 2
    assert update.caught_up
    assert update.cleared_dirty
    assert update.remote_hash == "remote"  # unchanged — no POST happened

    async with maker(engine)() as session:
        snap = await read_sync_state(session)
    # last_remote_attempt_at untouched: no remote call was made.
    assert snap.last_remote_attempt_at == T0 - datetime.timedelta(minutes=1)
    assert snap.last_remote_hash == "remote"
    assert snap.first_dirty_at is None
    assert snap.force_reload is False


# --------------------------------------------------------------------------- #
# Accepted POST then fatal diagnosis: no reload loop
# --------------------------------------------------------------------------- #


async def test_accepted_post_then_fatal_diagnosis_no_reload_loop(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await claim_lease(session, owner="c", token="T", ttl_seconds=300, now=T0)
        target = await capture_target(session)
        assert target.target_revision == 1

        accepted = await record_accepted_post(
            session, target_revision=1, remote_hash="H1", token="T", now=T0
        )
        assert accepted.caught_up
        assert accepted.cleared_dirty
        assert accepted.applied_revision == 1

        fatal = await record_diagnosis(
            session, state=DIAGNOSIS_FATAL, token="T", now=T0
        )
        await session.commit()

    assert fatal.state == DIAGNOSIS_FATAL
    assert fatal.failure_count == 1
    assert fatal.next_attempt_at == T0 + datetime.timedelta(minutes=1)
    # Fatal does not touch delivery-settled fields.
    assert fatal.snapshot.applied_revision == 1
    assert fatal.snapshot.desired_revision == 1
    assert fatal.snapshot.force_reload is False
    assert fatal.snapshot.first_dirty_at is None

    async with maker(engine)() as session:
        snap = await read_sync_state(session)

    # Even past the backoff window, the row is caught up → not due. No loop.
    elig_past = reconcile_eligibility(snap, now=T0 + datetime.timedelta(minutes=5))
    assert not elig_past.due
    assert elig_past.reason == "caught_up"
    # Within the backoff window it's also gated.
    elig_within = reconcile_eligibility(snap, now=T0 + datetime.timedelta(seconds=30))
    assert not elig_within.due


async def test_healthy_diagnosis_clears_backoff_and_records_healthy_hash(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await claim_lease(session, owner="c", token="T", ttl_seconds=300, now=T0)
        await record_accepted_post(
            session, target_revision=1, remote_hash="H1", token="T", now=T0
        )
        # Arm a backoff so clearing is observable.
        await session.execute(
            text(
                "UPDATE extension_sync_state SET failure_count = 3, "
                "next_attempt_at = :na WHERE extension_id = 'paisa'"
            ),
            {"na": T0 + datetime.timedelta(minutes=5)},
        )
        diag = await record_diagnosis(
            session,
            state=DIAGNOSIS_HEALTHY,
            healthy_hash="H1",
            token="T",
            now=T0,
        )
        await session.commit()
    assert diag.state == DIAGNOSIS_HEALTHY
    assert diag.last_healthy_hash == "H1"
    assert diag.failure_count == 0
    assert diag.next_attempt_at is None


async def test_unknown_diagnosis_records_state_only(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await claim_lease(session, owner="c", token="T", ttl_seconds=300, now=T0)
        # Arm backoff + a known healthy hash; unknown must leave them alone.
        await session.execute(
            text(
                "UPDATE extension_sync_state SET failure_count = 2, "
                "next_attempt_at = :na, last_healthy_hash = 'Hh' "
                "WHERE extension_id = 'paisa'"
            ),
            {"na": T0 + datetime.timedelta(minutes=2)},
        )
        diag = await record_diagnosis(
            session, state=DIAGNOSIS_UNKNOWN, token="T", now=T0
        )
        await session.commit()
    assert diag.state == DIAGNOSIS_UNKNOWN
    assert diag.failure_count == 2  # unchanged
    assert diag.next_attempt_at == T0 + datetime.timedelta(minutes=2)  # unchanged
    assert diag.last_healthy_hash == "Hh"  # unchanged


# --------------------------------------------------------------------------- #
# Force reload (operator) + reset (admin)
# --------------------------------------------------------------------------- #


async def test_set_force_reload_toggles_and_drives_eligibility(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        # Make the row fully caught up + not forced.
        await claim_lease(session, owner="c", token="T", ttl_seconds=300, now=T0)
        await record_accepted_post(
            session, target_revision=1, remote_hash="H", token="T", now=T0
        )
        await session.commit()
        snap = await read_sync_state(session)
    assert snap.force_reload is False
    assert not reconcile_eligibility(snap, now=T0).due

    async with maker(engine)() as session:
        upd = await set_force_reload(session, enabled=True, now=T0)
        await session.commit()
    assert upd.force_reload is True
    assert reconcile_eligibility(upd.snapshot, now=T0).due

    async with maker(engine)() as session:
        upd2 = await set_force_reload(session, enabled=False, now=T0)
        await session.commit()
    assert upd2.force_reload is False
    assert not reconcile_eligibility(upd2.snapshot, now=T0).due


async def test_reset_sync_state_returns_to_dirty_seed(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await claim_lease(session, owner="c", token="T", ttl_seconds=300, now=T0)
        await record_accepted_post(
            session, target_revision=1, remote_hash="H", token="T", now=T0
        )
        await session.execute(
            text(
                "UPDATE extension_sync_state SET desired_revision = 9, "
                "diagnosis_state = 'fatal', failure_count = 4 "
                "WHERE extension_id = 'paisa'"
            )
        )
        await session.commit()
        reset = await reset_sync_state(session, now=T0)
        await session.commit()
    snap = reset.snapshot
    assert snap.desired_revision == 1
    assert snap.applied_revision == 0
    assert snap.force_reload is True
    assert snap.failure_count == 0
    assert snap.diagnosis_state is None
    assert snap.last_remote_hash is None
    assert snap.lease_token is None
    assert snap.first_dirty_at is not None
    assert snap.last_dirty_at is not None


# --------------------------------------------------------------------------- #
# Backoff: deterministic 1/2/5/10/15 schedule (pure + persisted)
# --------------------------------------------------------------------------- #


def test_backoff_delta_schedule_and_cap():
    assert backoff_delta(1) == datetime.timedelta(minutes=1)
    assert backoff_delta(2) == datetime.timedelta(minutes=2)
    assert backoff_delta(3) == datetime.timedelta(minutes=5)
    assert backoff_delta(4) == datetime.timedelta(minutes=10)
    assert backoff_delta(5) == datetime.timedelta(minutes=15)
    # Capped at the last step.
    assert backoff_delta(6) == datetime.timedelta(minutes=15)
    assert backoff_delta(99) == datetime.timedelta(minutes=15)
    # Non-positive counts treated as the first step.
    assert backoff_delta(0) == datetime.timedelta(minutes=1)
    assert backoff_delta(-3) == datetime.timedelta(minutes=1)
    # Injectable schedule (deterministic testing).
    custom = (1, 3)
    assert backoff_delta(1, custom) == datetime.timedelta(minutes=1)
    assert backoff_delta(2, custom) == datetime.timedelta(minutes=3)
    assert backoff_delta(9, custom) == datetime.timedelta(minutes=3)


def test_backoff_minutes_constant_is_fixed_schedule():
    assert BACKOFF_MINUTES == (1, 2, 5, 10, 15)


async def test_record_pre_post_failure_escalates_backoff_and_keeps_dirty(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await claim_lease(session, owner="c", token="T", ttl_seconds=600, now=T0)

        expected_minutes = (1, 2, 5, 10, 15, 15)
        for i, mins in enumerate(expected_minutes, start=1):
            failure = await record_pre_post_failure(session, token="T", now=T0)
            assert failure.failure_count == i
            assert failure.next_attempt_at == T0 + datetime.timedelta(minutes=mins)
            assert failure.backoff_seconds == mins * 60
            assert failure.last_remote_attempt_at == T0
        await session.commit()

        snap = await read_sync_state(session)
    # Failures do not advance applied or clear dirty/force.
    assert snap.applied_revision == 0
    assert snap.force_reload is True
    assert snap.first_dirty_at is not None
    # And the failure arms the backoff gate.
    elig = reconcile_eligibility(snap, now=T0 + datetime.timedelta(seconds=10))
    assert not elig.due
    assert elig.reason == "backoff"
    assert elig.eligible_at == T0 + datetime.timedelta(minutes=15)


async def test_record_pre_post_failure_injectable_backoff(engine):
    async with maker(engine)() as session:
        await ensure_sync_state(session, now=T0)
        await claim_lease(session, owner="c", token="T", ttl_seconds=600, now=T0)
        failure = await record_pre_post_failure(
            session, token="T", now=T0, backoff_minutes=(7,)
        )
        assert failure.backoff_seconds == 7 * 60
        assert failure.next_attempt_at == T0 + datetime.timedelta(minutes=7)
