"""Paisa coordinator behavior tests.

Exercises :class:`financial_dashboard.services.paisa.coordinator.PaisaCoordinator`
end to end against a real WAL SQLite DB (with the ``init_db`` revision triggers)
using injectable seams (clock, sleep, client factory, orchestrator stages) so
tests are deterministic and never touch the network or sleep for real.

Coverage (one test per contract bullet):

* eligibility: disabled/connect/auto-off do no I/O; project+auto reconciles.
* debounce + max latency + remote floor coalescing (200-row one commit → one
  attempt; two commits inside debounce → one attempt; continuous dirty → max
  latency then remote floor).
* one pass: preflight before file write; generate once; hash-noop skips remote;
  publish-success/remote-failure retries the unchanged file (no old bug where
  local-unchanged suppresses retry after a remote failure); accepted POST +
  fatal diagnosis advances applied (no reload loop); periodic force reload
  calls remote even clean.
* single flight / lease: two coordinators claim → one winner; crash/expired
  lease recovery; stale fencing token rejected; commit during active sync →
  follow-up run; heartbeat spans blocked preflight/generation and is stopped on
  success, error, or cancellation.
* backoff: a pre-POST failure arms the deterministic schedule.
* manual vs automatic never overlap (shared lease).
* restart with pending state reconciles.
"""

import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.paisa.coordinator as coord_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import Transaction
from financial_dashboard.services.paisa.audit import (
    OPERATION_AUTOMATIC,
    OPERATION_GENERATE,
    OPERATION_SYNC,
    STATUS_FAILURE,
    recent_runs,
)
from financial_dashboard.services.paisa.coordinator import (
    HEARTBEAT_INTERVAL_SECONDS,
    LEASE_TTL_SECONDS,
    PaisaCoordinator,
    claim_manual_lease,
)
from financial_dashboard.services.paisa.orchestrator import (
    GenerateResult,
    PreflightReport,
    RemoteSyncReport,
)
from financial_dashboard.services.paisa.publisher import PublishResult
from financial_dashboard.services.paisa.sync_state import (
    DEFAULT_MAX_DIRTY_LATENCY,
    DEFAULT_QUIET_DEBOUNCE,
    read_sync_state,
)

pytestmark = pytest.mark.anyio

_TZ = datetime.UTC
UNUSED_PATH = "/tmp/paisa-coord-test-unused.journal"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """A WAL file DB with init_db schema + revision triggers, paisa row seeded.

    Mirrors test_paisa_sync_bulk_db's fixture: init_db seeds the singleton and
    installs the triggers; the tail cache-warming steps are stubbed.
    """
    from financial_dashboard.services import settings as settings_mod
    from financial_dashboard.services.categorization import merchant_rules

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", _noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", _noop)

    db_path = tmp_path / "paisa_coord.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    @event.listens_for(engine.sync_engine, "connect")
    def _wal(dbapi_connection, _record):
        cur = dbapi_connection.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=10000")
        finally:
            cur.close()

    try:
        await init_db(engine)
        # Start from a clean, caught-up, non-forced row so each test seeds its
        # own dirty window explicitly.
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE extension_sync_state "
                    "SET desired_revision = 1, applied_revision = 1, "
                    "    first_dirty_at = NULL, last_dirty_at = NULL, "
                    "    force_reload = 0, failure_count = 0, "
                    "    next_attempt_at = NULL, last_remote_attempt_at = NULL, "
                    "    last_published_hash = NULL, last_remote_hash = NULL, "
                    "    last_healthy_hash = NULL, diagnosis_state = NULL "
                    "WHERE extension_id = 'paisa'"
                )
            )
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def factory(db):
    return async_sessionmaker(db, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def settings_paisa(monkeypatch):
    """Project mode + auto on by default; tests override via the cache."""
    settings_mod._cache.update(
        {
            "paisa.mode": "project",
            "paisa.auto_sync_enabled": "true",
            "paisa.auto_sync_min_interval_minutes": "1",
            "paisa.notify_sync_failures": "false",
        }
    )
    # Make load_config return a minimal project config without depending on
    # every settings key.
    from financial_dashboard.services.paisa.config import PaisaProjectionConfig

    cfg = PaisaProjectionConfig(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path=UNUSED_PATH,
        selected_account_ids=(1,),
        cutover_date=datetime.date(2026, 1, 1),
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
    )
    monkeypatch.setattr(coord_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(coord_mod, "get_setting_bool", lambda key, default=False: True)


# --------------------------------------------------------------------------- #
# Fake seams
# --------------------------------------------------------------------------- #


class FakeClock:
    """A controllable tz-aware UTC clock."""

    def __init__(self, start: datetime.datetime):
        self.t = start.astimezone(_TZ)

    def __call__(self) -> datetime.datetime:
        return self.t

    def advance(self, delta: datetime.timedelta) -> None:
        self.t += delta


async def _preflight_ok(cfg, *, client=None):
    return PreflightReport(
        ok=True,
        outcome=None,
        capabilities=SimpleNamespace(ledger_cli="ledger"),
        reason=None,
    )


def _gen_result(*, ok=True, body_hash="hashbody", emitted=3, reason=None):
    report = SimpleNamespace(
        emitted_count=emitted,
        skipped=(),
        journal="...",
        entries=(),
    )
    publish = PublishResult(
        published=True,
        path=UNUSED_PATH,
        version="3",
        body_hash=body_hash,
        bytes_written=10,
    )
    return GenerateResult(
        ok=ok, report=report if ok else None, publish=publish, reason=reason
    )


def _remote_result(
    *,
    post_accepted=True,
    diagnosis_ok=True,
    outcome="synced",
    reason=None,
    diagnosis_expected=None,
    diagnosis_accepted=None,
    diagnosis_fatal=None,
):
    ok = post_accepted and diagnosis_ok is True
    return RemoteSyncReport(
        ok=ok,
        outcome=outcome
        if outcome is not None
        else ("synced" if ok else "sync_rejected"),
        post_accepted=post_accepted,
        diagnosis_ok=diagnosis_ok,
        reason=reason,
        diagnosis_expected=diagnosis_expected,
        diagnosis_accepted=diagnosis_accepted,
        diagnosis_fatal=diagnosis_fatal,
    )


class _FakeClient:
    """Minimal stand-in client whose aclose is a real coroutine method (no leak)."""

    async def aclose(self) -> None:
        return None


def _make_coordinator(
    factory,
    clock,
    *,
    preflight_fn=None,
    generate_fn=None,
    sync_remote_fn=None,
    min_interval_minutes=1,
    client_factory=None,
    heartbeat_interval=HEARTBEAT_INTERVAL_SECONDS,
    lease_ttl=LEASE_TTL_SECONDS,
):
    return PaisaCoordinator(
        session_factory=factory,
        now=clock,
        sleep=asyncio.sleep,
        preflight_fn=preflight_fn or _preflight_ok,
        generate_fn=generate_fn or _default_generate,
        sync_remote_fn=sync_remote_fn or _default_sync_remote,
        min_interval_minutes_fn=lambda: min_interval_minutes,
        client_factory=client_factory or (lambda cfg: _FakeClient()),
        poll_interval=0.01,
        heartbeat_interval=heartbeat_interval,
        lease_ttl=lease_ttl,
    )


async def _default_generate(sess, cfg):
    return _gen_result()


async def _default_sync_remote(report, cfg, *, client=None):
    return _remote_result()


async def _autowrite_count(factory) -> int:
    async with factory() as s:
        rows = await recent_runs(s, extension_id="paisa", operation=OPERATION_AUTOMATIC)
        return len(rows)


async def _dirty(factory, clock, *, n=1):
    """Commit ``n`` transactions in one outer transaction to bump desired_revision.

    The SQLite triggers stamp ``first_dirty_at``/``last_dirty_at`` with real
    wall-clock ``CURRENT_TIMESTAMP``; overwrite them to the fake clock's time so
    eligibility comparisons (which use the injected clock) are deterministic and
    decoupled from real time. Overwriting the state row fires no trigger
    (recursion guard), so desired_revision is not bumped again.
    """
    async with factory() as s:
        for i in range(n):
            s.add(
                Transaction(
                    bank="hdfc",
                    email_type="txn",
                    direction="debit",
                    amount=Decimal("1.00"),
                    reference_number=f"r-{i}-{clock().timestamp()}",
                )
            )
        await s.commit()
    ts = clock().replace(tzinfo=None).isoformat(sep=" ")
    async with factory() as s:
        await s.execute(
            text(
                "UPDATE extension_sync_state "
                "SET first_dirty_at = COALESCE(first_dirty_at, :ts), "
                "    last_dirty_at = :ts WHERE extension_id = 'paisa'"
            ),
            {"ts": ts},
        )
        await s.commit()


async def _snapshot(factory):
    async with factory() as s:
        snap = await read_sync_state(s)
        await s.rollback()
    return snap


# --------------------------------------------------------------------------- #
# Eligibility: disabled / connect / auto-off do no I/O
# --------------------------------------------------------------------------- #


async def test_disabled_mode_does_no_io(factory, monkeypatch):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    gen = AsyncMock(side_effect=AssertionError("no generate when disabled"))
    # Force disabled mode + auto on; the tick must return before generate.
    monkeypatch.setattr(
        coord_mod,
        "load_config",
        lambda: SimpleNamespace(can_project=False, mode="disabled"),
    )
    monkeypatch.setattr(coord_mod, "get_setting_bool", lambda k, d=False: True)
    c = _make_coordinator(factory, clock, generate_fn=gen)
    await c._tick()
    gen.assert_not_called()
    assert await _autowrite_count(factory) == 0


async def test_auto_off_does_no_io(factory, monkeypatch):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    gen = AsyncMock(side_effect=AssertionError("no generate when auto off"))
    monkeypatch.setattr(
        coord_mod,
        "load_config",
        lambda: SimpleNamespace(can_project=True, mode="project"),
    )

    calls = {"auto": False}

    def gsb(key, default=False):
        if key == "paisa.auto_sync_enabled":
            return calls["auto"]
        return default

    monkeypatch.setattr(coord_mod, "get_setting_bool", gsb)
    c = _make_coordinator(factory, clock, generate_fn=gen)
    await c._tick()
    gen.assert_not_called()


@pytest.mark.usefixtures("settings_paisa")
async def test_project_auto_reconciles_when_dirty(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result())
    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    gen.assert_awaited_once()
    remote.assert_awaited_once()
    snap = await _snapshot(factory)
    assert snap.applied_revision >= snap.desired_revision  # caught up
    assert snap.last_remote_hash == "hashbody"


# --------------------------------------------------------------------------- #
# Debounce + max latency + remote floor
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_200_row_one_commit_produces_one_attempt(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(
        factory, clock, n=200
    )  # one outer commit → +200 revision, one bump window
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result(emitted=200))
    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    gen.assert_awaited_once()  # one generate, one remote POST — one attempt
    remote.assert_awaited_once()


@pytest.mark.usefixtures("settings_paisa")
async def test_two_commits_inside_debounce_one_attempt(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(datetime.timedelta(seconds=1))  # inside 5s debounce
    await _dirty(factory, clock, n=1)
    # Inside the debounce window: a tick must NOT reconcile (coalescing).
    gen = AsyncMock(side_effect=AssertionError("must be debounced"))
    c = _make_coordinator(factory, clock, generate_fn=gen)
    await c._tick()
    gen.assert_not_called()
    # Once the quiet debounce elapses, a single tick reconciles both commits
    # in ONE attempt (one generate, one remote POST) — coalesced.
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen2 = AsyncMock(return_value=_gen_result())
    remote = AsyncMock(return_value=_remote_result())
    c2 = _make_coordinator(factory, clock, generate_fn=gen2, sync_remote_fn=remote)
    await c2._tick()
    gen2.assert_awaited_once()
    remote.assert_awaited_once()


@pytest.mark.usefixtures("settings_paisa")
async def test_caught_up_tick_does_not_take_writer_lock_behind_open_import(db, factory):
    """An eligibility poll stays read-only when the singleton already exists.

    A statement importer can hold SQLite's single writer lock for its outer
    transaction.  The coordinator must still read the prior committed state and
    return; an unconditional ``INSERT .. DO NOTHING`` of the existing singleton
    would wait on that writer lock and turn every harmless poll into contention.
    """
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    generated = AsyncMock(side_effect=AssertionError("uncommitted row is invisible"))
    coordinator = _make_coordinator(factory, clock, generate_fn=generated)

    async with db.connect() as writer:
        transaction = await writer.begin()
        await writer.execute(
            Transaction.__table__.insert(),
            {
                "bank": "hdfc",
                "email_type": "txn",
                "direction": "debit",
                "amount": Decimal("1.00"),
                "reference_number": "open-import-uncommitted",
            },
        )
        await asyncio.wait_for(coordinator._tick(), timeout=5)
        generated.assert_not_called()
        await transaction.rollback()


@pytest.mark.usefixtures("settings_paisa")
async def test_max_latency_overrides_debounce(factory):
    """Even with continuous commits (debounce never expires), max latency fires."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    # Advance just past max dirty latency (30s) — debounce (5s) never expired
    # because we did NOT keep dirtying, but max latency is measured from
    # first_dirty_at. 31s > 30s → due via max_latency.
    clock.advance(DEFAULT_MAX_DIRTY_LATENCY + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result())
    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    gen.assert_awaited_once()


@pytest.mark.usefixtures("settings_paisa")
async def test_remote_floor_throttles_dirty_driven_post(factory):
    """After a reload, a dirty event within the remote floor does not POST."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    # Distinct body hashes per generate so the second is NOT a hash-noop.
    hashes = iter(["h1", "h2", "h3"])

    def gen_factory():
        async def _g(sess, cfg):
            return _gen_result(body_hash=next(hashes))

        return AsyncMock(side_effect=_g)

    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(
        factory,
        clock,
        generate_fn=gen_factory(),
        sync_remote_fn=remote,
        min_interval_minutes=5,
    )
    # First reload succeeds.
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    await c._tick()
    assert remote.await_count == 1

    # A second dirty event 10s later is within the 5-minute remote floor and
    # past the debounce; the tick must NOT reconcile (floor not satisfied).
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    await c._tick()
    assert remote.await_count == 1  # no second attempt
    # Once the floor elapses, the next tick reconciles (body changed → POST).
    clock.advance(datetime.timedelta(minutes=5))
    await c._tick()
    assert remote.await_count == 2


@pytest.mark.usefixtures("settings_paisa")
async def test_continuous_dirty_avoids_starvation(factory):
    """Under continuous commits the remote floor still lets progress happen
    once it elapses (not starved by the ongoing dirty window)."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    # Distinct body hashes per generate so reloads are not treated as noops.
    hashes = iter([f"h{i}" for i in range(10)])

    def gen_factory():
        async def _g(sess, cfg):
            return _gen_result(body_hash=next(hashes))

        return AsyncMock(side_effect=_g)

    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(
        factory,
        clock,
        generate_fn=gen_factory(),
        sync_remote_fn=remote,
        min_interval_minutes=2,
    )
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    await c._tick()  # first reload; last_remote_attempt_at ≈ now
    assert remote.await_count == 1

    # Continuous commits, each 30s apart — within the 2-minute remote floor and
    # past the 5s debounce. Each tick must skip (floor not satisfied).
    for _ in range(3):
        await _dirty(factory, clock, n=1)
        clock.advance(datetime.timedelta(seconds=30))
        await c._tick()
    assert remote.await_count == 1
    # Floor (2 min) elapses → next tick reloads despite the ongoing dirty window.
    clock.advance(datetime.timedelta(seconds=120))
    await c._tick()
    assert remote.await_count == 2


# --------------------------------------------------------------------------- #
# Hash no-op
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_hash_noop_skips_remote_and_advances_applied(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    # Seed last_remote_hash to match the body the fake generate produces.
    async with factory() as s:
        await s.execute(
            text(
                "UPDATE extension_sync_state SET last_remote_hash = 'samehash' "
                "WHERE extension_id = 'paisa'"
            )
        )
        await s.commit()
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result(body_hash="samehash"))
    remote = AsyncMock(side_effect=AssertionError("must not POST on hash-noop"))
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    remote.assert_not_called()
    snap = await _snapshot(factory)
    assert snap.applied_revision >= snap.desired_revision


# --------------------------------------------------------------------------- #
# No old bug: local-unchanged does not suppress retry after remote failure
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_remote_failure_retries_unchanged_file(factory):
    """A remote failure leaves the row dirty; the next eligible tick re-POSTs
    even though the generated file is byte-identical (no hash-noop: the remote
    never accepted that hash)."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    # Generate produces an unchanged body across calls.
    gen = AsyncMock(return_value=_gen_result(body_hash="samebody"))
    # First remote call fails (pre-POST), second succeeds.
    remote_results = [
        _remote_result(
            post_accepted=False, diagnosis_ok=None, outcome="unreachable", reason="down"
        ),
        _remote_result(post_accepted=True, diagnosis_ok=True, outcome="synced"),
    ]
    remote = AsyncMock(side_effect=remote_results)
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    assert remote.await_count == 1
    snap = await _snapshot(factory)
    assert snap.failure_count == 1  # backoff armed
    assert snap.desired_revision > snap.applied_revision  # still dirty
    # Advance past the 1-min backoff floor.
    clock.advance(datetime.timedelta(minutes=2))
    await c._tick()
    assert remote.await_count == 2  # retried the SAME body — no hash-noop suppression
    snap2 = await _snapshot(factory)
    assert snap2.last_remote_hash == "samebody"
    assert snap2.failure_count == 0


@pytest.mark.usefixtures("settings_paisa")
async def test_last_remote_hash_gates_hash_noop_until_remote_acceptance(factory):
    """Persisted ``last_remote_hash`` (not ``last_published_hash``) gates the
    hash-noop skip. Regression for the old automation.py skip-unchanged bug
    where a locally-published include that the remote never accepted could be
    suppressed on the next attempt.

    Sequence (one constant body hash ``H`` across every generate):

    1. Attempt 1 — preflight OK, generate publishes ``H`` (so
       ``last_published_hash == H``), POST rejected (``sync_rejected``).
       ``last_remote_hash`` must stay ``None`` and the row must stay dirty.
    2. Attempt 2 (past backoff) — preflight OK, generate re-publishes the
       byte-identical ``H``, POST must NOT be skipped (no hash-noop: the
       remote never accepted ``H``); this attempt POST is accepted.
       ``last_remote_hash == H`` only now.
    3. Attempt 3 (a fresh dirty event, identical bytes ``H``) — now the
       hash-noop IS allowed to advance ``applied`` without a POST, because
       that exact hash was remotely accepted.
    """
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))

    body_hash = "the-one-hash"
    gen = AsyncMock(return_value=_gen_result(body_hash=body_hash))
    # POST sequence: rejected first, accepted second. The third attempt must
    # never reach the POST stage (hash-noop), so any third call is a bug.
    remote_results = [
        _remote_result(
            post_accepted=False,
            diagnosis_ok=None,
            outcome="sync_rejected",
            reason="503",
        ),
        _remote_result(post_accepted=True, diagnosis_ok=True, outcome="synced"),
        AssertionError("must NOT POST on hash-noop after remote acceptance"),
    ]
    remote = AsyncMock(side_effect=remote_results)
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)

    # Attempt 1: POST rejected. Remote hash must NOT be recorded; publish hash IS.
    await c._tick()
    assert remote.await_count == 1
    snap = await _snapshot(factory)
    assert snap.last_published_hash == body_hash  # publish checkpoint recorded
    assert snap.last_remote_hash is None  # remote never accepted
    assert snap.desired_revision > snap.applied_revision  # still dirty
    assert snap.failure_count == 1  # backoff armed

    # Attempt 2 (past the 1-min backoff): retry the SAME body. Hash-noop must
    # NOT fire because last_remote_hash is still None.
    clock.advance(datetime.timedelta(minutes=2))
    await c._tick()
    assert remote.await_count == 2  # retried identical bytes — no skip
    snap2 = await _snapshot(factory)
    assert snap2.last_remote_hash == body_hash  # only NOW remotely accepted
    assert snap2.applied_revision >= snap2.desired_revision  # caught up
    assert snap2.failure_count == 0  # backoff cleared

    # Attempt 3: a fresh dirty event with identical bytes. Hash-noop is now
    # allowed (remote has exactly this hash), so the POST must NOT be called.
    # Advance past both the 1-min remote floor (the last remote attempt was
    # attempt 2) and the debounce so the tick is eligible.
    await _dirty(factory, clock, n=1)
    clock.advance(datetime.timedelta(minutes=2))
    await c._tick()
    assert remote.await_count == 2  # no third POST — hash-noop advanced applied
    snap3 = await _snapshot(factory)
    assert snap3.applied_revision >= snap3.desired_revision  # advanced via hash-noop
    # The remote hash is unchanged (no new acceptance was needed).
    assert snap3.last_remote_hash == body_hash


# --------------------------------------------------------------------------- #
# Accepted POST + fatal diagnosis → no reload loop
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_accepted_post_with_fatal_diagnosis_advances_applied_no_loop(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result(body_hash="h-fatal"))
    remote = AsyncMock(
        return_value=_remote_result(
            post_accepted=True,
            diagnosis_ok=False,
            outcome="diagnosis_failed",
            reason="fatal danger",
            diagnosis_expected=1,
            diagnosis_accepted=0,
            diagnosis_fatal=1,
        )
    )
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    snap = await _snapshot(factory)
    # Accepted POST advanced applied + stamped remote hash.
    assert snap.applied_revision >= snap.desired_revision
    assert snap.last_remote_hash == "h-fatal"
    assert snap.diagnosis_state == "fatal"
    assert snap.failure_count == 1  # diagnosis fatal armed backoff
    # Next immediate tick: not dirty, not force → no reload loop.
    remote.reset_mock()
    await c._tick()
    # The row is caught up (applied>=desired) and not forced, so no POST.
    assert remote.await_count == 0


# --------------------------------------------------------------------------- #
# Periodic force reload calls remote even when clean
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_periodic_force_reload_calls_remote_even_clean(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    # Seed a clean, caught-up row whose last_remote_attempt_at is 7h ago.
    old = clock() - datetime.timedelta(hours=7)
    async with factory() as s:
        await s.execute(
            text(
                "UPDATE extension_sync_state SET applied_revision = desired_revision, "
                "last_remote_attempt_at = :old, first_dirty_at = NULL, "
                "last_dirty_at = NULL, force_reload = 0, last_remote_hash = 'clean', "
                "failure_count = 0, next_attempt_at = NULL "
                "WHERE extension_id = 'paisa'"
            ),
            {"old": old.replace(tzinfo=None).isoformat(sep=" ")},
        )
        await s.commit()
    gen = AsyncMock(return_value=_gen_result(body_hash="clean"))
    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    remote.assert_awaited_once()  # periodic reload POSTed despite clean hash


# --------------------------------------------------------------------------- #
# Backoff
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_pre_post_failure_arms_backoff(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result(body_hash="hb"))
    remote = AsyncMock(
        return_value=_remote_result(
            post_accepted=False, diagnosis_ok=None, outcome="unreachable", reason="down"
        )
    )
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    snap = await _snapshot(factory)
    assert snap.failure_count == 1
    assert snap.next_attempt_at is not None
    # Backoff schedule first step is 1 minute.
    assert snap.next_attempt_at >= clock()


# --------------------------------------------------------------------------- #
# Single flight / lease
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_two_coordinators_one_winner(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen1 = AsyncMock(return_value=_gen_result(body_hash="w1"))
    gen2 = AsyncMock(return_value=_gen_result(body_hash="w2"))
    remote1 = AsyncMock(return_value=_remote_result())
    remote2 = AsyncMock(return_value=_remote_result())
    c1 = _make_coordinator(factory, clock, generate_fn=gen1, sync_remote_fn=remote1)
    c2 = _make_coordinator(factory, clock, generate_fn=gen2, sync_remote_fn=remote2)
    # Run both attempts concurrently — exactly one wins the lease.
    await asyncio.gather(c1._tick(), c2._tick())
    total_remote = remote1.await_count + remote2.await_count
    assert total_remote == 1


@pytest.mark.usefixtures("settings_paisa")
async def test_expired_lease_recovered(factory):
    """A crashed coordinator's lease is reclaimed after its TTL expires."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    # Simulate a crashed coordinator: a held, expired lease.
    expired = clock() - datetime.timedelta(seconds=200)
    async with factory() as s:
        await s.execute(
            text(
                "UPDATE extension_sync_state SET lease_owner='ghost', "
                "lease_token='ghosttoken', lease_expires_at=:exp "
                "WHERE extension_id='paisa'"
            ),
            {"exp": expired.replace(tzinfo=None).isoformat(sep=" ")},
        )
        await s.commit()
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result())
    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    gen.assert_awaited_once()  # the new coordinator reclaimed and reconciled


@pytest.mark.usefixtures("settings_paisa")
async def test_stale_fencing_token_abandons_completion(factory):
    """When the lease is stolen mid-run, the token-guarded writes are abandoned
    (no commit) and the row stays dirty."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))

    stolen = {"done": False}

    async def stealing_remote(report, cfg, client=None):
        # Simulate another coordinator stealing the lease during the POST.
        async with factory() as s:
            await s.execute(
                text(
                    "UPDATE extension_sync_state SET lease_owner='thief', "
                    "lease_token='thieftoken', lease_expires_at=NULL "
                    "WHERE extension_id='paisa'"
                )
            )
            await s.commit()
        stolen["done"] = True
        return _remote_result()

    gen = AsyncMock(return_value=_gen_result(body_hash="stalebody"))
    c = _make_coordinator(
        factory, clock, generate_fn=gen, sync_remote_fn=stealing_remote
    )
    # Should not raise; the stale completion is abandoned.
    await c._tick()
    assert stolen["done"]
    snap = await _snapshot(factory)
    # The accepted-post completion was abandoned: remote hash NOT stamped by us.
    assert snap.last_remote_hash != "stalebody"


# --------------------------------------------------------------------------- #
# Commit during active sync → follow-up run
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_commit_during_sync_causes_followup(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))

    second_dirty = {"fired": False}
    # Changing body hash so the follow-up is not a hash-noop.
    seq = iter(["hd1", "hd2"])

    async def gen(sess, cfg):
        return _gen_result(body_hash=next(seq))

    async def remote_then_dirty(report, cfg, client=None):
        if not second_dirty["fired"]:
            # A concurrent commit lands during the POST, bumping desired past R.
            await _dirty(factory, clock, n=1)
            second_dirty["fired"] = True
        return _remote_result()

    g = AsyncMock(side_effect=gen)
    c = _make_coordinator(
        factory, clock, generate_fn=g, sync_remote_fn=remote_then_dirty
    )
    await c._tick()  # first run: post accepted, but desired bumped past R
    snap = await _snapshot(factory)
    # applied was clamped to the captured R, so desired > applied → still dirty.
    assert snap.desired_revision > snap.applied_revision
    # The follow-up reconciles once the remote floor + debounce elapse.
    clock.advance(datetime.timedelta(seconds=70))
    await c._tick()
    snap2 = await _snapshot(factory)
    assert snap2.applied_revision >= snap2.desired_revision


# --------------------------------------------------------------------------- #
# Restart with pending state
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_restart_pending_state_reconciles(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    # A fresh coordinator instance (simulate restart) picks up the pending state.
    gen = AsyncMock(return_value=_gen_result())
    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    gen.assert_awaited_once()
    snap = await _snapshot(factory)
    assert snap.applied_revision >= snap.desired_revision


# --------------------------------------------------------------------------- #
# Manual vs automatic never overlap (shared lease)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_manual_claims_lease_blocking_automatic(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    # Hold the lease as a "manual" operation would.
    async with factory() as s:
        from financial_dashboard.services.paisa.sync_state import claim_lease

        await claim_lease(s, owner="manual-sync", now=clock())
        await s.commit()
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(
        side_effect=AssertionError("auto must not run while manual holds lease")
    )
    c = _make_coordinator(factory, clock, generate_fn=gen)
    await c._tick()  # coordinator loses the lease → no work, no audit spam
    gen.assert_not_called()
    assert await _autowrite_count(factory) == 0


async def test_manual_lease_wait_returns_busy_when_held(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    # Hold the lease.
    async with factory() as s:
        from financial_dashboard.services.paisa.sync_state import claim_lease

        await claim_lease(s, owner="other", now=clock())
        await s.commit()
    # claim_manual_lease should time out → busy (with a tiny wait for the test).
    async with factory() as s:
        claim = await claim_manual_lease(
            s, owner="manual", wait_seconds=0.1, poll_seconds=0.02, now=clock()
        )
    assert claim.claimed is False
    assert claim.reason == "busy"


@pytest.mark.usefixtures("settings_paisa")
@pytest.mark.parametrize(
    ("operation", "audit_operation"),
    [
        ("generate", OPERATION_GENERATE),
        ("sync", OPERATION_SYNC),
    ],
)
async def test_manual_exception_releases_persisted_lease_immediately(
    factory, monkeypatch, operation, audit_operation
):
    """A failed manual call does not strand its committed lease for the TTL.

    ``_audited`` commits the running row before I/O, and the manual claim also
    commits. The exception cleanup must therefore roll back failed request
    state and release via a separate committed short transaction. A zero-wait
    claimant immediately after the exception proves reclaim does not depend on
    the 90-second expiry. Audit failure finalization must still survive.
    """
    from financial_dashboard.services.paisa import surface
    from financial_dashboard.services.paisa.sync_state import release_lease

    async def boom(*_args, **_kwargs):
        raise RuntimeError(f"{operation} exploded")

    if operation == "generate":
        monkeypatch.setattr(surface, "generate_now", boom)
        audited_call = surface.generate_now_audited
    else:
        monkeypatch.setattr(surface, "sync_now", boom)
        audited_call = surface.sync_now_audited

    async with factory() as session:
        with pytest.raises(RuntimeError, match=f"{operation} exploded"):
            await audited_call(session, trigger="test")

    released = await _snapshot(factory)
    assert released.lease_token is None
    assert released.lease_owner is None
    assert released.lease_expires_at is None

    async with factory() as session:
        reclaimed = await claim_manual_lease(
            session, owner="immediate-reclaimer", wait_seconds=0
        )
        assert reclaimed.claimed is True
        assert reclaimed.token is not None
        await release_lease(session, token=reclaimed.token)
        await session.commit()

    async with factory() as session:
        rows = await recent_runs(
            session, extension_id="paisa", operation=audit_operation
        )
    assert rows[0].status == STATUS_FAILURE
    assert rows[0].outcome == "error"
    assert rows[0].completed_at is not None
    assert f"{operation} exploded" in (rows[0].error or "")


# --------------------------------------------------------------------------- #
# Audit + notification (dedupe, truthful notify_sent)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_failure_writes_audit_row_and_notifies_when_opted_in(
    factory, monkeypatch
):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    settings_mod._cache["paisa.notify_sync_failures"] = "true"
    sent = []

    async def fake_send(app, *, chat_id, text):
        sent.append(text)

    import financial_dashboard.services.telegram as telegram_service

    monkeypatch.setattr(telegram_service, "tg_app", SimpleNamespace())
    monkeypatch.setattr(telegram_service, "_send_with_retry", fake_send)
    monkeypatch.setattr(
        "financial_dashboard.services.settings.is_telegram_configured", lambda: True
    )
    monkeypatch.setattr(
        "financial_dashboard.services.settings.get_telegram_chat_id", lambda: 123
    )
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result(body_hash="hf"))
    remote = AsyncMock(
        return_value=_remote_result(
            post_accepted=False, diagnosis_ok=None, outcome="unreachable", reason="down"
        )
    )
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    assert await _autowrite_count(factory) == 1
    assert len(sent) == 1
    assert "unreachable" in sent[0]
    # notify_sent truthfully reflects the attempted notification.
    async with factory() as s:
        rows = await recent_runs(s, extension_id="paisa", operation=OPERATION_AUTOMATIC)
        import json

        details = json.loads(rows[0].details)
        assert details["notify_sent"] is True


@pytest.mark.usefixtures("settings_paisa")
async def test_notify_dedupes_identical_failure_across_runs(factory, monkeypatch):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    settings_mod._cache["paisa.notify_sync_failures"] = "true"
    sent = []

    async def fake_send(app, *, chat_id, text):
        sent.append(text)

    import financial_dashboard.services.telegram as telegram_service

    monkeypatch.setattr(telegram_service, "tg_app", SimpleNamespace())
    monkeypatch.setattr(telegram_service, "_send_with_retry", fake_send)
    monkeypatch.setattr(
        "financial_dashboard.services.settings.is_telegram_configured", lambda: True
    )
    monkeypatch.setattr(
        "financial_dashboard.services.settings.get_telegram_chat_id", lambda: 123
    )
    gen = AsyncMock(return_value=_gen_result(body_hash="hd"))
    remote = AsyncMock(
        return_value=_remote_result(
            post_accepted=False, diagnosis_ok=None, outcome="unreachable", reason="down"
        )
    )
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)

    # First failing reconcile.
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    await c._tick()
    assert len(sent) == 1

    # Second identical failure after backoff: deduped.
    clock.advance(datetime.timedelta(minutes=2))
    await c._tick()
    assert len(sent) == 1  # same fingerprint → no re-notify


@pytest.mark.usefixtures("settings_paisa")
async def test_notify_sent_false_when_telegram_unconfigured(factory, monkeypatch):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    settings_mod._cache["paisa.notify_sync_failures"] = "true"
    # Telegram unconfigured: notify intent exists but send not attempted.
    monkeypatch.setattr(
        "financial_dashboard.services.settings.is_telegram_configured", lambda: False
    )
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result())
    remote = AsyncMock(
        return_value=_remote_result(
            post_accepted=False, diagnosis_ok=None, outcome="unreachable", reason="down"
        )
    )
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    async with factory() as s:
        rows = await recent_runs(s, extension_id="paisa", operation=OPERATION_AUTOMATIC)
        import json

        details = json.loads(rows[0].details)
        assert details["notify_sent"] is False  # truthful: not attempted


@pytest.mark.usefixtures("settings_paisa")
async def test_no_audit_spam_for_noop_ticks(factory):
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    # Clean, caught-up row → every tick is a no-op.
    gen = AsyncMock(side_effect=AssertionError("no generate when caught up"))
    c = _make_coordinator(factory, clock, generate_fn=gen)
    await c._tick()
    await c._tick()
    await c._tick()
    assert await _autowrite_count(factory) == 0


@pytest.mark.usefixtures("settings_paisa")
async def test_clean_runtime_polls_without_self_wake_or_writes(factory, monkeypatch):
    """A caught-up runtime tick rolls its SELECT back instead of committing it.

    The global ``after_commit`` hook wakes every running coordinator. If the
    eligibility read commits, the worker wakes itself after every clean tick
    and spins without honoring the poll interval. Exercise the real background
    loop and wake registry: two ticks must be separated by a real poll sleep,
    with no singleton mutation, seed, projection, or audit write.
    """
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    before = await _snapshot(factory)
    generated = AsyncMock(side_effect=AssertionError("caught-up row must stay idle"))
    ensure = AsyncMock(wraps=coord_mod.ensure_sync_state)
    monkeypatch.setattr(coord_mod, "ensure_sync_state", ensure)

    coordinator = PaisaCoordinator(
        session_factory=factory,
        now=clock,
        sleep=asyncio.sleep,
        preflight_fn=_preflight_ok,
        generate_fn=generated,
        sync_remote_fn=_default_sync_remote,
        min_interval_minutes_fn=lambda: 1,
        client_factory=lambda cfg: _FakeClient(),
        poll_interval=0.05,
    )
    original_tick = coordinator._tick
    tick_times: list[float] = []
    first_tick = asyncio.Event()
    second_tick = asyncio.Event()
    third_tick = asyncio.Event()

    async def counted_tick():
        tick_times.append(asyncio.get_running_loop().time())
        await original_tick()
        first_tick.set()
        if len(tick_times) >= 2:
            second_tick.set()
        if len(tick_times) >= 3:
            third_tick.set()

    monkeypatch.setattr(coordinator, "_tick", counted_tick)
    await coordinator.start()
    try:
        await asyncio.wait_for(first_tick.wait(), timeout=1)
        await asyncio.wait_for(second_tick.wait(), timeout=1)
        await asyncio.wait_for(third_tick.wait(), timeout=1)
        assert all(
            later - earlier >= 0.04
            for earlier, later in zip(tick_times[:2], tick_times[1:3], strict=True)
        )
    finally:
        await coordinator.stop()

    ensure.assert_not_awaited()
    generated.assert_not_called()
    assert await _snapshot(factory) == before
    assert await _autowrite_count(factory) == 0


# --------------------------------------------------------------------------- #
# Commit wake integration (real loop, tiny poll)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_commit_wakes_idle_coordinator(factory, monkeypatch):
    """A committed change fires the wake signal, which releases the coordinator's
    poll wait so it ticks sooner than its poll interval."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    attempts = []

    async def gen(sess, cfg):
        attempts.append("gen")
        return _gen_result()

    async def remote(report, cfg, client=None):
        attempts.append("remote")
        return _remote_result()

    # Use the real loop + real wake machinery: register the coordinator's wake
    # signal, then commit. The coordinator is built with the real start/stop.
    c = PaisaCoordinator(
        session_factory=factory,
        now=clock,
        sleep=asyncio.sleep,
        generate_fn=gen,
        sync_remote_fn=remote,
        preflight_fn=_preflight_ok,
        min_interval_minutes_fn=lambda: 1,
        client_factory=lambda cfg: _FakeClient(),
        poll_interval=10.0,  # large: only a wake should release the wait
    )
    await c.start()
    try:
        # Dirty + advance debounce so a tick is eligible.
        await _dirty(factory, clock, n=1)
        clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
        # Fire the wake (simulating a commit on this process).
        c.wake()
        # Give the loop a moment to tick.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if attempts:
                break
        assert attempts  # the wake released the wait → tick ran
    finally:
        await c.stop()


@pytest.mark.usefixtures("settings_paisa")
async def test_no_trigger_io_before_commit(factory):
    """Before any commit, a clean caught-up row triggers no orchestrator I/O."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    gen = AsyncMock(side_effect=AssertionError("no I/O before a dirtying commit"))
    remote = AsyncMock(side_effect=AssertionError("no I/O before a dirtying commit"))
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    gen.assert_not_called()
    remote.assert_not_called()


@pytest.mark.usefixtures("settings_paisa")
async def test_real_commit_fires_wake_through_listener(factory, monkeypatch):
    """A committed change on any session fires the global after_commit listener,
    which signals the coordinator's registered wake event (the commit-driven
    fast path). Verifies the wakeup registry ↔ coordinator wiring end to end."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    c = _make_coordinator(factory, clock)
    c._poll_interval = 10
    original_tick = c._tick
    tick_count = 0
    initial_tick = asyncio.Event()
    commit_tick = asyncio.Event()

    async def counted_tick():
        nonlocal tick_count
        tick_count += 1
        await original_tick()
        initial_tick.set()
        if tick_count >= 2:
            commit_tick.set()

    monkeypatch.setattr(c, "_tick", counted_tick)
    await c.start()
    try:
        # Wait for the initial tick to finish, leaving the coordinator in its
        # ten-second poll wait with the wake signal registered.
        await asyncio.wait_for(initial_tick.wait(), timeout=1)
        async with factory() as s:
            s.add(
                Transaction(
                    bank="hdfc",
                    email_type="txn",
                    direction="debit",
                    amount=Decimal("1.00"),
                    reference_number="wake-via-commit",
                )
            )
            await s.commit()
        # The commit wake is consumed by the run loop, so assert the observable
        # effect: a second tick starts well before the ten-second poll timeout.
        await asyncio.wait_for(commit_tick.wait(), timeout=1)
        assert tick_count == 2
    finally:
        await c.stop()


@pytest.mark.usefixtures("settings_paisa")
async def test_no_wake_signal_after_stop(factory):
    """Stopping the coordinator unregisters its wake signal (clean lifecycle)."""
    from financial_dashboard.services.paisa import wakeup

    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    c = _make_coordinator(factory, clock)
    await c.start()
    assert c._wake_signal in wakeup._wake_signals
    await c.stop()
    assert c._wake_signal not in wakeup._wake_signals


# --------------------------------------------------------------------------- #
# Stress / race: notify_sent stamps the correct run (by id, not "most recent")
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_stamp_notify_sent_targets_correct_run_id(factory, monkeypatch):
    """_stamp_notify_sent stamps by the captured run_id, not 'most recent'.
    If two automatic runs somehow exist, the stamp lands on the right one."""
    import json

    from financial_dashboard.db.models import ExtensionRun
    from financial_dashboard.services.paisa.audit import (
        complete_run,
        start_run,
    )
    from sqlalchemy import select

    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))

    # Write two automatic runs: the first one will be stamped, the second
    # should NOT receive the stamp.
    async with factory() as s:
        run1 = await start_run(
            s,
            extension_id="paisa",
            operation=OPERATION_AUTOMATIC,
            trigger="test",
        )
        run1_id = run1.id
        await complete_run(
            s,
            run1,
            status="failure",
            outcome="unreachable",
            details={"notify_fp": "abc123"},
            error="down",
        )
        run2 = await start_run(
            s,
            extension_id="paisa",
            operation=OPERATION_AUTOMATIC,
            trigger="test",
        )
        await complete_run(
            s,
            run2,
            status="success",
            outcome="synced",
            details={"note": "other"},
        )
        await s.commit()

    c = _make_coordinator(factory, clock)
    # Stamp the FIRST run (not "most recent").
    await c._stamp_notify_sent(run1_id, True)

    async with factory() as s:
        rows = list(
            (
                await s.execute(
                    select(ExtensionRun)
                    .where(ExtensionRun.extension_id == "paisa")
                    .order_by(ExtensionRun.id)
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) >= 2
    stamped = next(r for r in rows if r.id == run1_id)
    unstamped = next(r for r in rows if r.id != run1_id and r.outcome == "synced")
    details1 = json.loads(stamped.details) if stamped.details else {}
    details2 = json.loads(unstamped.details) if unstamped.details else {}
    assert details1.get("notify_sent") is True
    assert "notify_sent" not in details2


# --------------------------------------------------------------------------- #
# Stress: 200-row outer-commit integration (one bump, one coalesced reconcile)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_200_row_single_outer_commit_single_revision_bump(factory):
    """A bulk statement import of 200 rows in one outer commit bumps
    desired_revision by 200 (200 trigger firings in one txn), and the
    coordinator coalesces them into a single reconcile attempt (one generate +
    one POST) once the quiet debounce elapses."""
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))

    async with factory() as s:
        snap_before = await read_sync_state(s)
        await s.rollback()
    rev_before = snap_before.desired_revision

    # 200 rows in ONE outer commit → 200 trigger firings, all in one txn.
    await _dirty(factory, clock, n=200)

    async with factory() as s:
        snap_after = await read_sync_state(s)
        await s.rollback()
    # desired_revision advanced by exactly 200 (one per row, one outer commit).
    assert snap_after.desired_revision == rev_before + 200

    # A single tick (past debounce) reconciles all 200 rows in one attempt.
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result(emitted=200))
    remote = AsyncMock(return_value=_remote_result())
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()
    gen.assert_awaited_once()
    remote.assert_awaited_once()

    snap_final = await _snapshot(factory)
    assert snap_final.applied_revision >= snap_final.desired_revision


# --------------------------------------------------------------------------- #
# Stress: outer rollback drops the revision bump (post-commit transactional)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_outer_rollback_drops_revision_bump(factory):
    """If the dirtying transaction rolls back, the trigger's revision bump
    rolls back too (exact post-commit semantics: the bump shares the dirtying
    write's transaction). The coordinator never observes a revision for an
    uncommitted change."""

    async with factory() as s:
        snap_before = await read_sync_state(s)
        rev_before = snap_before.desired_revision

    # Add rows in a SAVEPOINT that rolls back.
    async with factory() as s:
        for i in range(5):
            s.add(
                Transaction(
                    bank="hdfc",
                    email_type="txn",
                    direction="debit",
                    amount=Decimal("1.00"),
                    reference_number=f"rb-{i}",
                )
            )
        sp = await s.begin_nested()
        try:
            await sp.rollback()
        except Exception:
            pass
        # The outer transaction still has the 5 rows (begin_nested rollback
        # only rolls back the savepoint). Roll back the whole thing.
        await s.rollback()

    async with factory() as s:
        snap_after = await read_sync_state(s)
        await s.rollback()
    # The full rollback dropped everything: revision unchanged.
    assert snap_after.desired_revision == rev_before


@pytest.mark.usefixtures("settings_paisa")
async def test_savepoint_rollback_drops_only_its_own_bump(factory):
    """A rolled-back SAVEPOINT drops only its own revision bump; the enclosing
    transaction's other bumps survive (exact post-commit semantics)."""

    async with factory() as s:
        snap_before = await read_sync_state(s)
        rev_before = snap_before.desired_revision

    # 3 rows in the outer txn, then 2 rows in a savepoint that rolls back,
    # then commit the outer.
    async with factory() as s:
        for i in range(3):
            s.add(
                Transaction(
                    bank="hdfc",
                    email_type="txn",
                    direction="debit",
                    amount=Decimal("1.00"),
                    reference_number=f"sp-outer-{i}",
                )
            )
        sp = await s.begin_nested()
        for i in range(2):
            s.add(
                Transaction(
                    bank="hdfc",
                    email_type="txn",
                    direction="debit",
                    amount=Decimal("1.00"),
                    reference_number=f"sp-inner-{i}",
                )
            )
        await sp.rollback()
        await s.commit()

    async with factory() as s:
        snap_after = await read_sync_state(s)
        await s.rollback()
    # Only the 3 outer rows committed; the 2 savepoint rows rolled back.
    assert snap_after.desired_revision == rev_before + 3


# --------------------------------------------------------------------------- #
# Stress: the automatic heartbeat protects every long attempt stage
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
@pytest.mark.parametrize(
    ("blocked_stage", "terminal"),
    [
        ("preflight", "success"),
        ("generation", "success"),
        ("preflight", "error"),
        ("generation", "cancel"),
    ],
)
async def test_attempt_heartbeat_spans_stages_and_cleans_up(
    factory, blocked_stage, terminal
):
    """Preflight/generation stay single-flight beyond the original lease TTL.

    A controlled heartbeat renews once before the original expiry, then the
    fake clock moves beyond that original expiry. A peer still cannot claim or
    publish. Success, an unexpected stage error, and task cancellation all stop
    the heartbeat and release the lease immediately.
    """
    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))

    entered = asyncio.Event()
    release_stage = asyncio.Event()
    heartbeat_waiting = asyncio.Event()
    second_heartbeat_sleep = asyncio.Event()
    heartbeat_cancelled = asyncio.Event()
    heartbeat_pulses: asyncio.Queue[None] = asyncio.Queue()
    heartbeat_sleep_calls = 0

    async def controlled_heartbeat_sleep(_seconds):
        nonlocal heartbeat_sleep_calls
        heartbeat_sleep_calls += 1
        heartbeat_waiting.set()
        if heartbeat_sleep_calls >= 2:
            second_heartbeat_sleep.set()
        try:
            await heartbeat_pulses.get()
        except asyncio.CancelledError:
            heartbeat_cancelled.set()
            raise

    async def block_selected_stage(stage):
        if blocked_stage != stage:
            return
        entered.set()
        await release_stage.wait()
        if terminal == "error":
            raise RuntimeError(f"blocked {stage} failed")

    async def staged_preflight(cfg, *, client=None):
        await block_selected_stage("preflight")
        return await _preflight_ok(cfg, client=client)

    async def staged_generate(session, cfg):
        await block_selected_stage("generation")
        return _gen_result(body_hash=f"{blocked_stage}-{terminal}")

    async def successful_remote(report, cfg, *, client=None):
        return _remote_result()

    coordinator = PaisaCoordinator(
        session_factory=factory,
        owner=f"blocked-{blocked_stage}-{terminal}",
        now=clock,
        sleep=controlled_heartbeat_sleep,
        preflight_fn=staged_preflight,
        generate_fn=staged_generate,
        sync_remote_fn=successful_remote,
        min_interval_minutes_fn=lambda: 1,
        client_factory=lambda cfg: _FakeClient(),
        heartbeat_interval=1,
        lease_ttl=2,
    )
    tick_task = asyncio.create_task(coordinator._tick())
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.wait_for(heartbeat_waiting.wait(), timeout=1)

    initial = await _snapshot(factory)
    assert initial.lease_token is not None
    assert initial.lease_expires_at is not None
    original_expiry = initial.lease_expires_at

    # Renew before the original TTL expires, then move past that original TTL.
    clock.advance(datetime.timedelta(seconds=1))
    heartbeat_pulses.put_nowait(None)
    await asyncio.wait_for(second_heartbeat_sleep.wait(), timeout=1)
    renewed = await _snapshot(factory)
    assert renewed.lease_expires_at is not None
    assert renewed.lease_expires_at > original_expiry
    clock.advance(datetime.timedelta(seconds=1, milliseconds=500))
    assert clock() > original_expiry
    assert clock() < renewed.lease_expires_at

    peer_preflight = AsyncMock(side_effect=AssertionError("peer must not claim"))
    peer_generate = AsyncMock(side_effect=AssertionError("peer must not publish"))
    peer = PaisaCoordinator(
        session_factory=factory,
        owner="blocked-stage-peer",
        now=clock,
        sleep=asyncio.sleep,
        preflight_fn=peer_preflight,
        generate_fn=peer_generate,
        sync_remote_fn=_default_sync_remote,
        min_interval_minutes_fn=lambda: 1,
        client_factory=lambda cfg: _FakeClient(),
        heartbeat_interval=1,
        lease_ttl=2,
    )
    await peer._tick()
    peer_preflight.assert_not_called()
    peer_generate.assert_not_called()

    if terminal == "cancel":
        tick_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(tick_task, timeout=2)
    else:
        release_stage.set()
        await asyncio.wait_for(tick_task, timeout=2)

    await asyncio.wait_for(heartbeat_cancelled.wait(), timeout=1)
    released = await _snapshot(factory)
    assert released.lease_owner is None
    assert released.lease_token is None
    assert released.lease_expires_at is None

    stopped_call_count = heartbeat_sleep_calls
    heartbeat_pulses.put_nowait(None)
    await asyncio.sleep(0)
    assert heartbeat_sleep_calls == stopped_call_count
    expected_audits = 0 if terminal == "cancel" else 1
    assert await _autowrite_count(factory) == expected_audits


# --------------------------------------------------------------------------- #
# Stress: notification dedupe fingerprint stored + survived across runs
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("settings_paisa")
async def test_notify_fp_stored_on_audit_row_for_cross_restart_dedupe(factory):
    """The notify_fp is stored in the audit details so dedupe survives
    restarts (the coordinator reads it from the prior run's details, not from
    an in-memory variable)."""
    import json

    clock = FakeClock(datetime.datetime(2026, 1, 1, 12, 0, tzinfo=_TZ))
    await _dirty(factory, clock, n=1)
    clock.advance(DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1))
    gen = AsyncMock(return_value=_gen_result(body_hash="fp-test"))
    remote = AsyncMock(
        return_value=_remote_result(
            post_accepted=False, diagnosis_ok=None, outcome="unreachable", reason="down"
        )
    )
    c = _make_coordinator(factory, clock, generate_fn=gen, sync_remote_fn=remote)
    await c._tick()

    async with factory() as s:
        rows = await recent_runs(s, extension_id="paisa", operation=OPERATION_AUTOMATIC)
        details = json.loads(rows[0].details)
    assert "notify_fp" in details  # persisted for dedupe across restarts
