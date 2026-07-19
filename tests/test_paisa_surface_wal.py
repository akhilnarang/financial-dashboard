"""WAL two-connection tests: audited network operations must not hold SQLite's
writer lock during file/network/projection I/O.

The Paisa coordinator/audit/surface contract requires that an audited manual
operation (probe/generate/sync) or an automatic reconcile **never** holds the
SQLite writer lock across the slow parts of its work. SQLite (even in WAL) has
exactly one writer per database; a transaction that opened an INSERT/UPDATE
and never commits will block every other write for the ``busy_timeout`` window
(5s in production — see ``financial_dashboard.db._set_sqlite_pragmas``).

The audit ``running`` row inserted by ``start_run`` is the obvious trap: if it
is opened in a transaction that is then held across ``await client.aclose()``,
``await client.fetch_config()``, ``await sync_remote(...)`` etc., the lock is
held for the duration of that call. A concurrent core write (a Transaction
insert, a settings save) would wait the full 5s busy_timeout and then fail
with ``database is locked``.

These tests use a WAL file DB with the production busy_timeout (5s) and block
a fake network call **beyond** the 5s window while a separate session commits
a core write. The core write must succeed well within the 5s budget — proving
no writer lock is held during the network/projection I/O.

Coverage:

* :func:`surface.probe_status_audited` — a probe's network call (no lease
  claim, so the only protective commit before the network call is the audit
  row's own short transaction).
* :func:`surface.generate_now_audited` — a generate's projection/file stage.
* :func:`surface.sync_now_audited` — a manual sync's POST (claim commits the
  lease early, so this passes regardless; still a useful regression guard).
* :class:`coordinator.PaisaCoordinator` — an automatic reconcile. The audit
  row is opened only AFTER work completes; this test pins that property.
"""

import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import financial_dashboard.services.paisa.coordinator as coord_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import Setting, Transaction
from financial_dashboard.schemas.extensions import (
    PaisaGenerateResponse,
    PaisaPublishInfo,
    PaisaStatusResponse,
    PaisaSyncResponse,
)
from financial_dashboard.services.paisa import surface
from financial_dashboard.services.paisa.audit import OPERATION_AUTOMATIC
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.coordinator import PaisaCoordinator
from financial_dashboard.services.paisa.orchestrator import (
    GenerateResult,
    PreflightReport,
    RemoteSyncReport,
)
from financial_dashboard.services.paisa.publisher import PublishResult
from financial_dashboard.services.paisa.sync_state import (
    DEFAULT_QUIET_DEBOUNCE,
    read_sync_state,
)

pytestmark = pytest.mark.anyio

_TZ = datetime.UTC
UNUSED_PATH = "/tmp/paisa-wal-test.journal"

#: Production-mirrored busy_timeout (5s). A blocked writer that cannot acquire
#: the lock within this window errors with SQLITE_BUSY. The fake network call
#: in these tests blocks indefinitely until released — well beyond 5s — so a
#: held audit transaction surfaces as SQLite's own lock error here.
BUSY_TIMEOUT_MS = 5000

#: Every fake slow stage remains in flight beyond SQLite's production
#: busy_timeout. A lock regression therefore reaches SQLite's own
#: ``database is locked`` error rather than succeeding because the test
#: released the fake operation too early.
SLOW_STAGE_SECONDS = BUSY_TIMEOUT_MS / 1000 + 0.25

#: Longer than SQLite's busy_timeout so a real SQLITE_BUSY is not hidden by an
#: earlier asyncio timeout. The elapsed assertion below still requires the
#: healthy writer path to complete promptly.
WRITER_TIMEOUT_SECONDS = BUSY_TIMEOUT_MS / 1000 + 2.0


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def wal_engine(tmp_path, monkeypatch):
    """A WAL file DB with init_db schema + revision triggers, busy_timeout=5s.

    The PRAGMAs mirror production (``financial_dashboard.db`` sets
    journal_mode=WAL + busy_timeout=5000 on every sqlite connect), so a lock
    held across a slow operation times out a concurrent write at the same 5s
    boundary the app would hit in production.
    """
    from financial_dashboard.services import settings as settings_mod
    from financial_dashboard.services.categorization import merchant_rules

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", _noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", _noop)

    db_path = tmp_path / "paisa_wal.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    @event.listens_for(engine.sync_engine, "connect")
    def _wal(dbapi_connection, _record):
        cur = dbapi_connection.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        finally:
            cur.close()

    try:
        await init_db(engine)
        # Start each test from a clean, caught-up, non-forced row so the
        # automatic coordinator only reconciles when a test explicitly dirties.
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
def factory(wal_engine):
    return async_sessionmaker(wal_engine, class_=AsyncSession, expire_on_commit=False)


def _cfg(mode="connect", **overrides) -> PaisaProjectionConfig:
    base = dict(
        mode=mode,
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
    base.update(overrides)
    return PaisaProjectionConfig(**base)


@pytest.fixture
def project_config(monkeypatch):
    """Project mode config so probe/generate/sync can proceed past mode gates."""
    cfg = _cfg(mode="project")
    monkeypatch.setattr(surface, "load_config", lambda: cfg)
    monkeypatch.setattr(coord_mod, "load_config", lambda: cfg)
    return cfg


@pytest.fixture
def connect_config(monkeypatch):
    """Connect mode config (probe-only)."""
    cfg = _cfg(mode="connect")
    monkeypatch.setattr(surface, "load_config", lambda: cfg)
    return cfg


async def _block_past_busy_timeout(
    entered: asyncio.Event,
    held_past_timeout: asyncio.Event,
    release: asyncio.Event,
) -> None:
    """Hold a fake slow stage for longer than SQLite's busy timeout.

    The final release event keeps the stage blocked even after the minimum
    duration, so the test controls exactly when operation finalization may
    write again.
    """
    entered.set()
    await asyncio.sleep(SLOW_STAGE_SECONDS)
    held_past_timeout.set()
    try:
        await asyncio.wait_for(release.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        raise AssertionError("test did not release the fake slow stage")


async def _commit_promptly(session: AsyncSession) -> None:
    """Commit without masking SQLite's own 5-second lock failure."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(session.commit(), timeout=WRITER_TIMEOUT_SECONDS)
    assert loop.time() - started < BUSY_TIMEOUT_MS / 1000


class _FakeClient:
    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Manual probe — audit row must not hold writer lock during the network call
# --------------------------------------------------------------------------- #


async def test_probe_audited_does_not_hold_writer_lock_during_network(
    wal_engine, factory, connect_config, monkeypatch
):
    """A manual probe commits the audit ``running`` row in a short transaction
    BEFORE awaiting the network call, so the SQLite writer lock is released
    before the probe's HTTP call. A concurrent core write (a Transaction
    insert) on a separate session must succeed well within the 5s busy_timeout
    while the probe is still blocked in its fake network call.
    """
    in_network = asyncio.Event()
    held_past_timeout = asyncio.Event()
    release_network = asyncio.Event()

    async def slow_probe_status(*, client=None):
        # Signal that we've entered the network call, then block until the
        # test releases us. The block is indefinite — well beyond the 5s
        # busy_timeout — so any held audit transaction would surface as a
        # timeout on the concurrent write below.
        await _block_past_busy_timeout(in_network, held_past_timeout, release_network)
        return PaisaStatusResponse(
            ok=True,
            reachable=True,
            mode="connect",
            can_connect=True,
            can_project=False,
            capabilities=None,
            diagnosis=None,
            reason=None,
        )

    monkeypatch.setattr(surface, "probe_status", slow_probe_status)

    async def run_probe():
        async with factory() as session:
            return await surface.probe_status_audited(session, trigger="test")

    probe_task = asyncio.create_task(run_probe())

    try:
        # Wait for the probe to enter its network call.
        await asyncio.wait_for(in_network.wait(), timeout=5.0)
        # The probe is now blocked in the network call. A concurrent core
        # write via a SEPARATE session must succeed within the busy_timeout
        # (5s). If the probe held the audit writer lock, this would time out.
        async with factory() as core_session:
            core_session.add(
                Transaction(
                    bank="hdfc",
                    email_type="txn",
                    direction="debit",
                    amount=Decimal("1.00"),
                    reference_number="concurrent-core-write",
                )
            )
            await _commit_promptly(core_session)

        # Prove the network call stayed blocked beyond SQLite's own timeout,
        # then release it and verify the probe completes.
        await asyncio.wait_for(held_past_timeout.wait(), timeout=WRITER_TIMEOUT_SECONDS)
        assert not probe_task.done()
        release_network.set()
        result = await asyncio.wait_for(probe_task, timeout=5.0)
        assert result.ok is True
    finally:
        release_network.set()
        if not probe_task.done():
            probe_task.cancel()
            try:
                await probe_task
            except asyncio.CancelledError, Exception:
                pass


# --------------------------------------------------------------------------- #
# Manual sync — same property for the network POST stage
# --------------------------------------------------------------------------- #


async def test_sync_audited_does_not_hold_writer_lock_during_network(
    wal_engine, factory, project_config, monkeypatch
):
    """A manual sync commits the audit ``running`` row (and the lease claim)
    in short transactions BEFORE awaiting the POST, so the SQLite writer lock
    is released before the network call. A concurrent core write on a separate
    session must succeed within the busy_timeout while the sync is blocked in
    its fake POST.
    """
    in_post = asyncio.Event()
    held_past_timeout = asyncio.Event()
    release_post = asyncio.Event()

    async def slow_sync_now(session, *, client=None):
        await _block_past_busy_timeout(in_post, held_past_timeout, release_post)
        return PaisaSyncResponse(
            ok=True,
            mode="project",
            outcome="synced",
            summary=None,
            publish=PaisaPublishInfo(
                published=True,
                skipped=False,
                path=UNUSED_PATH,
                version="3",
                body_hash="syncwal",
                bytes_written=10,
            ),
            diagnosis_ok=True,
            reason=None,
        )

    monkeypatch.setattr(surface, "sync_now", slow_sync_now)

    async def run_sync():
        async with factory() as session:
            return await surface.sync_now_audited(session, trigger="test")

    sync_task = asyncio.create_task(run_sync())

    try:
        # Wait for the sync to enter its POST stage.
        await asyncio.wait_for(in_post.wait(), timeout=5.0)
        # The sync is now blocked in the POST. A concurrent core write via a
        # SEPARATE session must succeed within the busy_timeout.
        async with factory() as core_session:
            core_session.add(
                Transaction(
                    bank="hdfc",
                    email_type="txn",
                    direction="debit",
                    amount=Decimal("2.00"),
                    reference_number="concurrent-core-write-sync",
                )
            )
            await _commit_promptly(core_session)

        await asyncio.wait_for(held_past_timeout.wait(), timeout=WRITER_TIMEOUT_SECONDS)
        assert not sync_task.done()
        release_post.set()
        result = await asyncio.wait_for(sync_task, timeout=5.0)
        assert result.ok is True
        # The target was captured before the fake POST. The concurrent core
        # commit advanced desired_revision while that POST was in flight, so
        # completion may advance only through the earlier R and must leave the
        # new write dirty for a follow-up reconcile.
        async with factory() as state_session:
            state = await read_sync_state(state_session)
        assert state is not None
        assert state.applied_revision == 1
        assert state.desired_revision == 2
        assert state.desired_revision > state.applied_revision
    finally:
        release_post.set()
        if not sync_task.done():
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError, Exception:
                pass


# --------------------------------------------------------------------------- #
# Manual generate — same property for the projection read + file write
# --------------------------------------------------------------------------- #


async def test_generate_audited_does_not_hold_writer_lock_during_projection(
    wal_engine, factory, project_config, monkeypatch
):
    """A manual generate commits the audit ``running`` row in a short
    transaction BEFORE awaiting the projection read + file write. A concurrent
    core write on a separate session must succeed within the busy_timeout
    while the generate is blocked in its fake projection.
    """
    in_generate = asyncio.Event()
    held_past_timeout = asyncio.Event()
    release_generate = asyncio.Event()

    async def slow_generate_now(session):
        await _block_past_busy_timeout(in_generate, held_past_timeout, release_generate)
        return PaisaGenerateResponse(
            ok=True,
            mode="project",
            summary=None,
            publish=PaisaPublishInfo(
                published=True,
                skipped=False,
                path=UNUSED_PATH,
                version="3",
                body_hash="genwal",
                bytes_written=10,
            ),
            reason=None,
        )

    monkeypatch.setattr(surface, "generate_now", slow_generate_now)

    async def run_generate():
        async with factory() as session:
            return await surface.generate_now_audited(session, trigger="test")

    generate_task = asyncio.create_task(run_generate())

    try:
        # Wait for the generate to enter its slow stage.
        await asyncio.wait_for(in_generate.wait(), timeout=5.0)
        # The generate is now blocked. A concurrent core write via a SEPARATE
        # session must succeed within the busy_timeout.
        async with factory() as core_session:
            core_session.add(
                Transaction(
                    bank="hdfc",
                    email_type="txn",
                    direction="debit",
                    amount=Decimal("3.00"),
                    reference_number="concurrent-core-write-generate",
                )
            )
            await _commit_promptly(core_session)

        await asyncio.wait_for(held_past_timeout.wait(), timeout=WRITER_TIMEOUT_SECONDS)
        assert not generate_task.done()
        release_generate.set()
        result = await asyncio.wait_for(generate_task, timeout=5.0)
        assert result.ok is True
    finally:
        release_generate.set()
        if not generate_task.done():
            generate_task.cancel()
            try:
                await generate_task
            except asyncio.CancelledError, Exception:
                pass


# --------------------------------------------------------------------------- #
# Automatic coordinator — audit row opened AFTER work, no lock during I/O
# --------------------------------------------------------------------------- #


async def test_coordinator_does_not_hold_audit_lock_during_network(
    wal_engine, factory, monkeypatch
):
    """The automatic coordinator opens its audit row in a short transaction
    AFTER the reconcile work completes (lease claim, preflight, generate,
    POST, diagnosis, state writes), so no audit write transaction is held
    during the network call. A concurrent core write on a separate session
    must succeed within the busy_timeout while the coordinator is blocked in
    its fake remote POST.
    """
    # Project + auto on by default for this test. Pin load_config to a fixed
    # project config (do not depend on the settings cache state, which can
    # carry stale values from prior tests in the same run).
    settings_mod._cache.update(
        {
            "paisa.mode": "project",
            "paisa.auto_sync_enabled": "true",
            "paisa.auto_sync_min_interval_minutes": "1",
            "paisa.notify_sync_failures": "false",
        }
    )
    cfg = _cfg(mode="project")
    monkeypatch.setattr(coord_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(coord_mod, "get_setting_bool", lambda k, d=False: True)

    clock_t = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_TZ)

    def fake_now():
        return clock_t

    in_post = asyncio.Event()
    held_past_timeout = asyncio.Event()
    release_post = asyncio.Event()

    async def slow_sync_remote(report, cfg, *, client=None):
        # Signal that the coordinator reached the remote POST stage, then
        # block until the test releases us. While blocked here, the audit
        # transaction MUST NOT be held — the audit row is only written after
        # we return and the attempt outcome is determined.
        await _block_past_busy_timeout(in_post, held_past_timeout, release_post)
        return RemoteSyncReport(
            ok=True,
            outcome="synced",
            post_accepted=True,
            diagnosis_ok=True,
            reason=None,
        )

    async def preflight_ok(cfg, *, client=None):
        return PreflightReport(
            ok=True,
            outcome=None,
            capabilities=SimpleNamespace(ledger_cli="ledger"),
            reason=None,
        )

    async def gen_ok(sess, cfg):
        report = SimpleNamespace(
            emitted_count=1,
            skipped=(),
            journal="...",
            entries=(),
        )
        publish = PublishResult(
            published=True,
            path=UNUSED_PATH,
            version="3",
            body_hash="coord-wal",
            bytes_written=10,
        )
        return GenerateResult(ok=True, report=report, publish=publish, reason=None)

    coordinator = PaisaCoordinator(
        session_factory=factory,
        now=fake_now,
        # Use a real cancellable sleep for the heartbeat loop. The generic
        # coordinator tests inject _no_sleep to advance quickly, but doing so
        # here would make the heartbeat busy-spin and continuously contend for
        # SQLite's writer slot, invalidating this lock-isolation test.
        sleep=asyncio.sleep,
        preflight_fn=preflight_ok,
        generate_fn=gen_ok,
        sync_remote_fn=slow_sync_remote,
        min_interval_minutes_fn=lambda: 1,
        client_factory=lambda cfg: _FakeClient(),
        poll_interval=10.0,  # large: we drive _tick() explicitly
        heartbeat_interval=SLOW_STAGE_SECONDS + 5.0,
    )

    # Seed a dirty event + advance past debounce so a tick is eligible.
    async with factory() as s:
        s.add(
            Transaction(
                bank="hdfc",
                email_type="txn",
                direction="debit",
                amount=Decimal("1.00"),
                reference_number="coord-wal-seed",
            )
        )
        await s.commit()
        # Overwrite trigger-stamped wall-clock dirty timestamps with the fake
        # clock so eligibility comparisons are deterministic.
        ts = clock_t.replace(tzinfo=None).isoformat(sep=" ")
        await s.execute(
            text(
                "UPDATE extension_sync_state "
                "SET first_dirty_at = :ts, last_dirty_at = :ts "
                "WHERE extension_id = 'paisa'"
            ),
            {"ts": ts},
        )
        await s.commit()

    clock_t = clock_t + DEFAULT_QUIET_DEBOUNCE + datetime.timedelta(seconds=1)

    tick_task = asyncio.create_task(coordinator._tick())

    try:
        # Wait for the coordinator to reach its remote POST stage. Generous
        # timeout — under test-suite load the tick can take several seconds
        # to clear eligibility, claim the lease, preflight, and generate
        # before reaching the POST. The lock-holding assertion below is about
        # the POST stage itself, not the pre-POST timing.
        await asyncio.wait_for(in_post.wait(), timeout=30.0)
        # The coordinator is now blocked in the POST. The audit row MUST NOT
        # be open yet (it is only written AFTER the POST returns). Verify a
        # concurrent core write via a SEPARATE session succeeds within budget.
        # Also verify no audit row exists yet (the attempt is still in flight).
        async with factory() as core_session:
            core_session.add(Setting(key="some.other.setting", value="concurrent"))
            await _commit_promptly(core_session)

        # No automatic audit row should exist yet (work is still in flight).
        async with factory() as s:
            rows = (
                await s.execute(
                    text(
                        "SELECT id FROM extension_runs "
                        "WHERE extension_id = 'paisa' "
                        "  AND operation = :op",
                    ),
                    {"op": OPERATION_AUTOMATIC},
                )
            ).all()
            assert rows == []

        # The POST must still be blocked after SQLite's 5-second timeout. No
        # heartbeat write has fired during that window, so this tests the
        # coordinator transaction boundary without synthetic writer churn.
        await asyncio.wait_for(held_past_timeout.wait(), timeout=WRITER_TIMEOUT_SECONDS)
        assert not tick_task.done()

        # Release the POST and let the tick complete.
        release_post.set()
        await asyncio.wait_for(tick_task, timeout=30.0)
    finally:
        release_post.set()
        if not tick_task.done():
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError, Exception:
                pass
