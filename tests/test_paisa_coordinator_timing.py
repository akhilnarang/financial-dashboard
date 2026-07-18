"""Coordinator-level debounce timing scenarios with real SQLite commits.

The default tests are deterministic: projected transactions are committed
through the real SQLite triggers, then only the trigger-owned dirty timestamps
are aligned to an injected clock before :meth:`PaisaCoordinator._tick` runs.
``desired_revision`` is never edited by a test; every revision is a real trigger
bump and is asserted after commit.  Boundary ordering is therefore explicit:
the t=5/t=10 commit returns before that boundary's eligibility decision.

An optional wall-clock version uses real trigger timestamps and takes about
16 seconds::

    PAISA_COORDINATOR_TIMING_STRESS=1 \
      uv run pytest -q tests/test_paisa_coordinator_timing.py -s

A production scheduler may legitimately tick just before a same-time boundary
commit.  That can produce an earlier sync, but not data loss: the later commit
bumps ``desired_revision`` and requires a follow-up.  These controlled tests pin
the requested commit-before-decision ordering without suppressing that valid
race in production.
"""

import asyncio
import datetime
import hashlib
import os
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.paisa.coordinator as coordinator_module
from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import Account, BalanceSnapshot, Transaction
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.coordinator import PaisaCoordinator
from financial_dashboard.services.paisa.orchestrator import (
    GenerateResult,
    PreflightReport,
    RemoteSyncReport,
    generate,
)
from financial_dashboard.services.paisa.projection import ProjectionReport, project
from financial_dashboard.services.paisa.sync_state import read_sync_state

TIMING_STRESS_ENV = "PAISA_COORDINATOR_TIMING_STRESS"
CUTOVER = datetime.date(2026, 1, 1)
T0 = datetime.datetime(2026, 7, 1, 12, 0, tzinfo=datetime.UTC)


class FakeClock:
    def __init__(self, value: datetime.datetime) -> None:
        self.value = value

    def __call__(self) -> datetime.datetime:
        return self.value

    def set_elapsed(self, seconds: float) -> None:
        self.value = T0 + datetime.timedelta(seconds=seconds)


class TimingHarness(NamedTuple):
    engine: object
    factory: async_sessionmaker
    config: PaisaProjectionConfig
    generated_path: Path


class SyncObservation(NamedTuple):
    timestamp: object
    body_hash: str
    journal: str


class RecordingRemote:
    def __init__(self, stamp: Callable[[], object]) -> None:
        self._stamp = stamp
        self.calls: list[SyncObservation] = []

    async def __call__(
        self,
        report: ProjectionReport,
        _config: PaisaProjectionConfig,
        *,
        client=None,
    ) -> RemoteSyncReport:
        del client
        body_hash = hashlib.sha256(report.journal.encode()).hexdigest()
        self.calls.append(SyncObservation(self._stamp(), body_hash, report.journal))
        return RemoteSyncReport(
            ok=True,
            outcome="synced",
            post_accepted=True,
            diagnosis_ok=True,
            reason=None,
            diagnosis_expected=0,
            diagnosis_accepted=0,
            diagnosis_fatal=0,
        )


class FakeClient:
    async def aclose(self) -> None:
        return None


@pytest.fixture
async def timing_harness(tmp_path, monkeypatch):
    from financial_dashboard.services import settings as settings_module
    from financial_dashboard.services.categorization import merchant_rules

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(settings_module, "load_all_settings", noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", noop)

    db_path = tmp_path / "paisa-coordinator-timing.db"
    generated_path = tmp_path / "paisa-timing.ledger"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    @event.listens_for(engine.sync_engine, "connect")
    def configure_sqlite(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=10000")
        finally:
            cursor.close()

    await init_db(engine)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.execute(
            Account.__table__.insert(),
            {
                "id": 1,
                "bank": "hdfc",
                "label": "Timing Savings",
                "type": "bank_account",
                "active": True,
            },
        )
        await connection.execute(
            BalanceSnapshot.__table__.insert(),
            {
                "id": 1,
                "account_id": 1,
                "kind": "asset",
                "category": "bank_balance",
                "as_of_date": CUTOVER - datetime.timedelta(days=1),
                "value": Decimal("100000.00"),
                "source": "bank_statement",
                "currency": "INR",
            },
        )

    config = PaisaProjectionConfig(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path=str(generated_path),
        selected_account_ids=(1,),
        cutover_date=CUTOVER,
        account_mappings={"1": "Assets:Bank:HDFC:Timing"},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
        ledger_cli="ledger",
    )
    monkeypatch.setattr(coordinator_module, "load_config", lambda: config)

    def get_setting_bool(key: str, default: bool = False) -> bool:
        if key == "paisa.auto_sync_enabled":
            return True
        if key == "paisa.notify_sync_failures":
            return False
        return default

    monkeypatch.setattr(coordinator_module, "get_setting_bool", get_setting_bool)
    try:
        yield TimingHarness(engine, factory, config, generated_path)
    finally:
        await engine.dispose()


async def _preflight_ok(_config, *, client=None) -> PreflightReport:
    del client
    return PreflightReport(
        ok=True,
        outcome=None,
        capabilities=SimpleNamespace(ledger_cli="ledger", readonly=False),
        reason=None,
    )


def _make_coordinator(
    harness: TimingHarness,
    *,
    now: Callable[[], datetime.datetime] | None,
    remote: RecordingRemote,
    generations: list[tuple[object, GenerateResult]],
    stamp: Callable[[], object],
) -> PaisaCoordinator:
    async def counted_generate(session, config):
        result = await generate(session, config)
        generations.append((stamp(), result))
        return result

    return PaisaCoordinator(
        session_factory=harness.factory,
        owner=f"timing-{id(remote)}",
        now=now,
        sleep=asyncio.sleep,
        client_factory=lambda _config: FakeClient(),
        preflight_fn=_preflight_ok,
        generate_fn=counted_generate,
        sync_remote_fn=remote,
        min_interval_minutes_fn=lambda: 1,
        heartbeat_interval=3_600,
        poll_interval=2,
    )


async def _state(harness: TimingHarness):
    async with harness.factory() as session:
        snapshot = await read_sync_state(session)
        await session.rollback()
    assert snapshot is not None
    return snapshot


async def _establish_baseline(
    harness: TimingHarness,
    coordinator: PaisaCoordinator,
    remote: RecordingRemote,
    generations: list[tuple[object, GenerateResult]],
) -> int:
    await coordinator._tick()
    assert len(remote.calls) == 1
    snapshot = await _state(harness)
    assert snapshot.desired_revision == snapshot.applied_revision
    assert snapshot.last_remote_hash == remote.calls[0].body_hash
    remote.calls.clear()
    generations.clear()
    return snapshot.desired_revision


async def _commit_at(
    harness: TimingHarness, clock: FakeClock, *, elapsed: float, serial: int
) -> int:
    """Commit a real projected row, then align only dirty timestamps to the clock."""
    clock.set_elapsed(elapsed)
    before = await _state(harness)
    async with harness.factory() as session:
        session.add(
            Transaction(
                account_id=1,
                bank="hdfc",
                email_type="timing_test",
                direction="debit",
                amount=Decimal(100 + serial),
                currency="INR",
                transaction_date=CUTOVER + datetime.timedelta(days=10 + serial),
                counterparty=f"Timing Merchant {serial}",
                reference_number=f"timing-{serial}",
                channel="test",
                category="groceries",
                source="timing-test",
            )
        )
        await session.commit()
    after_commit = await _state(harness)
    assert after_commit.desired_revision == before.desired_revision + 1

    # SQLite CURRENT_TIMESTAMP is real wall time and second-resolution.  Align
    # only the window timestamps to the injected clock so no sleep is required;
    # desired_revision remains the trigger-produced authoritative value.
    stored = clock().replace(tzinfo=None)
    async with harness.factory() as session:
        await session.execute(
            text(
                "UPDATE extension_sync_state "
                "SET first_dirty_at = CASE WHEN :new_window THEN :stamp "
                "                          ELSE first_dirty_at END, "
                "    last_dirty_at = :stamp "
                "WHERE extension_id = 'paisa'"
            ),
            {
                "new_window": before.desired_revision == before.applied_revision,
                "stamp": stored,
            },
        )
        await session.commit()
    return after_commit.desired_revision


def _published_body(path: Path) -> str:
    _header, separator, body = path.read_text().partition("\n\n")
    assert separator == "\n\n"
    return body


async def _assert_final_parity(
    harness: TimingHarness, remote: RecordingRemote, expected_revision: int
) -> str:
    async with harness.factory() as session:
        clean = await project(session, harness.config)
        await session.rollback()
    snapshot = await _state(harness)
    body = _published_body(harness.generated_path)
    body_hash = hashlib.sha256(clean.journal.encode()).hexdigest()
    assert snapshot.desired_revision == expected_revision
    assert snapshot.applied_revision == expected_revision
    assert snapshot.last_remote_hash == body_hash
    assert snapshot.last_healthy_hash == body_hash
    assert body == clean.journal == remote.calls[-1].journal
    return body_hash


@pytest.mark.anyio
async def test_boundary_burst_coalesces_once_after_final_quiet_window(timing_harness):
    """t=0/5/10 commits move the window; only the t=15 tick reconciles."""
    clock = FakeClock(T0 - datetime.timedelta(seconds=120))
    remote = RecordingRemote(clock)
    generations: list[tuple[object, GenerateResult]] = []
    coordinator = _make_coordinator(
        timing_harness,
        now=clock,
        remote=remote,
        generations=generations,
        stamp=clock,
    )
    base_revision = await _establish_baseline(
        timing_harness, coordinator, remote, generations
    )

    for serial, elapsed in enumerate((0, 5, 10), start=1):
        revision = await _commit_at(
            timing_harness, clock, elapsed=elapsed, serial=serial
        )
        assert revision == base_revision + serial
        await coordinator._tick()  # commit-before-decision at each boundary
        assert remote.calls == []
        assert generations == []

    clock.set_elapsed(14.999)
    await coordinator._tick()
    assert remote.calls == []
    clock.set_elapsed(15)
    await coordinator._tick()
    assert [call.timestamp for call in remote.calls] == [
        T0 + datetime.timedelta(seconds=15)
    ]
    assert len(generations) == 1
    await _assert_final_parity(timing_harness, remote, base_revision + 3)


@pytest.mark.anyio
async def test_gap_fires_first_debounce_then_retains_second_until_remote_floor(
    timing_harness,
):
    """Debounce permits t=5, while the one-minute remote floor delays t=10."""
    clock = FakeClock(T0 - datetime.timedelta(seconds=120))
    remote = RecordingRemote(clock)
    generations: list[tuple[object, GenerateResult]] = []
    coordinator = _make_coordinator(
        timing_harness,
        now=clock,
        remote=remote,
        generations=generations,
        stamp=clock,
    )
    base_revision = await _establish_baseline(
        timing_harness, coordinator, remote, generations
    )

    await _commit_at(timing_harness, clock, elapsed=0, serial=1)
    await coordinator._tick()
    clock.set_elapsed(5)
    await coordinator._tick()
    assert [call.timestamp for call in remote.calls] == [
        T0 + datetime.timedelta(seconds=5)
    ]

    await _commit_at(timing_harness, clock, elapsed=10, serial=2)
    await coordinator._tick()
    clock.set_elapsed(15)  # quiet debounce elapsed, remote floor has not
    await coordinator._tick()
    assert len(remote.calls) == 1
    assert len(generations) == 1  # floor gates before projection/publication
    pending = await _state(timing_harness)
    assert pending.desired_revision == base_revision + 2
    assert pending.applied_revision == base_revision + 1

    clock.set_elapsed(64.999)
    await coordinator._tick()
    assert len(remote.calls) == 1
    clock.set_elapsed(65)  # sixty seconds after the t=5 remote attempt
    await coordinator._tick()
    assert [call.timestamp for call in remote.calls] == [
        T0 + datetime.timedelta(seconds=5),
        T0 + datetime.timedelta(seconds=65),
    ]
    assert len(generations) == 2
    await _assert_final_parity(timing_harness, remote, base_revision + 2)


@pytest.mark.anyio
async def test_continuous_five_second_boundaries_hit_max_latency_then_follow_up(
    timing_harness,
):
    """No quiet gap through t=30; max latency reconciles, then floor retains later rows."""
    clock = FakeClock(T0 - datetime.timedelta(seconds=120))
    remote = RecordingRemote(clock)
    generations: list[tuple[object, GenerateResult]] = []
    coordinator = _make_coordinator(
        timing_harness,
        now=clock,
        remote=remote,
        generations=generations,
        stamp=clock,
    )
    base_revision = await _establish_baseline(
        timing_harness, coordinator, remote, generations
    )

    for serial, elapsed in enumerate(range(0, 31, 5), start=1):
        await _commit_at(timing_harness, clock, elapsed=float(elapsed), serial=serial)
        await coordinator._tick()
        if elapsed < 30:
            assert remote.calls == []
    assert [call.timestamp for call in remote.calls] == [
        T0 + datetime.timedelta(seconds=30)
    ]
    after_max_latency = await _state(timing_harness)
    assert after_max_latency.desired_revision == base_revision + 7
    assert after_max_latency.applied_revision == base_revision + 7

    for serial, elapsed in ((8, 35), (9, 40)):
        await _commit_at(timing_harness, clock, elapsed=float(elapsed), serial=serial)
        await coordinator._tick()
    clock.set_elapsed(45)  # debounce is satisfied; remote floor is not
    await coordinator._tick()
    pending = await _state(timing_harness)
    assert len(remote.calls) == 1
    assert pending.desired_revision == base_revision + 9
    assert pending.applied_revision == base_revision + 7

    clock.set_elapsed(89.999)
    await coordinator._tick()
    assert len(remote.calls) == 1
    clock.set_elapsed(90)
    await coordinator._tick()
    assert [call.timestamp for call in remote.calls] == [
        T0 + datetime.timedelta(seconds=30),
        T0 + datetime.timedelta(seconds=90),
    ]
    assert len(generations) == 2
    await _assert_final_parity(timing_harness, remote, base_revision + 9)


async def _sleep_until(start: float, elapsed: float) -> None:
    await asyncio.sleep(max(0.0, start + elapsed - time.monotonic()))


async def _commit_realtime(
    harness: TimingHarness, *, start: float, serial: int
) -> tuple[float, int]:
    before = await _state(harness)
    async with harness.factory() as session:
        session.add(
            Transaction(
                account_id=1,
                bank="hdfc",
                email_type="timing_realtime",
                direction="debit",
                amount=Decimal(500 + serial),
                currency="INR",
                transaction_date=CUTOVER + datetime.timedelta(days=100 + serial),
                counterparty=f"Realtime Merchant {serial}",
                reference_number=f"timing-realtime-{serial}",
                channel="test",
                category="groceries",
                source="timing-realtime",
            )
        )
        await session.commit()
    after = await _state(harness)
    assert after.desired_revision == before.desired_revision + 1
    return time.monotonic() - start, after.desired_revision


@pytest.mark.anyio
@pytest.mark.skipif(
    os.environ.get(TIMING_STRESS_ENV) != "1",
    reason=f"set {TIMING_STRESS_ENV}=1 to run the real-time debounce test",
)
async def test_realtime_t0_t5_t10_commits_sync_once_after_final_quiet(
    timing_harness,
):
    """Real clock/trigger timestamps; explicit post-commit ticks mimic 2s polling."""
    started_marker = time.monotonic()
    remote = RecordingRemote(lambda: time.monotonic() - started_marker)
    generations: list[tuple[object, GenerateResult]] = []
    coordinator = _make_coordinator(
        timing_harness,
        now=None,
        remote=remote,
        generations=generations,
        stamp=lambda: time.monotonic() - started_marker,
    )
    base_revision = await _establish_baseline(
        timing_harness, coordinator, remote, generations
    )

    # Baseline delivery is intentionally older than the timing run so the hard
    # one-minute remote floor does not obscure the debounce being measured.
    old_attempt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=2)
    async with timing_harness.factory() as session:
        await session.execute(
            text(
                "UPDATE extension_sync_state SET last_remote_attempt_at = :old "
                "WHERE extension_id = 'paisa'"
            ),
            {"old": old_attempt.replace(tzinfo=None)},
        )
        await session.commit()

    # Start just after a wall-clock second so SQLite's second-resolution
    # CURRENT_TIMESTAMP does not shorten the observed quiet interval materially.
    fraction = time.time() % 1
    await asyncio.sleep((1 - fraction) + 0.05)
    start = time.monotonic()
    started_marker = start
    commit_times: list[float] = []
    for serial, target in enumerate((0.0, 5.0, 10.0), start=1):
        await _sleep_until(start, target)
        committed_at, revision = await _commit_realtime(
            timing_harness, start=start, serial=serial
        )
        commit_times.append(committed_at)
        assert revision == base_revision + serial
        await coordinator._tick()  # boundary commit is already durable
        assert remote.calls == []

    poll_times: list[float] = []
    for target in (12.0, 14.0, 16.0, 18.0):
        await _sleep_until(start, target)
        poll_times.append(time.monotonic() - start)
        await coordinator._tick()
        if remote.calls:
            break
    assert len(remote.calls) == 1
    assert len(generations) == 1
    sync_time = float(remote.calls[0].timestamp)
    final_quiet = sync_time - commit_times[-1]
    assert 4.5 <= final_quiet <= 8.5
    final_hash = await _assert_final_parity(timing_harness, remote, base_revision + 3)
    print(
        "\n[paisa-coordinator-timing] "
        f"commit_seconds={[round(value, 3) for value in commit_times]} "
        f"poll_seconds={[round(value, 3) for value in poll_times]} "
        f"sync_seconds={[round(float(call.timestamp), 3) for call in remote.calls]} "
        f"sync_attempts={len(remote.calls)} final_quiet_seconds={final_quiet:.3f} "
        f"final_hash={final_hash}",
        flush=True,
    )
