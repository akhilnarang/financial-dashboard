"""Loader tests: idempotent reruns, safe-path refusal, reset confirmation.

These tests build their own temporary synthetic DBs under ``tmp_path`` so they
never touch production. Each DB path is placed under a ``synthetic`` directory
component as the safety guard requires.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from scripts.synth import build_scenario, load_scenario
from scripts.synth import constants as C
from scripts.synth.constants import RESET_CONFIRMATION_FLAG
from scripts.synth.loader import count_rows, drop_synthetic_db
from scripts.synth.safety import (
    UnsafeTargetError,
    assert_synthetic_db_path,
    confirm_reset,
)

pytestmark = pytest.mark.anyio


def _synthetic_db(tmp_path: Path, name: str = "synthetic.db") -> Path:
    """A DB path that satisfies the synthetic-path guard."""
    p = tmp_path / "synthetic" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Safe-path refusal
# ---------------------------------------------------------------------------


def test_refuses_production_db_name(tmp_path):
    with pytest.raises(UnsafeTargetError, match="production DB"):
        assert_synthetic_db_path(tmp_path / "financial_dashboard.db")


def test_refuses_non_synthetic_path(tmp_path):
    with pytest.raises(UnsafeTargetError, match="non-synthetic"):
        assert_synthetic_db_path(tmp_path / "foo.db")


def test_refuses_memory_target():
    with pytest.raises(UnsafeTargetError, match=":memory:"):
        assert_synthetic_db_path("sqlite+aiosqlite:///:memory:")


def test_accepts_synthetic_path(tmp_path):
    resolved = assert_synthetic_db_path(tmp_path / "synthetic" / "x.db")
    assert resolved.name == "x.db"


def test_reset_requires_confirmation_flag():
    with pytest.raises(UnsafeTargetError):
        confirm_reset(None)
    with pytest.raises(UnsafeTargetError):
        confirm_reset("yes-delete-the-synthetic-db-typo")
    confirm_reset(RESET_CONFIRMATION_FLAG)  # does not raise


async def test_reset_is_idempotent(tmp_path):
    db = _synthetic_db(tmp_path)
    db.write_bytes(b"x")  # pretend a DB exists
    confirm_reset(RESET_CONFIRMATION_FLAG)
    await drop_synthetic_db(db)
    assert not db.exists()
    # Dropping a non-existent DB is a no-op (idempotent).
    await drop_synthetic_db(db)
    assert not db.exists()


async def test_reset_refuses_non_synthetic_path(tmp_path):
    with pytest.raises(UnsafeTargetError):
        await drop_synthetic_db(tmp_path / "financial_dashboard.db")


# ---------------------------------------------------------------------------
# Idempotent loading
# ---------------------------------------------------------------------------


async def test_load_then_rerun_adds_zero_duplicates(tmp_path):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    after_first = await count_rows(db)

    second = await load_scenario(scenario, db)
    after_second = await count_rows(db)

    assert second["fidelity_transactions"] == 0
    assert second["bulk_transactions"] == 0
    assert second["bulk_emails"] == 0
    # Every populated table is byte-for-byte unchanged on rerun.
    assert after_first == after_second
    assert after_first["transactions"] == after_second["transactions"]
    assert after_first["transactions"] > 0


async def test_fidelity_lane_exercises_sms_email_merge(tmp_path):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    stats = await load_scenario(
        scenario, db, fidelity_txn_count=len(scenario.transactions)
    )
    # The whole scenario went through the fidelity lane; at least one paired
    # event was enriched via the SMS channel (the cross-channel merge path).
    assert stats["fidelity_sms_enriched"] >= 1
    counts = await count_rows(db)
    # No duplicate transactions despite email+SMS for the same event.
    assert counts["transactions"] == len(scenario.transactions)


async def test_bulk_lane_handles_high_volume(tmp_path):
    db = _synthetic_db(tmp_path)
    # A ci scenario with a tiny fidelity lane pushes almost everything to bulk.
    scenario = build_scenario(profile="ci")
    stats = await load_scenario(scenario, db, fidelity_txn_count=50)
    assert stats["bulk_transactions"] > 1000
    counts = await count_rows(db)
    assert counts["transactions"] == len(scenario.transactions)


def test_bulk_chunk_size_stays_under_sqlite_variable_limit():
    # The bulk lane's executemany must never produce a statement with more
    # bind variables than SQLite's default 999 ceiling, no matter how wide the
    # target table. Reproduce the loader's exact chunk computation against the
    # real Transaction column count to assert the invariant.
    from scripts.synth.loader import CHUNK_SIZE
    from financial_dashboard.db.models import Transaction

    num_cols = len(Transaction.__table__.columns)
    var_budget = 900
    chunk_size = min(CHUNK_SIZE, max(1, var_budget // num_cols))
    assert chunk_size >= 1
    # Every emitted statement uses at most chunk_size * num_cols binds.
    assert chunk_size * num_cols <= 999, (chunk_size, num_cols)


async def test_ci_profile_loads_idempotently_under_variable_limit(tmp_path):
    # The ci profile lands several thousand rows through the chunked bulk lane.
    # This confirms the adaptive chunking keeps the full profile under the
    # SQLite variable limit on a rerun too (idempotency + chunk safety together).
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="ci")
    await load_scenario(scenario, db, fidelity_txn_count=400)
    first = await count_rows(db)
    assert first["transactions"] == len(scenario.transactions)

    second_stats = await load_scenario(scenario, db, fidelity_txn_count=400)
    second = await count_rows(db)
    assert second_stats["bulk_transactions"] == 0
    assert second_stats["bulk_emails"] == 0
    assert second["transactions"] == first["transactions"]


async def test_db_counts_match_scenario_counts(tmp_path):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    expected = scenario.counts()
    await load_scenario(scenario, db)
    counts = await count_rows(db)
    for key, value in expected.items():
        assert counts[key] == value, f"{key}: expected {value}, got {counts[key]}"


async def test_count_rows_includes_investment_lots_and_extension_runs(tmp_path):
    """count_rows surfaces the investment_lots and extension_runs tables so
    manifest verify can police them. CAS ingestion persists exactly one lot for
    the scenario's single complete MF acquisition; the loader never runs an
    extension operation, so extension_runs stays zero after a load."""
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    counts = await count_rows(db)
    assert counts["investment_lots"] == scenario.counts()["investment_lots"] == 1
    assert counts["extension_runs"] == 0


async def test_verify_catches_lot_count_regression(tmp_path):
    """Manifest verify must flag a lot-count regression (e.g. a loader change
    that duplicated or dropped CAS lots) as tamper, while extension_runs is
    still expected zero before any extension operation runs."""
    import json

    from scripts.synth.manifest import TamperError, sha256_bytes, verify_manifest

    scenario = build_scenario(profile="smoke")
    expected = scenario.counts()
    assert expected["investment_lots"] == 1
    assert expected["extension_runs"] == 0

    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    # One real artefact so the artefact-checksum stage passes and verify
    # reaches the DB-count stage this test actually exercises.
    artefact = b"sentinel-bytes\n"
    (out_dir / "sentinel.txt").write_bytes(artefact)
    manifest = {
        "schema_version": "1",
        "generator_version": C.GENERATOR_VERSION,
        "seed": scenario.seed,
        "as_of": scenario.as_of.isoformat(),
        "profile": scenario.profile,
        "expected": dict(sorted(expected.items())),
        "invariants": {},
        "artefacts": {
            "sentinel.txt": {
                "sha256": sha256_bytes(artefact),
                "bytes": len(artefact),
            }
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # Correct counts → the count stage verifies cleanly (no drift).
    verify_manifest(out_dir, db_counts={**expected})

    # lot-count regression: 2 lots persisted instead of the expected 1.
    with pytest.raises(TamperError, match="investment_lots"):
        verify_manifest(
            out_dir,
            db_counts={**expected, "investment_lots": expected["investment_lots"] + 1},
        )

    # extension_runs appearing non-zero is also caught.
    with pytest.raises(TamperError, match="extension_runs"):
        verify_manifest(out_dir, db_counts={**expected, "extension_runs": 1})


async def test_load_refuses_via_load_scenario(tmp_path):
    scenario = build_scenario(profile="golden")
    with pytest.raises(UnsafeTargetError):
        await load_scenario(scenario, tmp_path / "not-synthetic.db")


# ---------------------------------------------------------------------------
# Shape-upgrade safety: loading a new generator/profile shape over an existing
# synthetic DB must be clean (no PK collision, no stale rows), and a same-shape
# rerun must remain an idempotent no-op. Regression for the
# ``UNIQUE constraint failed: emails.id`` defect.
# ---------------------------------------------------------------------------


def _all_message_ids(scenario) -> set[str]:
    """Every legitimate email message_id the scenario owns (transaction-linked
    + orphan), so a 'no stale rows' check can compare the loaded DB to it."""
    return {f"<{t.stable_id}@synthetic.local>" for t in scenario.transactions} | {
        oe.message_id for oe in scenario.orphan_emails
    }


async def _assert_db_matches_scenario(db, scenario) -> None:
    """Assert the loaded DB is *exactly* the current scenario: counts match,
    every email message_id is legitimate (no stale rows from a prior shape)."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from financial_dashboard.db.models import Email, Transaction
    from scripts.synth.loader import create_synthetic_engine

    counts = await count_rows(db)
    expected = scenario.counts()
    for key, value in expected.items():
        assert counts[key] == value, f"{key}: expected {value}, got {counts[key]}"

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            txns = (await session.execute(select(Transaction))).scalars().all()
            db_emails = (await session.execute(select(Email))).scalars().all()
            legit = _all_message_ids(scenario)
            db_mids = {e.message_id for e in db_emails}
            stale = db_mids - legit
            assert not stale, f"{len(stale)} stale emails survived: {sorted(stale)[:3]}"
            assert len(txns) == len(scenario.transactions)
    finally:
        await engine.dispose()


async def test_shape_upgrade_over_existing_db_is_clean(tmp_path):
    """Loading a different scenario shape (different seed → different
    message_ids, same overlapping email PK range) over an existing synthetic DB
    must not collide on emails.id and must leave the DB exactly matching the
    new scenario (no stale rows). This is the exact regression for the
    ``UNIQUE constraint failed: emails.id`` IntegrityError."""
    db = _synthetic_db(tmp_path)
    prior = build_scenario(seed=4242, profile="smoke")
    await load_scenario(prior, db)
    prior_counts = await count_rows(db)
    assert prior_counts["transactions"] == len(prior.transactions)

    # Different seed → fresh message_ids but the same low email PK range (100+).
    # Before the fix this raised IntegrityError on emails.id.
    current = build_scenario(seed=9999, profile="smoke")
    stats = await load_scenario(current, db)
    assert stats["fidelity_transactions"] > 0, "upgrade should have rebuilt rows"

    await _assert_db_matches_scenario(db, current)


async def test_same_shape_rerun_remains_idempotent_noop(tmp_path):
    """After a shape upgrade, a same-shape rerun must still be an idempotent
    no-op: the identity stamp matches, no reset happens, and stats report 0."""
    db = _synthetic_db(tmp_path)
    current = build_scenario(seed=9999, profile="smoke")
    await load_scenario(current, db)
    before = await count_rows(db)

    rerun = await load_scenario(current, db)
    after = await count_rows(db)
    assert rerun["fidelity_transactions"] == 0
    assert rerun["bulk_transactions"] == 0
    assert rerun["bulk_emails"] == 0
    assert before == after
    await _assert_db_matches_scenario(db, current)


async def test_upgrade_over_colliding_injected_ids_is_clean(tmp_path):
    """Inject old colliding email PKs (a pre-stamp, prior-version DB shape) and
    confirm the current load resets and rebuilds cleanly — simulating the
    reported 1.2.0→1.3.0 upgrade where old corpus emails own the same PKs."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    db = _synthetic_db(tmp_path)
    # Build a prior-version-shaped DB: load current, then *remove* the identity
    # stamp and inject a foreign email at a colliding PK with a different
    # message_id (exactly the 1.2.0→1.3.0 collision shape).
    prior = build_scenario(seed=4242, profile="smoke")
    await load_scenario(prior, db)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.begin() as conn:
            # Wipe the identity stamp so the next load cannot match it (mirrors
            # a DB written by a pre-identity loader version).
            await conn.execute(
                text("DELETE FROM settings WHERE key='synthetic.identity'")
            )
    finally:
        await engine.dispose()

    current = build_scenario(seed=4242, profile="smoke")
    await load_scenario(current, db)
    await _assert_db_matches_scenario(db, current)


async def test_failed_partial_load_recovers_on_rerun(tmp_path):
    """A failed/partial prior load must recover: the identity stamp is written
    only on full success, so a partial load leaves a stale/missing stamp and
    the next run resets + rebuilds to the exact current scenario."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    db = _synthetic_db(tmp_path)
    current = build_scenario(seed=4242, profile="smoke")
    await load_scenario(current, db)
    # Simulate a partial/failed subsequent load: a *different* shape loaded
    # partway then interrupted (here: delete the stamp so the next load treats
    # the DB as shape-mismatched and resets).
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM settings WHERE key='synthetic.identity'")
            )
    finally:
        await engine.dispose()

    # Re-run the SAME current shape: missing stamp → reset → full rebuild.
    await load_scenario(current, db)
    await _assert_db_matches_scenario(db, current)


async def test_identity_stamp_recorded_after_load(tmp_path):
    """A successful load stamps the scenario identity, so the next load can
    detect a match (no reset) or mismatch (reset)."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from financial_dashboard.db.models import Setting
    from scripts.synth.identity import IDENTITY_SETTING_KEY, load_identity
    from scripts.synth.loader import create_synthetic_engine

    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            stamp = (
                await session.execute(
                    select(Setting.value).where(Setting.key == IDENTITY_SETTING_KEY)
                )
            ).scalar_one()
            assert stamp == load_identity(scenario)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# CAS investment-lot ingestion (requirement: idempotent lot ingestion)
# ---------------------------------------------------------------------------


async def test_cas_complete_mf_fact_creates_exactly_one_lot(tmp_path):
    """The scenario's NSDL CAS upload carries one *complete* MF acquisition
    (units+nav+amount+date+isin) plus excluded demat/disposal/value-only facts.
    The loader — via the real ``ingest_cas_payload`` → ``create_investment_lots``
    — must persist exactly one InvestmentLot from the complete fact."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    from financial_dashboard.db.models import InvestmentLot
    from scripts.synth.loader import create_synthetic_engine

    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            lots = (await session.execute(select(InvestmentLot))).scalars().all()
            assert len(lots) == 1
            lot = lots[0]
            # The complete fact is the Synthetic Liquid Fund purchase.
            assert lot.instrument_id == "INE000A01020"
            assert lot.quantity == Decimal("500")
            assert lot.unit_cost == Decimal("100")
            assert lot.cost_basis == Decimal("50000")
            assert lot.currency == "INR"
    finally:
        await engine.dispose()


async def test_lot_ingestion_is_idempotent_on_rerun(tmp_path):
    """Re-loading the same scenario never duplicates lots (the CAS upsert
    deletes the prior upload's lots before re-creating them)."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    from financial_dashboard.db.models import InvestmentLot
    from scripts.synth.loader import create_synthetic_engine

    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)

    # Rerun: same seed/profile/as_of → idempotent.
    await load_scenario(scenario, db)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            lots = (await session.execute(select(InvestmentLot))).scalars().all()
            # Still exactly one lot, not two.
            assert len(lots) == 1
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Pristine Paisa sync-state singleton (mirrors production init_db's seed)
# ---------------------------------------------------------------------------
#
# Production init_db seeds exactly one pristine ``extension_sync_state`` row
# (desired_revision=1, applied_revision=0, force_reload=1, failure_count=0,
# every hash/retry/diagnosis/lease field NULL). The synthetic seed mirrors that
# so a loaded DB is representative of a freshly-initialized production DB, and
# downstream read paths see the singleton. The loader re-stamps the row to
# canonical values as the final write, so the recorded state stays deterministic
# even when SQLite triggers are active on the DB (init_db installs AFTER triggers
# on the core tables; the loader itself uses create_all, which installs none).


async def _sync_state_row(session):
    from financial_dashboard.db.models import ExtensionSyncState

    return await session.get(ExtensionSyncState, "paisa")


async def test_load_seeds_one_pristine_paisa_sync_state_row(tmp_path):
    """The loader seeds exactly one pristine Paisa sync-state singleton with
    the canonical fresh-init values (desired_revision=1, applied_revision=0,
    force_reload=1, failure_count=0, every hash/retry/diagnosis/lease field
    NULL), mirroring production init_db."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from scripts.synth.loader import create_synthetic_engine

    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            row = await _sync_state_row(session)
            assert row is not None, "paisa singleton missing"
            assert row.extension_id == "paisa"
            assert row.desired_revision == 1
            assert row.applied_revision == 0
            assert row.force_reload is True
            assert row.failure_count == 0
            # Every hash / retry / diagnosis / lease field is NULL on a fresh seed.
            for attr in (
                "last_published_hash",
                "last_remote_hash",
                "last_healthy_hash",
                "last_remote_attempt_at",
                "next_attempt_at",
                "diagnosis_state",
                "lease_owner",
                "lease_token",
                "lease_expires_at",
            ):
                assert getattr(row, attr) is None, attr
            # first/last_dirty_at are set (dirty + force_reload on fresh init).
            assert row.first_dirty_at is not None
            assert row.last_dirty_at is not None
            assert row.created_at is not None
    finally:
        await engine.dispose()


async def test_count_rows_and_scenario_counts_include_sync_state(tmp_path):
    """count_rows surfaces extension_sync_state and Scenario.counts() pins it
    at 1, so manifest verify can police a regression that drops the row or
    creates extras."""
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    expected = scenario.counts()
    assert expected["extension_sync_state"] == 1
    await load_scenario(scenario, db)
    counts = await count_rows(db)
    assert counts["extension_sync_state"] == 1


async def test_db_counts_match_scenario_counts_includes_sync_state(tmp_path):
    """The db-counts-match-scenario-counts contract now covers the sync-state
    singleton too (the existing test_db_counts_match_scenario_counts iterates
    over every key, so this is a belt-and-braces pin of the new key)."""
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="golden")
    await load_scenario(scenario, db)
    counts = await count_rows(db)
    assert (
        counts["extension_sync_state"] == scenario.counts()["extension_sync_state"] == 1
    )


async def test_verify_catches_sync_state_count_regression(tmp_path):
    """Manifest verify flags a sync-state count regression (row dropped or
    duplicated) as tamper, exactly like it flags investment_lots/extension_runs
    drift."""
    import json

    from scripts.synth.manifest import TamperError, sha256_bytes, verify_manifest

    scenario = build_scenario(profile="smoke")
    expected = scenario.counts()
    assert expected["extension_sync_state"] == 1

    out_dir = tmp_path / "corpus"
    out_dir.mkdir()
    artefact = b"sentinel-bytes\n"
    (out_dir / "sentinel.txt").write_bytes(artefact)
    manifest = {
        "schema_version": "1",
        "generator_version": C.GENERATOR_VERSION,
        "seed": scenario.seed,
        "as_of": scenario.as_of.isoformat(),
        "profile": scenario.profile,
        "expected": dict(sorted(expected.items())),
        "invariants": {},
        "artefacts": {
            "sentinel.txt": {
                "sha256": sha256_bytes(artefact),
                "bytes": len(artefact),
            }
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # Correct count verifies cleanly.
    verify_manifest(out_dir, db_counts={**expected})

    # Missing row → count 0 → tamper.
    with pytest.raises(TamperError, match="extension_sync_state"):
        verify_manifest(out_dir, db_counts={**expected, "extension_sync_state": 0})
    # Extra row → count 2 → tamper.
    with pytest.raises(TamperError, match="extension_sync_state"):
        verify_manifest(out_dir, db_counts={**expected, "extension_sync_state": 2})


async def test_rerun_keeps_sync_state_pristine(tmp_path):
    """A same-shape rerun leaves the pristine row pristine — the final re-stamp
    is idempotent and adds no second row."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from scripts.synth.loader import create_synthetic_engine

    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    await load_scenario(scenario, db)  # idempotent rerun

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            row = await _sync_state_row(session)
            assert row is not None
            assert row.desired_revision == 1
            assert row.applied_revision == 0
            assert row.force_reload is True
            assert row.failure_count == 0
            assert row.lease_owner is None
            assert row.last_remote_hash is None
            # No second row was created by the rerun.
            from sqlalchemy import func, select

            from financial_dashboard.db.models import ExtensionSyncState

            total = (
                await session.execute(
                    select(func.count()).select_from(ExtensionSyncState)
                )
            ).scalar_one()
            assert total == 1
    finally:
        await engine.dispose()


async def test_shape_upgrade_reset_wipes_stale_sync_state(tmp_path):
    """A shape-upgrade load wipes extension_sync_state along with every other
    loader-owned table, so no stale lease/hash/revision from the prior shape
    survives the rebuild — the new load re-seeds the singleton pristine."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    db = _synthetic_db(tmp_path)
    prior = build_scenario(seed=4242, profile="smoke")
    await load_scenario(prior, db)
    # Pollute the pristine row with stale coordinator-style fields, as if a
    # prior run had reconciled, leased, and failed.
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE extension_sync_state SET desired_revision = 99, "
                    "applied_revision = 98, force_reload = 0, failure_count = 5, "
                    "last_remote_hash = 'stalehash', lease_owner = 'stale-owner', "
                    "lease_token = 'stale-token', "
                    "lease_expires_at = '2035-01-01', "
                    "diagnosis_state = 'fatal', next_attempt_at = '2035-01-01' "
                    "WHERE extension_id = 'paisa'"
                )
            )
    finally:
        await engine.dispose()

    # Shape upgrade: different seed → reset + rebuild.
    current = build_scenario(seed=9999, profile="smoke")
    await load_scenario(current, db)
    await _assert_db_matches_scenario(db, current)

    # The rebuild re-seeded the pristine row — no stale fields survive.
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.connect() as conn:
            r = (
                await conn.execute(
                    text(
                        "SELECT desired_revision, applied_revision, force_reload, "
                        "failure_count, last_remote_hash, last_healthy_hash, "
                        "lease_owner, lease_token, lease_expires_at, "
                        "diagnosis_state, next_attempt_at "
                        "FROM extension_sync_state WHERE extension_id = 'paisa'"
                    )
                )
            ).one()
            assert tuple(r) == (1, 0, 1, 0, None, None, None, None, None, None, None)
    finally:
        await engine.dispose()


async def test_restamp_erases_trigger_revision_churn(tmp_path):
    """If SQLite triggers are active on this DB (e.g. init_db was invoked on
    it), the structural + bulk writes bump desired_revision non-deterministically.
    The loader's final re-stamp must force the row back to pristine so the
    seed's recorded state is deterministic regardless of trigger presence.

    This test simulates the churn directly (many bumps + coordinator pollution),
    then re-loads the same shape and asserts the re-stamp reset everything to
    canonical pristine values."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)

    # Simulate trigger-driven churn: many bumps + coordinator pollution that
    # a real coordinator run (or init_db's triggers firing on subsequent writes)
    # would leave behind.
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.begin() as conn:
            for _ in range(50):
                await conn.execute(
                    text(
                        "UPDATE extension_sync_state SET "
                        "desired_revision = desired_revision + 1 "
                        "WHERE extension_id = 'paisa'"
                    )
                )
            await conn.execute(
                text(
                    "UPDATE extension_sync_state SET applied_revision = 7, "
                    "force_reload = 0, failure_count = 4, "
                    "last_remote_hash = 'churned', last_healthy_hash = 'wasok', "
                    "lease_owner = 'coord', lease_token = 'tok', "
                    "lease_expires_at = '2035-01-01', "
                    "diagnosis_state = 'accepted', next_attempt_at = '2035-01-01' "
                    "WHERE extension_id = 'paisa'"
                )
            )
            before = (
                await conn.execute(
                    text(
                        "SELECT desired_revision, applied_revision, force_reload, "
                        "failure_count FROM extension_sync_state "
                        "WHERE extension_id = 'paisa'"
                    )
                )
            ).one()
            assert tuple(before) == (51, 7, 0, 4)
    finally:
        await engine.dispose()

    # Same-shape rerun (identity matches → no reset): the final re-stamp must
    # erase all churn and pollution.
    await load_scenario(scenario, db)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.connect() as conn:
            r = (
                await conn.execute(
                    text(
                        "SELECT desired_revision, applied_revision, force_reload, "
                        "failure_count, last_remote_hash, last_healthy_hash, "
                        "lease_owner, lease_token, lease_expires_at, "
                        "diagnosis_state, next_attempt_at "
                        "FROM extension_sync_state WHERE extension_id = 'paisa'"
                    )
                )
            ).one()
            assert tuple(r) == (1, 0, 1, 0, None, None, None, None, None, None, None)
    finally:
        await engine.dispose()


async def test_init_db_then_load_leaves_pristine_sync_state(tmp_path, monkeypatch):
    """When production init_db is invoked on the synthetic DB (seeding the
    singleton + installing the SQLite AFTER triggers) BEFORE the loader runs,
    the loader's structural + bulk writes bump desired_revision on every insert.
    The loader's final re-stamp must still leave the row pristine, so the seed
    is deterministic even under trigger-active conditions.

    This is the real path: it exercises init_db's trigger installation (21
    AFTER triggers on the core tables + settings) and confirms the loader's
    re-stamp wins over the resulting churn."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    # init_db warms two runtime caches through the *application's* global engine
    # at its tail; stub them to no-ops so the schema/migration work runs against
    # the engine the test owns (mirrors tests/test_paisa_sync_state_schema.py).
    from financial_dashboard.services import settings as settings_mod
    from financial_dashboard.services.categorization import merchant_rules

    async def _noop():
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", _noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", _noop)

    from financial_dashboard.db.init_db import init_db

    db = _synthetic_db(tmp_path, "initdb.db")
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    await init_db(engine)  # seeds pristine row + installs triggers
    await engine.dispose()

    # The loader runs over the trigger-active DB. Identity is unstamped → reset
    # + full rebuild; every structural/bulk insert bumps desired_revision.
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)

    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.connect() as conn:
            # Triggers really are installed (the precondition for churn).
            tcount = (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
                        "AND name LIKE 'ext_sync_dirty_%'"
                    )
                )
            ).scalar_one()
            assert tcount > 0, "init_db installed no triggers; test setup is wrong"
            r = (
                await conn.execute(
                    text(
                        "SELECT desired_revision, applied_revision, force_reload, "
                        "failure_count, last_remote_hash, lease_owner, "
                        "diagnosis_state, next_attempt_at "
                        "FROM extension_sync_state WHERE extension_id = 'paisa'"
                    )
                )
            ).one()
            assert tuple(r) == (1, 0, 1, 0, None, None, None, None)
    finally:
        await engine.dispose()
