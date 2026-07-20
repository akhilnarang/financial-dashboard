"""Schema + migration tests for the persisted Paisa sync-state foundation.

Covers:

* the ``extension_sync_state`` table is created by ``create_all`` with every
  refined-architecture column, sensible defaults, and the four invariants
  (non-negative revisions/failure_count, ``desired >= applied``);
* the Paisa singleton is seeded dirty + ``force_reload=true`` on fresh and
  legacy init so the next enabled coordinator reconciles once;
* ``init_db`` is idempotent — a coordinator-owned row is authoritative and
  is never re-dirtied by re-running migration;
* the SQLite triggers are installed on every relevant table/verb (and only
  those — never on ``extension_sync_state``/``extension_runs``);
* each trigger bumps ``desired_revision`` and manages
  ``first_dirty_at``/``last_dirty_at``;
* the settings trigger fires only for ``paisa.%`` keys and additionally
  resets the retry backoff;
* a rolled-back write (outer transaction or SAVEPOINT) leaves
  ``desired_revision`` untouched;
* ``INSERT ... ON CONFLICT DO NOTHING`` / ``INSERT OR IGNORE`` that inserts
  no row does not bump (AFTER trigger semantics);
* a direct write to ``extension_sync_state`` never recurses.

``init_db`` warms two runtime caches through the *application's* global engine
at its tail; those steps are monkeypatched to no-ops here so the schema and
migration work — the parts under test — run against an engine the test owns.
"""

import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import Base, ExtensionSyncState

pytestmark = pytest.mark.anyio


# Every core table whose INSERT/UPDATE/DELETE must bump desired_revision.
_TRIGGERED_TABLES = (
    "transactions",
    "accounts",
    "cards",
    "balance_snapshots",
    "investment_lots",
    "cas_uploads",
)

# The refined-architecture field set the table must carry.
_REQUIRED_COLUMNS = {
    "extension_id",
    "desired_revision",
    "applied_revision",
    "first_dirty_at",
    "last_dirty_at",
    "last_published_hash",
    "last_remote_hash",
    "last_healthy_hash",
    "last_remote_attempt_at",
    "next_attempt_at",
    "failure_count",
    "diagnosis_state",
    "force_reload",
    "lease_owner",
    "lease_token",
    "lease_expires_at",
    "created_at",
    "updated_at",
}


@pytest.fixture(autouse=True)
def _stub_runtime_cache_warming(monkeypatch):
    """Stub the two tail steps of init_db that use the global engine."""
    from financial_dashboard.services import settings as settings_mod
    from financial_dashboard.services.categorization import merchant_rules

    async def _noop():
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", _noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", _noop)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _engine(tmp_path, name="t.db"):
    return create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}")


async def _state(engine) -> ExtensionSyncState:
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        row = await session.get(ExtensionSyncState, "paisa")
        assert row is not None, "paisa singleton missing"
        return row


async def _set_state(
    engine, *, failure_count=0, next_attempt_at=None, last_remote_attempt_at=None
) -> None:
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        row = await session.get(ExtensionSyncState, "paisa")
        assert row is not None
        row.failure_count = failure_count
        row.next_attempt_at = next_attempt_at
        row.last_remote_attempt_at = last_remote_attempt_at
        await session.commit()


# --------------------------------------------------------------------------- #
# Table schema
# --------------------------------------------------------------------------- #


def test_model_has_required_columns():
    cols = {c.name for c in ExtensionSyncState.__table__.columns}
    assert cols == _REQUIRED_COLUMNS


def test_model_check_constraints_and_indexes_present():
    constraint_names = {
        c.name for c in ExtensionSyncState.__table__.constraints if c.name
    }
    for name in (
        "ck_extension_sync_state_desired_nonneg",
        "ck_extension_sync_state_applied_nonneg",
        "ck_extension_sync_state_failure_nonneg",
        "ck_extension_sync_state_desired_gte_applied",
    ):
        assert name in constraint_names, name
    index_names = {i.name for i in ExtensionSyncState.__table__.indexes}
    assert "ix_extension_sync_state_next_attempt" in index_names
    assert "ix_extension_sync_state_lease_expires" in index_names


async def test_create_all_builds_table_with_typed_defaults(tmp_path):
    """Fresh schema via create_all carries every column with the right
    nullability + server defaults so the Paisa singleton inserts cleanly."""
    engine = _engine(tmp_path)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with engine.begin() as conn:
            info = {
                row[1]: row
                for row in (
                    await conn.execute(text("PRAGMA table_info(extension_sync_state)"))
                )
            }
        for name in _REQUIRED_COLUMNS:
            assert name in info, f"{name} missing"
        # extension_id is the PK (pk flag index 5 == 1).
        assert info["extension_id"][5] == 1
        # NOT NULL columns the coordinator / triggers rely on.
        for notnull in (
            "desired_revision",
            "applied_revision",
            "failure_count",
            "force_reload",
        ):
            assert info[notnull][3] == 1, f"{notnull} should be NOT NULL"
            # server-side default exists so a raw trigger UPDATE on a missing
            # row never has to fall back to NULL.
            assert info[notnull][4] is not None, f"{notnull} needs a server default"
        # nullable tracking columns.
        for nullable in (
            "first_dirty_at",
            "last_dirty_at",
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
            assert info[nullable][3] == 0, f"{nullable} should be nullable"
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# Singleton seeding — fresh + legacy + idempotent
# --------------------------------------------------------------------------- #


async def test_paisa_singleton_seeded_dirty_force_reload_on_fresh_init(tmp_path):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        row = await _state(engine)
        assert row.desired_revision == 1
        assert row.applied_revision == 0
        assert row.force_reload is True
        assert row.failure_count == 0
        assert row.first_dirty_at is not None
        assert row.last_dirty_at is not None
        assert row.created_at is not None
    finally:
        await engine.dispose()


async def test_legacy_db_without_state_table_gains_table_and_singleton(tmp_path):
    """A DB built from create_all then stripped of extension_sync_state
    (simulating a pre-migration deployment) gets the table + dirty singleton
    on the first init_db run."""
    engine = _engine(tmp_path, "legacy.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("DROP TABLE extension_sync_state"))
        async with engine.begin() as conn:
            exists = (
                await conn.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name='extension_sync_state'"
                    )
                )
            ).first()
            assert exists is None

        await init_db(engine)
        row = await _state(engine)
        assert row.desired_revision == 1
        assert row.applied_revision == 0
        assert row.force_reload is True
    finally:
        await engine.dispose()


async def test_init_db_idempotent_does_not_redirty_coordinator_row(tmp_path):
    """Re-running init_db must not overwrite a coordinator-managed row —
    ON CONFLICT DO NOTHING leaves a reconciled row reconciled."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        # Simulate a successful reconcile: clear force_reload, advance applied.
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as session:
            row = await session.get(ExtensionSyncState, "paisa")
            assert row is not None
            row.force_reload = False
            row.applied_revision = row.desired_revision
            row.first_dirty_at = None
            await session.commit()

        await init_db(engine)
        row = await _state(engine)
        assert row.force_reload is False
        assert row.applied_revision == row.desired_revision
        assert row.first_dirty_at is None
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# Trigger presence / idempotency
# --------------------------------------------------------------------------- #


async def _trigger_names(engine) -> set[str]:
    async with engine.connect() as conn:
        return {
            r[0]
            for r in (
                await conn.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='trigger' "
                        "AND name LIKE 'ext_sync_dirty_%'"
                    )
                )
            )
        }


async def test_triggers_present_on_every_relevant_table_verb(tmp_path):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        names = await _trigger_names(engine)
        for table in _TRIGGERED_TABLES:
            for verb in ("insert", "update", "delete"):
                assert f"ext_sync_dirty_{table}_{verb}" in names
        for verb in ("insert", "update", "delete"):
            assert f"ext_sync_dirty_settings_{verb}" in names
        assert len(names) == len(_TRIGGERED_TABLES) * 3 + 3
    finally:
        await engine.dispose()


async def test_no_triggers_on_extension_state_tables(tmp_path):
    """Recursion guard: no trigger exists on extension_sync_state or
    extension_runs, so coordinator writes never bump themselves."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        names = await _trigger_names(engine)
        assert not any("ext_sync_dirty_extension_sync_state" in n for n in names)
        assert not any("ext_sync_dirty_extension_runs" in n for n in names)
        # And no trigger on tables the projection does not read.
        for excluded in (
            "emails",
            "fetch_rules",
            "email_sources",
            "sms_messages",
            "merchant_rules",
            "categories",
        ):
            assert not any(f"ext_sync_dirty_{excluded}_" in n for n in names)
    finally:
        await engine.dispose()


async def test_trigger_installation_idempotent_on_reinit(tmp_path):
    """init_db runs on every boot; trigger count must stay stable."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        first = await _trigger_names(engine)
        await init_db(engine)
        second = await _trigger_names(engine)
        assert first == second
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# Trigger effects — bump + dirty timestamps + backoff reset
# --------------------------------------------------------------------------- #


def _row_sql_for(table: str) -> tuple[str, dict]:
    """Return (INSERT sql, params) for one minimal row on each table.

    Raw-SQL inserts bypass the ORM, so columns that only have a Python-side
    ``default=`` (no ``server_default``) must be supplied explicitly here.
    """
    if table == "transactions":
        return (
            "INSERT INTO transactions (bank, email_type, direction, amount) "
            "VALUES ('hdfc', 'debit', 'OUT', 10.00)",
            {},
        )
    if table == "accounts":
        return (
            "INSERT INTO accounts (bank, label, type) "
            "VALUES ('hdfc', 'x', 'bank_account')",
            {},
        )
    if table == "cards":
        return (
            "INSERT INTO cards (account_id, card_mask) VALUES (1, '1234')",
            {},
        )
    if table == "balance_snapshots":
        return (
            "INSERT INTO balance_snapshots "
            "(account_id, kind, category, as_of_date, value, source, currency, "
            " created_at) "
            "VALUES (1, 'asset', 'bank_balance', '2026-01-01', 100.00, "
            "'bank_statement', 'INR', CURRENT_TIMESTAMP)",
            {},
        )
    if table == "investment_lots":
        return (
            "INSERT INTO investment_lots "
            "(cas_upload_id, instrument_id, instrument_name, quantity, unit_cost, "
            " cost_basis, currency, acquired_on, source_ref, created_at) "
            "VALUES (1, 'ISIN1', 'Fund', 10.0, 10.0, 100.0, 'INR', '2026-01-01', "
            " 'r1', CURRENT_TIMESTAMP)",
            {},
        )
    if table == "cas_uploads":
        return (
            "INSERT INTO cas_uploads "
            "(portfolio_key, depository_source, statement_date, grand_total, "
            " raw_holdings_json, portfolio_ok, created_at) "
            "VALUES ('pf1', 'cdsl', '2026-01-01', 100.00, '{}', 1, "
            " CURRENT_TIMESTAMP)",
            {},
        )
    raise AssertionError(table)


@pytest.mark.parametrize("table", _TRIGGERED_TABLES)
@pytest.mark.parametrize("verb", ["insert", "update", "delete"])
async def test_trigger_bumps_desired_revision_on_table_verb(tmp_path, table, verb):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        before = (await _state(engine)).desired_revision

        insert_sql, params = _row_sql_for(table)
        async with engine.begin() as conn:
            await conn.execute(text(insert_sql), params)
        after_insert = (await _state(engine)).desired_revision
        if verb == "insert":
            assert after_insert == before + 1
            return

        # UPDATE bumps too.
        async with engine.begin() as conn:
            await conn.execute(text(f"UPDATE {table} SET id = id WHERE id = 1"))
        after_update = (await _state(engine)).desired_revision
        if verb == "update":
            assert after_update == after_insert + 1
            return

        # DELETE bumps too.
        async with engine.begin() as conn:
            await conn.execute(text(f"DELETE FROM {table} WHERE id = 1"))
        after_delete = (await _state(engine)).desired_revision
        assert after_delete == after_update + 1
    finally:
        await engine.dispose()


async def test_bump_sets_first_dirty_only_once_and_advances_last(tmp_path):
    """first_dirty_at is set on the first dirtying write, then held; last_dirty_at
    advances on every subsequent one."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        # The migration seed already set first_dirty_at/last_dirty_at; clear them
        # so the trigger behavior is observable in isolation.
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as session:
            row = await session.get(ExtensionSyncState, "paisa")
            assert row is not None
            row.first_dirty_at = None
            row.last_dirty_at = None
            await session.commit()

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO accounts (bank, label, type) "
                    "VALUES ('hdfc', 'a', 'bank_account')"
                )
            )
        first = await _state(engine)
        assert first.first_dirty_at is not None
        assert first.last_dirty_at is not None
        first_first = first.first_dirty_at

        # Subsequent bump: first_dirty_at held, last_dirty_at advances.
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO accounts (bank, label, type) "
                    "VALUES ('icici', 'b', 'bank_account')"
                )
            )
        second = await _state(engine)
        assert second.first_dirty_at == first_first
        assert second.last_dirty_at >= first.last_dirty_at
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# Settings: key filter + backoff reset
# --------------------------------------------------------------------------- #


async def test_settings_trigger_skips_non_paisa_keys(tmp_path):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        before = (await _state(engine)).desired_revision
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) "
                    "VALUES ('telegram.enabled', 'true')"
                )
            )
        after = (await _state(engine)).desired_revision
        assert after == before
    finally:
        await engine.dispose()


async def test_settings_trigger_bumps_on_paisa_keys(tmp_path):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        before = (await _state(engine)).desired_revision
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) VALUES ('paisa.mode', 'connect')"
                )
            )
        after = (await _state(engine)).desired_revision
        assert after == before + 1
    finally:
        await engine.dispose()


async def test_settings_change_resets_retry_backoff(tmp_path):
    """A paisa.% settings change clears failure_count / next_attempt_at /
    last_remote_attempt_at so the config fix is retried immediately."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        future = datetime.datetime(2035, 1, 1)
        past = datetime.datetime(2020, 1, 1)
        await _set_state(
            engine,
            failure_count=7,
            next_attempt_at=future,
            last_remote_attempt_at=past,
        )

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) VALUES ('paisa.mode', 'connect')"
                )
            )
        row = await _state(engine)
        assert row.failure_count == 0
        assert row.next_attempt_at is None
        assert row.last_remote_attempt_at is None
    finally:
        await engine.dispose()


async def test_non_paisa_settings_change_does_not_reset_backoff(tmp_path):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        # SQLite stores datetimes as naive TEXT; use naive values so the
        # round-trip read-back compares equal.
        future = datetime.datetime(2035, 1, 1)
        past = datetime.datetime(2020, 1, 1)
        await _set_state(
            engine,
            failure_count=7,
            next_attempt_at=future,
            last_remote_attempt_at=past,
        )

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) "
                    "VALUES ('telegram.enabled', 'true')"
                )
            )
        row = await _state(engine)
        assert row.failure_count == 7
        assert row.next_attempt_at == future
        assert row.last_remote_attempt_at == past
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# Transaction / SAVEPOINT rollback + INSERT conflict-no-op
# --------------------------------------------------------------------------- #


async def test_outer_transaction_rollback_undoes_bump(tmp_path):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        before = (await _state(engine)).desired_revision
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO accounts (bank, label, type) "
                        "VALUES ('hdfc', 'rolledback', 'bank_account')"
                    )
                )
                raise RuntimeError("intentional rollback")
        except RuntimeError:
            pass
        after = (await _state(engine)).desired_revision
        assert after == before
    finally:
        await engine.dispose()


async def test_savepoint_rollback_undoes_only_inner_bump(tmp_path):
    """A rolled-back SAVEPOINT drops only its own bump; the outer write's
    bump survives the surrounding commit."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        before = (await _state(engine)).desired_revision

        async with engine.begin() as conn:
            # Outer write — bump must survive commit.
            await conn.execute(
                text(
                    "INSERT INTO accounts (bank, label, type) "
                    "VALUES ('hdfc', 'outer', 'bank_account')"
                )
            )
            # Inner SAVEPOINT — bump must be discarded on rollback. The
            # exception is contained so the outer transaction commits.
            try:
                async with conn.begin_nested():
                    await conn.execute(
                        text(
                            "INSERT INTO accounts (bank, label, type) "
                            "VALUES ('icici', 'inner', 'bank_account')"
                        )
                    )
                    raise RuntimeError("rollback savepoint")
            except RuntimeError:
                pass

        after = (await _state(engine)).desired_revision
        # Only the outer write survived → exactly one bump.
        assert after == before + 1
    finally:
        await engine.dispose()


async def test_insert_or_ignore_conflict_does_not_bump(tmp_path):
    """INSERT OR IGNORE on an existing row inserts nothing → AFTER trigger
    does not fire → desired_revision is unchanged."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) "
                    "VALUES ('paisa.mode', 'disabled')"
                )
            )
        before = (await _state(engine)).desired_revision

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT OR IGNORE INTO settings (key, value) "
                    "VALUES ('paisa.mode', 'connect')"
                )
            )
        after = (await _state(engine)).desired_revision
        assert after == before
    finally:
        await engine.dispose()


async def test_on_conflict_do_nothing_does_not_bump(tmp_path):
    """INSERT ... ON CONFLICT DO NOTHING on an existing row likewise does
    not bump."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) "
                    "VALUES ('paisa.mode', 'disabled')"
                )
            )
        before = (await _state(engine)).desired_revision

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) "
                    "VALUES ('paisa.mode', 'connect') "
                    "ON CONFLICT(key) DO NOTHING"
                )
            )
        after = (await _state(engine)).desired_revision
        assert after == before
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# No recursion on direct state writes
# --------------------------------------------------------------------------- #


async def test_direct_sync_state_update_does_not_recurse(tmp_path):
    """A coordinator UPDATE on extension_sync_state must not fire any trigger
    (there is none on it) and must not bump desired_revision."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        before = (await _state(engine)).desired_revision
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE extension_sync_state SET force_reload = 0, "
                    "applied_revision = desired_revision "
                    "WHERE extension_id = 'paisa'"
                )
            )
        after = await _state(engine)
        assert after.desired_revision == before
        assert after.force_reload is False
        assert after.applied_revision == before
    finally:
        await engine.dispose()


async def test_extension_run_write_does_not_bump(tmp_path):
    """An audit write to extension_runs is not a projection input and must
    not dirty the state."""
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        before = (await _state(engine)).desired_revision
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO extension_runs "
                    "(extension_id, operation, status, started_at) "
                    "VALUES ('paisa', 'probe', 'success', CURRENT_TIMESTAMP)"
                )
            )
        after = (await _state(engine)).desired_revision
        assert after == before
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# CHECK constraints
# --------------------------------------------------------------------------- #


async def test_desired_gte_applied_constraint_enforced(tmp_path):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        with pytest.raises(Exception):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "UPDATE extension_sync_state SET applied_revision = 999 "
                        "WHERE extension_id = 'paisa'"
                    )
                )
    finally:
        await engine.dispose()


async def test_failure_count_nonneg_constraint_enforced(tmp_path):
    engine = _engine(tmp_path)
    try:
        await init_db(engine)
        with pytest.raises(Exception):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "UPDATE extension_sync_state SET failure_count = -1 "
                        "WHERE extension_id = 'paisa'"
                    )
                )
    finally:
        await engine.dispose()
