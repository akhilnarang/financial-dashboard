"""Database-level behavioral tests for the persisted Paisa revision triggers.

These exercise the persisted-revision foundation in ``ExtensionSyncState``
(table ``extension_sync_state``) and the row-level ``AFTER INSERT/UPDATE/
DELETE`` SQLite triggers that ``init_db`` installs:

* core tables (``transactions``, ``accounts``, ``cards``,
  ``balance_snapshots``, ``investment_lots``, ``cas_uploads``) each bump
  ``desired_revision`` by one per affected row, plus stamp the dirty
  timestamps;
* a ``settings`` trigger does the same **only for ``paisa.*`` keys** and also
  resets the retry/backoff fields (``failure_count``→0, ``next_attempt_at``→
  NULL, ``last_remote_attempt_at``→NULL) so a config fix is retried at once.

Every bump runs inside the triggering statement's transaction, so it is
isolated, rolls back with the outer txn (and with an enclosing SAVEPOINT), and
a no-op ``INSERT ... ON CONFLICT DO NOTHING`` that inserts nothing never fires
the AFTER trigger. These tests prove all of that with real SQLite transactions
on a WAL file DB — no network, no DB-layer mocks — using minimal deterministic
data.
"""

import datetime
from decimal import Decimal
from typing import NamedTuple

import pytest
from sqlalchemy import event, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    Card,
    CasUpload,
    ExtensionSyncState,
    InvestmentLot,
    Setting,
    Transaction,
)

pytestmark = pytest.mark.anyio

#: Singleton under test (matches the manifest id and the seeded row).
EXTENSION_ID = "paisa"

_TZ = datetime.UTC


class BackoffState(NamedTuple):
    """Snapshot of the retry/backoff fields read off the state row."""

    failure_count: int
    next_attempt_at: object  # datetime | None; stored shape is driver-dependent
    last_remote_attempt_at: object


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _revision(conn) -> int:
    """Read ``desired_revision`` on ``conn`` (honors its open transaction)."""
    row = (
        await conn.execute(
            text(
                "SELECT desired_revision FROM extension_sync_state "
                "WHERE extension_id = :eid"
            ),
            {"eid": EXTENSION_ID},
        )
    ).one()
    return int(row[0])


async def _backoff(conn) -> BackoffState:
    """Read the retry/backoff fields for the paisa row."""
    row = (
        await conn.execute(
            text(
                "SELECT failure_count, next_attempt_at, last_remote_attempt_at "
                "FROM extension_sync_state WHERE extension_id = :eid"
            ),
            {"eid": EXTENSION_ID},
        )
    ).one()
    return BackoffState(int(row[0]), row[1], row[2])


async def _committed_revision(engine) -> int:
    """Revision as seen by a *fresh* connection — i.e. the published value."""
    async with engine.connect() as c:
        return await _revision(c)


async def _set_backoff(
    conn,
    *,
    failure_count: int,
    next_attempt_at: str | None,
    last_remote_attempt_at: str | None,
) -> None:
    """Directly set the retry/backoff fields (no trigger touches this table)."""
    await conn.execute(
        text(
            "UPDATE extension_sync_state "
            "SET failure_count = :fc, next_attempt_at = :na, "
            "last_remote_attempt_at = :lra WHERE extension_id = :eid"
        ),
        {
            "fc": failure_count,
            "na": next_attempt_at,
            "lra": last_remote_attempt_at,
            "eid": EXTENSION_ID,
        },
    )


def _txn_row(i: int, bank: str = "hdfc") -> dict:
    """A minimal, valid Transaction dict with a distinct natural key."""
    return {
        "bank": bank,
        "email_type": "txn",
        "direction": "debit",
        "amount": Decimal("1.00"),
        "reference_number": f"ref-{bank}-{i}",
    }


async def _scalar(conn, sql: str):
    return (await conn.execute(text(sql))).scalar_one()


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """A WAL file DB with init_db schema + revision triggers, paisa row seeded.

    ``init_db`` seeds the Paisa singleton and installs the triggers; the
    defensive ``INSERT OR IGNORE`` below is a no-op once that row exists and a
    guarantee if it ever does not. ``init_db``'s tail warms caches through the
    application engine, so those steps are stubbed (same pattern as
    ``test_cc_statement_email_summary`` / ``test_transactions_category_index``).
    """
    from financial_dashboard.services import settings as settings_mod
    from financial_dashboard.services.categorization import merchant_rules

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", _noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", _noop)

    db_path = tmp_path / "paisa_sync_bulk.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    # WAL lets a reader observe the last committed snapshot while another
    # connection holds an open (uncommitted) write transaction — the exact
    # visibility property the cross-connection test asserts.
    @event.listens_for(engine.sync_engine, "connect")
    def _wal(dbapi_connection, _record):
        cur = dbapi_connection.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
        finally:
            cur.close()

    try:
        await init_db(engine)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT OR IGNORE INTO extension_sync_state "
                    "(extension_id, desired_revision, applied_revision, "
                    " force_reload) VALUES (:eid, 1, 0, 0)"
                ),
                {"eid": EXTENSION_ID},
            )
        yield engine
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- #
# Expected model contract
# --------------------------------------------------------------------------- #


def test_extension_sync_state_model_contract():
    """Pin the model/table/column names the SQL helpers rely on."""
    assert ExtensionSyncState.__tablename__ == "extension_sync_state"
    cols = {c.name for c in ExtensionSyncState.__table__.columns}
    for required in (
        "extension_id",
        "desired_revision",
        "applied_revision",
        "failure_count",
        "next_attempt_at",
        "last_remote_attempt_at",
    ):
        assert required in cols, required


# --------------------------------------------------------------------------- #
# Visibility / transaction isolation
# --------------------------------------------------------------------------- #


async def test_bulk_insert_invisible_until_commit_then_visible(db):
    """200 inserts in one outer txn bump desired_revision, but only after COMMIT
    is the bump visible to a different connection."""
    engine = db
    base = await _committed_revision(engine)
    rows = [_txn_row(i) for i in range(200)]

    async with engine.connect() as writer:
        await writer.execute(Transaction.__table__.insert(), rows)

        # The writer observes its own uncommitted bump (200 per-row triggers).
        assert await _revision(writer) == base + 200

        # A separate connection must see the prior committed snapshot only.
        async with engine.connect() as reader:
            assert await _revision(reader) == base

        await writer.commit()

    # Published: a fresh connection now sees the full bump.
    assert await _committed_revision(engine) == base + 200


async def test_outer_rollback_leaves_revision_unchanged(db):
    """Rolling back the outer transaction discards every per-row bump."""
    engine = db
    base = await _committed_revision(engine)
    rows = [_txn_row(i) for i in range(200)]

    async with engine.connect() as writer:
        await writer.execute(Transaction.__table__.insert(), rows)
        assert await _revision(writer) == base + 200  # bumped within the txn
        await writer.rollback()

    assert await _committed_revision(engine) == base  # nothing published


async def test_per_row_savepoint_failures_increment_only_successful_rows(db):
    """Each row runs in its own SAVEPOINT; a rolled-back failure does not count,
    only released successes accumulate into the revision."""
    engine = db
    base = await _committed_revision(engine)

    async with engine.connect() as conn:
        # 100 rows with fresh natural keys — all succeed.
        for i in range(100):
            async with conn.begin_nested():
                await conn.execute(Transaction.__table__.insert(), _txn_row(i))

        # 100 rows reusing those keys — each collides on the partial unique
        # index and fails inside its savepoint.
        failures = 0
        for i in range(100):
            try:
                async with conn.begin_nested():
                    await conn.execute(Transaction.__table__.insert(), _txn_row(i))
            except IntegrityError:
                failures += 1

        assert failures == 100
        await conn.commit()

    assert await _committed_revision(engine) == base + 100


# --------------------------------------------------------------------------- #
# Core bulk DML detection
# --------------------------------------------------------------------------- #


async def test_core_bulk_insert_update_delete_all_detected(db):
    """SQLAlchemy Core executemany INSERT, plus bulk UPDATE and DELETE, each
    fire the per-row triggers for every matched row."""
    engine = db
    base = await _committed_revision(engine)
    rows = [_txn_row(i, bank="bulk") for i in range(5)]

    # bulk INSERT (executemany) -> +5
    async with engine.connect() as conn:
        await conn.execute(Transaction.__table__.insert(), rows)
        await conn.commit()
    assert await _committed_revision(engine) == base + 5

    # bulk UPDATE over the 5 rows -> +5
    async with engine.connect() as conn:
        await conn.execute(
            Transaction.__table__.update()
            .where(Transaction.__table__.c.bank == "bulk")
            .values(category="updated")
        )
        await conn.commit()
    assert await _committed_revision(engine) == base + 10

    # bulk DELETE over the 5 rows -> +5
    async with engine.connect() as conn:
        await conn.execute(
            Transaction.__table__.delete().where(Transaction.__table__.c.bank == "bulk")
        )
        await conn.commit()
    assert await _committed_revision(engine) == base + 15


async def test_on_conflict_do_nothing_does_not_increment(db):
    """A conflicting INSERT that is ignored inserts no row, so the trigger never
    fires and the revision is unchanged."""
    engine = db
    base = await _committed_revision(engine)
    row = _txn_row(0, bank="conf")

    async with engine.connect() as conn:
        await conn.execute(sqlite_insert(Transaction.__table__).values(**row))
        await conn.commit()
    after_first = await _committed_revision(engine)
    assert after_first == base + 1  # exactly one row landed

    # Same natural key, ignored on conflict.
    async with engine.connect() as conn:
        await conn.execute(
            sqlite_insert(Transaction.__table__).values(**row).on_conflict_do_nothing()
        )
        await conn.commit()

    assert await _committed_revision(engine) == after_first  # no new row

    # Authoritative: the conflict was ignored, so still exactly one row.
    async with engine.connect() as conn:
        count = await _scalar(
            conn,
            "SELECT count(*) FROM transactions WHERE bank = 'conf'",
        )
    assert count == 1


# --------------------------------------------------------------------------- #
# Monotonicity
# --------------------------------------------------------------------------- #


async def test_two_sequential_commits_yield_distinct_monotonic_revisions(db):
    """Two separate committed transactions produce strictly increasing,
    distinct revisions."""
    engine = db
    r0 = await _committed_revision(engine)

    async with engine.connect() as conn:
        await conn.execute(Transaction.__table__.insert(), _txn_row(1, bank="seq"))
        await conn.commit()
    r1 = await _committed_revision(engine)

    async with engine.connect() as conn:
        await conn.execute(Transaction.__table__.insert(), _txn_row(2, bank="seq"))
        await conn.commit()
    r2 = await _committed_revision(engine)

    assert r1 == r0 + 1
    assert r2 == r1 + 1
    assert r1 != r2 and r2 > r1


# --------------------------------------------------------------------------- #
# Core source tables dirty state
# --------------------------------------------------------------------------- #


async def test_account_card_snapshot_investment_cas_mutations_dirty_state(db):
    """A mutation on any of the five core source tables bumps the revision."""
    engine = db
    base = await _committed_revision(engine)

    # accounts
    async with engine.connect() as conn:
        await conn.execute(
            Account.__table__.insert().values(
                bank="hdfc", label="bank", type="bank_account"
            )
        )
        await conn.commit()
    assert await _committed_revision(engine) == base + 1

    # cards (needs an account)
    async with engine.connect() as conn:
        account_id = await _scalar(
            conn, "SELECT id FROM accounts ORDER BY id DESC LIMIT 1"
        )
        await conn.execute(
            Card.__table__.insert().values(account_id=account_id, card_mask="****1111")
        )
        await conn.commit()
    assert await _committed_revision(engine) == base + 2

    # cas_uploads
    async with engine.connect() as conn:
        await conn.execute(
            CasUpload.__table__.insert().values(
                portfolio_key="pf1",
                depository_source="cdsl",
                statement_date=datetime.date(2026, 1, 1),
                grand_total=Decimal("0"),
                raw_holdings_json="{}",
            )
        )
        await conn.commit()
    assert await _committed_revision(engine) == base + 3

    # investment_lots (needs a cas upload)
    async with engine.connect() as conn:
        cas_id = await _scalar(
            conn, "SELECT id FROM cas_uploads ORDER BY id DESC LIMIT 1"
        )
        await conn.execute(
            InvestmentLot.__table__.insert().values(
                cas_upload_id=cas_id,
                instrument_id="INE000A01012",
                instrument_name="Example Fund",
                quantity=Decimal("1"),
                unit_cost=Decimal("1"),
                cost_basis=Decimal("1"),
                currency="INR",
                acquired_on=datetime.date(2026, 1, 1),
                source_ref="src-1",
            )
        )
        await conn.commit()
    assert await _committed_revision(engine) == base + 4

    # balance_snapshots (account-scoped; exactly-one-source CHECK satisfied)
    async with engine.connect() as conn:
        account_id = await _scalar(
            conn, "SELECT id FROM accounts ORDER BY id DESC LIMIT 1"
        )
        await conn.execute(
            BalanceSnapshot.__table__.insert().values(
                account_id=account_id,
                kind="balance",
                category="bank",
                as_of_date=datetime.date(2026, 1, 1),
                value=Decimal("0"),
                source="manual",
            )
        )
        await conn.commit()
    assert await _committed_revision(engine) == base + 5


# --------------------------------------------------------------------------- #
# Settings trigger scoping
# --------------------------------------------------------------------------- #


async def test_non_paisa_setting_does_not_dirty_state(db):
    """Inserting or updating a non-paisa setting key leaves the revision alone."""
    engine = db
    base = await _committed_revision(engine)

    async with engine.connect() as conn:
        await conn.execute(
            Setting.__table__.insert().values(key="telegram.enabled", value="true")
        )
        await conn.commit()
    assert await _committed_revision(engine) == base

    async with engine.connect() as conn:
        await conn.execute(
            Setting.__table__.update()
            .where(Setting.__table__.c.key == "telegram.enabled")
            .values(value="false")
        )
        await conn.commit()
    assert await _committed_revision(engine) == base


async def test_paisa_setting_change_dirty_and_clears_backoff(db):
    """A paisa.* setting change bumps desired_revision AND resets the backoff
    fields (failure_count, next_attempt_at, last_remote_attempt_at)."""
    engine = db

    # Seed a backoff so clearing is observable. This direct write to the state
    # row does not pass through any trigger, so it must not bump the revision.
    future = datetime.datetime(2026, 12, 31, tzinfo=_TZ).isoformat()
    past = datetime.datetime(2026, 1, 1, tzinfo=_TZ).isoformat()
    async with engine.begin() as conn:
        await _set_backoff(
            conn,
            failure_count=3,
            next_attempt_at=future,
            last_remote_attempt_at=past,
        )
    async with engine.connect() as conn:
        seeded = await _backoff(conn)
    assert seeded.failure_count == 3
    assert seeded.next_attempt_at is not None
    assert seeded.last_remote_attempt_at is not None

    base = await _committed_revision(engine)

    async with engine.connect() as conn:
        await conn.execute(
            Setting.__table__.insert().values(key="paisa.mode", value="connect")
        )
        await conn.commit()

    assert await _committed_revision(engine) == base + 1  # dirtied

    async with engine.connect() as conn:
        cleared = await _backoff(conn)
    assert cleared.failure_count == 0
    assert cleared.next_attempt_at is None
    assert cleared.last_remote_attempt_at is None
