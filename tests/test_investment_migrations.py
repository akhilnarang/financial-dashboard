"""Inline investment schema migration and historical CAS lot backfill."""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import (
    Base,
    CasUpload,
    InvestmentLot,
    SnapshotHolding,
)
from financial_dashboard.services.investments import create_investment_lots

pytestmark = pytest.mark.anyio

#: The new nullable investment-detail columns the migration must add.
_NEW_COLUMNS = (
    "instrument_id",
    "quantity",
    "unit_price",
    "currency",
    "cost_basis",
    "acquired_on",
)

# The original SnapshotHolding schema, before Phase 4 added investment detail.
_OLD_SNAPSHOT_HOLDINGS = """
CREATE TABLE snapshot_holdings (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL,
    asset_class VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    value NUMERIC(16,2) NOT NULL
)
"""

#: The ALTER statements ``init_db`` runs for the new columns (mirrors the
#: migration block so the column types/nullable behavior is pinned).
_ALTER_TYPES = {
    "instrument_id": "VARCHAR",
    "quantity": "NUMERIC(20,6)",
    "unit_price": "NUMERIC(20,6)",
    "currency": "VARCHAR(3)",
    "cost_basis": "NUMERIC(18,4)",
    "acquired_on": "DATE",
}


def _stub_init_caches(monkeypatch):
    from financial_dashboard.services import settings as settings_mod
    from financial_dashboard.services.categorization import merchant_rules

    async def _noop():
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", _noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", _noop)


def _purchase_payload() -> dict:
    return {
        "transactions": [
            {
                "scope": "mf",
                "source_ref": "folio/1",
                "date": "2025-01-15",
                "description": "Legacy Fund",
                "isin": "INE000A01018",
                "transaction_type": "purchase",
                "units": "10",
                "nav": "50",
                "amount": "500",
                "reference": "LEGACY-1",
            }
        ]
    }


async def test_model_create_all_has_nullable_investment_columns(tmp_path):
    """A fresh schema built by ``create_all`` carries the new columns, all
    nullable, so existing aggregated holdings insert with them NULL."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/fresh.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with engine.begin() as conn:
            info = {
                row[1]: row
                for row in (
                    await conn.execute(text("PRAGMA table_info(snapshot_holdings)"))
                )
            }
            for col in _NEW_COLUMNS:
                assert col in info, f"{col} missing from snapshot_holdings"
                # notnull flag (index 3) must be 0 -> nullable.
                assert info[col][3] == 0, f"{col} should be nullable"
    finally:
        await engine.dispose()


async def test_old_style_aggregated_holding_inserts_with_null_detail(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/agg.db")
    maker = async_sessionmaker(engine)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with maker() as session:
            session.add(
                SnapshotHolding(
                    snapshot_id=0,  # FK off in tests; just exercises columns
                    asset_class="equity",
                    label="Equity",
                    value=Decimal("100.00"),
                )
            )
            await session.flush()
            # The new columns are NULL on an aggregated holding — no fabrication.
            row = await session.get(SnapshotHolding, 1)
            assert row.instrument_id is None
            assert row.quantity is None
            assert row.unit_price is None
            assert row.cost_basis is None
            assert row.acquired_on is None
            assert row.currency is None
    finally:
        await engine.dispose()


async def test_investment_lots_table_created_by_create_all(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/lots.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with engine.begin() as conn:
            cols = {
                row[1]
                for row in (
                    await conn.execute(text("PRAGMA table_info(investment_lots)"))
                )
            }
            assert {
                "cas_upload_id",
                "instrument_id",
                "instrument_name",
                "quantity",
                "unit_cost",
                "cost_basis",
                "currency",
                "acquired_on",
                "source_ref",
                "transaction_type",
                "reference",
                "source_occurrence",
            } <= cols
    finally:
        await engine.dispose()


async def test_alter_migration_adds_nullable_columns_to_old_schema(tmp_path):
    """A pre-Phase-4 database gains the columns via the same ALTER statements
    ``init_db`` runs; existing rows are preserved and the columns are nullable."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/old.db")
    try:
        async with engine.begin() as conn:
            await conn.execute(text(_OLD_SNAPSHOT_HOLDINGS))
            # An existing aggregated row must survive the migration.
            await conn.execute(
                text(
                    "INSERT INTO snapshot_holdings (snapshot_id, asset_class, label, value) "
                    "VALUES (1, 'equity', 'Equity', 100)"
                )
            )
            for col, typ in _ALTER_TYPES.items():
                await conn.execute(
                    text(f"ALTER TABLE snapshot_holdings ADD COLUMN {col} {typ}")
                )

        async with engine.begin() as conn:
            info = {
                row[1]: row
                for row in (
                    await conn.execute(text("PRAGMA table_info(snapshot_holdings)"))
                )
            }
            for col in _NEW_COLUMNS:
                assert col in info
                assert info[col][3] == 0  # nullable
            # The pre-existing row is intact and the new columns are NULL.
            row = (
                await conn.execute(
                    text(
                        "SELECT asset_class, label, value, instrument_id, quantity, "
                        "cost_basis, acquired_on FROM snapshot_holdings WHERE id = 1"
                    )
                )
            ).one()
            assert row.asset_class == "equity"
            assert row.label == "Equity"
            assert row.value == 100
            assert row.instrument_id is None
            assert row.quantity is None
            assert row.cost_basis is None
            assert row.acquired_on is None
    finally:
        await engine.dispose()


async def test_fresh_init_has_occurrence_schema_and_completed_backfill_marker(
    tmp_path, monkeypatch
):
    _stub_init_caches(monkeypatch)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/fresh-init.db")
    try:
        await init_db(engine)
        async with engine.connect() as conn:
            columns = {
                row[1]
                for row in (
                    await conn.execute(text("PRAGMA table_info(investment_lots)"))
                )
            }
            marker = (
                await conn.execute(
                    text(
                        "SELECT value FROM settings WHERE key = "
                        "'migrations.investment_lots_backfill_v1'"
                    )
                )
            ).scalar_one()
        assert "source_occurrence" in columns
        assert marker == "1"
    finally:
        await engine.dispose()


async def test_legacy_settings_gains_updated_at_before_cas_backfill_marker(
    tmp_path, monkeypatch
):
    """A key/value-only legacy settings table must not break the CAS marker.

    ``create_all`` does not ALTER existing tables, so init_db has to add the ORM
    bookkeeping column before the investment backfill reads/writes its marker.
    The second run pins idempotency.
    """
    _stub_init_caches(monkeypatch)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/legacy-settings.db")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
            )
            await conn.execute(
                text("INSERT INTO settings (key, value) VALUES ('legacy.key', 'kept')")
            )

        await init_db(engine)
        await init_db(engine)

        async with engine.connect() as conn:
            columns = {
                row[1]
                for row in (await conn.execute(text("PRAGMA table_info(settings)")))
            }
            rows = dict(
                (await conn.execute(text("SELECT key, value FROM settings"))).all()
            )
        assert "updated_at" in columns
        assert rows["legacy.key"] == "kept"
        assert rows["migrations.investment_lots_backfill_v1"] == "1"
    finally:
        await engine.dispose()


async def test_legacy_cas_payloads_backfill_once_and_isolate_malformed_json(
    tmp_path, monkeypatch, caplog
):
    _stub_init_caches(monkeypatch)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/legacy-backfill.db")
    try:
        # Simulate the deployed schema immediately before investment_lots:
        # existing CAS rows survive, but create_all must create the lot table.
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("DROP TABLE investment_lots"))
            await conn.execute(
                text(
                    "INSERT INTO cas_uploads "
                    "(id, portfolio_key, depository_source, statement_date, "
                    " grand_total, portfolio_ok, raw_holdings_json, created_at) "
                    "VALUES "
                    "(1, 'PAN-VALID', 'cdsl', '2025-01-31', 500, 1, :valid, "
                    " CURRENT_TIMESTAMP), "
                    "(2, 'PAN-BAD', 'nsdl', '2025-02-28', 100, 1, :bad, "
                    " CURRENT_TIMESTAMP)"
                ),
                {"valid": json.dumps(_purchase_payload()), "bad": "{broken"},
            )

        with caplog.at_level("WARNING"):
            await init_db(engine)
        await init_db(engine)

        maker = async_sessionmaker(engine)
        async with maker() as session:
            lots = (
                (
                    await session.execute(
                        select(InvestmentLot).order_by(InvestmentLot.id)
                    )
                )
                .scalars()
                .all()
            )
            marker_count = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM settings WHERE key = "
                        "'migrations.investment_lots_backfill_v1'"
                    )
                )
            ).scalar_one()
        assert len(lots) == 1
        assert lots[0].cas_upload_id == 1
        assert lots[0].source_occurrence == 0
        assert marker_count == 1
        assert "cas_upload_id=2" in caplog.text
        assert "{broken" not in caplog.text
    finally:
        await engine.dispose()


async def test_backfill_does_not_duplicate_existing_lots_and_rerun_is_idempotent(
    tmp_path, monkeypatch
):
    _stub_init_caches(monkeypatch)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/existing-lot.db")
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with maker() as session:
            upload = CasUpload(
                portfolio_key="PAN-EXISTING",
                depository_source="cdsl",
                statement_date=datetime.date(2025, 1, 31),
                grand_total=Decimal("500"),
                raw_holdings_json=json.dumps(_purchase_payload()),
            )
            session.add(upload)
            await session.flush()
            assert (
                await create_investment_lots(
                    session,
                    cas_upload_id=upload.id,
                    payload=_purchase_payload(),
                )
            )[0] == 1
            await session.commit()

        await init_db(engine)
        await init_db(engine)

        async with maker() as session:
            count = (
                await session.execute(text("SELECT count(*) FROM investment_lots"))
            ).scalar_one()
        assert count == 1
    finally:
        await engine.dispose()


async def test_interim_lot_table_rebuild_preserves_rows_and_enables_occurrences(
    tmp_path, monkeypatch
):
    _stub_init_caches(monkeypatch)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/interim-lots.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("DROP TABLE investment_lots"))
            await conn.execute(
                text(
                    "CREATE TABLE investment_lots ("
                    "id INTEGER PRIMARY KEY, cas_upload_id INTEGER NOT NULL, "
                    "instrument_id VARCHAR NOT NULL, instrument_name VARCHAR NOT NULL, "
                    "quantity NUMERIC(20,6) NOT NULL, unit_cost NUMERIC(20,6) NOT NULL, "
                    "cost_basis NUMERIC(18,4) NOT NULL, currency VARCHAR(3) NOT NULL, "
                    "acquired_on DATE NOT NULL, source_ref VARCHAR NOT NULL, "
                    "transaction_type VARCHAR, reference VARCHAR, created_at DATETIME, "
                    "CONSTRAINT uq_investment_lot_natural UNIQUE "
                    "(cas_upload_id, source_ref, instrument_id, acquired_on, reference), "
                    "FOREIGN KEY(cas_upload_id) REFERENCES cas_uploads(id))"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX ix_investment_lots_upload "
                    "ON investment_lots (cas_upload_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX ix_investment_lots_instrument "
                    "ON investment_lots (instrument_id)"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO cas_uploads "
                    "(id, portfolio_key, depository_source, statement_date, grand_total, "
                    " portfolio_ok, raw_holdings_json, created_at) VALUES "
                    "(1, 'PAN-INTERIM', 'cdsl', '2025-01-31', 500, 1, '{}', "
                    " CURRENT_TIMESTAMP)"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO investment_lots "
                    "(id, cas_upload_id, instrument_id, instrument_name, quantity, "
                    " unit_cost, cost_basis, currency, acquired_on, source_ref, "
                    " transaction_type, reference, created_at) VALUES "
                    "(1, 1, 'INE000A01018', 'Interim Fund', 10, 50, 500, 'INR', "
                    " '2025-01-15', 'folio/1', 'purchase', 'REF-1', CURRENT_TIMESTAMP)"
                )
            )

        await init_db(engine)

        async with engine.begin() as conn:
            preserved = (
                await conn.execute(
                    text(
                        "SELECT instrument_id, source_occurrence FROM investment_lots "
                        "WHERE id = 1"
                    )
                )
            ).one()
            await conn.execute(
                text(
                    "INSERT INTO investment_lots "
                    "(cas_upload_id, instrument_id, instrument_name, quantity, unit_cost, "
                    " cost_basis, currency, acquired_on, source_ref, transaction_type, "
                    " reference, source_occurrence, created_at) VALUES "
                    "(1, 'INE000A01018', 'Interim Fund', 10, 50, 500, 'INR', "
                    " '2025-01-15', 'folio/1', 'purchase', 'REF-1', 1, "
                    " CURRENT_TIMESTAMP)"
                )
            )
            count = (
                await conn.execute(text("SELECT count(*) FROM investment_lots"))
            ).scalar_one()
        assert preserved.instrument_id == "INE000A01018"
        assert preserved.source_occurrence == 0
        assert count == 2
    finally:
        await engine.dispose()
