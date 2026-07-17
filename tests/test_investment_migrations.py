"""Inline migration handling for the Phase 4 investment-grade holding columns.

The model definition and ``create_all`` produce the new nullable columns and
the new ``investment_lots`` table; the ALTER-style migration that ``init_db``
runs on a pre-existing database is verified by applying the same column-add
statements to an old-schema table. (``init_db`` itself reads the global
session at its tail, so it is not invoked here end-to-end.)
"""

from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from financial_dashboard.db.models import Base, SnapshotHolding

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
