"""Inline schema migrations and one-time backfills from stored JSON."""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import (
    Account,
    Base,
    CasUpload,
    InvestmentLot,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.investments import create_investment_lots

pytestmark = pytest.mark.anyio

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


async def test_legacy_tables_gain_new_columns_and_keep_rows(tmp_path, monkeypatch):
    """init_db upgrades legacy settings and snapshot_holdings tables in place.

    ``create_all`` does not ALTER existing tables. The settings table must gain
    ``updated_at`` before the CAS backfill writes its marker. Old aggregated
    holdings must survive with NULL investment detail. The second run pins
    idempotency.
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
            await conn.execute(text(_OLD_SNAPSHOT_HOLDINGS))
            await conn.execute(
                text(
                    "INSERT INTO snapshot_holdings (snapshot_id, asset_class, label, value) "
                    "VALUES (1, 'equity', 'Equity', 100)"
                )
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
            holding = (
                await conn.execute(
                    text(
                        "SELECT value, instrument_id, quantity, unit_price, currency, "
                        "cost_basis, acquired_on FROM snapshot_holdings WHERE id = 1"
                    )
                )
            ).one()
        assert "updated_at" in columns
        assert rows["legacy.key"] == "kept"
        assert rows["migrations.investment_lots_backfill_v1"] == "1"
        assert holding.value == 100
        assert holding[1:] == (None,) * 6
    finally:
        await engine.dispose()


async def test_legacy_cas_payloads_backfill_once_and_isolate_malformed_json(
    tmp_path, monkeypatch
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
    finally:
        await engine.dispose()


async def test_stored_reconciliations_backfill_card_holders_once(tmp_path, monkeypatch):
    """Old statement rows take the cardholder that their stored reconciliation
    names. A holder already set stays, malformed JSON does not stop boot, and
    the backfill runs once."""
    _stub_init_caches(monkeypatch)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/legacy-holder.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as session:
            session.add(Account(id=1, bank="hdfc", label="Card", type="credit_card"))
            txns = [
                Transaction(
                    account_id=1,
                    bank="hdfc",
                    email_type="cc_statement",
                    direction="debit",
                    amount=Decimal("10.00"),
                    currency="INR",
                    transaction_date=datetime.date(2026, 4, 7),
                    card_holder=holder,
                )
                for holder in (None, None, "Kept Holder")
            ]
            session.add_all(txns)
            await session.flush()
            matched, imported, kept = (txn.id for txn in txns)
            recon = {
                "matched": [
                    {"db_txn_id": matched, "person": "ADDON  HOLDER"},
                    {"db_txn_id": kept, "person": "OTHER HOLDER"},
                ],
                "missing": [{"imported_txn_id": imported, "person": "PRIMARY HOLDER"}],
            }
            session.add_all(
                StatementUpload(
                    account_id=1,
                    bank="hdfc",
                    filename=f"{name}.pdf",
                    file_path=f"/nonexistent/{name}.pdf",
                    reconciliation_data=data,
                )
                for name, data in (
                    ("good", json.dumps(recon)),
                    ("broken", "{broken"),
                    ("no-rows", '{"matched": null}'),
                    (
                        "bad-id",
                        '{"matched": [{"db_txn_id": {"id": 1}, "person": "X"}]}',
                    ),
                )
            )
            await session.commit()

        await init_db(engine)
        await init_db(engine)

        async with maker() as session:
            holders = dict(
                (await session.execute(select(Transaction.id, Transaction.card_holder)))
                .tuples()
                .all()
            )
            marker_count = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM settings WHERE key = "
                        "'migrations.card_holder_backfill'"
                    )
                )
            ).scalar_one()
        assert holders == {
            matched: "Addon Holder",
            imported: "Primary Holder",
            kept: "Kept Holder",
        }
        assert marker_count == 1
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
