import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db.models import (
    BalanceSnapshot,
    CasUpload,
    InvestmentLot,
    SnapshotHolding,
)
from financial_dashboard.services.cas_ingestion import (
    CasIngestError,
    ingest_cas_payload,
)

pytestmark = pytest.mark.anyio


def _mf_purchase(**overrides) -> dict:
    base = {
        "scope": "mf",
        "source_ref": "123/45",
        "date": "2026-01-15",
        "description": "Example Fund",
        "isin": "INE000A01018",
        "transaction_type": "purchase",
        "units": "1000",
        "nav": "50.00",
        "amount": "50000.00",
        "reference": "TXN001",
    }
    base.update(overrides)
    return base


async def _holdings(session, upload) -> tuple[BalanceSnapshot, list[SnapshotHolding]]:
    snapshot = (
        await session.execute(
            select(BalanceSnapshot).where(BalanceSnapshot.cas_upload_id == upload.id)
        )
    ).scalar_one()
    rows = (
        (
            await session.execute(
                select(SnapshotHolding).where(
                    SnapshotHolding.snapshot_id == snapshot.id
                )
            )
        )
        .scalars()
        .all()
    )
    return snapshot, list(rows)


async def test_ingest_emits_a_holding_row_per_asset_class_and_folio(
    session, cas_statement_payload
):
    """Each demat asset class and the summed folios land as one row each. A
    zero remainder adds no 'other' row. No transactions means no lots."""
    cas_statement_payload["accounts"] = [
        {
            "depository": "CDSL",
            "dp_id": "12088700",
            "client_id": "00000001",
            "dp_name": "Example DP",
            "total_value": "160000.00",
            "holdings": [
                {
                    "name": "Equity A",
                    "isin": "INE000A01012",
                    "asset_class": "equity",
                    "quantity": "100",
                    "price": "1000.00",
                    "value": "100000.00",
                    "flags": [],
                    "notes": None,
                },
                {
                    "name": "ETF B",
                    "isin": "INF000B01012",
                    "asset_class": "etf",
                    "quantity": "50",
                    "price": "1000.00",
                    "value": "50000.00",
                    "flags": [],
                    "notes": None,
                },
                {
                    "name": "Govt Bond",
                    "isin": "INE000G01012",
                    "asset_class": "govt_security",
                    "quantity": "10",
                    "price": "1000.00",
                    "value": "10000.00",
                    "flags": [],
                    "notes": None,
                },
            ],
        }
    ]
    cas_statement_payload["folios"] = [
        {
            "folio_number": "111",
            "amc": "AMC One",
            "total_value": "20000.00",
            "schemes": [],
        },
        {
            "folio_number": "222",
            "amc": "AMC Two",
            "total_value": "10000.00",
            "schemes": [],
        },
    ]
    cas_statement_payload["summary"]["grand_total"] = "190000.00"

    upload = await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()
    snapshot, rows = await _holdings(session, upload)

    assert upload.portfolio_key == "ABCDE1234F"
    assert upload.portfolio_ok is True
    assert upload.grand_total == Decimal("190000.00")
    assert snapshot.value == Decimal("190000.00")
    by_class = {r.asset_class: r.value for r in rows}
    assert by_class == {
        "equity": Decimal("100000.00"),
        "etf": Decimal("50000.00"),
        "govt_security": Decimal("10000.00"),
        "mutual_fund": Decimal("30000.00"),
    }
    assert (await session.execute(select(InvestmentLot))).scalars().all() == []


async def test_ingest_cas_payload_adds_other_only_for_positive_remainder(
    session, cas_statement_payload
):
    cas_statement_payload["summary"]["grand_total"] = "250000.00"

    upload = await ingest_cas_payload(session, cas_statement_payload)
    _snapshot, rows = await _holdings(session, upload)

    assert upload.grand_total == Decimal("250000.00")
    assert sum((row.value for row in rows), Decimal("0.00")) == Decimal("250000.00")
    assert any(
        row.asset_class == "other" and row.value == Decimal("50000.00") for row in rows
    )


async def test_ingest_cas_payload_does_not_emit_negative_other_when_unreconciled(
    session, cas_statement_payload
):
    cas_statement_payload["summary"]["grand_total"] = "150000.00"
    cas_statement_payload["reconciliation"]["portfolio_ok"] = False
    cas_statement_payload["reconciliation"]["portfolio_delta"] = "-50000.00"

    upload = await ingest_cas_payload(session, cas_statement_payload)
    _snapshot, rows = await _holdings(session, upload)

    assert upload.portfolio_ok is False
    assert not any(row.value < 0 for row in rows)


async def test_reingesting_same_portfolio_date_replaces_existing_rows(
    session, cas_statement_payload
):
    cas_statement_payload["transactions"] = [_mf_purchase()]
    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()
    cas_statement_payload["summary"]["grand_total"] = "210000.00"

    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    uploads = (await session.execute(select(CasUpload))).scalars().all()
    snapshots = (await session.execute(select(BalanceSnapshot))).scalars().all()
    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    assert len(uploads) == 1
    assert len(snapshots) == 1
    assert len(lots) == 1
    assert uploads[0].grand_total == Decimal("210000.00")
    assert snapshots[0].value == Decimal("210000.00")


async def test_ingest_cas_payload_requires_grand_total(session, cas_statement_payload):
    cas_statement_payload["summary"]["grand_total"] = None

    with pytest.raises(CasIngestError):
        await ingest_cas_payload(session, cas_statement_payload)


async def test_cdsl_refuses_to_replace_existing_nsdl_without_override(
    session, cas_statement_payload
):
    cas_statement_payload["meta"]["source"] = "nsdl"
    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    cas_statement_payload["meta"]["source"] = "cdsl"
    cas_statement_payload["summary"]["grand_total"] = "150000.00"

    with pytest.raises(CasIngestError):
        await ingest_cas_payload(session, cas_statement_payload)

    uploads = (await session.execute(select(CasUpload))).scalars().all()
    assert len(uploads) == 1
    assert uploads[0].depository_source == "nsdl"
    assert uploads[0].grand_total == Decimal("200000.00")


async def test_force_replace_overrides_nsdl_canonical_guard_and_recreates_lots(
    session, cas_statement_payload
):
    cas_statement_payload["meta"]["source"] = "nsdl"
    cas_statement_payload["transactions"] = [_mf_purchase()]
    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    cas_statement_payload["meta"]["source"] = "cdsl"
    cas_statement_payload["summary"]["grand_total"] = "150000.00"
    cas_statement_payload["transactions"] = [
        _mf_purchase(reference="TXN002", units="200", nav="10.00", amount="2000.00")
    ]
    await ingest_cas_payload(session, cas_statement_payload, force_replace=True)
    await session.flush()

    uploads = (await session.execute(select(CasUpload))).scalars().all()
    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    assert len(uploads) == 1
    assert uploads[0].depository_source == "cdsl"
    assert uploads[0].grand_total == Decimal("150000.00")
    assert [lot.reference for lot in lots] == ["TXN002"]


async def test_nsdl_replaces_existing_cdsl_without_override(
    session, cas_statement_payload
):
    cas_statement_payload["meta"]["source"] = "cdsl"
    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    cas_statement_payload["meta"]["source"] = "nsdl"
    cas_statement_payload["summary"]["grand_total"] = "250000.00"
    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    uploads = (await session.execute(select(CasUpload))).scalars().all()
    assert len(uploads) == 1
    assert uploads[0].depository_source == "nsdl"
    assert uploads[0].grand_total == Decimal("250000.00")


async def test_complete_lot_created_from_explicit_mf_purchase(
    session, cas_statement_payload
):
    txn = _mf_purchase(units="123.456789", nav="12.3456", amount="1524.15")
    cas_statement_payload["transactions"] = [txn]
    upload = await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    # The raw payload stays verbatim. The legacy lot backfill reads it.
    assert json.loads(upload.raw_holdings_json)["transactions"] == [txn]

    lot = (await session.execute(select(InvestmentLot))).scalar_one()
    assert lot.cas_upload_id == upload.id
    assert lot.instrument_id == "INE000A01018"
    assert lot.instrument_name == "Example Fund"
    assert lot.quantity == Decimal("123.456789")
    assert lot.unit_cost == Decimal("12.3456")
    assert lot.cost_basis == Decimal("1524.15")
    assert lot.currency == "INR"
    assert lot.acquired_on == datetime.date(2026, 1, 15)
    assert lot.source_ref == "123/45"
    assert lot.transaction_type == "purchase"
    assert lot.reference == "TXN001"
