from decimal import Decimal

import pytest
from sqlalchemy import func, select

from financial_dashboard.db.models import BalanceSnapshot, CasUpload, SnapshotHolding
from financial_dashboard.services.cas_ingestion import (
    CasIngestError,
    ingest_cas_payload,
)

pytestmark = pytest.mark.anyio


async def test_ingest_cas_payload_creates_upload_snapshot_and_holdings(
    session, cas_statement_payload
):
    upload = await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    snapshot = (
        await session.execute(
            select(BalanceSnapshot).where(BalanceSnapshot.cas_upload_id == upload.id)
        )
    ).scalar_one()
    holding_total = (
        await session.execute(
            select(func.sum(SnapshotHolding.value)).where(
                SnapshotHolding.snapshot_id == snapshot.id
            )
        )
    ).scalar_one()

    assert upload.portfolio_key == "ABCDE1234F"
    assert upload.portfolio_ok is True
    assert upload.grand_total == Decimal("200000.00")
    assert snapshot.value == Decimal("200000.00")
    assert holding_total == Decimal("200000.00")


async def test_ingest_cas_payload_adds_other_only_for_positive_remainder(
    session, cas_statement_payload
):
    cas_statement_payload["summary"]["grand_total"] = "250000.00"

    upload = await ingest_cas_payload(session, cas_statement_payload)
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

    assert upload.portfolio_ok is False
    assert not any(row.asset_class == "other" and row.value < 0 for row in rows)


async def test_reingesting_same_portfolio_date_replaces_existing_rows(
    session, cas_statement_payload
):
    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()
    cas_statement_payload["summary"]["grand_total"] = "210000.00"

    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    uploads = (await session.execute(select(CasUpload))).scalars().all()
    snapshots = (await session.execute(select(BalanceSnapshot))).scalars().all()
    assert len(uploads) == 1
    assert len(snapshots) == 1
    assert uploads[0].grand_total == Decimal("210000.00")
    assert snapshots[0].value == Decimal("210000.00")


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("summary", "grand_total"), "grand_total"),
        (("meta", "statement_period_end"), "statement period end"),
    ],
)
async def test_ingest_cas_payload_requires_grand_total_and_statement_date(
    session, cas_statement_payload, path, message
):
    section, key = path
    cas_statement_payload[section][key] = None

    with pytest.raises(CasIngestError, match=message):
        await ingest_cas_payload(session, cas_statement_payload)


async def test_cdsl_refuses_to_replace_existing_nsdl_without_override(
    session, cas_statement_payload
):
    cas_statement_payload["meta"]["source"] = "nsdl"
    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    cas_statement_payload["meta"]["source"] = "cdsl"
    cas_statement_payload["summary"]["grand_total"] = "150000.00"

    with pytest.raises(CasIngestError, match="NSDL"):
        await ingest_cas_payload(session, cas_statement_payload)

    uploads = (await session.execute(select(CasUpload))).scalars().all()
    assert len(uploads) == 1
    assert uploads[0].depository_source == "nsdl"
    assert uploads[0].grand_total == Decimal("200000.00")


async def test_force_replace_overrides_nsdl_canonical_guard(
    session, cas_statement_payload
):
    cas_statement_payload["meta"]["source"] = "nsdl"
    await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    cas_statement_payload["meta"]["source"] = "cdsl"
    cas_statement_payload["summary"]["grand_total"] = "150000.00"
    await ingest_cas_payload(session, cas_statement_payload, force_replace=True)
    await session.flush()

    uploads = (await session.execute(select(CasUpload))).scalars().all()
    assert len(uploads) == 1
    assert uploads[0].depository_source == "cdsl"
    assert uploads[0].grand_total == Decimal("150000.00")


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


async def test_ingest_emits_a_holding_row_per_asset_class_and_folio(
    session, cas_statement_payload
):
    """Every distinct holding class (asset_class) plus each folio's total lands
    as its own SnapshotHolding row, Title-cased, with folios bucketed as
    'Mutual Fund'. A non-positive remainder adds no spurious 'other' row."""
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
    # Two folios -> both roll up into the single "mutual_fund" class.
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
    # grand_total (190000) equals classified (160000 + 30000) -> no remainder.
    cas_statement_payload["summary"]["grand_total"] = "190000.00"

    upload = await ingest_cas_payload(session, cas_statement_payload)
    snapshot = (
        await session.execute(
            select(BalanceSnapshot).where(BalanceSnapshot.cas_upload_id == upload.id)
        )
    ).scalar_one()
    rows = (
        (
            await session.execute(
                select(SnapshotHolding)
                .where(SnapshotHolding.snapshot_id == snapshot.id)
                .order_by(SnapshotHolding.asset_class)
            )
        )
        .scalars()
        .all()
    )

    by_class = {r.asset_class: r for r in rows}
    # Demat holding classes each get their own row, summed by class.
    assert by_class["equity"].value == Decimal("100000.00")
    assert by_class["equity"].label == "Equity"
    assert by_class["etf"].value == Decimal("50000.00")
    assert by_class["etf"].label == "Etf"
    assert by_class["govt_security"].value == Decimal("10000.00")
    assert by_class["govt_security"].label == "Govt Security"
    # Both folios collapsed into one "mutual_fund" row.
    assert by_class["mutual_fund"].value == Decimal("30000.00")
    assert by_class["mutual_fund"].label == "Mutual Fund"
    # No remainder -> no fabricated 'other' row.
    assert "other" not in by_class
    # Holdings reconcile to the grand total.
    assert sum((r.value for r in rows), Decimal("0.00")) == Decimal("190000.00")
