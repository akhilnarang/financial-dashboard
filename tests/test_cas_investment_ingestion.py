"""CAS ingestion integration with investment lots: backward compatibility with
payloads that carry no transaction facts, complete-lot creation on re-ingest,
force-replace idempotency, and Decimal precision through the round-trip.
"""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from financial_dashboard.db.models import (
    BalanceSnapshot,
    CasUpload,
    InvestmentLot,
    SnapshotHolding,
)
from financial_dashboard.services.cas_ingestion import ingest_cas_payload

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


def _payload_with_transactions(*txns, grand_total="100000.00") -> dict:
    return {
        "meta": {
            "source": "cdsl",
            "investor_name": "Example Investor",
            "pan": "ABCDE1234F",
            "statement_period_start": "2026-04-01",
            "statement_period_end": "2026-04-30",
            "generated_on": "2026-05-02",
        },
        "accounts": [
            {
                "depository": "CDSL",
                "dp_id": "12088700",
                "client_id": "00000001",
                "dp_name": "Example DP",
                "total_value": grand_total,
                "holdings": [],
            }
        ],
        "folios": [],
        "transactions": list(txns),
        "summary": {"grand_total": grand_total},
        "reconciliation": {"portfolio_ok": True, "portfolio_delta": "0.00"},
    }


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


async def test_old_payload_without_transactions_still_works(
    session, cas_statement_payload
):
    """The fixture payload has transactions: [] — no lots, unchanged behavior."""
    upload = await ingest_cas_payload(session, cas_statement_payload)
    await session.flush()

    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    assert lots == []
    # Existing snapshot/holding behavior is unchanged.
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
    assert snapshot.value == Decimal("200000.00")
    assert holding_total == Decimal("200000.00")


async def test_raw_payload_preserved_verbatim(session):
    """The raw payload (including transactions) is preserved for diagnostics."""
    payload = _payload_with_transactions(_mf_purchase())
    upload = await ingest_cas_payload(session, payload)
    await session.flush()
    preserved = json.loads(upload.raw_holdings_json)
    assert preserved["transactions"] == [_mf_purchase()]


# ---------------------------------------------------------------------------
# Complete-lot creation
# ---------------------------------------------------------------------------


async def test_complete_lot_created_from_explicit_mf_purchase(session):
    payload = _payload_with_transactions(_mf_purchase())
    upload = await ingest_cas_payload(session, payload)
    await session.flush()

    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    assert len(lots) == 1
    lot = lots[0]
    assert lot.cas_upload_id == upload.id
    assert lot.instrument_id == "INE000A01018"
    assert lot.instrument_name == "Example Fund"
    assert lot.quantity == Decimal("1000")
    assert lot.unit_cost == Decimal("50")
    assert lot.cost_basis == Decimal("50000")
    assert lot.currency == "INR"
    assert lot.acquired_on == datetime.date(2026, 1, 15)
    assert lot.source_ref == "123/45"
    assert lot.transaction_type == "purchase"
    assert lot.reference == "TXN001"


async def test_incomplete_transactions_create_no_lots(session):
    payload = _payload_with_transactions(
        # demat movement: no cost in CAS
        {
            "scope": "demat",
            "source_ref": "d/1",
            "date": "2026-01-01",
            "isin": "INE000A01012",
            "transaction_type": "purchase",
            "quantity": "10",
        },
        # redemption: a disposal, not an acquisition
        _mf_purchase(
            transaction_type="redemption", units="-100", amount="-5200.00", nav="52.00"
        ),
    )
    await ingest_cas_payload(session, payload)
    await session.flush()
    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    assert lots == []


# ---------------------------------------------------------------------------
# Force-replace / idempotency
# ---------------------------------------------------------------------------


async def test_reingest_replaces_prior_lots_no_duplicates(session):
    payload = _payload_with_transactions(_mf_purchase())
    await ingest_cas_payload(session, payload)
    await session.flush()

    # Re-ingest the same period: the prior upload (and its lot) is deleted and
    # a fresh one created, so there is exactly one lot afterward.
    await ingest_cas_payload(session, payload)
    await session.flush()

    uploads = (await session.execute(select(CasUpload))).scalars().all()
    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    assert len(uploads) == 1
    assert len(lots) == 1


async def test_force_replace_recreates_lots(session):
    payload = _payload_with_transactions(_mf_purchase())
    payload["meta"]["source"] = "nsdl"
    await ingest_cas_payload(session, payload)
    await session.flush()

    payload["meta"]["source"] = "cdsl"
    payload["transactions"] = [
        _mf_purchase(reference="TXN002", units="200", nav="10.00", amount="2000.00")
    ]
    await ingest_cas_payload(session, payload, force_replace=True)
    await session.flush()

    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    uploads = (await session.execute(select(CasUpload))).scalars().all()
    assert len(uploads) == 1
    assert uploads[0].depository_source == "cdsl"
    assert len(lots) == 1
    assert lots[0].reference == "TXN002"


async def test_duplicate_transaction_within_one_payload_yields_one_lot(session):
    txn = _mf_purchase()
    payload = _payload_with_transactions(txn, txn)
    await ingest_cas_payload(session, payload)
    await session.flush()
    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    assert len(lots) == 1


# ---------------------------------------------------------------------------
# Decimal precision round-trip
# ---------------------------------------------------------------------------


async def test_decimal_precision_survives_ingest_round_trip(session):
    payload = _payload_with_transactions(
        _mf_purchase(
            units="123.456789",
            nav="12.3456",
            amount="1524.15",
            reference="P1",
        )
    )
    await ingest_cas_payload(session, payload)
    await session.flush()
    lot = (await session.execute(select(InvestmentLot))).scalar_one()
    assert lot.quantity == Decimal("123.456789")
    assert lot.unit_cost == Decimal("12.3456")
    assert lot.cost_basis == Decimal("1524.15")
