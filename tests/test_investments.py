"""Investment service: source-faithful lot classification, precision, and the
no-fabrication guarantee.

The classifier is exercised directly (pure) and through the persisted lot
table. A lot is built ONLY from an explicit, internally-consistent acquisition
fact; anything less is reported with a stable reason and never fabricated.
"""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db.models import CasUpload, InvestmentLot
from financial_dashboard.services.investments import (
    create_investment_lots,
    extract_lots_from_payload,
    get_current_valuations,
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


# ---------------------------------------------------------------------------
# Complete-lot classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        # demat movement: CAS carries no cost for securities
        (
            {
                "scope": "demat",
                "quantity": "10",
                "units": None,
                "nav": None,
                "amount": None,
            },
            "not_mutual_fund",
        ),
        # redemption is a disposal, never an acquisition lot
        (
            {"transaction_type": "redemption", "units": "-100", "amount": "-5200.00"},
            "disposal_transaction",
        ),
        # unknown type: cannot confirm the date is an acquisition date
        ({"transaction_type": "transfer"}, "ambiguous_transaction_type"),
        # missing nav (no per-unit cost)
        ({"nav": None}, "missing_lot_facts"),
        # inconsistent cost basis: amount != units*nav
        ({"amount": "40000.00"}, "cost_basis_inconsistent"),
    ],
)
def test_incomplete_or_excluded_transactions_are_reported_not_fabricated(
    overrides, reason
):
    lots, excluded = extract_lots_from_payload(
        {"transactions": [_mf_purchase(**overrides)]}
    )
    assert lots == []
    assert len(excluded) == 1
    assert excluded[0].reason == reason
    # No lot is ever fabricated: the exclusion carries the truth, not a guess.
    assert excluded[0].detail


# ---------------------------------------------------------------------------
# 1-paisa lot boundary: agreement gate + exact renderer consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("units", "nav", "amount", "accepted", "label"),
    [
        # difference 0.009 (sub-penny) -> accepted
        ("1", "100.009", "100.00", True, "diff_subpenny"),
        # difference exactly 0.01 -> REJECTED (the former 1-paisa grey zone)
        ("1", "100.01", "100.00", False, "diff_exactly_one_paisa"),
    ],
)
def test_lot_agreement_boundary(units, nav, amount, accepted, label):
    """The agreement gate accepts sub-penny disagreement (< 0.01, exclusive)
    and rejects a discrepancy of a full paisa or more, with a stable reason."""
    lots, excluded = extract_lots_from_payload(
        {"transactions": [_mf_purchase(units=units, nav=nav, amount=amount)]}
    )
    if accepted:
        assert len(lots) == 1, label
        assert excluded == [], label
    else:
        assert lots == [], label
        assert len(excluded) == 1, label
        # Rejected reason is stable across every boundary case.
        assert excluded[0].reason == "cost_basis_inconsistent", label
        assert excluded[0].detail, label


def test_accepted_lot_cost_basis_is_quantized_product_and_renders_balanced():
    """An accepted lot stores cost_basis == (quantity*unit_cost).quantize(0.01),
    not the printed amount. The renderer's lot guard then sees a zero diff in
    every backend."""
    units, nav, amount = "1", "100.009", "100.00"
    from financial_dashboard.services.paisa.renderers import render_document
    from financial_dashboard.services.paisa.renderers.base import (
        InvestmentLotEntry,
        LedgerDocument,
    )

    lots, excluded = extract_lots_from_payload(
        {"transactions": [_mf_purchase(units=units, nav=nav, amount=amount)]}
    )
    assert excluded == []
    lot = lots[0]
    # cost_basis is exactly the quantized product -> renderer guard is exact.
    assert lot.cost_basis == (lot.quantity * lot.unit_cost).quantize(Decimal("0.01"))
    entry = InvestmentLotEntry(
        instrument=lot.instrument_id,
        instrument_name=lot.instrument_name,
        quantity=lot.quantity,
        unit_cost=lot.unit_cost,
        cost_basis=lot.cost_basis,
        currency=lot.currency,
        acquired_on=lot.acquired_on,
    )
    doc = LedgerDocument(
        cutover_date=lot.acquired_on,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(entry,),
    )
    for backend in ("ledger", "hledger", "beancount"):
        # Rendering exercises check_lot_consistent; no UnbalancedEntry raised.
        assert render_document(doc, backend)


# ---------------------------------------------------------------------------
# Persisted lots
# ---------------------------------------------------------------------------


async def _upload(
    session, *, payload_txns, statement_date="2026-04-30", portfolio_key="PAN123"
):
    upload = CasUpload(
        portfolio_key=portfolio_key,
        depository_source="cdsl",
        statement_date=datetime.date.fromisoformat(statement_date),
        grand_total=Decimal("100000.00"),
        raw_holdings_json=json.dumps({"transactions": payload_txns}),
    )
    session.add(upload)
    await session.flush()
    return upload


async def test_create_investment_lots_direct_retry_is_idempotent(session):
    payload = {"transactions": [_mf_purchase(), _mf_purchase()]}
    upload = await _upload(session, payload_txns=payload["transactions"])

    assert (
        await create_investment_lots(session, cas_upload_id=upload.id, payload=payload)
    )[0] == 2
    assert (
        await create_investment_lots(session, cas_upload_id=upload.id, payload=payload)
    )[0] == 0

    rows = (
        (
            await session.execute(
                select(InvestmentLot).order_by(InvestmentLot.source_occurrence)
            )
        )
        .scalars()
        .all()
    )
    assert [row.source_occurrence for row in rows] == [0, 1]


# ---------------------------------------------------------------------------
# Current valuations
# ---------------------------------------------------------------------------


async def _upload_payload(
    session,
    *,
    portfolio_key: str,
    statement_date: str,
    payload: dict,
    source: str = "cdsl",
):
    upload = CasUpload(
        portfolio_key=portfolio_key,
        depository_source=source,
        statement_date=datetime.date.fromisoformat(statement_date),
        grand_total=Decimal("100000.00"),
        raw_holdings_json=json.dumps(payload),
    )
    session.add(upload)
    await session.flush()
    await create_investment_lots(
        session,
        cas_upload_id=upload.id,
        payload=payload,
    )
    return upload


async def test_current_valuations_preserve_same_isin_source_identities(session):
    isin = "INE000A01012"
    payload = {
        "accounts": [
            {
                "depository": "CDSL",
                "dp_id": "DP1",
                "client_id": "CLIENT1",
                "holdings": [
                    {
                        "name": "Shared Security",
                        "isin": isin,
                        "asset_class": "equity",
                        "quantity": "10",
                        "price": "100",
                        "value": "1000",
                    }
                ],
            },
            {
                "depository": "NSDL",
                "dp_id": "DP2",
                "client_id": "CLIENT2",
                "holdings": [
                    {
                        "name": "Shared Security",
                        "isin": isin,
                        "asset_class": "equity",
                        "quantity": "20",
                        "price": "100",
                        "value": "2000",
                    }
                ],
            },
        ],
        "folios": [
            {
                "folio_number": "FOLIO-1",
                "schemes": [
                    {
                        "scheme_name": "Shared Fund",
                        "isin": isin,
                        "units": "30",
                        "nav": "100",
                        "value": "3000",
                    }
                ],
            },
            {
                "folio_number": "FOLIO-2",
                "schemes": [
                    {
                        "scheme_name": "Shared Fund",
                        "isin": isin,
                        "units": "40",
                        "nav": "100",
                        "value": "4000",
                    }
                ],
            },
        ],
        "transactions": [],
    }
    await _upload_payload(
        session,
        portfolio_key="PAN-SOURCES",
        statement_date="2026-04-30",
        payload=payload,
    )

    valuations = await get_current_valuations(session)

    assert len(valuations) == 4
    assert {(value.scope, value.source_ref) for value in valuations} == {
        ("demat", "CDSL:DP1:CLIENT1"),
        ("demat", "NSDL:DP2:CLIENT2"),
        ("folio", "FOLIO-1"),
        ("folio", "FOLIO-2"),
    }
    assert all(value.instrument_id == isin for value in valuations)


async def test_current_valuations_read_the_latest_cas(session):
    old_payload = {
        "transactions": [_mf_purchase(units="10", nav="50", amount="500")],
        "folios": [
            {
                "folio_number": "123/45",
                "schemes": [
                    {
                        "scheme_name": "Example Fund",
                        "isin": "INE000A01018",
                        "units": "10",
                        "nav": "55",
                        "value": "550",
                    }
                ],
            }
        ],
    }
    new_payload = {
        **old_payload,
        "folios": [
            {
                "folio_number": "123/45",
                "schemes": [
                    {
                        "scheme_name": "Example Fund",
                        "isin": "INE000A01018",
                        "units": "10",
                        "nav": "80",
                        "value": "800",
                    }
                ],
            }
        ],
    }
    await _upload_payload(
        session,
        portfolio_key="PAN-VALUATION",
        statement_date="2026-01-31",
        payload=old_payload,
    )
    latest = await _upload_payload(
        session,
        portfolio_key="PAN-VALUATION",
        statement_date="2026-02-28",
        payload=new_payload,
    )

    valuations = await get_current_valuations(session)

    assert len(valuations) == 1
    assert valuations[0].cas_upload_id == latest.id
    assert valuations[0].unit_price == Decimal("80")
    assert valuations[0].value == Decimal("800")
