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
    get_canonical_lot_consumption,
    get_canonical_lots,
    get_complete_lots,
    get_current_valuations,
    get_incomplete_reasons,
    get_latest_values,
    get_positions,
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
# Disposal-history resolution (redemption safety)
# ---------------------------------------------------------------------------


def test_disposal_with_no_reference_is_always_unresolved():
    """A disposal with no source_ref (or no units) can never be explicitly tied
    to an acquisition, so the instrument is flagged unresolved — never quietly
    projected (which would overstate holdings)."""
    from financial_dashboard.services.investments import unresolved_disposal_instruments

    payloads = [
        {
            "transactions": [
                _mf_purchase(isin="INE000A01020", source_ref="a/1"),
                _mf_purchase(
                    isin="INE000A01020",
                    transaction_type="redemption",
                    units="-50",
                    amount="-5100.00",
                    source_ref=None,  # no reference -> un-tieable
                    reference="R1",
                ),
            ]
        }
    ]
    assert unresolved_disposal_instruments(payloads) == {"INE000A01020"}


def _disposal(**overrides):
    base = {
        "scope": "mf",
        "source_ref": "lot/ref",
        "date": "2026-03-01",
        "description": "Example Fund",
        "isin": "INE000A01018",
        "transaction_type": "redemption",
        "units": "-25",
        "nav": "12.00",
        "amount": "-300.00",
        "reference": "SALE-1",
    }
    base.update(overrides)
    return base


def _remaining_for(consumption, instrument_id):
    return {
        (key.acquired_on, key.reference, key.occurrence): quantity
        for key, quantity in consumption.remaining.items()
        if key.instrument_id == instrument_id
    }


def test_over_disposal_is_unresolved_and_never_clamped():
    from financial_dashboard.services.investments import resolve_lot_consumption

    payload = {
        "transactions": [
            _mf_purchase(
                source_ref="lot/ref",
                units="100",
                nav="10",
                amount="1000",
                reference="BUY-1",
            ),
            _disposal(units="-100.000001"),
        ]
    }
    consumption = resolve_lot_consumption([payload])
    assert consumption.unresolved_instruments == {"INE000A01018"}
    # Unresolved is an instrument-level suppression, never a fabricated
    # zero/clamp presented as a resolved remainder.
    assert consumption.remaining == {}


@pytest.mark.parametrize("buy_b_date", ["2026-01-01", "2026-02-01"])
def test_multi_lot_bucket_without_exact_reference_is_ambiguous(buy_b_date):
    """A shared source ref does not authorize FIFO or any other tie-break
    between acquisitions, on the same date or on distinct dates."""
    from financial_dashboard.services.investments import resolve_lot_consumption

    payload = {
        "transactions": [
            _mf_purchase(
                source_ref="lot/ref",
                date="2026-01-01",
                units="40",
                nav="10",
                amount="400",
                reference="BUY-A",
            ),
            _mf_purchase(
                source_ref="lot/ref",
                date=buy_b_date,
                units="60",
                nav="20",
                amount="1200",
                reference="BUY-B",
            ),
            _disposal(units="-50", reference="SALE"),
        ]
    }
    consumption = resolve_lot_consumption([payload])
    assert consumption.unresolved_instruments == {"INE000A01018"}
    assert consumption.remaining == {}


def test_partial_disposal_with_incomplete_acquisition_in_bucket_is_unresolved():
    """FIFO/cost allocation is not deterministic when the same explicit bucket
    contains another acquisition whose date or cost was absent from the source."""
    from financial_dashboard.services.investments import resolve_lot_consumption

    payload = {
        "transactions": [
            _mf_purchase(
                source_ref="lot/ref",
                units="100",
                nav="10",
                amount="1000",
                reference="BUY-COMPLETE",
            ),
            _mf_purchase(
                source_ref="lot/ref",
                units="50",
                nav=None,
                amount="500",
                reference="BUY-INCOMPLETE",
            ),
            _disposal(units="-25", reference="SALE"),
        ]
    }
    consumption = resolve_lot_consumption([payload])
    assert consumption.unresolved_instruments == {"INE000A01018"}
    assert consumption.remaining == {}


def test_exact_transaction_reference_disambiguates_same_date_lots():
    from financial_dashboard.services.investments import resolve_lot_consumption

    payload = {
        "transactions": [
            _mf_purchase(
                source_ref="lot/ref",
                units="40",
                nav="10",
                amount="400",
                reference="BUY-A",
            ),
            _mf_purchase(
                source_ref="lot/ref",
                units="60",
                nav="20",
                amount="1200",
                reference="BUY-B",
            ),
            _disposal(units="-25", reference="BUY-B"),
        ]
    }
    consumption = resolve_lot_consumption([payload])
    assert consumption.unresolved_instruments == set()
    assert _remaining_for(consumption, "INE000A01018") == {
        (datetime.date(2026, 1, 15), "BUY-B", 0): Decimal("35")
    }


def test_same_source_ref_is_scoped_by_instrument():
    """A ref shared by different instruments neither cross-consumes nor makes
    the independently exact disposal ambiguous."""
    from financial_dashboard.services.investments import resolve_lot_consumption

    payload = {
        "transactions": [
            _mf_purchase(
                isin="INE000A01018",
                source_ref="shared/ref",
                units="100",
                nav="10",
                amount="1000",
                reference="BUY-A",
            ),
            _mf_purchase(
                isin="INE000B01018",
                source_ref="shared/ref",
                units="70",
                nav="20",
                amount="1400",
                reference="BUY-B",
            ),
            _disposal(
                isin="INE000A01018",
                source_ref="shared/ref",
                units="-100",
                reference="SALE-A",
            ),
        ]
    }
    consumption = resolve_lot_consumption([payload])
    assert consumption.unresolved_instruments == set()
    assert list(_remaining_for(consumption, "INE000A01018").values()) == [Decimal("0")]
    # Untouched lots are intentionally absent from the adjustment map.
    assert _remaining_for(consumption, "INE000B01018") == {}


async def test_get_lot_consumption_is_read_only_and_returns_remaining_lots(session):
    """The DB accessor nets two equal partial disposals Decimal-exactly. It
    does not rewrite the persisted gross acquisition row."""
    from financial_dashboard.services.investments import get_lot_consumption

    transactions = [
        _mf_purchase(
            source_ref="lot/ref",
            units="1.234567",
            nav="12.3456",
            amount="15.24",
            reference="BUY-1",
        ),
        _disposal(units="-0.100001", reference="SALE-1"),
        _disposal(units="-0.100001", reference="SALE-2"),
    ]
    upload = await _upload(session, payload_txns=transactions)
    await create_investment_lots(
        session,
        cas_upload_id=upload.id,
        payload=json.loads(upload.raw_holdings_json),
    )

    consumption = await get_lot_consumption(session)

    assert consumption.unresolved_instruments == set()
    assert list(consumption.remaining.values()) == [Decimal("1.034565")]
    unchanged = (await session.execute(select(InvestmentLot))).scalar_one()
    assert unchanged.quantity == Decimal("1.234567")
    assert unchanged.cost_basis == Decimal("15.24")


# ---------------------------------------------------------------------------
# Persisted lots + read queries
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


async def test_get_incomplete_reasons_reads_preserved_raw_payload(session):
    upload = await _upload(
        session,
        payload_txns=[
            _mf_purchase(),
            _mf_purchase(transaction_type="redemption", units="-10", amount="-500.00"),
            {
                "scope": "demat",
                "source_ref": "d/1",
                "date": "2026-01-01",
                "isin": "INE000A01012",
                "transaction_type": "purchase",
                "quantity": "5",
            },
        ],
    )
    await create_investment_lots(
        session,
        cas_upload_id=upload.id,
        payload=json.loads(upload.raw_holdings_json),
    )
    reasons = await get_incomplete_reasons(session)
    assert {r.reason for r in reasons} == {"disposal_transaction", "not_mutual_fund"}
    assert len(await get_complete_lots(session)) == 1


async def test_positions_join_holdings_with_lot_cost_basis(session):
    """A holding with a complete lot carries its cost basis. A holding the CAS
    only priced gets zero lot fields, not a fabricated lot."""
    payload = {
        "transactions": [_mf_purchase()],
        "accounts": [
            {
                "holdings": [
                    {
                        "name": "Equity A",
                        "isin": "INE000A01012",
                        "asset_class": "equity",
                        "quantity": "100",
                        "price": "1000.00",
                        "value": "100000.00",
                    }
                ]
            }
        ],
        "folios": [
            {
                "folio_number": "123/45",
                "schemes": [
                    {
                        "scheme_name": "Example Fund",
                        "isin": "INE000A01018",
                        "units": "1000",
                        "nav": "55.00",
                        "value": "55000.00",
                    }
                ],
            }
        ],
    }
    upload = await _upload(session, payload_txns=[])
    upload.raw_holdings_json = json.dumps(payload)
    await create_investment_lots(session, cas_upload_id=upload.id, payload=payload)

    positions = {p.instrument_id: p for p in await get_positions(session)}
    fund = positions["INE000A01018"]
    assert fund.quantity == Decimal("1000")
    assert fund.unit_price == Decimal("55.00")
    assert fund.value == Decimal("55000.00")
    assert fund.lot_quantity == Decimal("1000")
    assert fund.lot_cost_basis == Decimal("50000.00")
    equity = positions["INE000A01012"]
    assert equity.value == Decimal("100000.00")
    assert equity.lot_quantity == Decimal("0")
    assert equity.lot_cost_basis == Decimal("0")
    assert await get_latest_values(session) == {
        "INE000A01012": Decimal("100000.00"),
        "INE000A01018": Decimal("55000.00"),
    }


# ---------------------------------------------------------------------------
# Multi-PAN lots + the persisted-lot vs projected-eligibility contract
# ---------------------------------------------------------------------------


async def test_persisted_lot_count_vs_projected_eligibility_contract(session):
    """Projection-eligibility contract, read from the same two accessors the
    projection uses (``get_complete_lots`` + ``get_lot_consumption``) WITHOUT
    calling the projection: unresolved instruments are suppressed and an exactly
    consumed persisted acquisition has zero remaining quantity.

    PAN1 contributes:
      - ISIN-A: clean purchase  -> lot persisted, eligible
      - ISIN-B: purchase + untied redemption -> lot persisted, SUPPRESSED
      - ISIN-D: linked switch (shared ref, matching magnitude) -> resolved,
        fully consumed
    PAN2 contributes:
      - ISIN-C: clean purchase -> lot persisted, eligible
    """
    from financial_dashboard.services.investments import (
        get_lot_consumption,
    )

    pan1 = [
        _mf_purchase(isin="INE000A01030", source_ref="p1/a", reference="PA1"),
        # ISIN-B: acquisition then a free-standing (untied) redemption.
        _mf_purchase(
            isin="INE000B01030",
            source_ref="p1/b",
            reference="PB1",
            units="100",
            nav="10.00",
            amount="1000.00",
        ),
        _mf_purchase(
            isin="INE000B01030",
            transaction_type="redemption",
            units="-40",
            amount="-440.00",
            nav="11.00",
            source_ref="p1/br",
            reference="RB1",
        ),
        # ISIN-D: a genuine linked switch — shared source_ref, matching magnitude.
        {
            "scope": "mf",
            "source_ref": "sw/d",
            "date": "2026-01-01",
            "description": "Switch Fund",
            "isin": "INE000D01030",
            "transaction_type": "switch_in",
            "units": "100",
            "nav": "10.00",
            "amount": "1000.00",
            "reference": "SD1",
        },
        {
            "scope": "mf",
            "source_ref": "sw/d",
            "date": "2026-01-01",
            "description": "Switch Fund",
            "isin": "INE000D01030",
            "transaction_type": "switch_out",
            "units": "-100",
            "nav": "10.00",
            "amount": "-1000.00",
            "reference": "SD1",
        },
    ]
    pan2 = [
        _mf_purchase(isin="INE000C01030", source_ref="p2/c", reference="PC1"),
    ]
    up1 = await _upload(session, portfolio_key="PAN1111A", payload_txns=pan1)
    up2 = await _upload(session, portfolio_key="PAN2222B", payload_txns=pan2)

    await create_investment_lots(
        session, cas_upload_id=up1.id, payload=json.loads(up1.raw_holdings_json)
    )
    await create_investment_lots(
        session, cas_upload_id=up2.id, payload=json.loads(up2.raw_holdings_json)
    )

    persisted = await get_complete_lots(session)
    consumption = await get_lot_consumption(session)
    unresolved = consumption.unresolved_instruments

    # Four complete lots persisted across both PANs (A, B, D from PAN1; C from PAN2).
    assert {lot.instrument_id for lot in persisted} == {
        "INE000A01030",
        "INE000B01030",
        "INE000C01030",
        "INE000D01030",
    }
    lot_c = next(lot for lot in persisted if lot.instrument_id == "INE000C01030")
    assert lot_c.cas_upload_id == up2.id
    assert (lot_c.source_ref, lot_c.reference) == ("p2/c", "PC1")
    # Only ISIN-B has an unresolvable (untied) disposal; the linked switch (D)
    # is exactly tied, so it is fully consumed rather than flagged.
    assert unresolved == {"INE000B01030"}
    assert {
        key.instrument_id
        for key, quantity in consumption.remaining.items()
        if quantity == 0
    } == {"INE000D01030"}

    # Re-derive this fixture's projected set from the accessor: one natural lot
    # per instrument, so a zero remaining instrument is removed as well.
    consumed = {
        key.instrument_id
        for key, quantity in consumption.remaining.items()
        if quantity == 0
    }
    eligible = [
        lot for lot in persisted if lot.instrument_id not in unresolved | consumed
    ]
    assert {lot.instrument_id for lot in eligible} == {
        "INE000A01030",
        "INE000C01030",
    }
    assert len(persisted) == 4
    assert len(eligible) == 2


# ---------------------------------------------------------------------------
# Canonical projection inputs: overlap, multiplicity, identity, valuation
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


async def test_canonical_lots_deduplicate_overlapping_monthly_cas(session):
    payload = {"transactions": [_mf_purchase()]}
    january = await _upload_payload(
        session,
        portfolio_key="PAN-OVERLAP",
        statement_date="2026-01-31",
        payload=payload,
    )
    february = await _upload_payload(
        session,
        portfolio_key="PAN-OVERLAP",
        statement_date="2026-02-28",
        payload=payload,
    )

    assert len(await get_complete_lots(session)) == 2
    canonical = await get_canonical_lots(session)

    assert len(canonical) == 1
    lot = canonical[0]
    assert lot.key.quantity == Decimal("1000")
    assert lot.key.cost_basis == Decimal("50000")
    assert lot.key.occurrence == 0
    assert lot.canonical_cas_upload_id == january.id
    assert tuple(item.cas_upload_id for item in lot.provenance) == (
        january.id,
        february.id,
    )


async def test_canonical_lots_keep_genuine_duplicate_multiplicity(session):
    payload = {"transactions": [_mf_purchase(), _mf_purchase()]}
    first = await _upload_payload(
        session,
        portfolio_key="PAN-MULTI",
        statement_date="2026-01-31",
        payload=payload,
    )
    second = await _upload_payload(
        session,
        portfolio_key="PAN-MULTI",
        statement_date="2026-02-28",
        payload=payload,
    )

    canonical = await get_canonical_lots(session)

    assert [lot.key.occurrence for lot in canonical] == [0, 1]
    assert all(
        tuple(item.cas_upload_id for item in lot.provenance) == (first.id, second.id)
        for lot in canonical
    )


async def test_canonical_disposal_state_deduplicates_overlapping_history(session):
    payload = {
        "transactions": [
            _mf_purchase(
                source_ref="lot/ref",
                units="100",
                nav="10",
                amount="1000",
                reference="BUY-1",
            ),
            _disposal(
                source_ref="lot/ref",
                units="-25",
                reference="SALE-1",
            ),
        ]
    }
    await _upload_payload(
        session,
        portfolio_key="PAN-DISPOSAL",
        statement_date="2026-03-31",
        payload=payload,
    )
    await _upload_payload(
        session,
        portfolio_key="PAN-DISPOSAL",
        statement_date="2026-04-30",
        payload=payload,
    )

    lots = await get_canonical_lots(session)
    consumption = await get_canonical_lot_consumption(session)

    assert len(lots) == 1
    assert consumption.unresolved == set()
    assert consumption.remaining == {lots[0].key: Decimal("75")}


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
    # The legacy aggregate sums instead of reverting to ISIN last-write-wins.
    aggregate = (await get_positions(session))[0]
    assert aggregate.quantity == Decimal("100")
    assert aggregate.value == Decimal("10000")


async def test_latest_valuation_changes_without_changing_acquisition_cost(session):
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

    canonical = await get_canonical_lots(session)
    valuations = await get_current_valuations(session)

    assert len(canonical) == 1
    assert canonical[0].key.unit_cost == Decimal("50")
    assert canonical[0].key.cost_basis == Decimal("500")
    assert len(valuations) == 1
    assert valuations[0].cas_upload_id == latest.id
    assert valuations[0].unit_price == Decimal("80")
    assert valuations[0].value == Decimal("800")
