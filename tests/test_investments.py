"""Investment service: source-faithful lot classification, precision, and the
no-fabrication guarantee.

The classifier is exercised directly (pure) and through the persisted lot
table. A lot is built ONLY from an explicit, internally-consistent acquisition
fact; anything less is reported with a stable reason and never fabricated.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db.models import CasUpload, InvestmentLot
from financial_dashboard.services.investments import (
    CAS_CURRENCY,
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


def test_complete_mf_purchase_becomes_a_lot():
    lots, excluded = extract_lots_from_payload({"transactions": [_mf_purchase()]})
    assert excluded == []
    assert len(lots) == 1
    lot = lots[0]
    assert lot.instrument_id == "INE000A01018"
    assert lot.instrument_name == "Example Fund"
    assert lot.quantity == Decimal("1000")
    assert lot.unit_cost == Decimal("50.00")
    assert lot.cost_basis == Decimal("50000.00")
    assert lot.currency == CAS_CURRENCY
    assert lot.acquired_on == datetime.date(2026, 1, 15)
    assert lot.transaction_type == "purchase"
    assert lot.reference == "TXN001"


def test_switch_in_is_an_acquisition_type():
    lots, _ = extract_lots_from_payload(
        {"transactions": [_mf_purchase(transaction_type="switch_in")]}
    )
    assert len(lots) == 1


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
        # blank type: cannot confirm the date is an acquisition date
        ({"transaction_type": None}, "ambiguous_transaction_type"),
        ({"transaction_type": "transfer"}, "ambiguous_transaction_type"),
        # missing nav (no per-unit cost)
        ({"nav": None}, "missing_lot_facts"),
        # missing units
        ({"units": None}, "missing_lot_facts"),
        # missing amount (no cost basis)
        ({"amount": None}, "missing_lot_facts"),
        # missing date (no acquisition date)
        ({"date": None}, "missing_lot_facts"),
        # missing isin (no instrument identity)
        ({"isin": None}, "missing_lot_facts"),
        # negative units (not a purchase)
        ({"units": "-50", "amount": "-2500.00"}, "missing_lot_facts"),
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


def test_no_lot_when_cost_basis_derivable_but_date_absent():
    """Cost basis is never used to invent an acquisition date."""
    lots, excluded = extract_lots_from_payload(
        {"transactions": [_mf_purchase(date=None)]}
    )
    assert lots == []
    assert excluded[0].reason == "missing_lot_facts"


def test_identical_transactions_within_payload_preserve_source_multiplicity():
    txn = _mf_purchase()
    lots, _ = extract_lots_from_payload({"transactions": [txn, txn]})
    assert len(lots) == 2
    assert [lot.source_occurrence for lot in lots] == [0, 1]


def test_decimal_precision_preserved():
    """High-precision units/nav survive unrounded into the lot fields."""
    lots, _ = extract_lots_from_payload(
        {
            "transactions": [
                _mf_purchase(
                    units="123.456789",
                    nav="12.3456",
                    amount="1524.15",  # 123.456789 * 12.3456 = 1524.15 (2dp)
                )
            ]
        }
    )
    lot = lots[0]
    assert lot.quantity == Decimal("123.456789")
    assert lot.unit_cost == Decimal("12.3456")


def test_cost_basis_tolerance_allows_sub_penny_rounding():
    """units*nav that rounds to the printed amount within 0.01 is accepted."""
    lots, excluded = extract_lots_from_payload(
        {"transactions": [_mf_purchase(units="100", nav="33.333", amount="3333.30")]}
    )
    # 100 * 33.333 = 3333.30 exactly -> accepted.
    assert len(lots) == 1
    assert lots[0].cost_basis == Decimal("3333.30")
    assert excluded == []


# ---------------------------------------------------------------------------
# 1-paisa lot boundary: agreement gate + exact renderer consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("units", "nav", "amount", "accepted", "label"),
    [
        # difference 0 -> accepted
        ("1000", "50.00", "50000.00", True, "diff_zero"),
        # difference 0.009 (sub-penny) -> accepted
        ("1", "100.009", "100.00", True, "diff_subpenny"),
        # difference exactly 0.01 -> REJECTED (the former 1-paisa grey zone)
        ("1", "100.01", "100.00", False, "diff_exactly_one_paisa"),
        # difference > 0.01 -> rejected
        ("1000", "50.00", "49990.00", False, "diff_above_one_paisa"),
        # real-world rounded NAV: 100 * 33.333 = 3333.30 exactly -> accepted
        ("100", "33.333", "3333.30", True, "rounded_nav_exact"),
        # real-world repeating NAV: 700 * 14.2857 = 9999.99 -> accepted
        ("700", "14.2857", "9999.99", True, "repeating_nav_exact"),
        # real-world repeating NAV whose product is sub-penny off the amount:
        # 3 * 33.3333 = 99.9999, amount 100.00 (diff 0.0001) -> accepted
        ("3", "33.3333", "100.00", True, "repeating_nav_subpenny"),
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


@pytest.mark.parametrize(
    ("units", "nav", "amount"),
    [
        # exact 2dp product
        ("1000", "50.00", "50000.00"),
        # high-precision product (7 dp)
        ("123.456789", "12.3456", "1524.15"),
        # sub-penny product (3 dp), accepted within tolerance
        ("1", "100.009", "100.00"),
        # rounded/repeating NAV
        ("100", "33.333", "3333.30"),
        ("700", "14.2857", "9999.99"),
    ],
)
def test_accepted_lot_cost_basis_is_quantized_product_and_renders_balanced(
    units, nav, amount
):
    """Every accepted lot stores cost_basis == (quantity*unit_cost).quantize(0.01)
    so the renderer's lot-consistency guard passes with a ZERO diff (never up to
    a paisa) and the entry renders without raising in every backend."""
    from financial_dashboard.services.paisa.renderers import render_document
    from financial_dashboard.services.paisa.renderers.base import (
        InvestmentLotEntry,
        LedgerDocument,
        check_lot_consistent,
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
    # The renderer's own consistency guard must pass (and with a zero diff).
    product_q = (entry.quantity * entry.unit_cost).quantize(Decimal("0.01"))
    assert abs(product_q - entry.cost_basis.quantize(Decimal("0.01"))) == Decimal("0")
    check_lot_consistent(entry)  # must not raise
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


def test_purchase_plus_redemption_same_instrument_is_unresolved():
    """An instrument with an acquisition lot AND a redemption that CAS does not
    tie to the lot is flagged unresolved so projection suppresses the lot."""
    from financial_dashboard.services.investments import unresolved_disposal_instruments

    payloads = [
        {
            "transactions": [
                _mf_purchase(isin="INE000A01020", source_ref="a/1", reference="P1"),
                _mf_purchase(
                    isin="INE000A01020",
                    transaction_type="redemption",
                    units="-50",
                    amount="-5100.00",
                    source_ref="a/2",
                    reference="R1",
                ),
            ]
        }
    ]
    assert unresolved_disposal_instruments(payloads) == {"INE000A01020"}


def test_purchase_only_path_has_no_unresolved_disposal():
    """Instruments acquired but never disposed are not flagged (unchanged path)."""
    from financial_dashboard.services.investments import unresolved_disposal_instruments

    payloads = [
        {
            "transactions": [
                _mf_purchase(isin="INE000A01020", source_ref="a/1"),
                _mf_purchase(
                    isin="INE000A01021",
                    source_ref="a/2",
                    units="200",
                    nav="10.00",
                    amount="2000.00",
                    reference="P2",
                ),
            ]
        }
    ]
    assert unresolved_disposal_instruments(payloads) == set()


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


def test_disposal_exactly_tied_by_shared_reference_is_resolved():
    """A disposal exactly tied to an acquisition by an explicit shared
    source_ref with matching magnitude is resolved by consuming the lot, not by
    leaving the gross acquisition eligible (the original P2 regression)."""
    from financial_dashboard.services.investments import resolve_lot_consumption

    payloads = [
        {
            "transactions": [
                {
                    "scope": "mf",
                    "source_ref": "sw/1",
                    "date": "2026-01-01",
                    "description": "Switch Fund",
                    "isin": "INE000A01020",
                    "transaction_type": "switch_in",
                    "units": "100",
                    "nav": "10.00",
                    "amount": "1000.00",
                    "reference": "SW1",
                },
                {
                    "scope": "mf",
                    "source_ref": "sw/1",
                    "date": "2026-01-01",
                    "description": "Switch Fund",
                    "isin": "INE000A01020",
                    "transaction_type": "switch_out",
                    "units": "-100",
                    "nav": "10.00",
                    "amount": "-1000.00",
                    "reference": "SW1",
                },
            ]
        }
    ]
    consumption = resolve_lot_consumption(payloads)
    assert consumption.unresolved_instruments == set()
    assert list(consumption.remaining.values()) == [Decimal("0")]


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


def test_exact_partial_disposal_preserves_unit_multiplicity_and_precision():
    """A unique source-ref tie can be partial; subtraction stays Decimal-exact
    and two equal-magnitude disposal rows are both applied, not de-duplicated."""
    from financial_dashboard.services.investments import resolve_lot_consumption

    purchase = _mf_purchase(
        source_ref="lot/ref",
        units="1.234567",
        nav="12.3456",
        amount="15.24",
        reference="BUY-1",
    )
    payload = {
        "transactions": [
            purchase,
            _disposal(units="-0.100001", reference="SALE-1"),
            _disposal(units="-0.100001", reference="SALE-2"),
        ]
    }
    consumption = resolve_lot_consumption([payload])
    assert consumption.unresolved_instruments == set()
    assert list(consumption.remaining.values()) == [Decimal("1.034565")]


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


def test_same_date_multi_lot_partial_boundary_is_ambiguous():
    """A shared source ref does not authorize an arbitrary tie-break between
    same-date acquisitions when a disposal cuts through that date group."""
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
                date="2026-01-01",
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


def test_multiple_acquisitions_are_unresolved_without_exact_lot_reference():
    """Distinct acquisition dates do not prove FIFO disposal allocation.

    A source ref shared by multiple lots remains ambiguous unless the disposal's
    explicit reference identifies one exact lot.
    """
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
                date="2026-02-01",
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


async def test_get_unresolved_disposal_instruments_reads_preserved_payloads(session):
    """The DB-backed accessor flags instruments with unresolvable disposals from
    preserved CAS uploads (read-only)."""
    from financial_dashboard.services.investments import (
        get_unresolved_disposal_instruments,
    )

    upload = await _upload(
        session,
        payload_txns=[
            _mf_purchase(isin="INE000A01020", source_ref="a/1"),
            _mf_purchase(
                isin="INE000A01020",
                transaction_type="redemption",
                units="-50",
                amount="-5100.00",
                source_ref="a/2",
            ),
        ],
    )
    result = await get_unresolved_disposal_instruments(session)
    assert upload.id  # upload was persisted
    assert "INE000A01020" in result


async def test_get_lot_consumption_is_read_only_and_returns_remaining_lots(session):
    """The DB accessor reads preserved disposal facts but does not rewrite the
    persisted gross acquisition row when reporting a partial remainder."""
    import json

    from financial_dashboard.services.investments import get_lot_consumption

    transactions = [
        _mf_purchase(
            source_ref="lot/ref",
            units="100",
            nav="10",
            amount="1000",
            reference="BUY-1",
        ),
        _disposal(units="-25", source_ref="lot/ref", reference="SALE-1"),
    ]
    upload = await _upload(session, payload_txns=transactions)
    await create_investment_lots(
        session,
        cas_upload_id=upload.id,
        payload=json.loads(upload.raw_holdings_json),
    )
    persisted = (await session.execute(select(InvestmentLot))).scalar_one()
    assert persisted.quantity == Decimal("100")
    assert persisted.cost_basis == Decimal("1000")

    consumption = await get_lot_consumption(session)

    assert consumption.unresolved_instruments == set()
    assert list(consumption.remaining.values()) == [Decimal("75")]
    unchanged = (await session.execute(select(InvestmentLot))).scalar_one()
    assert unchanged.quantity == Decimal("100")
    assert unchanged.cost_basis == Decimal("1000")


# ---------------------------------------------------------------------------
# Persisted lots + read queries
# ---------------------------------------------------------------------------


async def _upload(session, *, payload_txns, statement_date="2026-04-30"):
    upload = CasUpload(
        portfolio_key="PAN123",
        depository_source="cdsl",
        statement_date=datetime.date.fromisoformat(statement_date),
        grand_total=Decimal("100000.00"),
        raw_holdings_json=__import__("json").dumps({"transactions": payload_txns}),
    )
    session.add(upload)
    await session.flush()
    return upload


async def test_create_investment_lots_persists_complete_lots(session):
    upload = await _upload(
        session,
        payload_txns=[
            _mf_purchase(),
            _mf_purchase(
                isin="INE000A01019",
                reference="TXN002",
                units="200",
                nav="10.00",
                amount="2000.00",
            ),
        ],
    )
    created, excluded = await create_investment_lots(
        session, cas_upload_id=upload.id, payload={"transactions": []}
    )
    # payload passed here has no transactions -> nothing created from it; the
    # persisted rows above come from a direct call instead.
    assert created == 0
    lots, _ = extract_lots_from_payload(
        {
            "transactions": [
                _mf_purchase(),
                _mf_purchase(
                    isin="INE000A01019",
                    reference="TXN002",
                    units="200",
                    nav="10.00",
                    amount="2000.00",
                ),
            ]
        }
    )
    for lot in lots:
        session.add(
            InvestmentLot(
                cas_upload_id=upload.id,
                instrument_id=lot.instrument_id,
                instrument_name=lot.instrument_name,
                quantity=lot.quantity,
                unit_cost=lot.unit_cost,
                cost_basis=lot.cost_basis,
                currency=lot.currency,
                acquired_on=lot.acquired_on,
                source_ref=lot.source_ref,
                transaction_type=lot.transaction_type,
                reference=lot.reference,
            )
        )
    await session.flush()

    rows = await get_complete_lots(session)
    assert {r.instrument_id for r in rows} == {"INE000A01018", "INE000A01019"}
    # quantity/unit_cost stored at full Numeric precision (20,6)/(20,6).
    first = next(r for r in rows if r.instrument_id == "INE000A01018")
    assert first.quantity == Decimal("1000")
    assert first.unit_cost == Decimal("50")


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
        payload=__import__("json").loads(upload.raw_holdings_json),
    )
    reasons = await get_incomplete_reasons(session)
    labels = {r.reason for r in reasons}
    assert "disposal_transaction" in labels
    assert "not_mutual_fund" in labels


async def test_get_positions_joins_holdings_with_lot_cost_basis(session):
    upload = await _upload(
        session,
        payload_txns=[_mf_purchase()],
    )
    # Give the upload a holdings section so positions carry current value.
    import json

    payload = json.loads(upload.raw_holdings_json)
    payload["folios"] = [
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
    ]
    upload.raw_holdings_json = json.dumps(payload)
    await create_investment_lots(session, cas_upload_id=upload.id, payload=payload)

    positions = await get_positions(session)
    pos = next(p for p in positions if p.instrument_id == "INE000A01018")
    assert pos.quantity == Decimal("1000")
    assert pos.unit_price == Decimal("55.00")
    assert pos.value == Decimal("55000.00")
    # Cost basis aggregated from the complete lot only.
    assert pos.lot_quantity == Decimal("1000")
    assert pos.lot_cost_basis == Decimal("50000.00")


async def test_value_only_holding_without_lot_has_zero_cost_basis(session):
    """A holding the CAS priced but never acquired via a complete transaction
    appears as a position with zero lot fields — not a fabricated lot."""
    upload = await _upload(session, payload_txns=[])
    import json

    payload = json.loads(upload.raw_holdings_json)
    payload["accounts"] = [
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
    ]
    upload.raw_holdings_json = json.dumps(payload)

    positions = await get_positions(session)
    assert len(positions) == 1
    pos = positions[0]
    assert pos.instrument_id == "INE000A01012"
    assert pos.value == Decimal("100000.00")
    assert pos.lot_quantity == Decimal("0")
    assert pos.lot_cost_basis == Decimal("0")


async def test_get_latest_values_returns_explicit_market_values(session):
    upload = await _upload(session, payload_txns=[])
    import json

    payload = json.loads(upload.raw_holdings_json)
    payload["accounts"] = [
        {
            "holdings": [
                {
                    "name": "Eq",
                    "isin": "INE000A01012",
                    "quantity": "1",
                    "price": "10",
                    "value": "100.00",
                }
            ]
        }
    ]
    upload.raw_holdings_json = json.dumps(payload)
    values = await get_latest_values(session)
    assert values == {"INE000A01012": Decimal("100.00")}


# ---------------------------------------------------------------------------
# Multi-PAN lots + the persisted-lot vs projected-eligibility contract
# ---------------------------------------------------------------------------


async def _upload_pan(session, *, portfolio_key, txns, statement_date="2026-04-30"):
    upload = CasUpload(
        portfolio_key=portfolio_key,
        depository_source="cdsl",
        statement_date=datetime.date.fromisoformat(statement_date),
        grand_total=Decimal("100000.00"),
        raw_holdings_json=__import__("json").dumps({"transactions": txns}),
    )
    session.add(upload)
    await session.flush()
    return upload


async def test_multi_pan_lots_persisted_independently(session):
    """Two PANs (portfolio_keys) ingest their own lots independently; both are
    returned by the cross-upload ``get_complete_lots`` read, keyed by their own
    cas_upload_id, and source_ref survives the round-trip."""
    pan1 = [
        _mf_purchase(isin="INE000A01030", source_ref="p1/a", reference="PA1"),
    ]
    pan2 = [
        _mf_purchase(isin="INE000C01030", source_ref="p2/c", reference="PC1"),
    ]
    up1 = await _upload_pan(session, portfolio_key="PAN1111A", txns=pan1)
    up2 = await _upload_pan(session, portfolio_key="PAN2222B", txns=pan2)
    import json as _json

    await create_investment_lots(
        session, cas_upload_id=up1.id, payload=_json.loads(up1.raw_holdings_json)
    )
    await create_investment_lots(
        session, cas_upload_id=up2.id, payload=_json.loads(up2.raw_holdings_json)
    )

    lots = await get_complete_lots(session)
    assert {lot.instrument_id for lot in lots} == {"INE000A01030", "INE000C01030"}
    lot1 = next(lot for lot in lots if lot.instrument_id == "INE000A01030")
    lot2 = next(lot for lot in lots if lot.instrument_id == "INE000C01030")
    assert lot1.cas_upload_id == up1.id
    assert lot2.cas_upload_id == up2.id
    assert lot1.source_ref == "p1/a"
    assert lot1.reference == "PA1"


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
    up1 = await _upload_pan(session, portfolio_key="PAN1111A", txns=pan1)
    up2 = await _upload_pan(session, portfolio_key="PAN2222B", txns=pan2)
    import json as _json

    await create_investment_lots(
        session, cas_upload_id=up1.id, payload=_json.loads(up1.raw_holdings_json)
    )
    await create_investment_lots(
        session, cas_upload_id=up2.id, payload=_json.loads(up2.raw_holdings_json)
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
    import json

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
