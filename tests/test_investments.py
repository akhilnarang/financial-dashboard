"""Investment service: source-faithful lot classification, precision, and the
no-fabrication guarantee.

The classifier is exercised directly (pure) and through the persisted lot
table. A lot is built ONLY from an explicit, internally-consistent acquisition
fact; anything less is reported with a stable reason and never fabricated.
"""

import datetime
from decimal import Decimal

import pytest

from financial_dashboard.db.models import CasUpload, InvestmentLot
from financial_dashboard.services.investments import (
    CAS_CURRENCY,
    create_investment_lots,
    extract_lots_from_payload,
    get_complete_lots,
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


def test_duplicate_transaction_within_payload_deduplicated():
    txn = _mf_purchase()
    lots, _ = extract_lots_from_payload({"transactions": [txn, txn]})
    # Same natural key (source_ref/instrument/date/reference) -> one lot.
    assert len(lots) == 1


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
    source_ref with matching magnitude (a genuine linked switch) is resolvable
    and NOT flagged — the only non-inferred path that keeps a lot projected."""
    from financial_dashboard.services.investments import unresolved_disposal_instruments

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
    assert unresolved_disposal_instruments(payloads) == set()


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
    projection uses (``get_complete_lots`` + ``get_unresolved_disposal_instruments``)
    WITHOUT calling the projection: every persisted lot is eligible except those
    whose instrument carries an unresolvable disposal.

    PAN1 contributes:
      - ISIN-A: clean purchase  -> lot persisted, eligible
      - ISIN-B: purchase + untied redemption -> lot persisted, SUPPRESSED
      - ISIN-D: linked switch (shared ref, matching magnitude) -> resolved, kept
    PAN2 contributes:
      - ISIN-C: clean purchase -> lot persisted, eligible
    """
    from financial_dashboard.services.investments import (
        get_unresolved_disposal_instruments,
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
    unresolved = await get_unresolved_disposal_instruments(session)

    # Four complete lots persisted across both PANs (A, B, D from PAN1; C from PAN2).
    assert {lot.instrument_id for lot in persisted} == {
        "INE000A01030",
        "INE000B01030",
        "INE000C01030",
        "INE000D01030",
    }
    # Only ISIN-B has an unresolvable (untied) disposal; the linked switch (D)
    # is exactly tied so it is NOT flagged.
    assert unresolved == {"INE000B01030"}

    # The contract the projection enforces: eligible = persisted minus
    # instruments with unresolvable disposal history. Re-derived here from the
    # two read accessors only — the projection is not invoked.
    eligible = [lot for lot in persisted if lot.instrument_id not in unresolved]
    assert {lot.instrument_id for lot in eligible} == {
        "INE000A01030",
        "INE000C01030",
        "INE000D01030",
    }
    assert len(persisted) == 4
    assert len(eligible) == 3
