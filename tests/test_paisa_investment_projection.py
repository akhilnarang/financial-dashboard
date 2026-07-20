"""Investment-lot projection: the ``paisa.project_investments`` gate, conservative
lot-posting syntax for ledger/hledger/beancount, the read-only guarantee (no
core writes), and diagnostics counts/reasons.

No ``ledger``/``hledger``/``bean-check`` binary is required — the lot syntax is
asserted structurally, and a live ``beancount`` parse is exercised only when the
``beancount`` package happens to be importable (optional tooling).
"""

import datetime
import re
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db.models import CasUpload, InvestmentLot, Transaction
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.projection import project
from financial_dashboard.services.paisa.renderers import render_document
from financial_dashboard.services.paisa.renderers.base import (
    INVESTMENT_ASSET_ROOT,
    INVESTMENT_EQUITY_OPENING,
    InvestmentLotEntry,
    LedgerDocument,
    UnbalancedEntry,
)

pytestmark = pytest.mark.anyio

CUTOVER = datetime.date(2026, 1, 1)


def _config(**overrides) -> PaisaProjectionConfig:
    base = dict(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=(1,),
        cutover_date=CUTOVER,
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
        ledger_cli="ledger",
        fx_rates={},
        project_investments=True,
    )
    base.update(overrides)
    return PaisaProjectionConfig(**base)


def _lot(**overrides) -> InvestmentLotEntry:
    base = dict(
        instrument="INE000A01018",
        instrument_name="Example Fund",
        quantity=Decimal("1000"),
        unit_cost=Decimal("50.00"),
        cost_basis=Decimal("50000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 1, 15),
    )
    base.update(overrides)
    return InvestmentLotEntry(**base)


async def _seed_lot(session, **overrides):
    kwargs = dict(
        cas_upload_id=1,
        instrument_id="INE000A01018",
        instrument_name="Example Fund",
        quantity=Decimal("1000"),
        unit_cost=Decimal("50"),
        cost_basis=Decimal("50000"),
        currency="INR",
        acquired_on=datetime.date(2026, 1, 15),
        source_ref="123/45",
        transaction_type="purchase",
        reference="TXN001",
    )
    kwargs.update(overrides)
    session.add(InvestmentLot(**kwargs))
    await session.flush()


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


async def test_projection_gate_default_off_emits_no_lots(session):
    """Without project_investments, a persisted lot is never projected."""
    await _seed_lot(session)
    # Need a bank account + txn so project() runs the main path.
    from financial_dashboard.db.models import Account, BalanceSnapshot
    from financial_dashboard.db.enums import (
        SnapshotCategory,
        SnapshotKind,
        SnapshotSource,
    )

    session.add(Account(id=1, bank="hdfc", label="Savings", type="bank_account"))
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=CUTOVER,
            value=Decimal("1000.00"),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await session.flush()

    report = await project(session, _config(project_investments=False))
    assert report.investment_lot_count == 0
    assert "Assets:Investments" not in report.journal


async def test_projection_gate_on_emits_lots(session):
    import json

    session.add(
        CasUpload(
            id=1,
            portfolio_key="PAN",
            depository_source="cdsl",
            statement_date=datetime.date(2026, 4, 30),
            grand_total=Decimal("50000"),
            raw_holdings_json=json.dumps({"transactions": []}),
        )
    )
    await session.flush()
    await _seed_lot(session)
    from financial_dashboard.db.models import Account, BalanceSnapshot
    from financial_dashboard.db.enums import (
        SnapshotCategory,
        SnapshotKind,
        SnapshotSource,
    )

    session.add(Account(id=1, bank="hdfc", label="Savings", type="bank_account"))
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=CUTOVER,
            value=Decimal("1000.00"),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await session.flush()

    report = await project(session, _config(project_investments=True))
    assert report.investment_lot_count == 1
    assert "Assets:Investments:INE000A01018" in report.journal
    assert "Equity:Opening Balances:Investment" in report.journal


# ---------------------------------------------------------------------------
# Renderer syntax (structural, per backend)
# ---------------------------------------------------------------------------


def _render_lot(backend, lot=None):
    lot = lot or _lot()
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(lot,),
    )
    return render_document(doc, backend)


def test_ledger_lot_syntax():
    out = _render_lot("ledger")
    # dated at the acquisition date, with quantity COMMODITY, cost annotation,
    # the [date] lot note, and the equity contra with the explicit cost basis.
    # The instrument commodity is the ledger-family token everywhere: quoted in
    # the amount AND the declaration. Ledger 3.3.2 rejects a bare alphanumeric
    # in a posting amount; hledger rejects one in a declaration/P too, so the
    # declaration is quoted identically.
    assert "2026-01-15 * Investment Lot - Example Fund" in out
    assert re.search(
        r'Assets:Investments:INE000A01018\s+1000 "INE000A01018" \{50\.00 INR\} \[2026-01-15\]',
        out,
    )
    assert 'commodity "INE000A01018"' in out
    assert "Equity:Opening Balances:Investment" in out
    assert "-50000.00 INR" in out
    # No bank cash leg is inferred.
    assert "Assets:Bank" not in out


def test_hledger_lot_syntax():
    out = _render_lot("hledger")
    # Same quoted token as ledger — hledger rejects a bare alphanumeric in the
    # amount, declaration, and P directive, so every position is quoted.
    assert re.search(
        r'Assets:Investments:INE000A01018\s+1000 "INE000A01018" \{50\.00 INR\} \[2026-01-15\]',
        out,
    )
    assert 'commodity "INE000A01018"' in out
    assert "-50000.00 INR" in out


def test_beancount_lot_syntax():
    out = _render_lot("beancount")
    # beancount cost syntax {unit CURRENCY, date}; commodity + open directives.
    # Beancount's all-caps commodity grammar reads the ISIN bare (unchanged by
    # the ledger-family quoting fix).
    assert re.search(
        r"Assets:Investments:INE000A01018\s+1000 INE000A01018 \{50\.00 INR, 2026-01-15\}",
        out,
    )
    assert "commodity INE000A01018" in out
    assert "open Assets:Investments:INE000A01018 INE000A01018" in out
    assert "Equity:OpeningBalances:Investment" in out
    assert "open Equity:OpeningBalances:Investment INR" in out


def test_beancount_lot_parses_when_library_available():
    """Live-parse the beancount output only when the package is importable."""
    pytest.importorskip("beancount")
    from beancount import loader

    out = _render_lot("beancount")
    # loader.load_string returns a 3-tuple ``(entries, errors, options_map)``;
    # the prior ``loads_string``/``errors, _errors, _options`` unpack was a
    # latent no-op (it iterated entries, not errors).
    _entries, errors, _options = loader.load_string(out)
    # The cost syntax we emit is valid beancount: no hard parse errors.
    assert errors == [], errors


# ---------------------------------------------------------------------------
# No double counting / no fabrication
# ---------------------------------------------------------------------------


def test_lot_posts_only_to_investment_hierarchy():
    for backend in ("ledger", "hledger", "beancount"):
        out = _render_lot(backend)
        # The asset side is the dedicated Investments hierarchy; no cash leg.
        assert INVESTMENT_ASSET_ROOT in out
        assert "Assets:Bank" not in out
        assert "Assets:Cash" not in out
        # The contra is the dedicated investment equity, not the bank opening.
        if backend == "beancount":
            assert "Equity:OpeningBalances:Investment" in out
        else:
            assert INVESTMENT_EQUITY_OPENING in out


def test_inconsistent_lot_raises_rather_than_emitting_unbalanced():
    lot = _lot(cost_basis=Decimal("99999.00"))  # != 1000 * 50
    with pytest.raises(UnbalancedEntry):
        _render_lot("ledger", lot)


# ---------------------------------------------------------------------------
# Read-only guarantee + diagnostics
# ---------------------------------------------------------------------------


async def test_projection_writes_no_core_rows(session):
    """project() is read-only: it must not insert/update transactions or lots."""
    from financial_dashboard.db.models import Account, BalanceSnapshot
    from financial_dashboard.db.enums import (
        SnapshotCategory,
        SnapshotKind,
        SnapshotSource,
    )

    session.add(Account(id=1, bank="hdfc", label="Savings", type="bank_account"))
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=CUTOVER,
            value=Decimal("1000.00"),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await _seed_lot(session)

    txn_count_before = (await session.execute(select(Transaction))).scalars().all()
    await project(session, _config(project_investments=True))
    txn_count_after = (await session.execute(select(Transaction))).scalars().all()
    assert len(txn_count_after) == len(txn_count_before)
    # The single seeded lot is still there; projection created no new ones.
    lots = (await session.execute(select(InvestmentLot))).scalars().all()
    assert len(lots) == 1


async def test_projection_reports_excluded_reasons(session):
    """A CAS upload with an excluded transaction surfaces its reason label."""
    import json

    from financial_dashboard.db.models import Account, BalanceSnapshot, CasUpload
    from financial_dashboard.db.enums import (
        SnapshotCategory,
        SnapshotKind,
        SnapshotSource,
    )

    session.add(Account(id=1, bank="hdfc", label="Savings", type="bank_account"))
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=CUTOVER,
            value=Decimal("1000.00"),
            source=SnapshotSource.bank_statement.value,
        )
    )
    session.add(
        CasUpload(
            portfolio_key="PAN",
            depository_source="cdsl",
            statement_date=datetime.date(2026, 4, 30),
            grand_total=Decimal("0"),
            raw_holdings_json=json.dumps(
                {
                    "transactions": [
                        {
                            "scope": "demat",
                            "source_ref": "d/1",
                            "date": "2026-01-01",
                            "isin": "INE000A01012",
                            "transaction_type": "purchase",
                            "quantity": "5",
                        }
                    ]
                }
            ),
        )
    )
    await session.flush()

    report = await project(session, _config(project_investments=True))
    assert "not_mutual_fund" in report.investment_excluded


def test_explicit_unit_cost_stays_on_lot_cost_annotation():
    """Acquisition cost is carried by the lot, not synthesized as market NAV."""
    out = _render_lot("ledger")
    assert "{50.00 INR}" in out


# ---------------------------------------------------------------------------
# Disposal/redemption safety: conservative lot suppression by instrument
# ---------------------------------------------------------------------------


def _mf_purchase_raw(**overrides):
    """A complete MF acquisition transaction as preserved in a CAS payload."""
    base = {
        "scope": "mf",
        "source_ref": "mf/1",
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


def _mf_disposal_raw(**overrides):
    base = _mf_purchase_raw(
        date="2026-03-01",
        transaction_type="redemption",
        units="-25",
        nav="55.00",
        amount="-1375.00",
        reference="SALE001",
    )
    base.update(overrides)
    return base


async def _upload_with_lots(session, transactions, lots, *, payload_extra=None):
    import json

    from financial_dashboard.db.models import CasUpload

    payload = {"transactions": transactions}
    payload.update(payload_extra or {})
    upload = CasUpload(
        portfolio_key="PAN",
        depository_source="cdsl",
        statement_date=datetime.date(2026, 4, 30),
        grand_total=Decimal("0"),
        raw_holdings_json=json.dumps(payload),
    )
    session.add(upload)
    await session.flush()
    for lot in lots:
        await _seed_lot(session, cas_upload_id=upload.id, **lot)
    return upload


async def _bank_and_snapshot(session):
    """Seed the bank account + cutover snapshot the projection needs to run."""
    from financial_dashboard.db.models import Account, BalanceSnapshot
    from financial_dashboard.db.enums import (
        SnapshotCategory,
        SnapshotKind,
        SnapshotSource,
    )

    session.add(Account(id=1, bank="hdfc", label="Savings", type="bank_account"))
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=CUTOVER,
            value=Decimal("1000.00"),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await session.flush()


async def test_projection_suppresses_lot_when_instrument_has_redemption(session):
    """A complete acquisition lot is NOT projected when the preserved CAS facts
    contain a redemption for the same instrument that cannot be truthfully
    allocated to the lot — the default projection never overstates holdings."""
    import json

    from financial_dashboard.db.models import CasUpload

    await _bank_and_snapshot(session)
    await _seed_lot(session)  # INE000A01018 acquisition lot
    # Same instrument also has a redemption in the preserved CAS facts.
    session.add(
        CasUpload(
            portfolio_key="PAN",
            depository_source="cdsl",
            statement_date=datetime.date(2026, 4, 30),
            grand_total=Decimal("0"),
            raw_holdings_json=json.dumps(
                {
                    "transactions": [
                        _mf_purchase_raw(),
                        _mf_purchase_raw(
                            transaction_type="redemption",
                            units="-50",
                            amount="-2550.00",
                            source_ref="mf/2",
                            reference="RED001",
                        ),
                    ]
                }
            ),
        )
    )
    await session.flush()

    report = await project(session, _config(project_investments=True))
    # The acquisition lot is suppressed — never overstate holdings.
    assert report.investment_lot_count == 0
    assert "Assets:Investments:INE000A01018" not in report.journal
    assert "INE000A01018" in report.investment_disposal_unresolved
    assert "disposal_history_unresolved" in report.investment_excluded


async def test_projection_emits_lot_when_no_disposal_history(session):
    """The unchanged no-disposal path: a complete lot is projected when the
    preserved CAS facts contain no redemption for the instrument."""
    import json

    from financial_dashboard.db.models import CasUpload

    await _bank_and_snapshot(session)
    await _seed_lot(session)  # INE000A01018 acquisition lot
    # CasUpload carries only the purchase (no redemption) -> not suppressed.
    session.add(
        CasUpload(
            portfolio_key="PAN",
            depository_source="cdsl",
            statement_date=datetime.date(2026, 4, 30),
            grand_total=Decimal("0"),
            raw_holdings_json=json.dumps({"transactions": [_mf_purchase_raw()]}),
        )
    )
    await session.flush()

    report = await project(session, _config(project_investments=True))
    assert report.investment_lot_count == 1
    assert "Assets:Investments:INE000A01018" in report.journal
    assert report.investment_disposal_unresolved == ()
    assert "disposal_history_unresolved" not in report.investment_excluded


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_full_disposal_removes_lot_and_price_without_mutating_row(
    session, backend
):
    """An exact full tie consumes the acquisition in every projection backend;
    the gross persisted source row remains untouched and no price is orphaned."""
    await _bank_and_snapshot(session)
    purchase = _mf_purchase_raw(
        source_ref="full/ref",
        units="100",
        nav="10",
        amount="1000",
        reference="BUY-FULL",
    )
    disposal = _mf_disposal_raw(
        source_ref="full/ref",
        units="-100",
        reference="SELL-FULL",
    )
    await _upload_with_lots(
        session,
        [purchase, disposal],
        [
            {
                "source_ref": "full/ref",
                "reference": "BUY-FULL",
                "quantity": Decimal("100"),
                "unit_cost": Decimal("10"),
                "cost_basis": Decimal("1000"),
            }
        ],
    )

    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )

    assert report.investment_lot_count == 0
    assert not any(
        price.currency == "INE000A01018" for price in report.document.price_directives
    )
    assert "INE000A01018" not in report.journal
    persisted = (await session.execute(select(InvestmentLot))).scalar_one()
    assert persisted.quantity == Decimal("100")
    assert persisted.cost_basis == Decimal("1000")


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_partial_disposal_projects_proportional_remainder_and_funding_dedup(
    session, backend
):
    """A unique partial tie keeps exact unit cost, reduces quantity/cost basis,
    remains available to exact-ref funding dedup, and never writes the DB row."""
    await _bank_and_snapshot(session)
    purchase = _mf_purchase_raw(
        source_ref="partial/ref",
        units="100",
        nav="10",
        amount="1000",
        reference="BUY-PARTIAL",
    )
    disposal = _mf_disposal_raw(
        source_ref="partial/ref",
        units="-25",
        reference="SELL-PARTIAL",
    )
    await _upload_with_lots(
        session,
        [purchase, disposal],
        [
            {
                "source_ref": "partial/ref",
                "reference": "BUY-PARTIAL",
                "quantity": Decimal("100"),
                "unit_cost": Decimal("10"),
                "cost_basis": Decimal("1000"),
            }
        ],
    )
    session.add(
        Transaction(
            account_id=1,
            bank="hdfc",
            email_type="partial_funding",
            direction="debit",
            amount=Decimal("1000"),
            transaction_date=datetime.date(2026, 1, 15),
            category="investment",
            counterparty="MF Purchase",
            reference_number="partial/ref",
            currency="INR",
        )
    )
    await session.flush()

    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )

    assert report.investment_lot_count == 1
    lot = report.document.lot_postings[0]
    assert lot.quantity == Decimal("75")
    assert lot.unit_cost == Decimal("10")
    assert lot.cost_basis == Decimal("750.00")
    assert report.investment_funding_remapped == 1
    # Acquisition unit cost stays on the lot annotation; it is not mislabeled
    # as a market quote when the latest CAS has no explicit current holding NAV.
    assert not any(
        price.currency == "INE000A01018" for price in report.document.price_directives
    )
    persisted = (await session.execute(select(InvestmentLot))).scalar_one()
    assert persisted.quantity == Decimal("100")
    assert persisted.cost_basis == Decimal("1000")


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
@pytest.mark.parametrize("case", ["over", "same_date_ambiguous"])
async def test_non_deterministic_disposal_suppresses_affected_instrument(
    session, backend, case
):
    """Over-disposal and a same-date multi-lot boundary never guess an
    allocation; all affected lots and prices are suppressed in every backend."""
    await _bank_and_snapshot(session)
    if case == "over":
        transactions = [
            _mf_purchase_raw(
                source_ref="risk/ref",
                units="100",
                nav="10",
                amount="1000",
                reference="BUY-A",
            ),
            _mf_disposal_raw(
                source_ref="risk/ref", units="-100.000001", reference="SELL"
            ),
        ]
        lots = [
            {
                "source_ref": "risk/ref",
                "reference": "BUY-A",
                "quantity": Decimal("100"),
                "unit_cost": Decimal("10"),
                "cost_basis": Decimal("1000"),
            }
        ]
    else:
        transactions = [
            _mf_purchase_raw(
                source_ref="risk/ref",
                units="40",
                nav="10",
                amount="400",
                reference="BUY-A",
            ),
            _mf_purchase_raw(
                source_ref="risk/ref",
                units="60",
                nav="20",
                amount="1200",
                reference="BUY-B",
            ),
            _mf_disposal_raw(source_ref="risk/ref", units="-50", reference="SELL"),
        ]
        lots = [
            {
                "source_ref": "risk/ref",
                "reference": "BUY-A",
                "quantity": Decimal("40"),
                "unit_cost": Decimal("10"),
                "cost_basis": Decimal("400"),
            },
            {
                "source_ref": "risk/ref",
                "reference": "BUY-B",
                "quantity": Decimal("60"),
                "unit_cost": Decimal("20"),
                "cost_basis": Decimal("1200"),
            },
        ]
    await _upload_with_lots(session, transactions, lots)

    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )

    assert report.investment_lot_count == 0
    assert report.investment_disposal_unresolved == ("INE000A01018",)
    assert "disposal_history_unresolved" in report.investment_excluded
    assert "INE000A01018" not in report.journal
    assert not any(
        price.currency == "INE000A01018" for price in report.document.price_directives
    )


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_multi_lot_without_exact_reference_suppresses_instrument(
    session, backend
):
    """Acquisition dates do not authorize inferred FIFO in any backend.

    With two possible lots and no exact acquisition reference, the instrument
    and all of its price directives are suppressed rather than fabricated.
    """
    await _bank_and_snapshot(session)
    instrument = "INE000A01018"
    transactions = [
        _mf_purchase_raw(
            source_ref="fifo/ref",
            date="2026-01-10",
            units="40",
            nav="10",
            amount="400",
            reference="BUY-OLD",
        ),
        _mf_purchase_raw(
            source_ref="fifo/ref",
            date="2026-02-10",
            units="60",
            nav="20",
            amount="1200",
            reference="BUY-NEW",
        ),
        _mf_disposal_raw(
            source_ref="fifo/ref",
            date="2026-03-10",
            units="-50",
            reference="SELL-FIFO",
        ),
    ]
    await _upload_with_lots(
        session,
        transactions,
        [
            {
                "source_ref": "fifo/ref",
                "acquired_on": datetime.date(2026, 1, 10),
                "reference": "BUY-OLD",
                "quantity": Decimal("40"),
                "unit_cost": Decimal("10"),
                "cost_basis": Decimal("400"),
            },
            {
                "source_ref": "fifo/ref",
                "acquired_on": datetime.date(2026, 2, 10),
                "reference": "BUY-NEW",
                "quantity": Decimal("60"),
                "unit_cost": Decimal("20"),
                "cost_basis": Decimal("1200"),
            },
        ],
    )

    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )

    assert report.investment_lot_count == 0
    assert report.investment_disposal_unresolved == (instrument,)
    assert "disposal_history_unresolved" in report.investment_excluded
    assert instrument not in report.journal
    assert not any(
        price.currency == instrument for price in report.document.price_directives
    )


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_same_ref_different_instrument_consumes_only_matching_instrument(
    session, backend
):
    await _bank_and_snapshot(session)
    instrument_b = "INE000B01018"
    transactions = [
        _mf_purchase_raw(
            isin="INE000A01018",
            source_ref="shared/ref",
            units="100",
            nav="10",
            amount="1000",
            reference="BUY-A",
        ),
        _mf_purchase_raw(
            isin=instrument_b,
            source_ref="shared/ref",
            units="70",
            nav="20",
            amount="1400",
            reference="BUY-B",
        ),
        _mf_disposal_raw(
            isin="INE000A01018",
            source_ref="shared/ref",
            units="-100",
            reference="SELL-A",
        ),
    ]
    await _upload_with_lots(
        session,
        transactions,
        [
            {
                "instrument_id": "INE000A01018",
                "source_ref": "shared/ref",
                "reference": "BUY-A",
                "quantity": Decimal("100"),
                "unit_cost": Decimal("10"),
                "cost_basis": Decimal("1000"),
            },
            {
                "instrument_id": instrument_b,
                "instrument_name": "Fund B",
                "source_ref": "shared/ref",
                "reference": "BUY-B",
                "quantity": Decimal("70"),
                "unit_cost": Decimal("20"),
                "cost_basis": Decimal("1400"),
            },
        ],
    )

    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )

    assert report.investment_lot_count == 1
    assert report.document.lot_postings[0].instrument == instrument_b
    assert "INE000A01018" not in report.journal
    assert instrument_b in report.journal


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_partial_disposal_decimal_precision_renders_exactly(session, backend):
    await _bank_and_snapshot(session)
    quantity = Decimal("1.234567")
    unit_cost = Decimal("12.345678")
    original_cost = (quantity * unit_cost).quantize(Decimal("0.01"))
    transactions = [
        _mf_purchase_raw(
            source_ref="precision/ref",
            units=str(quantity),
            nav=str(unit_cost),
            amount=str(original_cost),
            reference="BUY-PRECISE",
        ),
        _mf_disposal_raw(
            source_ref="precision/ref",
            units="-0.234567",
            reference="SELL-PRECISE",
        ),
    ]
    await _upload_with_lots(
        session,
        transactions,
        [
            {
                "source_ref": "precision/ref",
                "reference": "BUY-PRECISE",
                "quantity": quantity,
                "unit_cost": unit_cost,
                "cost_basis": original_cost,
            }
        ],
    )

    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )

    lot = report.document.lot_postings[0]
    assert lot.quantity == Decimal("1.000000")
    assert lot.unit_cost == unit_cost
    assert lot.cost_basis == Decimal("12.35")
    assert (
        "1.000000 INE000A01018" in report.journal
        or '1.000000 "INE000A01018"' in report.journal
    )
    if backend == "beancount":
        pytest.importorskip("beancount")
        from beancount import loader

        _entries, errors, _options = loader.load_string(report.journal)
        assert errors == []


# ---------------------------------------------------------------------------
# Funding-suppression: no orphan price directive for a suppressed instrument
# ---------------------------------------------------------------------------


async def _seed_two_lots_same_date_amount(session):
    """Two complete lots on the same date with the same cost basis, each funded
    by a distinct CAS fact — the canonical funding-ambiguity setup."""
    import json

    from financial_dashboard.db.models import CasUpload

    await _bank_and_snapshot(session)
    for isin in ("INE000A01018", "INE000A01019"):
        await _seed_lot(
            session,
            instrument_id=isin,
            instrument_name=f"Fund {isin}",
            source_ref=f"mf/{isin}",
            reference=f"TXN_{isin}",
        )
    session.add(
        CasUpload(
            portfolio_key="PAN",
            depository_source="cdsl",
            statement_date=datetime.date(2026, 4, 30),
            grand_total=Decimal("0"),
            raw_holdings_json=json.dumps(
                {
                    "transactions": [
                        _mf_purchase_raw(
                            source_ref=f"mf/{isin}",
                            isin=isin,
                            reference=f"TXN_{isin}",
                        )
                        for isin in ("INE000A01018", "INE000A01019")
                    ]
                }
            ),
        )
    )
    await session.flush()


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_suppressed_funding_lot_drops_orphan_price_directive(session, backend):
    """A lot suppressed for funding ambiguity must not leave a price directive
    behind — no orphan ``P``/``price`` line for the suppressed instrument in any
    backend. (Regression: the suppressed instruments were absent from the
    filtered instrument set, so their prices slipped through the drop filter.)"""
    await _seed_two_lots_same_date_amount(session)
    # Bank investment txn matching date+amount of both lots (ambiguous → both
    # suppressed). The seeded lots are dated 2026-01-15 / cost 50000.
    session.add(
        Transaction(
            account_id=1,
            bank="hdfc",
            email_type="test_funding_ambig",
            direction="debit",
            amount=Decimal("50000.00"),
            transaction_date=datetime.date(2026, 1, 15),
            category="investment",
            counterparty="MF Purchase",
            reference_number="SHARED_REF",
            currency="INR",
        )
    )
    await session.flush()
    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )
    # Both lots suppressed (the ref is shared and date+amount is ambiguous).
    assert report.investment_lot_count == 0
    assert len(report.investment_funding_unresolved) == 2
    # No price directive for either suppressed instrument in the document.
    price_currencies = {p.currency for p in report.document.price_directives}
    for isin in ("INE000A01018", "INE000A01019"):
        assert isin not in price_currencies
    # And no orphan price / asset-account line in the rendered journal either.
    for isin in ("INE000A01018", "INE000A01019"):
        assert isin not in report.journal, (
            f"orphan reference to suppressed instrument {isin} in {backend}"
        )
    # An FX/lot price directive for a SURVIVING instrument still appears when
    # present — here there are none, so the price block is empty.
    assert not any(
        p.currency.startswith("INE") for p in report.document.price_directives
    )


# ---------------------------------------------------------------------------
# Canonical overlap + independent latest valuation + closed source population
# ---------------------------------------------------------------------------


async def _persist_payload(
    session,
    *,
    payload,
    statement_date,
    portfolio_key="PAN-CANONICAL",
    investor_name="Primary Portfolio",
):
    import json

    from financial_dashboard.services.investments import create_investment_lots

    upload = CasUpload(
        portfolio_key=portfolio_key,
        depository_source="cdsl",
        investor_name=investor_name,
        statement_date=statement_date,
        grand_total=Decimal("10000"),
        raw_holdings_json=json.dumps(payload),
    )
    session.add(upload)
    await session.flush()
    await create_investment_lots(session, cas_upload_id=upload.id, payload=payload)
    return upload


async def test_projection_canonicalizes_overlap_but_preserves_true_multiplicity(
    session,
):
    await _bank_and_snapshot(session)
    repeated = _mf_purchase_raw(units="10", nav="50", amount="500")
    for statement_date in (
        datetime.date(2026, 3, 31),
        datetime.date(2026, 4, 30),
    ):
        await _persist_payload(
            session,
            statement_date=statement_date,
            payload={
                # Two identical source occurrences are genuine multiplicity;
                # the later overlapping statement repeats both.
                "transactions": [repeated, repeated],
                "folios": [
                    {
                        "folio_number": "mf/1",
                        "schemes": [
                            {
                                "scheme_name": "Example Fund",
                                "isin": "INE000A01018",
                                "units": "20",
                                "nav": "80",
                                "value": "1600",
                            }
                        ],
                    }
                ],
            },
        )

    report = await project(session, _config(project_investments=True))

    assert report.investment_lot_count == 2
    assert [lot.quantity for lot in report.document.lot_postings] == [
        Decimal("10"),
        Decimal("10"),
    ]
    assert [
        dict(lot.meta)["dashboard_source_occurrence"]
        for lot in report.document.lot_postings
    ] == ["0", "1"]
    assert all(
        len(dict(lot.meta)["dashboard_cas_upload_ids"].split("|")) == 2
        for lot in report.document.lot_postings
    )


async def test_canonical_disposal_consumption_dedupes_overlapping_history(session):
    await _bank_and_snapshot(session)
    purchase = _mf_purchase_raw(
        source_ref="exact/ref", units="10", nav="50", amount="500"
    )
    disposal = _mf_disposal_raw(
        source_ref="exact/ref", units="-10", reference="SELL-EXACT"
    )
    payload = {
        "transactions": [purchase, disposal],
        "folios": [
            {
                "folio_number": "exact/ref",
                "schemes": [
                    {
                        "scheme_name": "Example Fund",
                        "isin": "INE000A01018",
                        "units": "0",
                        "nav": "80",
                        "value": "0",
                    }
                ],
            }
        ],
    }
    await _persist_payload(
        session, payload=payload, statement_date=datetime.date(2026, 3, 31)
    )
    await _persist_payload(
        session, payload=payload, statement_date=datetime.date(2026, 4, 30)
    )

    report = await project(session, _config(project_investments=True))

    assert report.investment_lot_count == 0
    assert not any(
        price.currency == "INE000A01018" for price in report.document.price_directives
    )


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_latest_cas_nav_is_market_price_not_acquisition_cost(session, backend):
    await _bank_and_snapshot(session)
    payload = {
        "transactions": [_mf_purchase_raw(units="30", nav="50", amount="1500")],
        # Same commodity in two folios remains two diagnostic identities, but
        # contributes one compatible commodity/date market price and no extra
        # holding posting.
        "folios": [
            {
                "folio_number": "FOLIO-1",
                "schemes": [
                    {
                        "scheme_name": "Example Fund",
                        "isin": "INE000A01018",
                        "units": "10",
                        "nav": "80",
                        "value": "800",
                    }
                ],
            },
            {
                "folio_number": "FOLIO-2",
                "schemes": [
                    {
                        "scheme_name": "Example Fund",
                        "isin": "INE000A01018",
                        "units": "20",
                        "nav": "80",
                        "value": "1600",
                    }
                ],
            },
        ],
    }
    await _persist_payload(
        session, payload=payload, statement_date=datetime.date(2026, 4, 30)
    )

    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )

    assert len(report.document.lot_postings) == 1
    assert report.document.lot_postings[0].unit_cost == Decimal("50")
    market = [
        price
        for price in report.document.price_directives
        if price.currency == "INE000A01018"
    ]
    assert len(market) == 1
    assert market[0].date == datetime.date(2026, 4, 30)
    assert market[0].rate == Decimal("80")
    assert market[0].unit == "INR"
    assert report.investment_market_price_count == 1
    assert report.investment_current_valuation_count == 2
    assert report.investment_quantity_mismatch_count == 0
    assert report.investment_valuation_sources == (
        "PAN-CANONICAL/folio/FOLIO-1/INE000A01018#0",
        "PAN-CANONICAL/folio/FOLIO-2/INE000A01018#0",
    )
    assert report.cas_investment_scope == "included"
    if backend == "beancount":
        pytest.importorskip("beancount")
        from beancount import loader

        _entries, errors, _options = loader.load_string(report.journal)
        assert errors == []


async def test_conflicting_same_date_current_prices_are_suppressed(session):
    await _bank_and_snapshot(session)
    payload = {
        "transactions": [_mf_purchase_raw(units="30", nav="50", amount="1500")],
        "folios": [
            {
                "folio_number": "FOLIO-1",
                "schemes": [
                    {
                        "isin": "INE000A01018",
                        "units": "10",
                        "nav": "80",
                        "value": "800",
                    }
                ],
            },
            {
                "folio_number": "FOLIO-2",
                "schemes": [
                    {
                        "isin": "INE000A01018",
                        "units": "20",
                        "nav": "81",
                        "value": "1620",
                    }
                ],
            },
        ],
    }
    await _persist_payload(
        session, payload=payload, statement_date=datetime.date(2026, 4, 30)
    )

    report = await project(session, _config(project_investments=True))

    assert report.investment_lot_count == 1
    assert report.investment_market_price_count == 0
    assert report.investment_market_price_conflicts == ("INE000A01018@2026-04-30",)
    assert "current_price_conflict" in report.investment_excluded
    assert not any(
        price.currency == "INE000A01018" for price in report.document.price_directives
    )
    assert report.cas_investment_scope == "partial"


async def test_closed_population_scope_names_non_account_sources(session):
    from financial_dashboard.db.models import ManualItem

    await _bank_and_snapshot(session)
    payload = {
        "transactions": [_mf_purchase_raw(units="10", nav="50", amount="500")],
        "folios": [
            {
                "folio_number": "FOLIO-1",
                "schemes": [
                    {
                        "isin": "INE000A01018",
                        "units": "10",
                        "nav": "80",
                        "value": "800",
                    }
                ],
            }
        ],
    }
    await _persist_payload(
        session,
        portfolio_key="PAN-SCOPE",
        investor_name="Scope Investor",
        payload=payload,
        statement_date=datetime.date(2026, 4, 30),
    )
    session.add_all(
        [
            ManualItem(
                id=10, name="Private Property", kind="asset", category="real_estate"
            ),
            ManualItem(id=11, name="Family Loan", kind="liability", category="loan"),
            ManualItem(
                id=12,
                name="Inactive Decoy",
                kind="asset",
                category="other",
                active=False,
            ),
        ]
    )
    await session.flush()

    excluded = await project(session, _config(project_investments=False))
    included = await project(session, _config(project_investments=True))

    assert excluded.cas_portfolio_count == 1
    assert excluded.cas_portfolio_labels == ("PAN-SCOPE (Scope Investor)",)
    assert excluded.cas_investment_scope == "excluded"
    assert excluded.manual_asset_labels == ("10: Private Property",)
    assert excluded.manual_liability_labels == ("11: Family Loan",)
    assert excluded.net_worth_scope_complete is False
    assert included.cas_investment_scope == "included"
    assert included.net_worth_scope_complete is False  # manual rows remain outside
    assert "Private Property" not in included.journal
    assert "Family Loan" not in included.journal
    assert "Inactive Decoy" not in included.manual_asset_labels
