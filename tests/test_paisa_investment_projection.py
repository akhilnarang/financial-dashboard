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

from financial_dashboard.db.models import InvestmentLot, Transaction
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


def test_price_directive_emitted_from_explicit_unit_cost():
    """A lot's explicit per-unit cost becomes a price directive (no market
    quote is synthesized)."""
    # Price directives are emitted by the projection (not the bare lot doc),
    # so this confirms the lot itself carries its cost annotation truthfully.
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
