"""Investment-lot *renderer* syntax and the core lot service.

Paisa projection no longer consumes ``InvestmentLot`` rows — CAS is projected
from authoritative portfolio aggregates only (see
``tests/test_paisa_cas_valuation.py``). The lot renderer capability and the core
normalization service are retained and still covered here: lots remain a
first-class ingestion fact for the dashboard and for a future, separately
reviewed cost-basis feature.

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


async def test_core_service_reports_lot_exclusion_reasons(session):
    """Lot-classification exclusions live on the CORE service, not the projection.

    Paisa projects CAS as an authoritative aggregate that *includes* these
    holdings' value, so surfacing them as projection "exclusions" would imply
    value was omitted. The reasons remain available from the investment service
    for the dashboard's own investment surface.
    """
    from financial_dashboard.services.investments import get_incomplete_reasons

    await _bank_and_snapshot(session)
    await _upload_with_lots(
        session,
        transactions=[
            _mf_purchase_raw(),
            _mf_purchase_raw(
                source_ref="mf/2", nav=None, amount=None, reference="TXN002"
            ),
            {
                "scope": "demat",
                "source_ref": "d/1",
                "date": "2026-02-01",
                "description": "Equity",
                "isin": "INE000A01019",
                "transaction_type": "buy",
                "quantity": "5",
                "reference": "DP1",
            },
        ],
        lots=[],
    )

    reasons = {excl.reason for excl in await get_incomplete_reasons(session)}

    assert "not_mutual_fund" in reasons
    assert "missing_lot_facts" in reasons

    # The projection itself reports only its own policy diagnostics.
    report = await project(session, _config(project_investments=True))
    assert "not_mutual_fund" not in report.investment_excluded
    assert "missing_lot_facts" not in report.investment_excluded


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
