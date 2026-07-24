"""Aggregate-only CAS valuation projection.

Every CAS portfolio is projected from its authoritative INR ``BalanceSnapshot``
history and nothing else. The projection does not consume ``InvestmentLot``
rows, emits no commodity lots and no CAS market prices, and never nets a bank
investment leg against a portfolio aggregate.

That boundary is deliberate. Reconciling per-lot cost basis against
per-statement CAS aggregates is not determined by this data: a bank row does not
name the portfolio it funded, and a CAS value change is purchases minus
redemptions *plus market movement*. Every attempt to bridge that gap produced a
correctness defect — double counting, deleted assets, negative asset balances,
or a later statement rewriting an earlier one.

Every portfolio key, ISIN and amount here is synthetic.
"""

import datetime
import json
from decimal import Decimal

import pytest

from financial_dashboard.db.enums import (
    SnapshotCategory,
    SnapshotKind,
    SnapshotSource,
)
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    CasUpload,
    Transaction,
)
from financial_dashboard.services.paisa.accounting import KIND_VALUATION
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.portfolio_identity import (
    PORTFOLIO_TOKEN_SECRET_KEY,
    normalize_portfolio_key,
    portfolio_token,
)
from financial_dashboard.services.paisa.projection import project
from financial_dashboard.services.paisa.renderers.base import (
    EQUITY_REVALUATION,
    INVESTMENT_VALUATION_ROOT,
)

pytestmark = pytest.mark.anyio

CUTOVER = datetime.date(2026, 1, 1)
SECRET = "a" * 64

#: Synthetic portfolio identifiers — never a real PAN.
PORTFOLIO_A = "SYNTHETIC-A"
PORTFOLIO_B = "SYNTHETIC-B"

#: Synthetic ISIN-shaped instrument ids.
ISIN_DEMAT = "INE000D01011"
ISIN_FUND = "INE000A01018"


def _config(**overrides) -> PaisaProjectionConfig:
    base: dict[str, object] = dict(
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


@pytest.fixture(autouse=True)
def _portfolio_secret(monkeypatch):
    """A deterministic installation secret so account tokens are stable."""
    from financial_dashboard.services import settings as settings_module

    monkeypatch.setitem(settings_module._cache, PORTFOLIO_TOKEN_SECRET_KEY, SECRET)


async def _bank_and_snapshot(session, opening="1000.00"):
    session.add(Account(id=1, bank="hdfc", label="Savings", type="bank_account"))
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=CUTOVER,
            value=Decimal(opening),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await session.flush()


async def _persist_cas(
    session,
    *,
    payload,
    statement_date,
    grand_total,
    portfolio_key=PORTFOLIO_A,
    investor_name="Synthetic Portfolio",
    create_lots=True,
):
    """Persist a CAS upload the way ingestion does: upload + snapshot (+ lots).

    ``BalanceSnapshot.value`` carries ``grand_total`` verbatim, which is what
    makes the ledger's CAS component equal the dashboard's net-worth component.
    """
    upload = CasUpload(
        portfolio_key=portfolio_key,
        depository_source="cdsl",
        investor_name=investor_name,
        statement_date=statement_date,
        grand_total=Decimal(grand_total),
        raw_holdings_json=json.dumps(payload),
    )
    session.add(upload)
    await session.flush()
    session.add(
        BalanceSnapshot(
            cas_upload_id=upload.id,
            portfolio_key=portfolio_key,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.investment.value,
            as_of_date=statement_date,
            value=Decimal(grand_total),
            source=SnapshotSource.cas.value,
        )
    )
    await session.flush()
    if create_lots:
        from financial_dashboard.services.investments import create_investment_lots

        await create_investment_lots(session, cas_upload_id=upload.id, payload=payload)
    return upload


def _demat_payload(*, holdings):
    return {
        "accounts": [
            {
                "dp_id": "DP-1",
                "client_id": "CL-1",
                "depository": "cdsl",
                "holdings": holdings,
            }
        ]
    }


def _holding(isin=ISIN_DEMAT, *, quantity="100", price="250", value="25000"):
    return {
        "isin": isin,
        "name": "Synthetic Equity",
        "asset_class": "equity",
        "quantity": quantity,
        "price": price,
        "value": value,
    }


def _mf_purchase(**overrides):
    base = {
        "scope": "mf",
        "source_ref": "mf/1",
        "date": "2026-01-15",
        "description": "Synthetic Fund",
        "isin": ISIN_FUND,
        "transaction_type": "purchase",
        "units": "1000",
        "nav": "50.00",
        "amount": "50000.00",
        "reference": "TXN001",
    }
    base.update(overrides)
    return base


def _fund_folio(*, units="1000", nav="80", value="80000"):
    return {
        "folio_number": "FOLIO-1",
        "schemes": [
            {
                "scheme_name": "Synthetic Fund",
                "isin": ISIN_FUND,
                "units": units,
                "nav": nav,
                "value": value,
            }
        ],
    }


async def _investment_txn(session, *, amount, on, direction="debit", category=None):
    session.add(
        Transaction(
            account_id=1,
            transaction_date=on,
            amount=Decimal(amount),
            direction=direction,
            category=category
            or ("investment" if direction == "debit" else "investment_redemption"),
            raw_description="investment",
            currency="INR",
            bank="hdfc",
            email_type="txn",
        )
    )
    await session.flush()


def _token(portfolio: str = PORTFOLIO_A) -> str:
    token = portfolio_token(portfolio, SECRET)
    assert token is not None
    return token


def _valuation_account(portfolio: str = PORTFOLIO_A) -> str:
    return f"{INVESTMENT_VALUATION_ROOT}:{_token(portfolio)}"


def _totals(report):
    out: dict[str, Decimal] = {}
    for entry in report.entries:
        for posting in entry.postings:
            out[posting.account] = (
                out.get(posting.account, Decimal("0.00")) + posting.amount
            )
    return out


def _balance_at(report, prefix, as_of):
    return sum(
        (
            p.amount
            for e in report.entries
            if e.date <= as_of
            for p in e.postings
            if p.account.startswith(prefix)
        ),
        Decimal("0.00"),
    )


# ---------------------------------------------------------------------------
# Aggregate valuation is projected, with no cost basis
# ---------------------------------------------------------------------------


async def test_demat_portfolio_value_is_projected_without_a_lot(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )

    report = await project(session, _config())

    assert report.investment_valuation_portfolios == (PORTFOLIO_A,)
    assert report.investment_valuation_total == Decimal("25000.00")
    assert report.cas_investment_scope == "included"
    assert report.cas_investment_coverage == "valuation_only"
    assert _valuation_account() in report.journal
    assert EQUITY_REVALUATION in report.journal
    # No cost annotation, no commodity, no market price anywhere.
    assert "{" not in report.journal
    assert ISIN_DEMAT not in report.journal
    assert report.document.price_directives == ()
    assert report.document.lot_postings == ()
    assert report.investment_lot_count == 0


async def test_valuation_postings_are_plain_inr(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )

    report = await project(session, _config())

    valuation = [e for e in report.entries if e.kind == KIND_VALUATION]
    assert len(valuation) == 1
    for posting in valuation[0].postings:
        assert posting.commodity == "INR"
    assert "dashboard_cost_basis_available: false" in report.journal


# ---------------------------------------------------------------------------
# THE boundary regression: populated InvestmentLot rows change nothing
# ---------------------------------------------------------------------------


async def test_populated_investment_lots_do_not_affect_the_journal(session):
    """Lots remain an ingestion fact; the journal must be identical without them.

    This is the load-bearing guarantee of the aggregate-only boundary: CAS
    accounting depends solely on ``BalanceSnapshot`` history, so whether
    ``InvestmentLot`` rows exist is irrelevant to the emitted ledger.
    """
    from sqlalchemy import func, select

    from financial_dashboard.db.models import InvestmentLot

    await _bank_and_snapshot(session)
    # A fully lot-complete MF payload — the case that previously drove the
    # cost-basis path — WITH lot normalization enabled.
    await _persist_cas(
        session,
        payload={"transactions": [_mf_purchase()], "folios": [_fund_folio()]},
        statement_date=datetime.date(2026, 4, 30),
        grand_total="80000.00",
        create_lots=True,
    )
    lot_count = await session.scalar(select(func.count()).select_from(InvestmentLot))
    assert lot_count >= 1, "fixture must actually create lots to be meaningful"

    with_lots = await project(session, _config())

    # Remove every lot row and re-project.
    await session.execute(InvestmentLot.__table__.delete())
    await session.flush()
    assert (await session.scalar(select(func.count()).select_from(InvestmentLot))) == 0

    without_lots = await project(session, _config())

    # Byte-identical journal, and identical CAS accounting.
    assert with_lots.journal == without_lots.journal
    assert with_lots.investment_lot_count == 0
    assert without_lots.investment_lot_count == 0
    assert with_lots.cas_investment_coverage == "valuation_only"
    assert with_lots.investment_valuation_total == Decimal("80000.00")
    # No commodity lot or market price was emitted even though lots existed.
    assert with_lots.document.lot_postings == ()
    assert with_lots.document.price_directives == ()
    assert ISIN_FUND not in with_lots.journal


async def test_lot_complete_portfolio_still_projects_only_the_aggregate(session):
    """A portfolio whose holdings are fully lot-covered is still aggregate-only."""
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload={"transactions": [_mf_purchase()], "folios": [_fund_folio()]},
        statement_date=datetime.date(2026, 4, 30),
        grand_total="80000.00",
    )

    report = await project(session, _config())

    assert report.investment_cost_basis_portfolios == ()
    assert report.investment_valuation_portfolios == (PORTFOLIO_A,)
    assert report.cas_investment_coverage == "valuation_only"
    assert report.investment_lot_count == 0


# ---------------------------------------------------------------------------
# Statement-effective history
# ---------------------------------------------------------------------------


async def test_opening_at_cutover_then_forward_deltas(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="20000")]),
        statement_date=datetime.date(2025, 12, 31),
        grand_total="20000.00",
    )
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="26000")]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="26000.00",
    )

    report = await project(session, _config())

    valuation = sorted(
        (e for e in report.entries if e.kind == KIND_VALUATION), key=lambda e: e.date
    )
    assert len(valuation) == 2
    assert valuation[0].date == CUTOVER
    assert valuation[0].postings[0].amount == Decimal("20000.00")
    assert valuation[1].date == datetime.date(2026, 3, 31)
    assert valuation[1].postings[0].amount == Decimal("6000.00")


async def test_future_statement_never_changes_an_earlier_balance(session):
    """The forward-only invariant, asserted by appending a future statement."""
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="20000")]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="20000.00",
    )
    before = await project(session, _config())
    earlier = _balance_at(before, INVESTMENT_VALUATION_ROOT, datetime.date(2026, 4, 15))
    assert earlier == Decimal("20000.00")

    # Append a much later, much larger statement.
    await _persist_cas(
        session,
        payload={"transactions": [_mf_purchase()], "folios": [_fund_folio()]},
        statement_date=datetime.date(2026, 9, 30),
        grand_total="90000.00",
    )

    after = await project(session, _config())

    assert (
        _balance_at(after, INVESTMENT_VALUATION_ROOT, datetime.date(2026, 4, 15))
        == earlier
    )
    assert _balance_at(
        after, INVESTMENT_VALUATION_ROOT, datetime.date(2026, 12, 31)
    ) == Decimal("90000.00")


async def test_portfolio_falling_to_zero_clears(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="20000")]),
        statement_date=datetime.date(2025, 12, 31),
        grand_total="20000.00",
    )
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[]),
        statement_date=datetime.date(2026, 6, 30),
        grand_total="0.00",
    )

    report = await project(session, _config())

    valuation = sorted(
        (e for e in report.entries if e.kind == KIND_VALUATION), key=lambda e: e.date
    )
    assert [e.postings[0].amount for e in valuation] == [
        Decimal("20000.00"),
        Decimal("-20000.00"),
    ]


async def test_unchanged_value_emits_no_entry(session):
    await _bank_and_snapshot(session)
    for statement_date in (datetime.date(2026, 3, 31), datetime.date(2026, 4, 30)):
        await _persist_cas(
            session,
            payload=_demat_payload(holdings=[_holding()]),
            statement_date=statement_date,
            grand_total="25000.00",
        )

    report = await project(session, _config())

    valuation = [e for e in report.entries if e.kind == KIND_VALUATION]
    assert len(valuation) == 1


async def test_aggregate_is_preserved_when_holdings_do_not_sum_to_it(session):
    """The authoritative grand total wins; no allocation is invented."""
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="25000")]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="31000.00",
    )

    report = await project(session, _config())

    assert report.investment_valuation_total == Decimal("31000.00")


async def test_portfolio_without_snapshot_is_diagnosed_not_invented(session):
    await _bank_and_snapshot(session)
    upload = CasUpload(
        portfolio_key=PORTFOLIO_A,
        depository_source="cdsl",
        investor_name="Synthetic Portfolio",
        statement_date=datetime.date(2026, 3, 31),
        grand_total=Decimal("25000.00"),
        raw_holdings_json=json.dumps(_demat_payload(holdings=[_holding()])),
    )
    session.add(upload)
    await session.flush()

    report = await project(session, _config())

    assert report.investment_valuation_portfolios == ()
    assert report.investment_valuation_unrepresented == (PORTFOLIO_A,)
    assert "portfolio_value_unavailable" in report.investment_excluded
    assert report.cas_investment_scope == "partial"


async def test_multiple_portfolios_each_get_their_own_account(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
        portfolio_key=PORTFOLIO_A,
    )
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="9000")]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="9000.00",
        portfolio_key=PORTFOLIO_B,
    )

    report = await project(session, _config())

    assert report.investment_valuation_portfolios == (PORTFOLIO_A, PORTFOLIO_B)
    assert _valuation_account(PORTFOLIO_A) in report.journal
    assert _valuation_account(PORTFOLIO_B) in report.journal
    assert report.investment_valuation_total == Decimal("34000.00")


# ---------------------------------------------------------------------------
# Bank investment legs are unresolved by policy
# ---------------------------------------------------------------------------


async def test_purchase_keeps_its_unallocated_asset_and_is_reported(session):
    await _bank_and_snapshot(session)
    await _investment_txn(session, amount="25000.00", on=datetime.date(2026, 2, 10))
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )

    report = await project(session, _config())

    totals = _totals(report)
    assert totals["Assets:Investments:Unallocated"] == Decimal("25000.00")
    assert totals["Assets:Bank:Hdfc:Savings"] == Decimal("-25000.00")
    assert report.investment_unresolved_purchases == 1
    # Sources are present, but the total is not exact.
    assert report.net_worth_sources_complete is True
    assert report.net_worth_scope_complete is False


async def test_redemption_never_drives_an_asset_negative(session):
    await _bank_and_snapshot(session)
    await _investment_txn(session, amount="10000.00", on=datetime.date(2026, 2, 10))
    await _investment_txn(
        session, amount="40000.00", on=datetime.date(2026, 3, 5), direction="credit"
    )
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )

    report = await project(session, _config())

    assert report.investment_unresolved_redemptions == 1
    balances: dict[str, Decimal] = {}
    for entry in sorted(report.entries, key=lambda e: e.date):
        for posting in entry.postings:
            balances[posting.account] = (
                balances.get(posting.account, Decimal("0.00")) + posting.amount
            )
        for account, value in balances.items():
            if account.startswith("Assets:Investments"):
                assert value >= 0, f"{account} negative ({value}) on {entry.date}"


async def test_ledger_investments_never_fall_below_native(session):
    from financial_dashboard.services.networth import current_networth

    await _bank_and_snapshot(session)
    await _investment_txn(session, amount="12000.00", on=datetime.date(2026, 2, 10))
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="20000")]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="20000.00",
    )

    report = await project(session, _config())
    for as_of in (
        datetime.date(2026, 2, 28),
        datetime.date(2026, 3, 31),
        datetime.date(2026, 8, 31),
    ):
        summary = await current_networth(session, today=as_of)
        native = sum(
            (
                row.value
                for group in summary.groups
                if group.category == SnapshotCategory.investment.value
                for row in group.rows
            ),
            Decimal("0.00"),
        )
        ledger = _balance_at(report, "Assets:Investments", as_of)
        assert ledger >= native, f"ledger under-reports at {as_of}"


async def test_cas_component_matches_current_networth(session):
    from financial_dashboard.services.networth import current_networth

    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="20000")]),
        statement_date=datetime.date(2025, 12, 31),
        grand_total="20000.00",
    )
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="26500")]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="26500.00",
    )

    as_of = datetime.date(2026, 6, 30)
    summary = await current_networth(session, today=as_of)
    native = sum(
        (
            row.value
            for group in summary.groups
            if group.category == SnapshotCategory.investment.value
            for row in group.rows
        ),
        Decimal("0.00"),
    )
    report = await project(session, _config())
    assert _balance_at(report, INVESTMENT_VALUATION_ROOT, as_of) == native


# ---------------------------------------------------------------------------
# Operator mappings outrank every generated policy
# ---------------------------------------------------------------------------


async def test_explicit_mapping_wins_for_redemption(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )
    await _investment_txn(
        session, amount="5000.00", on=datetime.date(2026, 4, 5), direction="credit"
    )

    report = await project(
        session,
        _config(
            category_mappings={"investment_redemption": "Assets:Investments:MyBroker"}
        ),
    )

    totals = _totals(report)
    assert "Assets:Investments:MyBroker" in totals
    assert "Equity:Transfers In" not in totals
    assert report.investment_unresolved_redemptions == 0


async def test_explicit_mapping_wins_for_purchase(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )
    await _investment_txn(session, amount="5000.00", on=datetime.date(2026, 2, 10))

    report = await project(
        session,
        _config(category_mappings={"investment": "Assets:Investments:MyBroker"}),
    )

    totals = _totals(report)
    assert "Assets:Investments:MyBroker" in totals
    assert "Assets:Investments:Unallocated" not in totals
    assert report.investment_unresolved_purchases == 0


# ---------------------------------------------------------------------------
# Gate, privacy, determinism, backends
# ---------------------------------------------------------------------------


async def test_gate_off_emits_no_valuation(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )

    report = await project(session, _config(project_investments=False))

    assert report.investment_valuation_portfolios == ()
    assert report.cas_investment_scope == "excluded"
    assert report.cas_investment_coverage == "excluded"
    assert INVESTMENT_VALUATION_ROOT not in report.journal


async def test_gate_off_leaves_investment_taxonomy_untouched(session):
    """With the feature off, investment rows keep their long-standing contra."""
    await _bank_and_snapshot(session)
    await _investment_txn(
        session, amount="100.00", on=datetime.date(2026, 2, 5), direction="credit"
    )

    report = await project(session, _config(project_investments=False))

    totals = _totals(report)
    assert "Assets:Investments:Unallocated" in totals
    assert report.investment_unresolved_redemptions == 0


async def test_journal_never_contains_the_raw_portfolio_key(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
        portfolio_key="ABCDE1234F",
    )

    report = await project(session, _config())

    assert "ABCDE1234F" not in report.journal
    assert "Synthetic Portfolio" not in report.journal
    assert "dashboard_portfolio_key" not in report.journal


def test_portfolio_token_is_stable_and_opaque():
    token = portfolio_token(PORTFOLIO_A, SECRET)
    assert token is not None
    assert token == portfolio_token(PORTFOLIO_A, SECRET)
    assert PORTFOLIO_A not in token
    assert normalize_portfolio_key(PORTFOLIO_A) not in token
    assert len(token) - 2 >= 16  # >= 80 bits of base32
    assert portfolio_token(PORTFOLIO_A, SECRET) != portfolio_token(PORTFOLIO_B, SECRET)
    assert portfolio_token(PORTFOLIO_A, None) is None


async def test_without_a_secret_the_shared_account_is_used(session, monkeypatch):
    from financial_dashboard.services import settings as settings_module

    monkeypatch.delitem(
        settings_module._cache, PORTFOLIO_TOKEN_SECRET_KEY, raising=False
    )
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )

    report = await project(session, _config())

    assert PORTFOLIO_A not in report.journal
    assert INVESTMENT_VALUATION_ROOT in report.journal
    assert report.investment_valuation_total == Decimal("25000.00")


async def test_projection_is_byte_deterministic(session):
    await _bank_and_snapshot(session)
    for index, key in enumerate([PORTFOLIO_B, PORTFOLIO_A]):
        await _persist_cas(
            session,
            payload=_demat_payload(holdings=[_holding(value=str(1000 + index))]),
            statement_date=datetime.date(2026, 3, 31),
            grand_total=f"{1000 + index}.00",
            portfolio_key=key,
        )

    runs = {(await project(session, _config())).journal for _ in range(5)}
    assert len(runs) == 1


async def test_projection_writes_no_core_rows(session):
    from sqlalchemy import func, select

    from financial_dashboard.db.models import InvestmentLot

    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload={"transactions": [_mf_purchase()], "folios": [_fund_folio()]},
        statement_date=datetime.date(2026, 4, 30),
        grand_total="80000.00",
    )

    async def counts():
        return (
            await session.scalar(select(func.count()).select_from(InvestmentLot)),
            await session.scalar(select(func.count()).select_from(BalanceSnapshot)),
            await session.scalar(select(func.count()).select_from(CasUpload)),
        )

    before = await counts()
    await project(session, _config())
    assert await counts() == before


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_renders_on_every_backend(session, backend):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )

    report = await project(session, _config(ledger_cli=backend))

    assert report.investment_valuation_entry_count == 1
    assert "25000.00 INR" in report.journal
    if backend == "beancount":
        # beancount account segments disallow "-"; the normalizer strips it.
        assert _token().replace("-", "") in report.journal
        pytest.importorskip("beancount")
        from beancount import loader

        _entries, errors, _options = loader.load_string(report.journal)
        assert errors == []
    else:
        assert _valuation_account() in report.journal


async def test_manual_items_still_block_full_parity(session):
    from financial_dashboard.db.models import ManualItem

    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )
    session.add(
        ManualItem(id=10, name="Synthetic Asset", kind="asset", category="other")
    )
    await session.flush()

    report = await project(session, _config())

    assert report.cas_investment_scope == "included"
    assert report.net_worth_sources_complete is False
    assert report.net_worth_scope_complete is False


# ---------------------------------------------------------------------------
# Closed-population net-worth scope (restored: covers still-active behavior)
# ---------------------------------------------------------------------------


async def test_closed_population_scope_names_non_account_sources(session):
    """Every active native net-worth source outside Account selection is named.

    The account picker cannot select CAS portfolios or manual items, so the
    summary must enumerate them deterministically — including that inactive
    manual items are excluded and that manual labels never reach the journal.
    """
    from financial_dashboard.db.models import ManualItem

    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 4, 30),
        grand_total="25000.00",
        portfolio_key="PAN-SCOPE",
        investor_name="Scope Investor",
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
    # An inactive manual item is never surfaced.
    assert "Inactive Decoy" not in excluded.manual_asset_labels

    assert included.cas_investment_scope == "included"
    # Manual rows remain outside projection, so completeness stays false.
    assert included.net_worth_scope_complete is False
    assert included.net_worth_sources_complete is False
    # Private manual labels never enter the generated journal.
    assert "Private Property" not in included.journal
    assert "Family Loan" not in included.journal


# ---------------------------------------------------------------------------
# Consumer propagation: report -> surface/reconciliation DTO -> serialized JSON
# ---------------------------------------------------------------------------
#
# The two full-suite failures during this rewrite were both consumer breakage in
# files the change did not touch. These tests pin the whole chain so a field
# added to ProjectionReport cannot silently stop at the projection boundary.


async def _report_with_unresolved(session):
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )
    await _investment_txn(session, amount="7000.00", on=datetime.date(2026, 2, 10))
    await _investment_txn(
        session, amount="3000.00", on=datetime.date(2026, 4, 5), direction="credit"
    )
    return await project(session, _config())


async def test_unresolved_counters_reach_the_surface_summary(session):
    from financial_dashboard.services.paisa.surface import _projection_summary

    report = await _report_with_unresolved(session)
    assert report.investment_unresolved_purchases == 1
    assert report.investment_unresolved_redemptions == 1

    summary = _projection_summary(report)

    assert summary is not None
    assert summary.investment_unresolved_purchases == 1
    assert summary.investment_unresolved_redemptions == 1
    # Either counter must make exactness false at the exposed layer too.
    assert summary.net_worth_scope_complete is False
    assert summary.net_worth_sources_complete is True
    assert summary.cas_investment_coverage == "valuation_only"


async def test_unresolved_counters_reach_the_reconciliation_diag(session):
    from financial_dashboard.services.paisa.reconciliation import _projection_diag

    report = await _report_with_unresolved(session)

    diag = _projection_diag(report)

    assert diag is not None
    assert diag.investment_unresolved_purchases == 1
    assert diag.investment_unresolved_redemptions == 1
    assert diag.net_worth_scope_complete is False


async def test_unresolved_counters_survive_json_serialization(session):
    import json as _json

    from financial_dashboard.services.paisa.surface import _projection_summary

    report = await _report_with_unresolved(session)

    payload = _json.loads(_projection_summary(report).model_dump_json())

    assert payload["investment_unresolved_purchases"] == 1
    assert payload["investment_unresolved_redemptions"] == 1
    assert payload["net_worth_scope_complete"] is False
    assert payload["net_worth_sources_complete"] is True
    assert payload["cas_investment_coverage"] == "valuation_only"
    assert payload["investment_valuation_portfolios"] == [PORTFOLIO_A]


async def test_legacy_lot_fields_stay_empty_through_every_consumer(session):
    """Backward-compatible fields must never carry stale UI diagnostics."""
    import json as _json

    from financial_dashboard.services.paisa.reconciliation import _projection_diag
    from financial_dashboard.services.paisa.surface import _projection_summary

    await _bank_and_snapshot(session)
    # A fully lot-complete payload — the case that used to populate these.
    await _persist_cas(
        session,
        payload={"transactions": [_mf_purchase()], "folios": [_fund_folio()]},
        statement_date=datetime.date(2026, 4, 30),
        grand_total="80000.00",
    )

    report = await project(session, _config())
    summary = _projection_summary(report)
    diag = _projection_diag(report)
    payload = _json.loads(summary.model_dump_json())

    assert report.investment_lot_count == 0
    assert report.investment_funding_remapped == 0
    assert report.investment_funding_unresolved == ()
    assert report.investment_disposal_unresolved == ()
    assert report.investment_market_price_count == 0
    assert report.investment_cost_basis_portfolios == ()

    for surfaced in (summary, diag):
        assert surfaced.investment_lot_count == 0
        assert surfaced.investment_funding_unresolved == []
    assert summary.investment_market_price_count == 0
    assert summary.investment_cost_basis_portfolios == []
    assert diag.investment_disposal_unresolved_count == 0

    assert payload["investment_lot_count"] == 0
    assert payload["investment_funding_unresolved"] == []
    assert payload["investment_cost_basis_portfolios"] == []


async def test_projection_excluded_reasons_are_policy_only(session):
    """Lot-normalization reasons must not appear as projection exclusions.

    The authoritative aggregate INCLUDES the value of holdings that cannot
    become lots, so labelling them "excluded" here would imply value was
    omitted while ``cas_investment_scope`` says ``included``.
    """
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding()]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="25000.00",
    )

    report = await project(session, _config())

    assert report.cas_investment_scope == "included"
    assert set(report.investment_excluded) <= {
        "valuation_only_no_cost_basis",
        "portfolio_value_unavailable",
    }
    for stale in ("not_mutual_fund", "missing_lot_facts", "disposal_transaction"):
        assert stale not in report.investment_excluded


# ---------------------------------------------------------------------------
# Source identity must mirror native net worth exactly
# ---------------------------------------------------------------------------


async def test_raw_key_variants_match_native_at_every_date(session):
    """Two raw portfolio keys differing only by case/whitespace.

    Native net worth groups by the RAW ``portfolio_key``, so it treats these as
    separate sources and sums them. Grouping by a normalized key here merged the
    series and — for same-date rows — silently dropped one, under-reporting the
    ledger against native. They must post to the same private token account and
    still sum to the native total.
    """
    from financial_dashboard.services.networth import current_networth

    await _bank_and_snapshot(session)
    for key, value, when in (
        ("PAN-X", "10000.00", datetime.date(2026, 3, 31)),
        (" pan-x ", "7000.00", datetime.date(2026, 3, 31)),
        ("PAN-X", "12000.00", datetime.date(2026, 5, 31)),
    ):
        await _persist_cas(
            session,
            payload=_demat_payload(holdings=[_holding(value=value.split(".")[0])]),
            statement_date=when,
            grand_total=value,
            portfolio_key=key,
        )

    report = await project(session, _config())

    for as_of in (
        datetime.date(2026, 4, 15),
        datetime.date(2026, 6, 30),
        datetime.date(2026, 12, 31),
    ):
        summary = await current_networth(session, today=as_of)
        native = sum(
            (
                row.value
                for group in summary.groups
                if group.category == SnapshotCategory.investment.value
                for row in group.rows
            ),
            Decimal("0.00"),
        )
        ledger = _balance_at(report, INVESTMENT_VALUATION_ROOT, as_of)
        assert ledger == native, f"divergence at {as_of}: {ledger} != {native}"

    # Both raw identities collapse to one private token account, and the raw
    # keys never reach the journal.
    assert report.investment_valuation_portfolios == ("PAN-X",)
    assert "pan-x" not in report.journal
    assert "PAN-X" not in report.journal


async def test_same_date_raw_variants_are_not_dropped(session):
    """The same-date case specifically: neither series may be overwritten."""
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="4000")]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="4000.00",
        portfolio_key="PAN-Y",
    )
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="6000")]),
        statement_date=datetime.date(2026, 3, 31),
        grand_total="6000.00",
        portfolio_key="pan-y",
    )

    report = await project(session, _config())

    assert _balance_at(
        report, INVESTMENT_VALUATION_ROOT, datetime.date(2026, 6, 30)
    ) == Decimal("10000.00")


# ---------------------------------------------------------------------------
# Opening metadata retains the real observation date
# ---------------------------------------------------------------------------


async def test_opening_metadata_keeps_the_snapshot_date(session):
    """The entry is dated at the cutover; the metadata keeps the source date.

    Recording the cutover as ``dashboard_as_of`` would make a stale valuation
    look like it was observed at cutover and lose audit provenance.
    """
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="5000")]),
        statement_date=datetime.date(2025, 10, 31),
        grand_total="5000.00",
    )
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="8000")]),
        statement_date=datetime.date(2025, 12, 15),
        grand_total="8000.00",
    )

    report = await project(session, _config())

    opening = [e for e in report.entries if e.kind == KIND_VALUATION][0]
    meta = dict(opening.meta)
    # The latest eligible pre-cutover snapshot supplies BOTH value and date.
    assert opening.date == CUTOVER
    assert opening.postings[0].amount == Decimal("8000.00")
    assert meta["dashboard_as_of"] == "2025-12-15"
    assert meta["dashboard_valuation_kind"] == "opening"


async def test_opening_metadata_uses_cutover_when_snapshot_is_on_it(session):
    """A snapshot dated exactly at the cutover reports the cutover, correctly."""
    await _bank_and_snapshot(session)
    await _persist_cas(
        session,
        payload=_demat_payload(holdings=[_holding(value="5000")]),
        statement_date=CUTOVER,
        grand_total="5000.00",
    )

    report = await project(session, _config())

    opening = [e for e in report.entries if e.kind == KIND_VALUATION][0]
    assert opening.date == CUTOVER
    assert dict(opening.meta)["dashboard_as_of"] == CUTOVER.isoformat()
