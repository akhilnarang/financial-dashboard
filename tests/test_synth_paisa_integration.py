"""Integration coverage: load a synthetic DB, then drive the **production**
``services.paisa`` projection/renderer against selected seeded accounts.

This is the bridge test between the synthetic seed and the real Paisa
projection. It constructs a :class:`PaisaProjectionConfig` directly (bypassing
``load_config``, which reads the live ``paisa.*`` settings registry), points it
at a freshly loaded synthetic DB, and asserts:

* the projected journal renders and every emitted entry balances to zero;
* the projection is stable — two runs produce byte-identical output;
* the projection is a pure read: no core row (transactions / accounts /
  snapshots) is inserted or mutated by the call;
* long and/or spaced account names survive rendering verbatim and obey
  ledger's account/amount separation contract (>=2 spaces), validated both
  structurally and, when the optional ``ledger`` CLI is on PATH, against the
  real ledger parser.

It does **not** modify any production module to do this — the projection is
already a documented read over the ORM.
"""

import datetime
import os
import re
import shutil
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    InvestmentLot,
    Transaction,
)
from financial_dashboard.services.paisa import PaisaProjectionConfig, project
from financial_dashboard.services.paisa.config import FxRate
from scripts.synth import build_scenario, load_scenario
from scripts.synth.loader import create_synthetic_engine

pytestmark = pytest.mark.anyio


def _synthetic_db(tmp_path: Path) -> Path:
    p = tmp_path / "synthetic" / "synthetic.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


async def _counts(session: AsyncSession) -> dict[str, int]:
    """Core-table row counts the projection must not change."""
    return {
        "transactions": (
            await session.execute(select(func.count()).select_from(Transaction))
        ).scalar_one(),
        "accounts": (
            await session.execute(select(func.count()).select_from(Account))
        ).scalar_one(),
        "snapshots": (
            await session.execute(select(func.count()).select_from(BalanceSnapshot))
        ).scalar_one(),
    }


def _config(
    selected: tuple[int, ...],
    cutover: datetime.date,
    *,
    account_mappings: dict[str, str] | None = None,
    category_mappings: dict[str, str] | None = None,
    non_inr_policy: str = "skip",
    fx_rates: dict[str, tuple[FxRate, ...]] | None = None,
    project_investments: bool = False,
    ledger_cli: str = "ledger",
) -> PaisaProjectionConfig:
    # Built directly — no settings round-trip, so the test is hermetic.
    return PaisaProjectionConfig(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=selected,
        cutover_date=cutover,
        account_mappings=account_mappings or {},
        category_mappings=category_mappings or {},
        non_inr_policy=non_inr_policy,  # type: ignore[arg-type]
        request_timeout_seconds=15,
        fx_rates=fx_rates or {},
        project_investments=project_investments,
        ledger_cli=ledger_cli,
    )


def _fx_rates_from_scenario(scenario) -> dict[str, tuple[FxRate, ...]]:
    """Build a PaisaProjectionConfig.fx_rates map straight from the scenario's
    deterministic historical rates (requirement: canonical scenarios include
    explicit historical FX rates)."""
    by_ccy: dict[str, list[FxRate]] = {}
    for fx in scenario.fx_rates:
        by_ccy.setdefault(fx.currency.upper(), []).append(
            FxRate(date=fx.date, rate=fx.rate)
        )
    return {
        ccy: tuple(sorted(rates, key=lambda r: r.date)) for ccy, rates in by_ccy.items()
    }


def _assert_ledger_posting_contract(journal: str) -> None:
    """Every posting line must split into ``<account>  <amount>`` on >=2 spaces.

    This is ledger's own parse rule: the account name and the amount must be
    separated by at least two spaces (or a tab), otherwise ledger absorbs the
    amount into the account name. Mirroring that rule here checks the contract
    on every posting line, not just the ones the test injected.
    """
    for line in journal.splitlines():
        if not line.startswith("    ") or "INR" not in line:
            continue
        body = line.lstrip()
        parts = re.split(r" {2,}", body, maxsplit=1)
        assert len(parts) == 2, (
            f"posting not split into account + amount on >=2 spaces: {line!r}"
        )
        assert parts[1].endswith("INR"), f"unexpected amount field: {parts[1]!r}"


def _ledger_available() -> bool:
    return shutil.which("ledger") is not None


def _ledger_parses_journal(journal: str) -> None:
    """Optional real-contract check: run the ``ledger`` CLI over the journal.

    No-op when ``ledger`` is not on PATH (it is not a project dependency). When
    present, it is the strongest validation that the production renderer's
    output actually parses — including the long/spaced account names the
    structural check above can only approximate.
    """
    if not _ledger_available():
        return
    with tempfile.NamedTemporaryFile("w", suffix=".ledger", delete=False) as fh:
        fh.write(journal)
        path = fh.name
    try:
        completed = subprocess.run(
            ["ledger", "-f", path, "balanced"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, (
            f"ledger rejected the production journal:\n{completed.stderr}"
        )
    finally:
        os.unlink(path)


async def test_production_projection_against_synthetic_db_is_balanced_and_readonly(
    tmp_path,
):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            # Pick every active bank account the loader actually created.
            bank_ids = tuple(
                (
                    await session.execute(
                        select(Account.id)
                        .where(Account.type == "bank_account", Account.active.is_(True))
                        .order_by(Account.id)
                    )
                )
                .scalars()
                .all()
            )
            assert bank_ids, "expected seeded bank accounts to project"

            # Cutover just inside the scenario window so a meaningful slice is
            # projected, with opening balances resolvable from seeded snapshots.
            cutover = scenario.as_of - datetime.timedelta(days=20)

            # Inject a long, spaced ledger name for the first bank account so
            # the test asserts such names survive rendering. This name
            # overruns the renderer's alignment column (48 chars of name
            # budget), which is exactly the case that previously got a
            # single-space gap ledger would mis-parse as a single mangled
            # account name.
            long_name = "Assets:Bank:HDFC:Salary Plus Premium Savings Account"
            assert len(long_name) >= 47 and " " in long_name
            account_mappings = {str(bank_ids[0]): long_name}
            config = _config(bank_ids, cutover, account_mappings=account_mappings)

            before = await _counts(session)
            report_a = await project(session, config)
            after = await _counts(session)

        # --- read-only invariant: zero core writes --------------------------
        assert before == after, "projection mutated core rows"

        # --- balanced invariant: every emitted entry sums to zero -----------
        assert report_a.emitted_count > 0, "expected a non-empty projection"
        for entry in report_a.entries:
            total = sum((p.amount for p in entry.postings), Decimal("0"))
            assert total == 0, f"unbalanced entry on {entry.date}: {entry.payee!r}"

        # Non-INR rows are skipped (the scenario deliberately seeds one USD row).
        assert report_a.non_inr_count >= 1
        # Self-transfers collapse to single paired entries, not two.
        assert report_a.self_transfer_pairs >= 1

        # --- ledger posting contract: every posting splits on >=2 spaces -----
        _assert_ledger_posting_contract(report_a.journal)

        # --- long/spaced account name survives rendering verbatim ------------
        # The exact name (spaces included) must appear in the journal and be
        # recoverable as the part before the first >=2-space run — not absorbed
        # into the amount the way a single-space gap would cause. Pick a posting
        # line (one carrying an amount), not the bare ``account`` declaration.
        posting_with_long = [
            ln
            for ln in report_a.journal.splitlines()
            if long_name in ln and "INR" in ln
        ]
        assert posting_with_long, (
            "long/spaced account name never posted with an amount (only declared?)"
        )
        long_line = posting_with_long[0]
        parts = re.split(r" {2,}", long_line.lstrip(), maxsplit=1)
        assert parts[0] == long_name, (
            f"long account name mangled/absorbed into amount: {parts[0]!r}"
        )

        # --- optional real-contract check: ledger CLI parses the journal -----
        _ledger_parses_journal(report_a.journal)

        # --- stable invariant: a second run is byte-identical ---------------
        async with maker() as session:
            report_b = await project(
                session,
                _config(bank_ids, cutover, account_mappings=account_mappings),
            )
        assert report_a.journal == report_b.journal
        assert report_a.emitted_count == report_b.emitted_count
    finally:
        await engine.dispose()


async def test_projection_reports_missing_selected_account_without_writing(tmp_path):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="golden")
    await load_scenario(scenario, db)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            # A real account plus a deliberately-missing id.
            real_id = (
                await session.execute(
                    select(Account.id)
                    .where(Account.type == "bank_account")
                    .order_by(Account.id)
                    .limit(1)
                )
            ).scalar_one()
            config = _config((real_id, 9_999_999), scenario.as_of)
            before = await _counts(session)
            report = await project(session, config)
            after = await _counts(session)

        assert before == after
        assert any(s.reason == "unknown_account" for s in report.skipped)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Multi-currency FX projection: skip vs priced, price directives, missing-rate
# diagnostics, byte-stability. Exercises the production projection against the
# scenario's explicit historical FX rates (covered + missing-rate rows).
# ---------------------------------------------------------------------------


async def _bank_ids(session) -> tuple[int, ...]:
    return tuple(
        (
            await session.execute(
                select(Account.id)
                .where(Account.type == "bank_account", Account.active.is_(True))
                .order_by(Account.id)
            )
        )
        .scalars()
        .all()
    )


async def test_priced_policy_emits_covered_foreign_entries_and_price_directives(
    tmp_path,
):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    fx = _fx_rates_from_scenario(scenario)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            bank_ids = await _bank_ids(session)
            cutover = scenario.as_of - datetime.timedelta(days=20)
            report = await project(
                session,
                _config(
                    bank_ids,
                    cutover,
                    non_inr_policy="priced",
                    fx_rates=fx,
                ),
            )
            # The scenario seeds covered USD + EUR rows and missing-rate GBP +
            # backdated-USD rows. Priced emits the covered ones and skips the
            # uncovered ones as missing_fx_rate.
            assert report.projected_foreign_count >= 1
            assert report.missing_fx_rate_count >= 1
            assert "USD" in report.source_currencies
            # Covered rows are emitted in their own commodity, never relabelled.
            assert " USD" in report.journal
            # Price directives are emitted (deduplicated per currency/date).
            assert "P " in report.journal and "INR" in report.journal
            # Missing-rate rows surface the diagnostic reason.
            assert any(s.reason == "missing_fx_rate" for s in report.skipped)
    finally:
        await engine.dispose()


async def test_skip_policy_skips_every_non_inr_row(tmp_path):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    fx = _fx_rates_from_scenario(scenario)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            bank_ids = await _bank_ids(session)
            cutover = scenario.as_of - datetime.timedelta(days=20)
            report = await project(
                session,
                _config(
                    bank_ids,
                    cutover,
                    non_inr_policy="skip",
                    fx_rates=fx,  # rates present but unused under skip
                ),
            )
            # Under skip, NO foreign row is emitted even though rates exist.
            assert report.projected_foreign_count == 0
            assert report.missing_fx_rate_count == 0
            assert report.non_inr_count >= 1
            assert all(s.reason != "missing_fx_rate" for s in report.skipped)
    finally:
        await engine.dispose()


async def test_priced_projection_is_byte_stable_across_runs(tmp_path):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    fx = _fx_rates_from_scenario(scenario)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        cutover = scenario.as_of - datetime.timedelta(days=20)
        async with maker() as session:
            bank_ids = await _bank_ids(session)
            report_a = await project(
                session,
                _config(bank_ids, cutover, non_inr_policy="priced", fx_rates=fx),
            )
        async with maker() as session:
            report_b = await project(
                session,
                _config(bank_ids, cutover, non_inr_policy="priced", fx_rates=fx),
            )
        # Stable bytes: price-directive dedup + deterministic sort make the
        # priced journal byte-identical across runs.
        assert report_a.journal == report_b.journal
        assert report_a.projected_foreign_count == report_b.projected_foreign_count
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Investment-lot projection: project_investments gate, complete-lot output,
# excluded reasons, no fabricated cost. The scenario's NSDL CAS upload carries
# one complete MF acquisition (→ InvestmentLot) plus excluded demat/disposal/
# value-only facts.
# ---------------------------------------------------------------------------


async def test_investment_projection_gate_off_emits_no_lots(tmp_path):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            bank_ids = await _bank_ids(session)
            cutover = scenario.as_of - datetime.timedelta(days=20)
            report = await project(
                session, _config(bank_ids, cutover, project_investments=False)
            )
            # Gate off → no lot entries counted and no per-instrument lot
            # account. (``Assets:Investments:Unallocated`` may still appear from
            # ordinary investment contribution postings — that is not a lot.)
            assert report.investment_lot_count == 0
            assert "Assets:Investments:INE000A01020" not in report.journal
    finally:
        await engine.dispose()


async def test_investment_projection_gate_on_emits_complete_lot_and_excluded_reasons(
    tmp_path,
):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            # The loader creates exactly one complete lot from the MF
            # acquisition fact (INE000A01020). The portfolio it belongs to also
            # holds value-only positions, so the lot alone cannot represent the
            # portfolio's worth — projection falls the portfolio back to its
            # authoritative aggregate value and drops the lot, rather than
            # emitting a partial lot set that omits the rest of the value.
            lots_before = (await session.execute(select(InvestmentLot))).scalars().all()
            assert len(lots_before) == 1
            assert lots_before[0].instrument_id == "INE000A01020"

            bank_ids = await _bank_ids(session)
            cutover = scenario.as_of - datetime.timedelta(days=20)
            before = await _counts(session)
            report = await project(
                session, _config(bank_ids, cutover, project_investments=True)
            )
            after = await _counts(session)

        # Read-only: projection changed no core rows.
        assert before == after
        # The portfolio is represented by its aggregate value, not by a partial
        # lot set: no lot, no commodity holding, and no fabricated cost.
        assert report.investment_lot_count == 0
        assert "Assets:Investments:INE000A01020" not in report.journal
        assert report.investment_valuation_portfolios
        assert report.investment_valuation_total > 0
        assert "valuation_only_no_cost_basis" in report.investment_excluded
        # Lot-level disposal tracking is no longer part of the projection:
        # CAS is projected from portfolio aggregates, so per-instrument
        # suppression state is always empty here (the core lot service still
        # computes it for the dashboard).
        assert report.investment_disposal_unresolved == ()
        # Paisa reports only its own policy diagnostics. Lot-classification
        # reasons (not_mutual_fund, missing_lot_facts, ...) belong to the core
        # investment service: the authoritative aggregate INCLUDES those
        # holdings' value, so listing them here would imply value was omitted.
        excluded = set(report.investment_excluded)
        assert excluded == {"valuation_only_no_cost_basis"}
    finally:
        await engine.dispose()


async def test_investment_projection_suppression_invents_no_cost(tmp_path):
    """No cost basis or asset account is invented for a suppressed instrument.

    The disposal-only instrument (INE000A01045) is suppressed conservatively.
    The portfolio also holds value-only positions, so it uses the valuation-only
    path: its value is represented in aggregate and NO commodity lot is emitted.
    Asserted structurally — a bare amount substring would also match the
    aggregate valuation and so could pass without a lot existing at all.
    """
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            bank_ids = await _bank_ids(session)
            cutover = scenario.as_of - datetime.timedelta(days=20)
            report = await project(
                session, _config(bank_ids, cutover, project_investments=True)
            )
        # No commodity lot account is emitted for ANY instrument.
        assert "Assets:Investments:INE000A01045" not in report.journal
        assert report.investment_disposal_unresolved == ()
        # No commodity lot at all under the valuation-only path — so no cost
        # basis is asserted for any instrument, invented or otherwise.
        assert report.investment_lot_count == 0
        assert report.document.lot_postings == ()
        assert not any(
            line.lstrip().startswith("Assets:Investments:INE")
            for line in report.journal.splitlines()
        )
        # The value is still represented, as an aggregate carrying no cost.
        assert report.investment_valuation_portfolios
        assert report.investment_valuation_total > 0
        assert "{" not in report.journal  # no cost annotation anywhere
    finally:
        await engine.dispose()
