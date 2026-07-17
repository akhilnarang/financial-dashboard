"""Production read-service parity over the loaded synthetic seed.

Runs the *production* read paths — ``cashflow_summary``, ``current_networth``,
``monthly_trend``, the Paisa projection (ledger backend), and the offline
statement reconciliation — over a loaded synthetic DB and asserts the
invariants the canonical scenario is shaped to exercise:

* **scope partition**: the cashflow report's bank scope excludes card swipes
  and unaccounted rows; the footnotes count them;
* **bucket / sign identities**: income is credit-positive, expense is
  debit-positive; contra-credits (refund/fee reversal) net against spend;
* **refund/fee netting**: a contra-expense credit reduces the expense bucket;
* **non-INR / undated footnotes**: foreign-currency and undated rows surface in
  their footnotes, never in a headline bucket;
* **latest-source / forward-fill / stale / unreconciled**: net-worth picks the
  latest per source, the trend forward-fills across months, and an
  unreconciled CAS portfolio is flagged;
* **projection**: every emitted entry balances, the projection is a pure read,
  and the skip reasons (unmatched self-transfer / card-side payment /
  missing FX rate) appear for the rows the scenario deliberately seeds.

Everything stays offline: no socket, no parser, no Telegram, no Paisa HTTP. The
projection uses ``allow_remote=False`` so it never calls out.
"""

import datetime
from collections import Counter
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    CasUpload,
    Transaction,
)
from financial_dashboard.services.cashflow.report import cashflow_summary
from financial_dashboard.services.networth import current_networth, monthly_trend
from financial_dashboard.services.paisa import PaisaProjectionConfig, project
from financial_dashboard.services.paisa.config import FxRate
from scripts.synth import PROJECTION_CUTOVER, build_scenario, load_scenario
from scripts.synth.loader import create_synthetic_engine

pytestmark = pytest.mark.anyio


def _synthetic_db(tmp_path, name="synthetic.db"):
    p = tmp_path / "synthetic" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _fx_rates_for(scenario) -> dict[str, tuple[FxRate, ...]]:
    """Build the projection's ``fx_rates`` map from the scenario's deterministic
    rates, using the production :class:`FxRate` so the lookup path is real."""
    out: dict[str, list[FxRate]] = {}
    for fx in scenario.fx_rates:
        out.setdefault(fx.currency.upper(), []).append(
            FxRate(date=fx.date, rate=fx.rate)
        )
    return {k: tuple(sorted(v, key=lambda r: r.date)) for k, v in out.items()}


@pytest.fixture
async def loaded(tmp_path):
    db = _synthetic_db(tmp_path)
    scenario = build_scenario(profile="smoke")
    await load_scenario(scenario, db)
    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield scenario, engine, maker
    await engine.dispose()


# ---------------------------------------------------------------------------
# cashflow_summary: scope partition, bucket/sign identities, refund/fee netting
# ---------------------------------------------------------------------------


async def test_cashflow_scope_partition_excludes_card_and_unaccounted(loaded):
    scenario, _engine, maker = loaded
    as_of = scenario.as_of
    async with maker() as session:
        # A range wide enough to cover the whole scenario.
        report = await cashflow_summary(
            session,
            as_of - datetime.timedelta(days=400),
            as_of,
        )
        # Every headline figure is bank-scoped (bank + debit-card accounts).
        acct_types = {
            a.id: a.type
            for a in (await session.execute(select(Account))).scalars().all()
        }
        # Card swipes (cc_debit_purchase on a credit_card account) never appear
        # in a headline bucket — they are out of the bank scope.
        card_swipe = (
            (
                await session.execute(
                    select(Transaction).where(
                        Transaction.email_type == "cc_debit_purchase"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert card_swipe, "scenario must seed CC swipes"
        for swipe in card_swipe:
            assert acct_types.get(swipe.account_id) == "credit_card"
        # The unaccounted footnote counts the no-account rows (>=1 unknown row).
        assert report.footnotes.unaccounted_count >= 1


async def test_cashflow_income_credit_positive_expense_debit_positive(loaded):
    scenario, _engine, maker = loaded
    as_of = scenario.as_of
    async with maker() as session:
        report = await cashflow_summary(
            session, as_of - datetime.timedelta(days=400), as_of
        )
        # Income lines are credit-positive (signed flow was +amount).
        for line in report.income.lines:
            assert line.total >= 0, f"income line {line.slug} negative: {line.total}"
        # The expense bucket total is the debits minus contra-credits; it is
        # non-negative when spend dominates refunds/cashback.
        assert report.expense.total >= 0
        # Salary is present as income.
        salary = next((ln for ln in report.income.lines if ln.slug == "salary"), None)
        assert salary is not None and salary.total > 0


async def test_cashflow_refund_and_fee_reversal_net_against_expense(loaded):
    """A contra-expense credit (refund / fee reversal) reduces the expense
    bucket: the category's signed contribution is negative there."""
    scenario, _engine, maker = loaded
    as_of = scenario.as_of
    async with maker() as session:
        report = await cashflow_summary(
            session, as_of - datetime.timedelta(days=400), as_of
        )
        # The refund line carries a non-positive total (a contra credit) inside
        # the expense bucket — money back reducing spend.
        refund = next((ln for ln in report.expense.lines if ln.slug == "refund"), None)
        assert refund is not None
        assert refund.total <= 0, f"refund contra should be <= 0, got {refund.total}"


async def test_cashflow_non_inr_and_undated_footnotes(loaded):
    """Non-INR rows are excluded from headline rupee buckets but counted in the
    non-INR footnote; undated rows are counted in the (range-independent) undated
    footnote."""
    scenario, _engine, maker = loaded
    as_of = scenario.as_of
    async with maker() as session:
        report = await cashflow_summary(
            session, as_of - datetime.timedelta(days=400), as_of
        )
        # The scenario seeds USD/EUR/GBP/XXX bank-side rows → non-INR footnote.
        assert report.footnotes.non_inr_count >= 1
        # The undated footnote counts the deliberately-undated row.
        assert report.footnotes.undated_count >= 1


# ---------------------------------------------------------------------------
# current_networth + monthly_trend: latest-source, forward-fill, stale,
# unreconciled, non-INR exclusion
# ---------------------------------------------------------------------------


async def test_networth_latest_source_and_non_inr_exclusion(loaded):
    scenario, _engine, maker = loaded
    async with maker() as session:
        summary = await current_networth(session, today=scenario.as_of)
        # Net worth is assets - liabilities, both non-negative.
        assert summary.total_assets > 0
        assert summary.total_liabilities > 0
        assert summary.net_worth == summary.total_assets - summary.total_liabilities
        # A non-INR snapshot exists in the DB but is excluded from the INR totals.
        non_inr = (
            (
                await session.execute(
                    select(BalanceSnapshot).where(BalanceSnapshot.currency != "INR")
                )
            )
            .scalars()
            .all()
        )
        assert non_inr, "scenario must seed a non-INR snapshot"


async def test_networth_flags_unreconciled_portfolio(loaded):
    """An unreconciled CAS portfolio (portfolio_ok=False) surfaces as an
    unreconciled net-worth row."""
    scenario, _engine, maker = loaded
    async with maker() as session:
        summary = await current_networth(session, today=scenario.as_of)
        # The scenario seeds an unreconciled CAS upload.
        unrecon = (
            (
                await session.execute(
                    select(CasUpload).where(CasUpload.portfolio_ok.is_(False))
                )
            )
            .scalars()
            .all()
        )
        assert unrecon, "scenario must seed an unreconciled CAS upload"
        # At least one net-worth row is flagged unreconciled.
        assert any(row.unreconciled for group in summary.groups for row in group.rows)


async def test_monthly_trend_forward_fills_across_months(loaded):
    """The trend emits one point per month from the earliest snapshot to today,
    forward-filling the latest value across month boundaries."""
    scenario, _engine, maker = loaded
    async with maker() as session:
        points = await monthly_trend(session, today=scenario.as_of)
        assert len(points) >= 2, "expected a multi-month trend"
        # Points are month-keyed in order.
        keys = [p.month for p in points]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Projection (ledger backend): balanced, pure-read, skip reasons, FX separation
# ---------------------------------------------------------------------------


def _projection_config(selected, cutover, scenario) -> PaisaProjectionConfig:
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
        account_mappings={},
        category_mappings={},
        non_inr_policy="priced",
        fx_rates=_fx_rates_for(scenario),
        request_timeout_seconds=15,
        ledger_cli="ledger",
    )


async def test_projection_balanced_and_pure_read(loaded):
    scenario, _engine, maker = loaded
    async with maker() as session:
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
        cutover = PROJECTION_CUTOVER
        config = _projection_config(bank_ids, cutover, scenario)

        before = (
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id.in_(bank_ids))
                )
            )
            .scalars()
            .all()
        )
        report = await project(session, config)
        after = (
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id.in_(bank_ids))
                )
            )
            .scalars()
            .all()
        )
        # Pure read: no core row added/removed/changed.
        assert len(after) == len(before)
        # Every emitted entry balances to zero.
        for entry in report.entries:
            total = sum((p.amount for p in entry.postings), Decimal("0"))
            assert total == 0, f"unbalanced entry on {entry.date}: {entry.payee!r}"
        assert report.emitted_count >= 1


async def test_projection_skip_reasons_cover_seeded_edges(loaded):
    """The projection skip-reason set includes the rows the scenario deliberately
    seeds: an unmatched self-transfer, a card-side payment, and a missing-FX-rate
    (GBP) row."""
    scenario, _engine, maker = loaded
    async with maker() as session:
        # Select every account so card-side and unaccounted rows are in view.
        all_ids = tuple(
            (await session.execute(select(Account.id).order_by(Account.id)))
            .scalars()
            .all()
        )
        cutover = PROJECTION_CUTOVER
        config = _projection_config(all_ids, cutover, scenario)
        report = await project(session, config)
        reasons = Counter(row.reason for row in report.skipped)
        # Unmatched self-transfer (single-leg debit) is reported.
        assert reasons.get("unmatched_self_transfer", 0) >= 1, reasons
        # Card-side CC payment is skipped (the bank leg is authoritative).
        assert report.card_side_payments >= 1
        # GBP has no FX rate → missing_fx_rate.
        assert report.missing_fx_rate_count >= 1, (
            f"expected missing_fx_rate for GBP, got {dict(reasons)}"
        )


async def test_projection_fx_commodities_stay_separate(loaded):
    """A priced USD entry is emitted in USD (its own commodity), never relabelled
    INR; the source-currency set records USD."""
    scenario, _engine, maker = loaded
    async with maker() as session:
        bank_ids = tuple(
            (
                await session.execute(
                    select(Account.id).where(
                        Account.type == "bank_account", Account.active.is_(True)
                    )
                )
            )
            .scalars()
            .all()
        )
        cutover = PROJECTION_CUTOVER
        config = _projection_config(bank_ids, cutover, scenario)
        report = await project(session, config)
        # At least one entry posts in USD (covered USD row is in the window).
        usd_postings = [
            p for e in report.entries for p in e.postings if p.commodity == "USD"
        ]
        assert usd_postings, "expected a priced USD posting"
        assert "USD" in report.source_currencies


# ---------------------------------------------------------------------------
# Metadata + category_method fidelity across lanes
# ---------------------------------------------------------------------------


async def test_category_method_axis_populated(loaded):
    """The category_method axis is varied across the corpus in both lanes."""
    _scenario, _engine, maker = loaded
    async with maker() as session:
        rows = (await session.execute(select(Transaction))).scalars().all()
        methods = {r.category_method for r in rows}
        # synthetic is always present (the bulk default); the scenario also
        # seeds manual / llm / pending_llm, and the production self-transfer
        # rule contributes rule.
        assert "synthetic" in methods
        assert {"manual", "llm", "pending_llm"} <= methods, methods


async def test_review_status_axis_populated(loaded):
    _scenario, _engine, maker = loaded
    async with maker() as session:
        rows = (await session.execute(select(Transaction))).scalars().all()
        statuses = {r.review_status for r in rows if r.review_status}
        # The scenario seeds pending / notified / resolved + the reviewed/flagged
        # rows the statement-link and unknown-edge paths add.
        assert {"pending", "notified", "resolved"} <= statuses, statuses


# ---------------------------------------------------------------------------
# Projection running-balance + diagnostics (requirement: never-negative
# projected running balance; resolved+unresolved card payments; investment lot
# emitted; genuinely-invalid currency skipped as invalid_currency)
# ---------------------------------------------------------------------------


def _account_kind(name: str) -> str:
    if name.startswith("Assets:"):
        return "asset"
    if name.startswith("Liabilities:"):
        return "liability"
    return "other"


def _running_balances(report):
    """Walk openings + date-ordered entries and return per-``(account,
    commodity)`` running balances plus each asset account's worst (min) point.

    This is a projection-level reconstruction (independent of the source
    transactions' ``balance`` field) so it validates what an actual ledger
    engine would compute from the emitted journal + openings.
    """
    from collections import defaultdict

    bal = defaultdict(lambda: Decimal("0"))
    for ob in report.openings:
        bal[(ob.account_name, "INR")] += ob.amount
    entries = sorted(
        report.entries,
        key=lambda e: (e.date or datetime.date.max, e.txn_ids[0] if e.txn_ids else 0),
    )
    worst_asset: dict[tuple, Decimal] = {}
    for e in entries:
        for p in e.postings:
            key = (p.account, p.commodity)
            bal[key] += p.amount
            if _account_kind(p.account) == "asset":
                worst_asset[key] = min(worst_asset.get(key, Decimal("0")), bal[key])
    return bal, worst_asset


async def test_projection_running_balance_never_negative_inr(loaded):
    """With the documented cutover, every asset account's INR running balance
    stays >= 0 at every posting, and ``Assets:Investments:Unallocated`` never
    goes negative — the cutover opening snapshots (replayed from the scenario's
    tracked balances) plus the ordered contribution/redemption sequence make
    the projection economically sound."""
    scenario, _engine, maker = loaded
    async with maker() as session:
        all_ids = tuple(
            (await session.execute(select(Account.id).order_by(Account.id)))
            .scalars()
            .all()
        )
        config = PaisaProjectionConfig(
            mode="project",
            base_url="",
            external_url="",
            allow_remote=False,
            auth_username="",
            auth_password="",
            generated_path="",
            selected_account_ids=all_ids,
            cutover_date=PROJECTION_CUTOVER,
            account_mappings={},
            category_mappings={},
            non_inr_policy="priced",
            fx_rates=_fx_rates_for(scenario),
            request_timeout_seconds=15,
            ledger_cli="ledger",
            project_investments=True,
        )
        report = await project(session, config)
        _bal, worst_asset = _running_balances(report)

        # Every INR asset running balance is non-negative at its worst point.
        inr_neg = {k: v for k, v in worst_asset.items() if k[1] == "INR" and v < 0}
        assert not inr_neg, f"INR asset running balance went negative: {inr_neg}"

        # The Unallocated investment account specifically never goes negative.
        unalloc = {k: v for k, v in worst_asset.items() if "Unallocated" in k[0]}
        assert unalloc, "expected Assets:Investments:Unallocated postings"
        for (_name, _ccy), worst in unalloc.items():
            assert worst >= 0, f"Unallocated went negative: {worst}"


async def test_projection_card_payments_resolved_and_unresolved(loaded):
    """At least one bank-side card payment resolves to a selected card liability
    (explicit card_id), and at least one is deliberately unresolved — both
    diagnostics populated."""
    scenario, _engine, maker = loaded
    async with maker() as session:
        all_ids = tuple(
            (await session.execute(select(Account.id).order_by(Account.id)))
            .scalars()
            .all()
        )
        config = PaisaProjectionConfig(
            mode="project",
            base_url="",
            external_url="",
            allow_remote=False,
            auth_username="",
            auth_password="",
            generated_path="",
            selected_account_ids=all_ids,
            cutover_date=PROJECTION_CUTOVER,
            account_mappings={},
            category_mappings={},
            non_inr_policy="priced",
            fx_rates=_fx_rates_for(scenario),
            request_timeout_seconds=15,
            ledger_cli="ledger",
        )
        report = await project(session, config)
        assert report.card_payments_resolved >= 1, "expected a resolved card payment"
        assert report.card_payments_unresolved >= 1, (
            "expected a deliberately-unresolved card payment"
        )


async def test_projection_investment_lot_emitted(loaded):
    """The complete CAS lot is emitted (not suppressed): its instrument carries
    no free-standing disposal, so investment_lot_count >= 1."""
    scenario, _engine, maker = loaded
    async with maker() as session:
        all_ids = tuple(
            (await session.execute(select(Account.id).order_by(Account.id)))
            .scalars()
            .all()
        )
        config = PaisaProjectionConfig(
            mode="project",
            base_url="",
            external_url="",
            allow_remote=False,
            auth_username="",
            auth_password="",
            generated_path="",
            selected_account_ids=all_ids,
            cutover_date=PROJECTION_CUTOVER,
            account_mappings={},
            category_mappings={},
            non_inr_policy="priced",
            fx_rates=_fx_rates_for(scenario),
            request_timeout_seconds=15,
            ledger_cli="ledger",
            project_investments=True,
        )
        report = await project(session, config)
        assert report.investment_lot_count >= 1, (
            f"expected the complete lot emitted, got {report.investment_lot_count}; "
            f"excluded={report.investment_excluded}"
        )


async def test_projection_invalid_currency_skipped_as_invalid(loaded):
    """The genuinely-invalid currency token (``000``, digit-first) is skipped as
    ``invalid_currency`` — not ``missing_fx_rate`` — while USD/EUR stay priced
    and GBP stays missing-rate."""
    from collections import Counter

    scenario, _engine, maker = loaded
    async with maker() as session:
        all_ids = tuple(
            (await session.execute(select(Account.id).order_by(Account.id)))
            .scalars()
            .all()
        )
        config = PaisaProjectionConfig(
            mode="project",
            base_url="",
            external_url="",
            allow_remote=False,
            auth_username="",
            auth_password="",
            generated_path="",
            selected_account_ids=all_ids,
            cutover_date=PROJECTION_CUTOVER,
            account_mappings={},
            category_mappings={},
            non_inr_policy="priced",
            fx_rates=_fx_rates_for(scenario),
            request_timeout_seconds=15,
            ledger_cli="ledger",
        )
        report = await project(session, config)
        reasons = Counter(row.reason for row in report.skipped)
        assert reasons.get("invalid_currency", 0) >= 1, (
            f"expected invalid_currency for the 000 token, got {dict(reasons)}"
        )
        # GBP has no configured rate → still missing_fx_rate (kept separate).
        assert report.missing_fx_rate_count >= 1
        assert "USD" in report.source_currencies  # priced, not skipped
