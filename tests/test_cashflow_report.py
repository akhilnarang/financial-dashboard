import datetime
from decimal import Decimal

import pytest
from sqlalchemy import null, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.cashflow.report import (
    cashflow_summary,
    cashflow_trend,
    trend_ranges,
)

pytestmark = pytest.mark.anyio
D = Decimal
JUN = datetime.date(2026, 6, 1)
JUN_END = datetime.date(2026, 6, 30)


async def _add(session, **kw):
    base = dict(
        bank="hdfc",
        email_type="x",
        currency="INR",
        transaction_date=datetime.date(2026, 6, 15),
    )
    base.update(kw)
    session.add(Transaction(**base))
    await session.flush()


async def test_income_excludes_repayment(session: AsyncSession):
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(
        session,
        direction="credit",
        amount=D("5000"),
        category="repayment",
        counterparty="MOM",
    )
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.income.total == D("1000")
    assert s.transfers_in.total == D("5000")
    assert s.transfers_in.lines[0].counterparty == "MOM"


async def test_refund_and_cashback_reduce_expense(session: AsyncSession):
    await _add(session, direction="debit", amount=D("300"), category="groceries")
    await _add(session, direction="credit", amount=D("50"), category="refund")
    await _add(session, direction="credit", amount=D("20"), category="cashback_rewards")
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.expense.total == D("230")  # 300 - 50 - 20


async def test_investment_gross_and_net(session: AsyncSession):
    await _add(session, direction="debit", amount=D("1000"), category="investment")
    await _add(
        session, direction="credit", amount=D("400"), category="investment_redemption"
    )
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.investment.contributions == D("1000")
    assert s.investment.redemptions == D("400")
    assert s.investment.net == D("600")


async def test_investment_same_slug_both_directions_are_distinct_lines(session):
    # `investment` appears as BOTH a contribution (debit) and a redemption
    # (credit); the two must be separate lines distinguished by `kind`.
    await _add(session, direction="debit", amount=D("1000"), category="investment")
    await _add(session, direction="credit", amount=D("300"), category="investment")
    s = await cashflow_summary(session, JUN, JUN_END)
    by_kind = {ln.kind: ln for ln in s.investment.lines if ln.slug == "investment"}
    assert by_kind["contribution"].total == D("1000")  # displayed positive
    assert by_kind["redemption"].total == D("-300")  # displayed negative
    assert s.investment.net == D("700")


async def test_breakdown_line_display_signs(session: AsyncSession):
    # Line totals are each line's contribution to its displayed bucket:
    # income positive, expense spend positive, contra (refund) negative.
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(session, direction="debit", amount=D("300"), category="groceries")
    await _add(session, direction="credit", amount=D("50"), category="refund")
    s = await cashflow_summary(session, JUN, JUN_END)
    salary = next(ln for ln in s.income.lines if ln.slug == "salary")
    groceries = next(ln for ln in s.expense.lines if ln.slug == "groceries")
    refund = next(ln for ln in s.expense.lines if ln.slug == "refund")
    assert salary.total == D("1000") and salary.kind is None
    assert groceries.total == D("300")  # spend shown positive
    assert refund.total == D("-50")  # contra shown negative


async def test_internal_excluded_but_footnoted(session: AsyncSession):
    await _add(
        session, direction="debit", amount=D("9999"), category="credit_card_payment"
    )
    await _add(session, direction="debit", amount=D("100"), category="dining")
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.expense.total == D("100")
    assert s.footnotes.internal_count == 1


async def test_cc_no_double_count(session: AsyncSession):
    await _add(session, direction="debit", amount=D("500"), category="dining")  # swipe
    await _add(
        session, direction="debit", amount=D("500"), category="credit_card_payment"
    )  # bank leg
    await _add(
        session, direction="credit", amount=D("500"), category="credit_card_payment"
    )  # card leg
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.expense.total == D("500")


async def test_uncategorized_line_is_currency_agnostic(session: AsyncSession):
    # The uncategorized line applies no currency filter, so a non-INR
    # uncategorized row is included and the line matches /transactions?uncategorized=1.
    await _add(session, direction="debit", amount=D("11"), category=None)
    await _add(session, direction="debit", amount=D("22"), category="unknown")
    await _add(session, direction="debit", amount=D("33"), category="mystery_slug")
    await _add(session, direction="debit", amount=D("5"), category=None, currency="USD")
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.uncategorized.count == 4  # includes the non-INR row
    assert s.uncategorized.total == D("-71")  # -(11+22+33+5)


async def test_non_inr_excluded_from_buckets_and_counted(session: AsyncSession):
    # A non-INR *bucketed* row is excluded from the bucket but the non-INR row
    # that is ALSO uncategorized (above) appears in the uncategorized line too —
    # informational overlap, no double-count (neither feeds a headline bucket).
    await _add(
        session, direction="debit", amount=D("5"), category="dining", currency="USD"
    )
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.expense.total == D("0")
    assert s.footnotes.non_inr_count == 1


async def test_null_currency_treated_as_inr(session: AsyncSession):
    # A NULL currency must be bucketed as INR, not dropped. `currency=None` would
    # NOT exercise that: the column has an "INR" default, so the ORM writes "INR"
    # and nothing NULL ever reaches the query. `null()` forces the real SQL NULL,
    # and the row is re-read below so the test can never quietly stop testing it.
    await _add(
        session, direction="debit", amount=D("80"), category="dining", currency=null()
    )
    stored = (await session.execute(select(Transaction.currency))).scalars().all()
    assert stored == [None]

    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.expense.total == D("80")
    assert s.footnotes.non_inr_count == 0


async def test_undated_line(session: AsyncSession):
    # transaction_date IS NULL rows are excluded from every month/range bucket
    # and surfaced only in the Undated line (net + count) — no created_at guess.
    await _add(
        session,
        direction="debit",
        amount=D("40"),
        category="dining",
        transaction_date=None,
        created_at=datetime.datetime(2026, 6, 10, tzinfo=datetime.UTC),
    )
    await _add(
        session,
        direction="debit",
        amount=D("50"),
        category="dining",
        transaction_date=None,
        created_at=None,
    )
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.expense.total == D("0")  # neither undated row is bucketed
    assert s.footnotes.undated_count == 2  # both surfaced in the Undated line
    assert s.footnotes.undated_net == D("-90")  # signed net of the two debits


async def test_boundaries_inclusive_and_zero_amount(session: AsyncSession):
    await _add(
        session,
        direction="debit",
        amount=D("0"),
        category="dining",
        transaction_date=JUN,
    )
    await _add(
        session,
        direction="debit",
        amount=D("70"),
        category="dining",
        transaction_date=JUN_END,
    )
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.expense.count == 2
    assert s.expense.total == D("70")


async def test_reconciliation_identity(session: AsyncSession):
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(session, direction="credit", amount=D("200"), category="repayment")
    await _add(session, direction="debit", amount=D("300"), category="dining")
    await _add(session, direction="debit", amount=D("100"), category="investment")
    s = await cashflow_summary(session, JUN, JUN_END)
    # income + transfers_in - expense - net_invested
    assert s.net_cash_retained == D("1000") + D("200") - D("300") - D("100")


async def test_trend_zero_filled_and_salary_count(session: AsyncSession):
    # A fixed `today` keeps the trailing-window assertion deterministic.
    today = datetime.date(2026, 6, 15)
    # Both salary rows are on/before `today` (June 5 and June 12).
    await _add(
        session,
        direction="credit",
        amount=D("1000"),
        category="salary",
        transaction_date=datetime.date(2026, 6, 5),
    )
    await _add(
        session,
        direction="credit",
        amount=D("1000"),
        category="salary",
        transaction_date=datetime.date(2026, 6, 12),
    )
    # A repayment credit in June must NOT inflate trend income.
    await _add(
        session,
        direction="credit",
        amount=D("9000"),
        category="repayment",
        transaction_date=datetime.date(2026, 6, 8),
    )
    pts = await cashflow_trend(session, months=3, today=today)
    assert len(pts) == 3  # no gaps; trailing 3 incl. current partial month
    assert [p.month for p in pts] == ["2026-04", "2026-05", "2026-06"]
    jun = next(p for p in pts if p.month == "2026-06")
    assert jun.salary_count == 2
    assert jun.income == D("2000")  # repayment excluded from trend income


async def test_trend_excludes_future_rows(session: AsyncSession):
    # A row dated after `today` (still the current calendar month) is excluded
    # from that month's partial-month figures.
    today = datetime.date(2026, 6, 15)
    await _add(
        session,
        direction="credit",
        amount=D("1000"),
        category="salary",
        transaction_date=datetime.date(2026, 6, 5),
    )
    await _add(
        session,
        direction="credit",
        amount=D("500"),
        category="salary",
        transaction_date=datetime.date(2026, 6, 25),
    )  # future vs today
    pts = await cashflow_trend(session, months=1, today=today)
    jun = next(p for p in pts if p.month == "2026-06")
    assert jun.income == D("1000")  # the June 25 row is excluded
    assert jun.salary_count == 1


async def test_trend_salary_count_ignores_currency(session: AsyncSession):
    # salary_count counts every salary row in the month; only the three monetary
    # series (income / expense / net_invested) drop foreign-currency rows.
    today = datetime.date(2026, 6, 15)
    await _add(
        session,
        direction="credit",
        amount=D("1000"),
        category="salary",
        transaction_date=datetime.date(2026, 6, 5),
    )
    await _add(
        session,
        direction="credit",
        amount=D("2000"),
        category="salary",
        currency="USD",
        transaction_date=datetime.date(2026, 6, 6),
    )
    pts = await cashflow_trend(session, months=1, today=today)
    jun = next(p for p in pts if p.month == "2026-06")
    assert jun.salary_count == 2  # the USD salary is counted
    assert jun.income == D("1000")  # but it does not reach the rupee sum


async def test_trend_months_clamped(session: AsyncSession):
    today = datetime.date(2026, 6, 15)
    assert len(await cashflow_trend(session, months=1, today=today)) == 1
    assert len(await cashflow_trend(session, months=999, today=today)) == 60


def test_resolve_range_independent_bounds():
    from financial_dashboard.services.cashflow.report import resolve_range

    today = datetime.date(2026, 6, 15)
    # both missing -> first-of-month .. today
    assert resolve_range(None, None, today=today) == (datetime.date(2026, 6, 1), today)
    # invalid date_from does NOT reset a valid date_to, and vice-versa
    assert resolve_range("garbage", "2026-06-20", today=today) == (
        datetime.date(2026, 6, 1),
        datetime.date(2026, 6, 20),
    )
    assert resolve_range("2026-06-03", "nonsense", today=today) == (
        datetime.date(2026, 6, 3),
        today,
    )


async def test_blank_category_joins_the_null_uncategorized_line(session: AsyncSession):
    """A category stored as "" is not a slug — it is the same absence a NULL is.

    Left as its own line it would be a line whose drill-through link (the
    category-less filter) lists every row except the one it counted.
    """
    await _add(session, direction="debit", amount=D("800"), category=null())
    await _add(session, direction="debit", amount=D("600"), category="")
    await _add(session, direction="debit", amount=D("200"), category="unknown")
    await _add(session, direction="debit", amount=D("90"), category="crypto")

    s = await cashflow_summary(session, JUN, JUN_END)

    by_slug = {ln.slug: ln for ln in s.uncategorized.lines}
    # Three lines, not four: NULL and "" are one.
    assert set(by_slug) == {None, "unknown", "crypto"}
    assert by_slug[None].count == 2
    assert by_slug[None].total == D("-1400")
    assert by_slug[None].label == "(uncategorized)"
    assert by_slug["crypto"].label == "unmapped: crypto"
    # The tile still counts every uncategorized row.
    assert s.uncategorized.count == 4
    assert s.uncategorized.total == D("-1690")
    # And a blank category is in no bucket.
    assert s.expense.total == D("0")


def test_trend_ranges_give_each_month_its_own_days():
    """Clicking a month sets the range to that month, so each month needs bounds
    computed over the same window the trend draws — and the newest month is a
    partial one, so its upper bound is today, not a month end in the future."""
    today = datetime.date(2026, 6, 17)
    ranges = trend_ranges(12, today=today)

    assert len(ranges) == 12
    assert list(ranges)[-1] == "2026-06"
    assert list(ranges)[0] == "2025-07"
    # The current, partial month stops at today.
    assert ranges["2026-06"] == ["2026-06-01", "2026-06-17"]
    # A whole past month runs to its real last day — 31sts, 30ths and February.
    assert ranges["2026-05"] == ["2026-05-01", "2026-05-31"]
    assert ranges["2025-11"] == ["2025-11-01", "2025-11-30"]
    assert ranges["2026-02"] == ["2026-02-01", "2026-02-28"]


async def test_every_month_the_trend_draws_has_a_range_to_click(session: AsyncSession):
    """A month with bars but no bounds is a month the user can click to nowhere."""
    today = datetime.date(2026, 6, 17)
    points = await cashflow_trend(session, 12, today=today)
    assert [p.month for p in points] == list(trend_ranges(12, today=today))


async def test_trend_pre_seeds_every_month_so_empty_history_is_zeros_not_no_points(
    session: AsyncSession,
):
    """The trend never returns an empty array — a history with nothing in it comes
    back as a full window of zeros. An empty state gated on the array's length is
    therefore unreachable, and the page has to gate on the values being zero.
    """
    today = datetime.date(2026, 6, 17)
    points = await cashflow_trend(session, 12, today=today)

    assert len(points) == 12
    assert all(
        p.income == 0 and p.expense == 0 and p.net_invested == 0 and p.salary_count == 0
        for p in points
    )
