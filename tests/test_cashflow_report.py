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
from tests.conftest import (
    MISSING_ACCOUNT_ID,
    bank_account,
    card_account,
)

pytestmark = pytest.mark.anyio
D = Decimal
JUN = datetime.date(2026, 6, 1)
JUN_END = datetime.date(2026, 6, 30)


async def _add(session, **kw):
    """Seed one transaction, linked to the bank account unless ``account_id`` says else.

    The report is bank-scoped, so an unlinked row is *unaccounted* and reaches no
    headline figure. Defaulting the link here is what keeps every test below about
    the thing it is named for; a test that wants a card row, or a row on no account
    at all, passes ``account_id`` and says so.
    """
    base = dict(
        bank="hdfc",
        email_type="x",
        currency="INR",
        transaction_date=datetime.date(2026, 6, 15),
        account_id=await bank_account(session),
    )
    base.update(kw)
    session.add(Transaction(**base))
    await session.flush()


async def test_refund_and_cashback_reduce_expense(session: AsyncSession):
    # Line totals show each line's effect on its bucket: spend and income
    # positive, contra (refund) negative.
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(session, direction="debit", amount=D("300"), category="groceries")
    await _add(session, direction="credit", amount=D("50"), category="refund")
    await _add(session, direction="credit", amount=D("20"), category="cashback_rewards")
    s = await cashflow_summary(session, JUN, JUN_END)
    assert s.expense.total == D("230")  # 300 - 50 - 20
    salary = next(ln for ln in s.income.lines if ln.slug == "salary")
    groceries = next(ln for ln in s.expense.lines if ln.slug == "groceries")
    refund = next(ln for ln in s.expense.lines if ln.slug == "refund")
    assert salary.total == D("1000") and salary.kind is None
    assert groceries.total == D("300")
    assert refund.total == D("-50")


async def test_investment_same_slug_both_directions_are_distinct_lines(session):
    # `investment` appears as BOTH a contribution (debit) and a redemption
    # (credit); the two must be separate lines distinguished by `kind`.
    await _add(session, direction="debit", amount=D("1000"), category="investment")
    await _add(session, direction="credit", amount=D("300"), category="investment")
    await _add(
        session, direction="credit", amount=D("400"), category="investment_redemption"
    )
    s = await cashflow_summary(session, JUN, JUN_END)
    by_kind = {ln.kind: ln for ln in s.investment.lines if ln.slug == "investment"}
    assert by_kind["contribution"].total == D("1000")  # displayed positive
    assert by_kind["redemption"].total == D("-300")  # displayed negative
    assert s.investment.contributions == D("1000")
    assert s.investment.redemptions == D("700")
    assert s.investment.net == D("300")


async def test_the_swipe_is_out_of_the_bank_view_and_the_bill_is_not_double_counted(
    session: AsyncSession,
):
    """The two views count different rows, which is what stops the double count.

    Bank: the bill paid. All accounts: the swipe it settled. The bill is internal
    in the detail — counting it there as well would charge the same rupee twice.
    """
    card = await card_account(session)
    await _add(
        session,
        direction="debit",
        amount=D("500"),
        category="dining",
        account_id=card,
    )  # the swipe
    await _add(
        session, direction="debit", amount=D("500"), category="credit_card_payment"
    )  # the bank leg: paying that swipe's bill
    await _add(
        session,
        direction="credit",
        amount=D("500"),
        category="credit_card_payment",
        account_id=card,
    )  # the card leg of the same payment

    s = await cashflow_summary(session, JUN, JUN_END)

    # Bank: the bill, and only the bill. The card's two rows are out of scope.
    assert s.expense.total == D("500")
    assert [ln.slug for ln in s.expense.lines] == ["credit_card_payment"]
    assert s.expense.count == 1

    # All accounts: the swipe, and only the swipe.
    assert s.expense_detail.total == D("500")
    assert [ln.slug for ln in s.expense_detail.lines] == ["dining"]
    assert s.expense_detail.count == 1


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
    """Every term pinned to a figure computed from the fixture, not from the code.

    The decoys are the point: a card row, an unlinked row and a dangling one are
    all in range and all carry bucketable slugs, so a summary that forgot its
    scope anywhere would land them in a headline and be caught here rather than
    quietly reconciling with itself.
    """
    card = await card_account(session)

    # Bank income: 1000 + 50 = 1050.
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(session, direction="credit", amount=D("50"), category="interest")
    # Transfers in: 200.
    await _add(session, direction="credit", amount=D("200"), category="repayment")
    # Bank expense: 300 dining - 50 refund + (400 - 100) of card bills = 550.
    await _add(session, direction="debit", amount=D("300"), category="dining")
    await _add(session, direction="credit", amount=D("50"), category="refund")
    await _add(
        session, direction="debit", amount=D("400"), category="credit_card_payment"
    )
    await _add(
        session, direction="credit", amount=D("100"), category="credit_card_payment"
    )
    # Net invested: 100 - 40 = 60.
    await _add(session, direction="debit", amount=D("100"), category="investment")
    await _add(
        session, direction="credit", amount=D("40"), category="investment_redemption"
    )
    # Internal: excluded from every term above; gross 900, net -500.
    await _add(session, direction="debit", amount=D("700"), category="self_transfer")
    await _add(session, direction="credit", amount=D("200"), category="self_transfer")
    # Decoys. The card rows are out of the bank view; the unlinked and the
    # dangling row are in no scope at all.
    await _add(
        session, direction="debit", amount=D("900"), category="dining", account_id=card
    )
    await _add(
        session,
        direction="credit",
        amount=D("5000"),
        category="salary",
        account_id=card,
    )
    await _add(
        session, direction="debit", amount=D("111"), category="dining", account_id=None
    )
    await _add(
        session,
        direction="credit",
        amount=D("222"),
        category="salary",
        account_id=MISSING_ACCOUNT_ID,
    )

    s = await cashflow_summary(session, JUN, JUN_END)

    assert s.income.total == D("1050")
    assert s.transfers_in.total == D("200")
    assert s.expense.total == D("550")
    assert s.investment.net == D("60")
    # 1050 + 200 - 550 - 60, to the paisa.
    assert s.net_cash_retained == D("640")

    # The detail is a different question over a different population: every
    # account's expense-bucket rows (300 - 50 + 900 + 111), card bills internal.
    assert s.expense_detail.total == D("1261")

    assert s.footnotes.internal_count == 2
    assert s.footnotes.internal_gross == D("900")
    assert s.footnotes.internal_net == D("-500")
    assert s.footnotes.unaccounted_count == 2
    assert s.footnotes.unaccounted_net == D("111")  # -111 + 222


async def test_trend_and_salary_count_are_both_bank_scoped(session: AsyncSession):
    # The salary count is a second, independent query: a scope applied to the
    # monetary series alone would leave the count contradicting the bar above it.
    today = datetime.date(2026, 6, 15)
    card = await card_account(session)
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
        amount=D("7000"),
        category="salary",
        account_id=card,
        transaction_date=datetime.date(2026, 6, 6),
    )
    await _add(
        session,
        direction="debit",
        amount=D("400"),
        category="dining",
        account_id=card,
        transaction_date=datetime.date(2026, 6, 7),
    )
    await _add(
        session,
        direction="debit",
        amount=D("2500"),
        category="credit_card_payment",
        transaction_date=datetime.date(2026, 6, 3),
    )

    pts = await cashflow_trend(session, months=1, today=today)
    jun = next(p for p in pts if p.month == "2026-06")

    assert jun.income == D("1000")  # the card salary is not in the bank series
    assert jun.salary_count == 1  # nor in the count beside it
    assert jun.expense == D("2500")  # the bank's card bill, not the card swipe


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
    # The June 25 row is out. The USD salary counts but adds no rupees.
    assert jun.income == D("1000")
    assert jun.salary_count == 2


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


async def test_trend_pre_seeds_every_month_so_empty_history_is_zeros_not_no_points(
    session: AsyncSession,
):
    """The trend never returns an empty array — a history with nothing in it comes
    back as a full window of zeros. An empty state gated on the array's length is
    therefore unreachable, and the page has to gate on the values being zero.
    """
    today = datetime.date(2026, 6, 17)
    points = await cashflow_trend(session, 12, today=today)

    # Each drawn month has a range to click.
    assert [p.month for p in points] == list(trend_ranges(12, today=today))
    assert all(
        p.income == 0 and p.expense == 0 and p.net_invested == 0 and p.salary_count == 0
        for p in points
    )


def test_presets_cover_month_quarter_and_year_ranges():
    """The one-click ranges the filter bar shows, with concrete bounds.

    A current period runs to today. A completed period (last month, last
    quarter) runs to its own last day, not today.
    """
    from financial_dashboard.web.cashflow import _presets

    presets = _presets(datetime.date(2026, 8, 18))

    assert presets == [
        {"label": "This month", "date_from": "2026-08-01", "date_to": "2026-08-18"},
        {"label": "Last month", "date_from": "2026-07-01", "date_to": "2026-07-31"},
        {"label": "This quarter", "date_from": "2026-07-01", "date_to": "2026-08-18"},
        {"label": "Last quarter", "date_from": "2026-04-01", "date_to": "2026-06-30"},
        {"label": "Financial year", "date_from": "2026-04-01", "date_to": "2026-08-18"},
        {"label": "Calendar year", "date_from": "2026-01-01", "date_to": "2026-08-18"},
    ]

    # In January to March, the financial year and last quarter roll back a year.
    jan = {item["label"]: item for item in _presets(datetime.date(2026, 2, 15))}
    assert jan["Financial year"]["date_from"] == "2025-04-01"
    assert jan["Last quarter"]["date_from"] == "2025-10-01"
    assert jan["Last quarter"]["date_to"] == "2025-12-31"
