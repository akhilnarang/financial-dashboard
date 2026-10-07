import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import delete, null, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    BalanceSnapshot,
    BankStatementUpload,
    Transaction,
)
from financial_dashboard.services.cashflow.bridge import cash_bridge
from financial_dashboard.services.cashflow.buckets import BUCKET_BY_SLUG
from financial_dashboard.services.cashflow.report import (
    cashflow_summary,
    cashflow_trend,
    trend_ranges,
)
from financial_dashboard.services.categorization.polarity import (
    EXPENSE_SLUGS,
    INCOME_SLUGS,
)
from financial_dashboard.services.categorization.vocabulary import SEED_CATEGORIES
from tests.conftest import (
    MISSING_ACCOUNT_ID,
    bank_account,
    card_account,
    ensure_account,
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
    txn = Transaction(**base)
    session.add(txn)
    await session.flush()
    return txn


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
    # Bank expense: 300 dining - 50 refund - 20 cashback + (400 - 100) of card
    # bills = 530.
    await _add(session, direction="debit", amount=D("300"), category="dining")
    await _add(session, direction="credit", amount=D("50"), category="refund")
    await _add(session, direction="credit", amount=D("20"), category="cashback_rewards")
    await _add(
        session, direction="debit", amount=D("400"), category="credit_card_payment"
    )
    await _add(
        session, direction="credit", amount=D("100"), category="credit_card_payment"
    )
    # Net invested: 100 - 15 - 40 = 45. One slug in both directions is two lines.
    await _add(session, direction="debit", amount=D("100"), category="investment")
    await _add(session, direction="credit", amount=D("15"), category="investment")
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
    assert s.expense.total == D("530")
    assert s.investment.net == D("45")
    assert s.investment.contributions == D("100")
    assert s.investment.redemptions == D("55")
    # 1050 + 200 - 530 - 45, to the paisa.
    assert s.net_cash_retained == D("675")

    # A line total is its effect on the bucket: a contra line is negative.
    expense = {ln.slug: ln.total for ln in s.expense.lines}
    assert expense["dining"] == D("300")
    assert expense["refund"] == D("-50")
    assert expense["credit_card_payment"] == D("300")
    by_kind = {
        ln.kind: ln.total for ln in s.investment.lines if ln.slug == "investment"
    }
    assert by_kind == {"contribution": D("100"), "redemption": D("-15")}

    # The detail is a different question over a different population: every
    # account's expense-bucket rows (300 - 50 - 20 + 900 + 111). The card bill is
    # internal here, or the swipe it settled would count twice.
    assert s.expense_detail.total == D("1241")
    assert "credit_card_payment" not in {ln.slug for ln in s.expense_detail.lines}

    assert s.footnotes.internal_count == 2
    assert s.footnotes.internal_gross == D("900")
    assert s.footnotes.internal_net == D("-500")
    assert s.footnotes.unaccounted_count == 2
    assert s.footnotes.unaccounted_net == D("111")  # -111 + 222


async def test_trend_is_bank_scoped_zero_filled_and_stops_at_today(
    session: AsyncSession,
):
    # The salary count is a second, independent query: a scope applied to the
    # monetary series alone would leave the count contradicting the bar above it.
    today = datetime.date(2026, 6, 15)
    card = await card_account(session)
    for day in (5, 12):
        await _add(
            session,
            direction="credit",
            amount=D("1000"),
            category="salary",
            transaction_date=datetime.date(2026, 6, day),
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
    # A repayment is not income.
    await _add(
        session,
        direction="credit",
        amount=D("9000"),
        category="repayment",
        transaction_date=datetime.date(2026, 6, 8),
    )
    # A row after today is out of the partial month.
    await _add(
        session,
        direction="credit",
        amount=D("500"),
        category="salary",
        transaction_date=datetime.date(2026, 6, 25),
    )
    # A USD salary counts as a salary but adds no rupees.
    await _add(
        session,
        direction="credit",
        amount=D("2000"),
        category="salary",
        currency="USD",
        transaction_date=datetime.date(2026, 6, 6),
    )

    pts = await cashflow_trend(session, months=3, today=today)

    assert [p.month for p in pts] == ["2026-04", "2026-05", "2026-06"]
    for empty in pts[:2]:
        assert (empty.income, empty.expense, empty.net_invested) == (0, 0, 0)
        assert empty.salary_count == 0
    jun = pts[-1]
    assert jun.income == D("2000")
    assert jun.salary_count == 3
    assert jun.expense == D("2500")  # the bank's card bill, not the card swipe


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


def test_every_seed_slug_has_one_bucket_consistent_with_its_polarity():
    # A new seed slug with no bucket would read as uncategorized. The four
    # re-homed income slugs are the only ones that leave their polarity bucket.
    assert set(BUCKET_BY_SLUG) == {s for s in SEED_CATEGORIES if s != "unknown"}
    rehomed = {
        "refund": "expense",
        "cashback_rewards": "expense",
        "investment_redemption": "investment",
        "repayment": "transfers_in",
    }
    for slug in EXPENSE_SLUGS:
        assert BUCKET_BY_SLUG[slug] == "expense", slug
    for slug in INCOME_SLUGS:
        assert BUCKET_BY_SLUG[slug] == rehomed.get(slug, "income"), slug


async def test_summary_api_returns_the_range_it_used(client, session: AsyncSession):
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(session, direction="debit", amount=D("250"), category="groceries")
    await _add(
        session,
        direction="credit",
        amount=D("500"),
        category="repayment",
        counterparty="MOM",
    )

    r = await client.get(
        "/api/cashflow/summary?date_from=2026-06-01&date_to=2026-06-30"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["date_from"] == "2026-06-01"
    assert body["date_to"] == "2026-06-30"
    assert D(str(body["income"]["total"])) == D("1000")
    assert body["income"]["lines"][0]["slug"] == "salary"
    assert D(str(body["expense"]["total"])) == D("250")
    assert D(str(body["transfers_in"]["total"])) == D("500")
    assert body["transfers_in"]["lines"][0]["counterparty"] == "MOM"
    assert D(str(body["net_cash_retained"])) == D("1250")


async def test_trend_api_clamps_months(client):
    low = await client.get("/api/cashflow/trend?months=0")
    assert low.status_code == 200
    assert len(low.json()) == 1

    high = await client.get("/api/cashflow/trend?months=999")
    assert high.status_code == 200
    assert len(high.json()) == 60


async def test_bridge_closes_on_complete_data_and_a_missing_row_is_the_gap(
    session: AsyncSession,
):
    """Every bank-side row lands in exactly one bridge term.

    The card-side family and excluded rows and the unlinked row are decoys. A
    bridge that took the all-account footnotes, or the unaccounted net, would
    miss the statement closing by their amount. An older statement that
    overlaps the newer one and implies another opening balance loses to it,
    with a warning.
    """
    bank = await bank_account(session)
    card = await card_account(session)
    session.add(
        BankStatementUpload(
            account_id=bank,
            bank="hdfc",
            filename="older.pdf",
            file_path="/synthetic/older.pdf",
            opening_balance="99,000.00",
            closing_balance="98,775.00",
            statement_period_start="15/05/2026",
            statement_period_end="15/06/2026",
        )
    )
    session.add(
        BankStatementUpload(
            account_id=bank,
            bank="hdfc",
            filename="synthetic.pdf",
            file_path="/synthetic/statement.pdf",
            opening_balance="1,00,000.00",
            closing_balance="99,775.00",
            statement_period_start="01/06/2026",
            statement_period_end="30/06/2026",
        )
    )
    for direction, amount, category in [
        ("credit", "1000", "salary"),
        ("credit", "50", "interest"),
        ("credit", "200", "repayment"),
        ("debit", "300", "dining"),
        ("credit", "50", "refund"),
        ("debit", "400", "credit_card_payment"),
        ("debit", "100", "investment"),
        ("credit", "40", "investment_redemption"),
        ("debit", "700", "self_transfer"),
        ("credit", "200", "self_transfer"),
        ("debit", "150", "family"),
        ("debit", "25", None),
    ]:
        await _add(session, direction=direction, amount=D(amount), category=category)
    await _add(
        session,
        direction="debit",
        amount=D("80"),
        category="self_transfer",
        exclude_from_cashflow=True,
    )
    await _add(
        session, direction="debit", amount=D("10"), category="dining", currency="USD"
    )
    await _add(
        session, direction="debit", amount=D("500"), category="family", account_id=card
    )
    await _add(
        session,
        direction="debit",
        amount=D("60"),
        category="dining",
        account_id=card,
        exclude_from_cashflow=True,
    )
    await _add(
        session, direction="debit", amount=D("111"), category="dining", account_id=None
    )

    bridge = await cash_bridge(session, await cashflow_summary(session, JUN, JUN_END))
    assert bridge.opening == D("100000")
    assert bridge.actual_closing == D("99775")
    assert bridge.expected_closing == D("99775")
    assert bridge.gap == 0
    assert {line.key: line.amount for line in bridge.other} == {
        "internal": D("-500"),
        "family": D("-150"),
        "excluded": D("-80"),
        "uncategorized": D("-25"),
        "non_inr": D("-10"),
        "no_balance": 0,
    }
    (account,) = bridge.accounts
    assert (account.opening.source, account.closing.source) == (
        "statement",
        "statement",
    )
    assert account.gap == 0
    assert any("100000.00, 99000.00" in w for w in bridge.warnings)

    await session.execute(
        delete(Transaction)
        .where(Transaction.category == "self_transfer")
        .where(Transaction.direction == "credit")
    )
    bridge = await cash_bridge(session, await cashflow_summary(session, JUN, JUN_END))
    assert bridge.gap == D("200")
    assert bridge.accounts[0].gap == D("200")


async def test_bridge_api_estimates_from_a_running_balance_and_skips_unknown(
    client, session: AsyncSession
):
    """A running balance is moved to each boundary with the rows between, the
    later rows of its own day included. A statement that covers a day wins over
    a nearer running balance, whose same-day order can be wrong. Statements that
    do not cover the day do not hide a nearer snapshot. An account with no
    balance at all stays out of the totals, and its rows do too. A row that a
    statement prints after its period end counts once, on its own date.
    """
    await _add(
        session,
        direction="debit",
        amount=D("100"),
        category="dining",
        transaction_date=datetime.date(2026, 6, 10),
        transaction_time=datetime.time(10),
        balance=D("900"),
    )
    await _add(
        session,
        direction="credit",
        amount=D("30"),
        category="refund",
        transaction_date=datetime.date(2026, 6, 10),
        transaction_time=datetime.time(12),
    )
    await _add(
        session,
        direction="credit",
        amount=D("50"),
        category="salary",
        transaction_date=datetime.date(2026, 6, 20),
    )
    no_balance = await ensure_account(session, 3, "bank_account")
    await _add(
        session,
        direction="credit",
        amount=D("500"),
        category="salary",
        account_id=no_balance,
    )

    bracketed = await ensure_account(session, 4, "bank_account")
    session.add(
        BankStatementUpload(
            account_id=bracketed,
            bank="hdfc",
            filename="half-year.pdf",
            file_path="/synthetic/half-year.pdf",
            opening_balance="0.00",
            closing_balance="200.00",
            statement_period_start="01/04/2026",
            statement_period_end="30/09/2026",
        )
    )
    # The alert times run in the reverse of the posting order.
    for time, balance in ((datetime.time(10), D("100")), (datetime.time(9), D("200"))):
        await _add(
            session,
            direction="credit",
            amount=D("100"),
            category="salary",
            account_id=bracketed,
            transaction_time=time,
            balance=balance,
        )

    around = await ensure_account(session, 5, "bank_account")
    for start, end in (("01/04/2026", "30/04/2026"), ("01/08/2026", "31/08/2026")):
        session.add(
            BankStatementUpload(
                account_id=around,
                bank="hdfc",
                filename="around.pdf",
                file_path="/synthetic/around.pdf",
                opening_balance="0.00",
                closing_balance="0.00",
                statement_period_start=start,
                statement_period_end=end,
            )
        )
    for as_of in (datetime.date(2026, 5, 31), datetime.date(2026, 6, 30)):
        session.add(
            BalanceSnapshot(
                account_id=around,
                kind="asset",
                category="bank_balance",
                as_of_date=as_of,
                value=D("500"),
                source="bank_statement",
            )
        )

    # The May statement prints June 1 rows: one it imported, one it matched to an
    # alert, and one it imported that a merge folded into a June row. Its closing
    # and the June opening both include them. It is the newest, so its closing
    # gives the opening. The June statement and an older statement that also ends
    # on May 31 must agree with it.
    late = await ensure_account(session, 6, "bank_account")
    older, june, may = (
        BankStatementUpload(
            account_id=late,
            bank="hdfc",
            filename="late.pdf",
            file_path="/synthetic/late.pdf",
            opening_balance=opening,
            closing_balance=closing,
            statement_period_start=start,
            statement_period_end=end,
        )
        for start, end, opening, closing in (
            ("01/04/2026", "31/05/2026", "965.00", "965.00"),
            ("01/06/2026", "30/06/2026", "1,000.00", "1,070.00"),
            ("01/05/2026", "31/05/2026", "1,000.00", "1,000.00"),
        )
    )
    for upload in (older, june, may):
        session.add(upload)
        await session.flush()
    rows = [
        await _add(
            session,
            direction="credit",
            amount=amount,
            category="interest",
            account_id=late,
            transaction_date=JUN,
            bank_statement_upload_id=upload_id,
        )
        for amount, upload_id in (
            (D("20"), may.id),
            (D("10"), None),
            (D("5"), june.id),
            (D("70"), None),
        )
    ]
    may.reconciliation_data = json.dumps(
        {
            "matched": [{"db_txn_id": rows[1].id}],
            "missing": [{"imported_txn_id": rows[2].id}],
        }
    )
    await session.flush()

    r = await client.get("/api/cashflow/bridge?date_from=2026-06-01&date_to=2026-06-30")
    assert r.status_code == 200
    body = r.json()
    known, unknown, statement, snapshot, spilled = body["accounts"]
    assert (known["opening"]["source"], D(str(known["opening"]["amount"]))) == (
        "estimated",
        D("1000"),
    )
    assert D(str(known["closing"]["amount"])) == D("980")
    assert unknown["opening"]["source"] == "unknown"
    assert unknown["gap"] is None
    assert D(str(statement["closing"]["amount"])) == D("200")
    assert D(str(statement["gap"])) == 0
    assert statement["closing"]["as_of"] == "2026-03-31"
    assert D(str(snapshot["opening"]["amount"])) == D("500")
    assert snapshot["opening"]["as_of"] == "2026-05-31"
    assert D(str(spilled["opening"]["amount"])) == D("965")
    assert D(str(spilled["gap"])) == 0
    assert not any("disagree" in w for w in body["warnings"])
    assert D(str(body["opening"])) == D("2465")
    assert D(str(body["actual_closing"])) == D("2750")
    assert D(str(body["gap"])) == 0
    assert any(unknown["label"] in w for w in body["warnings"])
