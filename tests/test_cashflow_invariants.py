"""Cross-cutting invariants of the cashflow report, over a rich seeded DB.

``test_cashflow_report.py`` tests one figure per case. This file checks that
every total, footnote and line count equals a hand-computed sum over the seed.
A bucket that drops a row or counts one twice breaks the equality.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.cashflow.report import (
    cashflow_summary,
    cashflow_trend,
)
from tests.conftest import (
    MISSING_ACCOUNT_ID,
    bank_account,
    card_account,
    ensure_account,
)

pytestmark = pytest.mark.anyio
D = Decimal

JUN = datetime.date(2026, 6, 1)
JUL = datetime.date(2026, 7, 1)
JUN_END = datetime.date(2026, 6, 30)


async def _add(session: AsyncSession, **kw) -> Transaction:
    """Seed one transaction, linked to the bank account unless it says else.

    The default link keeps a row in the bank scope, where the headline figures
    live. Tests about card rows, unaccounted rows or rows on no account at all
    pass ``account_id=`` and say so — that is the population the invariants
    care about, so the helper makes the choice explicit per row.
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


async def _seed_rich_population(session: AsyncSession) -> None:
    """Seed one row of every kind the invariants are over.

    Bank income, transfers-in, expense with a contra credit (refund, cashback,
    fee reversal), investment contributions and redemptions, internal
    self-transfers, card swipes, card credits, unaccounted rows, NULL/blank/
    whitespace/unknown/unmapped categories, blank counterparties, INR/NULL/
    non-INR currencies, undated rows, an excluded row, out-of-range rows, and a
    row on a debit_card (the other bank-side account type).
    """
    card = await card_account(session)
    debit_card = await ensure_account(session, 7, "debit_card")
    weird = await ensure_account(session, 42, "prepaid_wallet")

    # --- bank income ---
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(session, direction="credit", amount=D("50"), category="interest")
    await _add(session, direction="credit", amount=D("30"), category="other_income")

    # --- transfers-in (repayment, its own line) ---
    await _add(
        session,
        direction="credit",
        amount=D("200"),
        category="repayment",
        counterparty="MOM",
    )
    await _add(
        session,
        direction="credit",
        amount=D("100"),
        category="repayment",
        counterparty="   ",  # whitespace-only counterparty
    )

    # --- bank expense with contra-credits ---
    await _add(session, direction="debit", amount=D("300"), category="groceries")
    await _add(session, direction="debit", amount=D("120"), category="dining")
    await _add(session, direction="debit", amount=D("80"), category="fees_charges")
    # Three contra-expense credits net against spend.
    await _add(session, direction="credit", amount=D("50"), category="refund")
    await _add(session, direction="credit", amount=D("20"), category="cashback_rewards")
    # A credit fees_charges is the fee-reversal path; nets against fees_charges.
    await _add(session, direction="credit", amount=D("30"), category="fees_charges")

    # --- investment contributions + redemptions ---
    await _add(session, direction="debit", amount=D("100"), category="investment")
    await _add(
        session, direction="credit", amount=D("40"), category="investment_redemption"
    )
    await _add(
        session, direction="credit", amount=D("15"), category="investment"
    )  # a credit 'investment' is also a redemption in the bucket map's eyes

    # --- internal (bank-side self_transfer) ---
    await _add(session, direction="debit", amount=D("700"), category="self_transfer")
    await _add(session, direction="credit", amount=D("200"), category="self_transfer")

    # --- credit_card_payment: bank leg (debit) and card leg (credit) ---
    await _add(
        session, direction="debit", amount=D("500"), category="credit_card_payment"
    )
    await _add(
        session,
        direction="credit",
        amount=D("500"),
        category="credit_card_payment",
        account_id=card,
    )

    # --- card swipes + card credits: out of every bank headline figure ---
    await _add(
        session,
        direction="debit",
        amount=D("250"),
        category="dining",
        account_id=card,
    )
    await _add(
        session,
        direction="credit",
        amount=D("8000"),
        category="salary",
        account_id=card,
    )

    # --- debit-card row (still bank-scoped) ---
    await _add(
        session,
        direction="debit",
        amount=D("60"),
        category="transport",
        account_id=debit_card,
    )

    # --- uncategorized, on the bank: NULL / blank / whitespace / unknown / unmapped ---
    await _add(session, direction="debit", amount=D("11"), category=None)
    await _add(session, direction="debit", amount=D("22"), category="")
    await _add(session, direction="debit", amount=D("33"), category="   ")
    await _add(session, direction="debit", amount=D("44"), category="unknown")
    await _add(session, direction="debit", amount=D("55"), category="crypto")

    # --- blank counterparty in a bucketed category (not just transfers-in) ---
    await _add(
        session,
        direction="debit",
        amount=D("70"),
        category="utilities",
        counterparty="",
    )

    # --- non-INR rows: out of every headline bucket, surfaced in footnotes ---
    await _add(
        session, direction="debit", amount=D("5"), category="dining", currency="USD"
    )
    await _add(
        session,
        direction="debit",
        amount=D("7"),
        category=None,  # uncategorized AND non-INR — both lines list it
        currency="USD",
    )

    # --- undated rows: in no range, surfaced only in the Undated footnote ---
    await _add(
        session,
        direction="debit",
        amount=D("40"),
        category="dining",
        transaction_date=None,
        created_at=datetime.datetime(2026, 6, 10, tzinfo=datetime.UTC),
    )

    # --- unaccounted rows: in no scope ---
    await _add(
        session,
        direction="debit",
        amount=D("111"),
        category="dining",
        account_id=None,
    )
    await _add(
        session,
        direction="credit",
        amount=D("222"),
        category="salary",
        account_id=MISSING_ACCOUNT_ID,
    )
    await _add(
        session,
        direction="debit",
        amount=D("30"),
        category="rent",
        account_id=weird,
    )

    # --- excluded row: in no figure but the Excluded footnote ---
    await _add(
        session,
        direction="debit",
        amount=D("999"),
        category="groceries",
        exclude_from_cashflow=True,
    )

    # --- out-of-range rows: in scope and bucketable, but not in this range ---
    await _add(
        session,
        direction="credit",
        amount=D("9999"),
        category="salary",
        transaction_date=JUL,
    )
    await _add(
        session,
        direction="debit",
        amount=D("8888"),
        category="dining",
        transaction_date=JUL,
    )


async def test_every_figure_equals_a_direct_sum_over_its_rows(session: AsyncSession):
    """Each total, footnote, line count and trend month equals a sum over the
    seed, computed by hand. A bucket that drops a row or counts one twice
    breaks the equality.

    Under the bank scope a credit_card_payment debit is expense.
    """
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    # Salary 1000 + interest 50 + other_income 30.
    assert (s.income.total, s.income.count) == (D("1080"), 3)
    assert {ln.slug: (ln.total, ln.count) for ln in s.income.lines} == {
        "salary": (D("1000"), 1),
        "interest": (D("50"), 1),
        "other_income": (D("30"), 1),
    }

    # 300 + 120 + (80 - 30) - 50 - 20 + 500 + 60 + 70.
    assert (s.expense.total, s.expense.count) == (D("1030"), 9)
    assert {ln.slug: (ln.total, ln.count) for ln in s.expense.lines} == {
        "credit_card_payment": (D("500"), 1),
        "groceries": (D("300"), 1),
        "dining": (D("120"), 1),
        "utilities": (D("70"), 1),
        "transport": (D("60"), 1),
        "fees_charges": (D("50"), 2),
        "refund": (D("-50"), 1),
        "cashback_rewards": (D("-20"), 1),
    }

    # Blank and whitespace counterparties collapse into one line.
    assert (s.transfers_in.total, s.transfers_in.count) == (D("300"), 2)
    assert {ln.counterparty: ln.count for ln in s.transfers_in.lines} == {
        "MOM": 1,
        None: 1,
    }

    # Contribution 100 less redemptions 40 and 15.
    assert (s.investment.net, s.investment.count) == (D("45"), 3)
    assert {(ln.slug, ln.kind): ln.count for ln in s.investment.lines} == {
        ("investment", "contribution"): 1,
        ("investment_redemption", "redemption"): 1,
        ("investment", "redemption"): 1,
    }

    # Uncategorized applies no currency clause, so the USD 7 is in it.
    assert (s.uncategorized.total, s.uncategorized.count) == (D("-172"), 6)

    assert s.net_cash_retained == D("305")

    f = s.footnotes
    assert (f.internal_count, f.internal_gross, f.internal_net) == (
        2,
        D("900"),
        D("-500"),
    )
    assert f.non_inr_count == 2
    assert (f.undated_count, f.undated_net) == (1, D("-40"))
    # -111 + 222 - 30.
    assert (f.unaccounted_count, f.unaccounted_net) == (3, D("81"))
    assert (f.excluded_count, f.excluded_gross, f.excluded_net) == (
        1,
        D("999"),
        D("-999"),
    )

    today = datetime.date(2026, 6, 15)
    pts = await cashflow_trend(session, months=1, today=today)
    jun_trend = next(p for p in pts if p.month == "2026-06")

    jun_summary = await cashflow_summary(session, JUN, today)
    assert jun_trend.income == jun_summary.income.total
    assert jun_trend.expense == jun_summary.expense.total
    assert jun_trend.net_invested == jun_summary.investment.net
    assert jun_trend.salary_count == 1
