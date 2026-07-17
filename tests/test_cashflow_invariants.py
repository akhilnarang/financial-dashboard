"""Cross-cutting invariants of the cashflow report, over a rich seeded DB.

``test_cashflow_report.py`` tests one figure per case — the income total for
repayment exclusion, the expense total for a contra credit, and so on. This
file owns the *cross-cutting* properties that hold over a population rich
enough that a quiet double-count or a missing population would distort every
figure at once, and only an invariant over the whole population can catch:

* **No double count.** Every bank-side in-range row is in exactly one
  headline bucket (income / expense / investment / transfers_in) or the
  internal footnote, and never in two of them; a card / unaccounted row is in
  none of them.
* **Report totals equal direct source-row sums.** Each headline total is the
  signed sum of the same rows a direct query over the source tables selects,
  with the same scope, currency and date clauses. A bucket that quietly
  dropped a row, or counted one twice, is caught by the difference.
* **Drill-through identities.** The line count of every bucket and footnote
  equals the row count of the population its drill-through selects, so a tile
  and the list behind its link cannot drift apart.
* **Scope flip.** The one slug whose bucket depends on scope —
  credit_card_payment — is internal over every account and expense over the
  bank, so the same row is in the bank expense line and NOT in the
  all-accounts expense detail.

The cases the per-figure tests already own (NULL currency, undated rows, blank
counterparty) are seeded here too because the invariants above are only
meaningful over a population that includes them; the assertions here are over
the totals, not the per-figure shape.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.cashflow.buckets import bucket_for_slug
from financial_dashboard.services.cashflow.report import (
    BLANK_COUNTERPARTY,
    NON_INR,
    UNCATEGORIZED,
    cashflow_summary,
    cashflow_trend,
)
from financial_dashboard.services.cashflow.scope import (
    BANK_SCOPE,
    CARD_SCOPE,
    UNACCOUNTED_SCOPE,
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
IN_RANGE = Transaction.transaction_date.between(JUN, JUN_END)

# The un-aggregated signed-flow expression. ``report.SIGNED_FLOW`` is already
# wrapped in ``func.sum`` for the report's ``GROUP BY`` queries; the invariants
# below compose their own aggregate, so they need the inner expression.
SIGNED = case(
    (Transaction.direction == "credit", Transaction.amount),
    else_=-Transaction.amount,
)
# A NULL currency is an INR row whose default did not backfill, so the bucket
# clauses treat it as INR; the parity queries here must do the same.
INR_OR_NULL = Transaction.currency.is_(None) | (Transaction.currency == "INR")


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
    non-INR currencies, undated rows, out-of-range rows, and a row on a
    debit_card (the other bank-side account type).
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


# ---------------------------------------------------------------------------
# No double count: scope partition
# ---------------------------------------------------------------------------


async def test_bank_card_unaccounted_partition_is_exhaustive_and_disjoint(
    session: AsyncSession,
):
    """Every in-range row is in exactly one of {bank, card, unaccounted}.

    That partition is what every no-double-count invariant below stands on: a
    row a headline bucket counts must be bank-side, and a row in two scopes
    would be counted twice. Seeded over the rich population so an unknown
    account type, a NULL link and a dangling link are all in the test.
    """
    await _seed_rich_population(session)

    bank_ids = set(
        (await session.execute(select(Transaction.id).where(IN_RANGE, BANK_SCOPE)))
        .scalars()
        .all()
    )
    card_ids = set(
        (await session.execute(select(Transaction.id).where(IN_RANGE, CARD_SCOPE)))
        .scalars()
        .all()
    )
    unaccounted_ids = set(
        (
            await session.execute(
                select(Transaction.id).where(IN_RANGE, UNACCOUNTED_SCOPE)
            )
        )
        .scalars()
        .all()
    )
    all_ids = set(
        (await session.execute(select(Transaction.id).where(IN_RANGE))).scalars().all()
    )

    # Exhaustive: every in-range row is in some scope.
    assert bank_ids | card_ids | unaccounted_ids == all_ids
    # Disjoint: no row is in two scopes.
    assert bank_ids.isdisjoint(card_ids)
    assert bank_ids.isdisjoint(unaccounted_ids)
    assert card_ids.isdisjoint(unaccounted_ids)


async def test_no_bank_in_range_row_is_double_counted_across_buckets(
    session: AsyncSession,
):
    """Every in-range, INR-or-NULL, bank-side row is in exactly one of the four
    headline buckets OR the internal footnote — never two.

    A row's category resolves to one bucket under the bank scope, and the
    four headline buckets + internal are mutually exclusive: a row that landed
    in two of them is a double count by construction. Uncategorized rows are
    in a fifth bucket (also exclusive of the other four).
    """
    await _seed_rich_population(session)

    rows = (
        await session.execute(
            select(Transaction.id, Transaction.category).where(IN_RANGE, BANK_SCOPE)
        )
    ).all()

    seen: dict[int, str] = {}
    for txn_id, category in rows:
        bucket = bucket_for_slug(category, scope="bank")
        # The same row must not have been placed in two buckets.
        assert txn_id not in seen, (
            f"txn {txn_id} placed in both {seen[txn_id]} and {bucket}"
        )
        seen[txn_id] = bucket
    # Every bank-side in-range row resolved to exactly one bucket.
    assert len(seen) == len(rows)


# ---------------------------------------------------------------------------
# Report totals equal direct source-row sums
# ---------------------------------------------------------------------------


async def _bank_in_range_signed_sum(session: AsyncSession, *extra_where) -> Decimal:
    """Signed sum of in-range, bank-side, INR-or-NULL rows, optionally filtered.

    The same predicate shape the report uses (IN_RANGE + BANK_SCOPE +
    INR_OR_NULL + an extra category clause), so a difference between this and a
    report total is a real discrepancy, not a clause-shape mismatch.
    """
    row = (
        await session.execute(
            select(func.sum(SIGNED)).where(
                IN_RANGE, BANK_SCOPE, INR_OR_NULL, *extra_where
            )
        )
    ).scalar()
    return D(row or 0)


async def test_income_total_equals_direct_signed_sum_of_income_bucket_rows(
    session: AsyncSession,
):
    """The headline income total is the signed sum of the bank-side in-range
    rows whose slug the bucket map puts in income — no more, no less. A row the
    bucket dropped, or a row counted twice, would make the two disagree."""
    from financial_dashboard.services.cashflow.buckets import INCOME_BUCKET

    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct = await _bank_in_range_signed_sum(
        session, Transaction.category.in_(tuple(INCOME_BUCKET))
    )
    assert s.income.total == direct
    # And the line count agrees with the row count (INR-or-NULL, like the
    # bucket itself — a non-INR salary row reaches the salary_count trend
    # figure but not this rupee total).
    direct_count = (
        await session.execute(
            select(func.count()).where(
                IN_RANGE,
                BANK_SCOPE,
                INR_OR_NULL,
                Transaction.category.in_(tuple(INCOME_BUCKET)),
            )
        )
    ).scalar_one()
    assert s.income.count == direct_count


async def test_expense_total_equals_direct_signed_sum_of_expense_bucket_rows(
    session: AsyncSession,
):
    """Expense is the signed sum NEGATED (so spend reads positive and a contra
    credit reads negative). The contra credits (refund, cashback_rewards,
    fees_charges credit) are in the bucket, so a fee-reversal credit is
    netted into this figure too. Under the bank scope, a ``credit_card_payment``
    debit is the bill — the moment cash leaves — and is in the expense bucket
    here too, even though over every account it is internal."""
    from financial_dashboard.services.cashflow.buckets import BUCKET_BY_SLUG

    # The set of slugs that resolve to "expense" under the bank scope. This is
    # the scope-flipped view: credit_card_payment is in here (the bill is the
    # bank-side expense), even though BUCKET_BY_SLUG alone says it is internal.
    expense_slugs_bank = tuple(
        slug
        for slug in BUCKET_BY_SLUG
        if bucket_for_slug(slug, scope="bank") == "expense"
    )

    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct = await _bank_in_range_signed_sum(
        session, Transaction.category.in_(expense_slugs_bank)
    )
    assert s.expense.total == -direct
    direct_count = (
        await session.execute(
            select(func.count()).where(
                IN_RANGE,
                BANK_SCOPE,
                INR_OR_NULL,
                Transaction.category.in_(expense_slugs_bank),
            )
        )
    ).scalar_one()
    assert s.expense.count == direct_count


async def test_expense_total_includes_the_fee_reversal_contra_credit(
    session: AsyncSession,
):
    """The fee-reversal credit is in the expense bucket, so the expense total
    is the GROSS spend minus the contra — exactly what netting fees_charges
    means. Pinned with the exact numbers so a regression that re-homed
    fees_charges out of the bucket would change both totals."""
    await _add(session, direction="debit", amount=D("80"), category="fees_charges")
    await _add(session, direction="credit", amount=D("30"), category="fees_charges")
    s = await cashflow_summary(session, JUN, JUN_END)
    # 80 - 30 = 50, on the fees_charges line of the expense bucket.
    fees_line = next(ln for ln in s.expense.lines if ln.slug == "fees_charges")
    assert fees_line.total == D("50")
    assert fees_line.count == 2
    assert s.expense.total == D("50")


async def test_transfers_in_total_equals_direct_signed_sum_of_repayment_rows(
    session: AsyncSession,
):
    """Transfers-in (repayment) is its own bucket so it never inflates income;
    the headline total is the signed sum of its rows."""
    from financial_dashboard.services.cashflow.buckets import TRANSFERS_IN_SLUG

    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct = await _bank_in_range_signed_sum(
        session, Transaction.category == TRANSFERS_IN_SLUG
    )
    assert s.transfers_in.total == direct
    direct_count = (
        await session.execute(
            select(func.count()).where(
                IN_RANGE,
                BANK_SCOPE,
                INR_OR_NULL,
                Transaction.category == TRANSFERS_IN_SLUG,
            )
        )
    ).scalar_one()
    assert s.transfers_in.count == direct_count


async def test_investment_net_equals_direct_signed_sum_of_investment_rows(
    session: AsyncSession,
):
    """Investment is split by direction in the bucket but net=contributions -
    redemptions, which is the signed sum of the same rows NEGATED."""
    from financial_dashboard.services.cashflow.buckets import INVESTMENT_BUCKET

    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct = await _bank_in_range_signed_sum(
        session, Transaction.category.in_(tuple(INVESTMENT_BUCKET))
    )
    # net = contributions - redemptions = -(signed sum) because a contribution
    # is a debit (negative signed flow) and a redemption is a credit.
    assert s.investment.net == -direct
    direct_count = (
        await session.execute(
            select(func.count()).where(
                IN_RANGE,
                BANK_SCOPE,
                INR_OR_NULL,
                Transaction.category.in_(tuple(INVESTMENT_BUCKET)),
            )
        )
    ).scalar_one()
    assert s.investment.count == direct_count


async def test_uncategorized_total_equals_direct_signed_sum_of_uncategorized_rows(
    session: AsyncSession,
):
    """Uncategorized is currency-agnostic and bank-scoped. Its total is the
    signed sum of the same rows, with no INR_OR_NULL clause — a non-INR
    uncategorized row is included here too."""
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct_signed = D(
        (
            await session.execute(
                select(func.sum(SIGNED)).where(IN_RANGE, BANK_SCOPE, UNCATEGORIZED)
            )
        ).scalar()
        or 0
    )
    assert s.uncategorized.total == direct_signed
    direct_count = (
        await session.execute(
            select(func.count()).where(IN_RANGE, BANK_SCOPE, UNCATEGORIZED)
        )
    ).scalar_one()
    assert s.uncategorized.count == direct_count


async def test_internal_footnote_count_matches_self_transfer_rows(
    session: AsyncSession,
):
    """Under the bank scope, only self_transfer is internal — a card bill is
    expense there. The footnote's count and gross are the direct sum of those
    rows, with no currency clause (the footnotes are rough figures)."""
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct_count, direct_gross = (
        await session.execute(
            select(func.count(), func.sum(Transaction.amount)).where(
                IN_RANGE, BANK_SCOPE, Transaction.category == "self_transfer"
            )
        )
    ).one()
    assert s.footnotes.internal_count == direct_count
    assert s.footnotes.internal_gross == D(direct_gross or 0)


async def test_non_inr_footnote_count_matches_the_non_inr_bank_rows(
    session: AsyncSession,
):
    """The non-INR footnote counts the bank-side in-range rows the buckets
    dropped — the rows that would otherwise be a silent gap between the bank
    total and the sum of the buckets."""
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct = (
        await session.execute(select(func.count()).where(IN_RANGE, BANK_SCOPE, NON_INR))
    ).scalar_one()
    assert s.footnotes.non_inr_count == direct


async def test_undated_footnote_matches_every_undated_row_regardless_of_scope(
    session: AsyncSession,
):
    """Undated rows fall into no range and no scope on the report; the footnote
    is range- and scope-independent, and counts every undated row."""
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct_count, direct_net = (
        await session.execute(
            select(func.count(), func.sum(SIGNED)).where(
                Transaction.transaction_date.is_(None)
            )
        )
    ).one()
    assert s.footnotes.undated_count == direct_count
    assert s.footnotes.undated_net == D(direct_net or 0)


async def test_unaccounted_footnote_matches_the_unaccounted_rows(session: AsyncSession):
    """Unaccounted is the complement of {bank, card} over the in-range rows."""
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    direct_count, direct_net = (
        await session.execute(
            select(func.count(), func.sum(SIGNED)).where(IN_RANGE, UNACCOUNTED_SCOPE)
        )
    ).one()
    assert s.footnotes.unaccounted_count == direct_count
    assert s.footnotes.unaccounted_net == D(direct_net or 0)


# ---------------------------------------------------------------------------
# Scope flip: credit_card_payment over the bank vs every account
# ---------------------------------------------------------------------------


async def test_credit_card_payment_bank_expense_vs_internal_in_the_detail(
    session: AsyncSession,
):
    """The one slug whose bucket depends on scope: a credit_card_payment debit
    on the bank lands in the bank expense total, and the same row is INTERNAL
    in the all-accounts expense detail — counting it there too would charge
    the same rupee twice. The detail's exclusion of it IS the no-double-count
    invariant for card bills."""
    await _add(
        session, direction="debit", amount=D("500"), category="credit_card_payment"
    )
    await _add(session, direction="debit", amount=D("100"), category="dining")
    s = await cashflow_summary(session, JUN, JUN_END)

    # Bank expense: the bill is in.
    assert s.expense.total == D("600")
    assert {ln.slug for ln in s.expense.lines} == {"credit_card_payment", "dining"}
    # All-accounts detail: the bill is out — only the swipe-side dining.
    assert s.expense_detail.total == D("100")
    assert [ln.slug for ln in s.expense_detail.lines] == ["dining"]


# ---------------------------------------------------------------------------
# Drill-through identities: tile counts equal their drill-through row counts
# ---------------------------------------------------------------------------


async def test_every_bucket_count_matches_its_drill_through_row_count(
    session: AsyncSession,
):
    """A bucket's ``count`` IS the row count its drill-through link
    (?category=... or the scope-appropriate filter) lists, so a tile and the
    list behind its link cannot drift apart. Pinned over the rich population
    so every bucket has rows."""
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    for ln in s.income.lines:
        n = (
            await session.execute(
                select(func.count()).where(
                    IN_RANGE,
                    BANK_SCOPE,
                    INR_OR_NULL,
                    Transaction.category == ln.slug,
                )
            )
        ).scalar_one()
        assert ln.count == n, f"income line {ln.slug}"

    for ln in s.expense.lines:
        n = (
            await session.execute(
                select(func.count()).where(
                    IN_RANGE,
                    BANK_SCOPE,
                    INR_OR_NULL,
                    Transaction.category == ln.slug,
                )
            )
        ).scalar_one()
        assert ln.count == n, f"expense line {ln.slug}"

    # The investment bucket splits a slug by direction, so the line count is
    # the row count over (slug, direction).
    for ln in s.investment.lines:
        direction = "debit" if ln.kind == "contribution" else "credit"
        n = (
            await session.execute(
                select(func.count()).where(
                    IN_RANGE,
                    BANK_SCOPE,
                    INR_OR_NULL,
                    Transaction.category == ln.slug,
                    Transaction.direction == direction,
                )
            )
        ).scalar_one()
        assert ln.count == n, f"investment line {ln.slug}/{ln.kind}"


async def test_transfers_in_drill_through_lists_each_counterparty_once(
    session: AsyncSession,
):
    """The transfers-in lines are per-counterparty (NULL collapsed from blank/
    whitespace), and each line's count is the row count for that counterparty —
    the link lists exactly the rows the tile counted."""
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    for ln in s.transfers_in.lines:
        if ln.counterparty is None:
            clause = BLANK_COUNTERPARTY
        else:
            clause = Transaction.counterparty == ln.counterparty
        n = (
            await session.execute(
                select(func.count()).where(
                    IN_RANGE,
                    BANK_SCOPE,
                    INR_OR_NULL,
                    Transaction.category == "repayment",
                    clause,
                )
            )
        ).scalar_one()
        assert ln.count == n, f"transfers_in line {ln.counterparty!r}"


# ---------------------------------------------------------------------------
# net_cash_retained invariant
# ---------------------------------------------------------------------------


async def test_net_cash_retained_is_the_sum_of_its_four_terms(session: AsyncSession):
    """net_cash_retained = income + transfers_in - expense - investment.net,
    over the bank. The four terms are each pinned to a direct source-row sum
    above; this asserts they compose to the headline figure exactly, so a
    bucket that quietly changed sign or dropped a row would change either its
    own total or this composition."""
    await _seed_rich_population(session)
    s = await cashflow_summary(session, JUN, JUN_END)

    assert s.net_cash_retained == (
        s.income.total + s.transfers_in.total - s.expense.total - s.investment.net
    )


# ---------------------------------------------------------------------------
# Trend identities: salary count + each month reconciles with its summary
# ---------------------------------------------------------------------------


async def test_trend_salary_count_matches_direct_count_per_month(session: AsyncSession):
    """salary_count per month is the row count of the bank-side salary rows in
    that month, with no currency filter. Pinned so a future change that added
    a currency filter to the count (to match the monetary series) would change
    the USD-salary case the per-figure test already covers."""
    await _seed_rich_population(session)
    today = datetime.date(2026, 6, 15)
    pts = await cashflow_trend(session, months=1, today=today)
    jun = next(p for p in pts if p.month == "2026-06")

    direct = (
        await session.execute(
            select(func.count()).where(
                Transaction.transaction_date.between(datetime.date(2026, 6, 1), today),
                BANK_SCOPE,
                Transaction.category == "salary",
            )
        )
    ).scalar_one()
    assert jun.salary_count == direct


async def test_each_trend_month_reconciles_with_its_own_summary(session: AsyncSession):
    """A trend month's income / expense / net_invested are the same figures
    the summary would produce for that month's range — same buckets, same
    scope, same exclusions. A change to one and not the other would change
    this difference to non-zero."""
    await _seed_rich_population(session)
    today = datetime.date(2026, 6, 15)
    pts = await cashflow_trend(session, months=1, today=today)
    jun_trend = next(p for p in pts if p.month == "2026-06")

    jun_summary = await cashflow_summary(session, datetime.date(2026, 6, 1), today)
    assert jun_trend.income == jun_summary.income.total
    assert jun_trend.expense == jun_summary.expense.total
    assert jun_trend.net_invested == jun_summary.investment.net
