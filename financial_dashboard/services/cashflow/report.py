"""Read-time aggregation for the cashflow report.

Two pure readers over ``transactions`` — a summary for a date range and a
month-by-month trend — plus the date-range parser both cashflow routes share.
Nothing here writes; the numbers are aggregated in SQL and returned as the
pydantic response models, so the JSON API and the HTML page render the same
figures from the same source.

The report is a **cash-basis** one: every headline figure, the trend and the
footnotes are scoped to the **bank** (``scope.BANK_SCOPE``). Money is spent when
it leaves the bank, which for a card is the day the *bill* is paid, not the day
of the swipe — so ``credit_card_payment`` is an expense here, and the swipes
themselves are out of scope entirely. That answers "did I spend more than I
earned".

The one figure that is *not* bank-scoped is the expense **detail**, which
aggregates every account and answers a different question — "what did I actually
buy" — by counting the swipes. There, a card bill is internal churn again, or it
would charge the same rupee twice. The two are therefore never added together,
and they do not reconcile: over any range their difference is exactly the timing
gap between what was bought and what has been paid for.

Nothing is dropped for being out of scope. A row on no account, or on an account
whose type the code does not know, is *unaccounted* and is counted in a footnote
of its own, outside the reconciliation arithmetic.

The one rule everything else falls out of: a row's signed flow is ``+amount``
for a credit and ``-amount`` for a debit. Contra behaviour needs no special
case — a refund or cashback credit lands in the expense bucket with a positive
signed flow, which *reduces* spend once the bucket is displayed; an investment
redemption credit likewise nets against contributions; and a reversed card bill
nets off the bills it was reversed from, with no machinery of its own.

Date basis is ``transaction_date`` alone (never a ``created_at`` fallback): it
is the column ``/transactions`` filters on, so every figure here covers exactly
the rows its drill-through link lists. Rows with no ``transaction_date`` fall
into no range and no month, and are surfaced only by the Undated footnote.
"""

import datetime
from decimal import Decimal
from typing import NamedTuple

from sqlalchemy import ColumnElement, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import (
    DEFAULT_TREND_MONTHS,
    month_end,
    month_key as month_key_of,
    trailing_month_starts,
)
from financial_dashboard.db.models import Transaction
from financial_dashboard.schemas.cashflow import (
    BucketSummary,
    CashflowSummary,
    CategoryLine,
    CounterpartyLine,
    Footnotes,
    InvestmentSummary,
    TransfersInSummary,
    TrendPoint,
)
from financial_dashboard.services.cashflow.buckets import (
    BUCKET_BY_SLUG,
    TRANSFERS_IN_SLUG,
    bucket_for_slug,
    internal_slugs_for_scope,
    label_for_slug,
)
from financial_dashboard.services.cashflow.scope import (
    BANK_SCOPE,
    UNACCOUNTED_SCOPE,
    Scope,
)
from financial_dashboard.services.categorization.slugs import UNKNOWN_SLUG

MAX_TREND_MONTHS = 60
NO_COUNTERPARTY_LABEL = "(no counterparty)"

#: The scope every headline figure, footnote and trend point is drawn over. Named
#: once so the report, its drill-through links and the bucket map cannot come to
#: disagree about which rows a figure counted.
REPORT_SCOPE: Scope = "bank"
BANK_INTERNAL_SLUGS = tuple(internal_slugs_for_scope(REPORT_SCOPE))

# A NULL currency is an INR row that predates the column's default, not an
# unknown one: bucket it as INR rather than dropping it.
INR_OR_NULL = Transaction.currency.is_(None) | (Transaction.currency == "INR")
NON_INR = Transaction.currency.is_not(None) & (Transaction.currency != "INR")

SIGNED_FLOW = func.sum(
    case(
        (Transaction.direction == "credit", Transaction.amount),
        else_=-Transaction.amount,
    )
)
ROW_COUNT = func.count()

# NULL, empty and whitespace-only are the same absence of a value, and one line of
# the report must never list rows a different line's link would claim. So every
# blank test — SQL and Python, counterparty and category — is written over one
# character set: bare SQL `trim()` strips spaces only, while Python's bare
# `.strip()` strips every whitespace character, and that difference alone was
# enough to give a tab-only counterparty its own report line whose link listed the
# *blank* rows instead of the row it counted.
BLANK_CHARS = " \t\n\r\x0b\x0c"

# The SQL half: the population a blank `?counterparty=` selects.
BLANK_COUNTERPARTY = Transaction.counterparty.is_(None) | (
    func.trim(Transaction.counterparty, BLANK_CHARS) == ""
)


def is_blank_counterparty(value: str | None) -> bool:
    """The Python half: True when a query-string counterparty names nobody."""
    return value is None or value.strip(BLANK_CHARS) == ""


# Grouping key for the transfers-in lines: every blank spelling collapses onto
# the one None group the blank drill-through lists.
NORMALIZED_COUNTERPARTY = case(
    (BLANK_COUNTERPARTY, None),
    else_=Transaction.counterparty,
)

# A category stored as "" (or as whitespace) is not a slug — it is the same "no
# category" a NULL is, so it must be one population on both sides of the drill:
# the aggregation collapses it onto the NULL line, and `?category_null=1` selects
# exactly that same set. Kept apart, an empty-string row forms its own report line
# whose link — a NULL-only filter — lists every row *except* the one it counted.
BLANK_CATEGORY = Transaction.category.is_(None) | (
    func.trim(Transaction.category, BLANK_CHARS) == ""
)

# Grouping key for the uncategorized lines, so blank and NULL are the one line
# that `?category_null=1` lists.
NORMALIZED_CATEGORY = case(
    (BLANK_CATEGORY, None),
    else_=Transaction.category,
)

# The uncategorized population: no category at all, the 'unknown' sentinel, or a
# runtime slug the bucket map has never heard of. Defined once and used by both
# the report's uncategorized line and the `?uncategorized=1` drill-through, so the
# count on the tile and the rows on the list cannot drift apart.
UNCATEGORIZED = (
    BLANK_CATEGORY
    | (Transaction.category == UNKNOWN_SLUG)
    | Transaction.category.not_in(list(BUCKET_BY_SLUG))
)


class CashflowRange(NamedTuple):
    """The inclusive bounds a cashflow figure covers. Both ends are counted."""

    date_from: datetime.date
    date_to: datetime.date


def resolve_range(
    date_from: str | None,
    date_to: str | None,
    *,
    today: datetime.date | None = None,
) -> CashflowRange:
    """Normalize the two query-string bounds into a concrete inclusive range.

    Each bound is parsed independently, so a typo in one never silently moves
    the other: a missing or unparseable ``date_from`` defaults to the first of
    ``today``'s month, a missing or unparseable ``date_to`` defaults to
    ``today``.
    """
    today = today or datetime.date.today()
    try:
        start = datetime.date.fromisoformat(date_from or "")
    except ValueError:
        start = today.replace(day=1)
    try:
        end = datetime.date.fromisoformat(date_to or "")
    except ValueError:
        end = today
    return CashflowRange(start, end)


def _decimal(value: Decimal | None) -> Decimal:
    # An empty group aggregates to NULL, which reads as zero flow.
    return Decimal(0) if value is None else value


def _by_magnitude(line: CategoryLine | CounterpartyLine) -> tuple[Decimal, str]:
    # Biggest contribution first, label breaking ties so the order is stable.
    return (-abs(line.total), line.label)


async def cashflow_summary(
    session: AsyncSession,
    date_from: datetime.date,
    date_to: datetime.date,
) -> CashflowSummary:
    """Aggregate one inclusive ``transaction_date`` range into report buckets.

    Every figure on the returned summary is bank-scoped except ``expense_detail``,
    which spans all accounts, and the unaccounted/undated footnotes, which exist
    precisely to count what the bank scope leaves out.
    """
    in_range = Transaction.transaction_date.between(date_from, date_to)

    # The headline buckets share one scan of the in-range, bank-side INR-or-null
    # rows. Grouping by direction as well as category is what lets the investment
    # bucket split: `investment` is direction-neutral and can be both a
    # contribution and a redemption in the same range, which category alone
    # cannot tell apart.
    grouped = (
        await session.execute(
            select(
                Transaction.category,
                Transaction.direction,
                SIGNED_FLOW,
                ROW_COUNT,
            )
            .where(in_range, INR_OR_NULL, BANK_SCOPE)
            .group_by(Transaction.category, Transaction.direction)
        )
    ).all()

    income: dict[str | None, CategoryLine] = {}
    expense: dict[str | None, CategoryLine] = {}
    investment: list[CategoryLine] = []

    for slug, direction, flow, count in grouped:
        signed = _decimal(flow)
        bucket = bucket_for_slug(slug, scope=REPORT_SCOPE)
        if bucket == "income":
            _merge(income, slug, signed, count)
        elif bucket == "expense":
            # Negated so spend reads positive and a contra credit reads negative.
            _merge(expense, slug, -signed, count)
        elif bucket == "investment":
            contribution = direction == "debit"
            investment.append(
                CategoryLine(
                    slug=slug,
                    label=label_for_slug(slug),
                    total=-signed,
                    count=count,
                    kind="contribution" if contribution else "redemption",
                )
            )
        # internal, transfers_in and uncategorized rows are deliberately not
        # merged here: internal movements are excluded from every bucket sum
        # (and only counted in the footnotes), while the other two are
        # aggregated below on their own keys.

    income_lines = sorted(income.values(), key=_by_magnitude)
    expense_lines = sorted(expense.values(), key=_by_magnitude)
    investment_lines = sorted(investment, key=_by_magnitude)

    contributions = sum(
        (ln.total for ln in investment_lines if ln.kind == "contribution"),
        Decimal(0),
    )
    redemptions = -sum(
        (ln.total for ln in investment_lines if ln.kind == "redemption"),
        Decimal(0),
    )
    investment_summary = InvestmentSummary(
        contributions=contributions,
        redemptions=redemptions,
        net=contributions - redemptions,
        count=sum(ln.count for ln in investment_lines),
        lines=investment_lines,
    )

    transfers_in = await _transfers_in(session, in_range)
    uncategorized = await _uncategorized(session, in_range)
    expense_detail = await _expense_detail(session, in_range)
    footnotes = await _footnotes(session, in_range)

    income_summary = _bucket(income_lines)
    expense_summary = _bucket(expense_lines)
    return CashflowSummary(
        date_from=date_from,
        date_to=date_to,
        income=income_summary,
        expense=expense_summary,
        expense_detail=expense_detail,
        investment=investment_summary,
        transfers_in=transfers_in,
        uncategorized=uncategorized,
        net_cash_retained=(
            income_summary.total
            + transfers_in.total
            - expense_summary.total
            - investment_summary.net
        ),
        footnotes=footnotes,
    )


def _merge(
    lines: dict[str | None, CategoryLine], slug: str | None, total: Decimal, count: int
) -> None:
    # One group per direction arrives for the same slug; a bucket that does not
    # split by direction folds them into a single line.
    existing = lines.get(slug)
    if existing is None:
        lines[slug] = CategoryLine(
            slug=slug, label=label_for_slug(slug), total=total, count=count
        )
        return
    existing.total += total
    existing.count += count


def _bucket(lines: list[CategoryLine]) -> BucketSummary:
    return BucketSummary(
        total=sum((ln.total for ln in lines), Decimal(0)),
        count=sum(ln.count for ln in lines),
        lines=lines,
    )


async def _transfers_in(
    session: AsyncSession, in_range: ColumnElement[bool]
) -> TransfersInSummary:
    """Money handed back by a person — its own line, so it never inflates income."""
    rows = (
        await session.execute(
            select(NORMALIZED_COUNTERPARTY, SIGNED_FLOW, ROW_COUNT)
            .where(
                in_range,
                INR_OR_NULL,
                BANK_SCOPE,
                Transaction.category == TRANSFERS_IN_SLUG,
            )
            .group_by(NORMALIZED_COUNTERPARTY)
        )
    ).all()
    lines = sorted(
        (
            CounterpartyLine(
                counterparty=counterparty,
                label=counterparty or NO_COUNTERPARTY_LABEL,
                total=_decimal(flow),
                count=count,
            )
            for counterparty, flow, count in rows
        ),
        key=_by_magnitude,
    )
    return TransfersInSummary(
        total=sum((ln.total for ln in lines), Decimal(0)),
        count=sum(ln.count for ln in lines),
        lines=lines,
    )


async def _uncategorized(
    session: AsyncSession, in_range: ColumnElement[bool]
) -> BucketSummary:
    """Rows no bucket can place: NULL, the 'unknown' sentinel, or an unmapped slug.

    Bank-scoped, like the buckets it is the error bar on: only a bank-side
    uncategorized row can distort a bank-basis identity. A card-side one is out
    of the view entirely and would inflate the error bar with rows no figure here
    ever counted.

    No currency clause — a non-INR row with no category is still uncategorized,
    and the drill-through link (``/transactions?uncategorized=1``) has no
    currency clause either, so the tile's count and the list's row count agree.
    A non-INR row can therefore appear here and in the non-INR footnote; both
    are informational and neither feeds a headline bucket, so nothing is
    double-counted.

    Grouping is on the *normalized* category, so a row whose category is an empty
    string joins the NULL line rather than forming a line of its own — the two are
    the same absence of a category, and one drill-through link lists them both.
    """
    rows = (
        await session.execute(
            select(NORMALIZED_CATEGORY, SIGNED_FLOW, ROW_COUNT)
            .where(in_range, BANK_SCOPE, UNCATEGORIZED)
            .group_by(NORMALIZED_CATEGORY)
        )
    ).all()
    lines = sorted(
        (
            CategoryLine(
                slug=slug,
                label=label_for_slug(slug),
                total=_decimal(flow),
                count=count,
            )
            for slug, flow, count in rows
        ),
        key=_by_magnitude,
    )
    return _bucket(lines)


async def _expense_detail(
    session: AsyncSession, in_range: ColumnElement[bool]
) -> BucketSummary:
    """What was actually bought, over every account — the swipes, by category.

    Deliberately *not* bank-scoped: a card swipe is where a purchase is legible,
    and it never touches the bank. Deliberately *not* the headline either: the
    headline is what left the bank, and a swipe has not left it until the bill is
    paid. The difference between the two is the timing gap, not an error, so the
    two figures must never be added together or expected to reconcile.

    A ``credit_card_payment`` is internal here — over every account it settles
    swipes this very aggregation already counted, and counting both would charge
    the same rupee twice. Only the scope-free bucket map can say that, so this is
    the one caller that asks it without a scope.

    INR-or-null, like every other rupee figure, so the drill-through link that
    lists these rows can carry ``non_inr=0`` and match the total on the page.
    Direction does not split an expense line, so one group per category is
    enough; the signed sum is negated, which is what makes a refund read as the
    contra it is.
    """
    rows = (
        await session.execute(
            select(Transaction.category, SIGNED_FLOW, ROW_COUNT)
            .where(in_range, INR_OR_NULL)
            .group_by(Transaction.category)
        )
    ).all()
    lines: dict[str | None, CategoryLine] = {}
    for slug, flow, count in rows:
        if bucket_for_slug(slug) == "expense":
            _merge(lines, slug, -_decimal(flow), count)
    return _bucket(sorted(lines.values(), key=_by_magnitude))


async def _footnotes(session: AsyncSession, in_range: ColumnElement[bool]) -> Footnotes:
    """The excluded-from-the-headline populations, counted so they are visible.

    The internal footnote is bank-scoped and, under that scope, counts
    ``self_transfer`` alone: a card bill is expense here, so counting it as
    internal would both double-count it and list rows the headline claims.

    It carries a **signed** net as well as the unsigned gross, and the signed one
    is the informative figure: money the owner sends themselves inside the tracked
    accounts nets to zero, so a net that is *not* zero says the tracked perimeter
    leaks — money left for accounts this dashboard cannot see. That is exactly why
    ``net_cash_retained`` cannot be read as "the tracked bank balances went up by
    this much", and the page has to be able to say so.

    ``unaccounted`` is the rest of the table: a row on no account, on an account
    row that is gone, or on an account whose type nothing recognizes. It is the
    complement of the bank and card scopes, so no row can fall between the two and
    vanish. It is range-scoped, currency-agnostic and sits outside the
    reconciliation arithmetic entirely — it is a data-quality figure, not a term.

    The monetary footnotes are currency-agnostic sums (they add any non-INR rows
    in as plain rupees), which is why they are rough figures kept out of every
    headline bucket. The counts stay currency-agnostic too, so each one matches the
    row count of its drill-through link.
    """
    internal_count, internal_gross, internal_net = (
        await session.execute(
            select(ROW_COUNT, func.sum(Transaction.amount), SIGNED_FLOW).where(
                in_range,
                BANK_SCOPE,
                Transaction.category.in_(BANK_INTERNAL_SLUGS),
            )
        )
    ).one()
    non_inr_count = (
        await session.execute(select(ROW_COUNT).where(in_range, BANK_SCOPE, NON_INR))
    ).scalar_one()
    # Undated rows match no range, so this figure is range-independent. It stays
    # global for the same reason: a row with no date is a data problem wherever it
    # sits, and its drill-through carries neither a range nor a scope.
    undated_count, undated_net = (
        await session.execute(
            select(ROW_COUNT, SIGNED_FLOW).where(Transaction.transaction_date.is_(None))
        )
    ).one()
    unaccounted_count, unaccounted_net = (
        await session.execute(
            select(ROW_COUNT, SIGNED_FLOW).where(in_range, UNACCOUNTED_SCOPE)
        )
    ).one()
    return Footnotes(
        internal_count=internal_count,
        internal_gross=_decimal(internal_gross),
        internal_net=_decimal(internal_net),
        non_inr_count=non_inr_count,
        undated_count=undated_count,
        undated_net=_decimal(undated_net),
        unaccounted_count=unaccounted_count,
        unaccounted_net=_decimal(unaccounted_net),
    )


def trend_ranges(
    months: int = DEFAULT_TREND_MONTHS, *, today: datetime.date | None = None
) -> dict[str, list[str]]:
    """Each trend month mapped to the inclusive range that selecting it means.

    Clicking a month on the trend chart sets the report's range to that month, so
    the bounds have to be computed somewhere both the chart and a bookmarked URL
    agree on. Computing them here — over the same window and the same ``today``
    the trend itself uses — is what keeps a clicked month and the month whose bars
    were clicked describing the same days.

    The newest month is a partial one, so its upper bound is ``today`` rather than
    a month end in the future: selecting it means the same days the "This month"
    preset does.
    """
    today = today or datetime.date.today()
    months = max(1, min(months, MAX_TREND_MONTHS))
    ranges: dict[str, list[str]] = {}
    for start in trailing_month_starts(today, months):
        end = min(month_end(start), today)
        ranges[month_key_of(start)] = [start.isoformat(), end.isoformat()]
    return ranges


async def cashflow_trend(
    session: AsyncSession,
    months: int = DEFAULT_TREND_MONTHS,
    *,
    today: datetime.date | None = None,
) -> list[TrendPoint]:
    """Month-by-month income / expense / net-invested over a trailing window.

    Uses the same bucket map, the same bank scope and the same exclusions as
    ``cashflow_summary``, so a month here always reconciles with the summary for
    that month: income is the income bucket's slugs, never every credit, which is
    what keeps a ``repayment`` out of it, and the scope is the bank, which is what
    keeps the chart from telling a different story month by month than the tiles
    above it tell for the selected range.
    """
    today = today or datetime.date.today()
    months = max(1, min(months, MAX_TREND_MONTHS))
    window = trailing_month_starts(today, months)
    month_key = func.strftime("%Y-%m", Transaction.transaction_date)
    # The upper bound keeps a future-dated row from landing in the current
    # month's partial figures.
    in_window = Transaction.transaction_date.between(window[0], today)

    rows = (
        await session.execute(
            select(
                month_key,
                Transaction.category,
                Transaction.direction,
                SIGNED_FLOW,
            )
            .where(in_window, INR_OR_NULL, BANK_SCOPE)
            .group_by(month_key, Transaction.category, Transaction.direction)
        )
    ).all()

    # Counted on its own, with no currency clause: the three monetary series are
    # rupee sums and so must drop foreign-currency rows, but a salary paid in
    # another currency is still a salary credit that month, and the count has to
    # match the row count of the `category=salary` drill-through, which applies
    # no currency filter either. It is a separate query, so it needs the scope
    # said again: a card-side salary is out of the bank view, and a count that
    # kept it would contradict the income bar it sits under.
    salary_counts = (
        await session.execute(
            select(month_key, ROW_COUNT)
            .where(in_window, BANK_SCOPE, Transaction.category == "salary")
            .group_by(month_key)
        )
    ).all()

    keys = [month_key_of(start) for start in window]
    points = {
        key: TrendPoint(
            month=key,
            income=Decimal(0),
            expense=Decimal(0),
            net_invested=Decimal(0),
            salary_count=0,
        )
        for key in keys
    }

    for month, slug, _direction, flow in rows:
        point = points[month]
        signed = _decimal(flow)
        bucket = bucket_for_slug(slug, scope=REPORT_SCOPE)
        if bucket == "income":
            point.income += signed
        elif bucket == "expense":
            point.expense += -signed
        elif bucket == "investment":
            point.net_invested += -signed

    for month, count in salary_counts:
        points[month].salary_count = count

    # The window is pre-seeded, so a month with no rows is a zero, not a gap.
    return [points[key] for key in keys]
