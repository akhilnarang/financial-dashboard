"""Response shapes for the cashflow report.

``report.py`` builds these directly: the service is the single source of the
numbers and both the JSON and HTML routes render what it returns. They are the
public contract of ``/api/cashflow/*``, so the sign conventions below are
documented on the fields themselves and reach an OpenAPI consumer.

Every ``total`` is the line's contribution to the bucket *as displayed*, so the
lines of a bucket sum to the bucket's headline figure and bars read intuitively:
income and transfers-in are signed (credits positive), expense lines are negated
(spend positive, so a refund/cashback credit comes out negative), and investment
lines are negated (contributions positive, redemptions negative).
"""

import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field

#: ``YYYY-MM``. The trend keys its points by calendar month, and the format is
#: part of the contract: a client groups, sorts and labels on this string.
MonthKey = Annotated[
    str,
    Field(
        description="Calendar month as YYYY-MM (zero-padded, e.g. 2026-07).",
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
        examples=["2026-07"],
    ),
]

InvestmentKind = Literal["contribution", "redemption"]


class CategoryLine(BaseModel):
    """One category's contribution to a bucket, over the summary's range."""

    slug: str | None = Field(
        description=(
            "The row's category slug, or null for the group of rows that carry no "
            "category at all (the NULL/blank group of the uncategorized bucket)."
        ),
    )
    label: str = Field(
        description=(
            "Display name for the line. '(uncategorized)' for the null-slug group, "
            "'unmapped: <slug>' for a slug the report's bucket map does not know."
        ),
    )
    total: Decimal = Field(
        description=(
            "The line's contribution to its bucket as displayed, not the raw signed "
            "flow. Income and transfers-in: signed (credits positive). Expense: "
            "negated, so spend is positive and a refund/cashback credit is negative. "
            "Investment: negated, so a contribution is positive and a redemption is "
            "negative. Uncategorized: raw signed flow. A bucket's lines therefore sum "
            "to its headline figure."
        ),
    )
    count: int = Field(description="Transactions aggregated into this line.")
    kind: InvestmentKind | None = Field(
        default=None,
        description=(
            "Investment lines only: which direction of the slug this line is. The "
            "investment bucket is the one that splits a single slug by direction, so "
            "the slug alone does not identify the line. Null on every other bucket."
        ),
    )


class CounterpartyLine(BaseModel):
    """One counterparty's contribution to the transfers-in bucket."""

    counterparty: str | None = Field(
        description=(
            "The counterparty, or null for the group of rows that name nobody. Blank "
            "and whitespace-only counterparties are collapsed onto that null group."
        ),
    )
    label: str = Field(
        description="Display name; '(no counterparty)' for the null group.",
    )
    total: Decimal = Field(
        description="Signed rupee flow for this counterparty; credits are positive.",
    )
    count: int = Field(description="Transactions aggregated into this line.")


class BucketSummary(BaseModel):
    """A headline bucket: its total, its row count and the lines it is made of."""

    total: Decimal = Field(
        description="Sum of the bucket's line totals, in the bucket's display sign.",
    )
    count: int = Field(description="Transactions in the bucket.")
    lines: list[CategoryLine] = Field(
        description="Per-category lines, largest absolute contribution first.",
    )


class InvestmentSummary(BaseModel):
    """The investment bucket, split by direction rather than netted flat."""

    contributions: Decimal = Field(
        description="Money put in over the range; positive.",
    )
    redemptions: Decimal = Field(
        description="Money taken out over the range; positive.",
    )
    net: Decimal = Field(
        description="contributions - redemptions. Negative in a redemption-heavy range.",
    )
    count: int = Field(description="Investment transactions in the range.")
    lines: list[CategoryLine] = Field(
        description=(
            "One line per (slug, direction) pair, so the same slug can appear twice — "
            "once as a contribution and once as a redemption. Contribution totals are "
            "positive, redemption totals negative."
        ),
    )


class TransfersInSummary(BaseModel):
    """Money handed back by a person: its own bucket, so it never inflates income."""

    total: Decimal = Field(description="Signed rupee flow; credits positive.")
    count: int = Field(description="Transfers-in transactions in the range.")
    lines: list[CounterpartyLine] = Field(
        description="Per-counterparty lines, largest absolute contribution first.",
    )


class Footnotes(BaseModel):
    """The populations deliberately excluded from every headline bucket.

    They are counted rather than dropped, so money the buckets do not sum is
    still visible. The monetary figures are currency-agnostic sums — they add
    any foreign-currency rows in as plain rupees — which is exactly why they are
    rough figures kept out of the headline.
    """

    internal_count: int = Field(
        description=(
            "Bank-side self-transfers in the range. A credit-card payment is NOT here: "
            "on a cash basis, paying the card bill is when the money leaves the bank, "
            "so it is counted as expense."
        ),
    )
    internal_gross: Decimal = Field(
        description=(
            "Gross (unsigned) amount of those internal movements. Rough figure — may "
            "mix currencies."
        ),
    )
    internal_net: Decimal = Field(
        description=(
            "Signed net flow of those internal movements (credits positive). Money the "
            "owner sends between their own tracked accounts nets to zero, so a non-zero "
            "figure here is money that left the tracked accounts for accounts the "
            "dashboard does not see — which is why net_cash_retained is not the change "
            "in tracked bank balances. Rough figure — may mix currencies."
        ),
    )
    non_inr_count: int = Field(
        description="Bank-side rows in the range with a currency other than INR.",
    )
    undated_count: int = Field(
        description=(
            "Rows with no transaction_date. They fall into no range, so this figure is "
            "range-independent — and, unlike every other figure here, it is not scoped "
            "to the bank either: a row with no date is a data problem on any account."
        ),
    )
    undated_net: Decimal = Field(
        description=(
            "Signed net flow of the undated rows. Range-independent, and a rough "
            "figure — may mix currencies."
        ),
    )
    unaccounted_count: int = Field(
        description=(
            "Rows in the range on no account at all, or on an account whose type is "
            "neither a bank nor a card. They are in no scope, so they reach no headline "
            "figure; counted here so they are visible rather than dropped."
        ),
    )
    unaccounted_net: Decimal = Field(
        description=(
            "Signed net flow of the unaccounted rows (credits positive). Outside the "
            "reconciliation arithmetic — a data-quality figure, not a term in it. Rough "
            "figure — may mix currencies."
        ),
    )
    family_count: int = Field(
        description=(
            "Rows categorized `family` in the range, over all accounts: money that "
            "moved through the owner's accounts for a family member. Excluded from "
            "every headline bucket."
        ),
    )
    family_net: Decimal = Field(
        description=(
            "Signed net flow of the family rows (credits positive). A running family "
            "position, not expected to net to zero. Rough figure; may mix currencies."
        ),
    )
    excluded_count: int = Field(
        default=0,
        description=(
            "Rows flagged out of the cashflow in the range, over all accounts. The "
            "figures above drop them. This line counts them so they stay visible. An "
            "undated flagged row is in no range and shows in the Undated line."
        ),
    )
    excluded_gross: Decimal = Field(
        default=Decimal(0),
        description=(
            "Gross (unsigned) amount of the excluded rows. Read this when a flagged "
            "pair nets to zero. Rough figure. It may mix currencies."
        ),
    )
    excluded_net: Decimal = Field(
        default=Decimal(0),
        description=(
            "Signed net flow of the excluded rows (credits positive). Rough figure. "
            "It may mix currencies."
        ),
    )


class CashflowSummary(BaseModel):
    """Every cashflow figure for one inclusive ``transaction_date`` range.

    Cash basis, scoped to the bank: income is what landed there and expense is
    what left it, so a card bill is spend and a card swipe is not. Two figures
    span every account instead: ``expense_detail`` (see its own description) and
    the ``family`` footnote, which counts a family member's card swipe alongside
    their bank transfer.
    """

    date_from: datetime.date = Field(
        description="First day of the range; inclusive. Rows on this date are counted.",
    )
    date_to: datetime.date = Field(
        description="Last day of the range; inclusive. Rows on this date are counted.",
    )
    income: BucketSummary = Field(
        description="Earnings only: transfers-in and investment redemptions are not here.",
    )
    expense: BucketSummary = Field(
        description=(
            "Money that left the bank, net of refunds and cashback, which appear as "
            "negative lines. On a cash basis this counts the card BILLS paid, not the "
            "swipes they settled: a credit-card payment is an expense line here."
        ),
    )
    expense_detail: BucketSummary = Field(
        description=(
            "What was bought, over EVERY account: the card swipes by category, where a "
            "credit-card payment is internal instead (counting both would charge the "
            "same rupee twice). This is a different question from `expense` and will "
            "NOT reconcile to it — the difference is the timing gap between buying and "
            "paying the bill. Never add the two together. INR-or-null, like the other "
            "rupee figures."
        ),
    )
    investment: InvestmentSummary = Field(
        description=(
            "Money moved in and out of investments, split by direction rather than "
            "netted flat: it is neither income nor spend, so it is its own bucket."
        ),
    )
    transfers_in: TransfersInSummary = Field(
        description=(
            "Credits that are somebody handing money back, broken out per "
            "counterparty so they never inflate income."
        ),
    )
    uncategorized: BucketSummary = Field(
        description=(
            "Rows no bucket can place: no category, the 'unknown' sentinel, or an "
            "unmapped slug. Unlike the rupee buckets this one has no currency filter, "
            "so it is a rough figure that may mix currencies — it feeds no headline and "
            "is the error bar on net_cash_retained rather than a term in it."
        ),
    )
    net_cash_retained: Decimal = Field(
        description=(
            "income + transfers_in - expense - investment.net, over the bank. "
            "Informational, not an enforced invariant, and NOT the change in tracked "
            "bank balances: single-leg transfers and untracked accounts mean it will "
            "not reconcile exactly. The uncategorized total is its error bar, and a "
            "non-zero footnotes.internal_net says how much left the tracked perimeter."
        ),
    )
    footnotes: Footnotes = Field(
        description=(
            "The populations every headline bucket excludes — internal movements, "
            "non-INR rows, undated rows and rows on no known account — counted rather "
            "than silently dropped."
        ),
    )


class TrendPoint(BaseModel):
    """One calendar month of the trailing trend, on the same buckets as the summary."""

    month: MonthKey
    income: Decimal = Field(
        description="The month's income bucket; earnings only, credits positive.",
    )
    expense: Decimal = Field(
        description="The month's spend, positive, net of refunds and cashback.",
    )
    net_invested: Decimal = Field(
        description="Contributions minus redemptions; negative in a redemption-heavy month.",
    )
    salary_count: int = Field(
        description=(
            "Salary credits that month. A ~14-day pay cycle puts 2 or 3 paychecks in a "
            "calendar month, so income can swing by half with nothing having changed; "
            "this count is what tells that apart from a real swing. Counted without a "
            "currency filter, unlike the three monetary figures."
        ),
    )


BalanceSource = Literal["statement", "estimated", "unknown"]


class BridgeBalance(BaseModel):
    """One account's balance at the end of one day."""

    amount: Decimal | None = Field(
        description="The balance, or null when no source gives one.",
    )
    source: BalanceSource = Field(
        description=(
            "'statement': a bank statement starts the next day or ends on this day. "
            "'estimated': the nearest other balance (statement, snapshot or a "
            "transaction's running balance), moved to this day with the rows between. "
            "'unknown': no balance exists for the account."
        ),
    )
    as_of: datetime.date | None = Field(
        description=(
            "End of the day that the source balance is for. It is the requested day "
            "for a statement balance. Null when the source is unknown."
        ),
    )


class BridgeAccount(BaseModel):
    """One bank account's part of the cash bridge."""

    account_id: int
    label: str
    bank: str
    opening: BridgeBalance = Field(description="Balance at the start of date_from.")
    closing: BridgeBalance = Field(description="Balance at the end of date_to.")
    net_flow: Decimal = Field(
        description=(
            "Signed sum of every dated row on the account in the range, excluded rows "
            "included. Credits are positive."
        ),
    )
    gap: Decimal | None = Field(
        description=(
            "closing - opening - net_flow. Zero when the rows explain the balance "
            "change. Null when a balance is unknown. The totals then leave the "
            "account out."
        ),
    )


class BridgeLine(BaseModel):
    """One 'Other' term of the bridge."""

    key: str
    label: str
    amount: Decimal = Field(
        description="Signed change to bank cash. A credit adds, a debit subtracts.",
    )


class CashBridge(BaseModel):
    """Opening bank cash, the cashflow terms, and the closing bank cash.

    ``expected_closing = opening + earned - spent - net_invested + transfers_in +
    sum(other)``. ``gap = actual_closing - expected_closing``. A gap of zero means
    the rows explain the balance change. The gap is the sum of the account gaps.
    """

    date_from: datetime.date
    date_to: datetime.date
    opening: Decimal = Field(description="Sum of known opening balances.")
    earned: Decimal = Field(description="The income bucket of the bank-scope report.")
    spent: Decimal = Field(description="The expense bucket of the bank-scope report.")
    net_invested: Decimal = Field(
        description="Investment contributions minus redemptions, from the report.",
    )
    transfers_in: Decimal = Field(description="The transfers-in bucket of the report.")
    other: list[BridgeLine] = Field(
        description="Every other bank-cash movement in the range, signed.",
    )
    expected_closing: Decimal
    actual_closing: Decimal = Field(description="Sum of known closing balances.")
    gap: Decimal = Field(description="actual_closing - expected_closing.")
    accounts: list[BridgeAccount]
    warnings: list[str] = Field(
        description="Plain-language notes on what the totals leave out.",
    )
