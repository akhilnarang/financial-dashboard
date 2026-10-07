"""Cash bridge: opening bank cash, plus the cashflow terms, gives closing bank cash.

The flow terms come from the bank-scope ``CashflowSummary``. This module does not
bucket a row again. It adds only the bank-side flows that the summary does not
sum over the bank: family rows, excluded rows and foreign-currency rows in a
summing bucket. The terms then cover every dated bank-scope row in the range
exactly once, so the total gap is the sum of the account gaps.
"""

import datetime
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.enums import SnapshotCategory
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    BankStatementUpload,
    Transaction,
)
from financial_dashboard.schemas.cashflow import (
    BridgeAccount,
    BridgeBalance,
    BridgeLine,
    CashBridge,
    CashflowSummary,
)
from financial_dashboard.services.cashflow.buckets import FAMILY_SLUG
from financial_dashboard.services.cashflow.report import (
    BANK_INTERNAL_SLUGS,
    EXCLUDED,
    NON_INR,
    SIGNED_AMOUNT,
    UNCATEGORIZED,
)
from financial_dashboard.services.cashflow.scope import BANK_ACCOUNT_TYPES, BANK_SCOPE
from financial_dashboard.services.statements.bank import _parse_amount, _parse_date

ONE_DAY = datetime.timedelta(days=1)

# The part of a bank row's flow that the summary does not sum. "report" rows
# are in a summary figure already: a bucket, the internal footnote or the
# uncategorized bucket. The order matters: an excluded row is in no summary
# figure, whatever its category.
FLOW_KIND = case(
    (EXCLUDED, "excluded"),
    (Transaction.category == FAMILY_SLUG, "family"),
    (Transaction.category.in_(BANK_INTERNAL_SLUGS), "report"),
    (UNCATEGORIZED, "report"),
    (NON_INR, "non_inr"),
    else_="report",
).label("kind")


class Anchor(NamedTuple):
    """A known balance at the end of ``day``."""

    day: datetime.date
    value: Decimal
    kind: str


def _statement_anchors(upload: BankStatementUpload) -> list[Anchor]:
    anchors = []
    for raw_day, raw_value, shift in (
        (upload.statement_period_start, upload.opening_balance, -ONE_DAY),
        (upload.statement_period_end, upload.closing_balance, datetime.timedelta()),
    ):
        if not raw_day or not raw_value:
            continue
        try:
            anchors.append(
                Anchor(
                    _parse_date(raw_day) + shift, _parse_amount(raw_value), "statement"
                )
            )
        except InvalidOperation, ValueError:
            continue
    return anchors


def _flow_between(
    daily: dict[datetime.date, Decimal], after: datetime.date, upto: datetime.date
) -> Decimal:
    """Signed flow of the days in ``(after, upto]``."""
    return sum((flow for day, flow in daily.items() if after < day <= upto), Decimal(0))


class Period(NamedTuple):
    """One statement: the day before it starts, its last day, and both balances."""

    start: datetime.date
    end: datetime.date
    opening: Decimal
    closing: Decimal


class Balance(NamedTuple):
    """A balance, and the balances each covering statement implies, newest first."""

    balance: BridgeBalance
    implied: list[Decimal]


def _from_period(
    period: Period, day: datetime.date, daily: dict[datetime.date, Decimal]
) -> Anchor:
    """The balance a statement implies at the end of ``day``, from its nearer end."""
    if day - period.start <= period.end - day:
        flow = _flow_between(daily, period.start, day)
        return Anchor(period.start, period.opening + flow, "statement")
    flow = _flow_between(daily, day, period.end)
    return Anchor(period.end, period.closing - flow, "statement")


def balance_at(
    day: datetime.date,
    periods: list[Period],
    anchors: list[Anchor],
    daily: dict[datetime.date, Decimal],
) -> Balance:
    """Give one account's balance at the end of ``day``.

    Args:
        day: The day whose closing balance is needed.
        periods: The account's statements, newest first.
        anchors: The account's known balances: statement ends newest first, then
            snapshots, then running balances. The first wins a tie.
        daily: The account's signed flow per day.

    Returns:
        The newest statement that covers ``day``, moved from its nearer end with
        the rows between. Otherwise the nearest anchor, moved the same way.
        A balance moved over one day or more is estimated. With no anchor, the
        balance is unknown.
    """
    # ponytail: a hole between two statements is rolled across with the rows.
    # Where no statement covers the day, the order of rows in a day is a guess.
    covering = [_from_period(p, day, daily) for p in periods if p.start <= day <= p.end]
    if covering:
        best = covering[0]
        amount = best.value
    elif anchors:
        best = min(anchors, key=lambda a: abs((a.day - day).days))
        if best.day <= day:
            amount = best.value + _flow_between(daily, best.day, day)
        else:
            amount = best.value - _flow_between(daily, day, best.day)
    else:
        return Balance(BridgeBalance(amount=None, source="unknown", as_of=None), [])
    source = (
        "statement" if best.kind == "statement" and best.day == day else "estimated"
    )
    return Balance(
        BridgeBalance(amount=amount, source=source, as_of=best.day),
        [c.value for c in covering],
    )


class BankFlows(NamedTuple):
    """The bank-scope rows, read once."""

    daily: dict[int, dict[datetime.date, Decimal]]
    net_flow: dict[int, Decimal]
    extra: dict[str, Decimal]
    day_end: dict[tuple[int, datetime.date], Decimal]


async def _bank_flows(
    session: AsyncSession, date_from: datetime.date, date_to: datetime.date
) -> BankFlows:
    """Sum the dated bank rows per account and day, and find each day's end balance.

    A running balance is the balance after its row. The rows after it on the
    same day move it on to the end of that day.
    """
    flows = BankFlows(
        defaultdict(lambda: defaultdict(Decimal)),
        defaultdict(Decimal),
        defaultdict(Decimal),
        {},
    )
    # ponytail: rows with no time sort by id, so the order in a day is a guess.
    rows = await session.execute(
        select(
            Transaction.account_id,
            Transaction.transaction_date,
            FLOW_KIND,
            SIGNED_AMOUNT,
            Transaction.balance,
        )
        .where(BANK_SCOPE, Transaction.transaction_date.is_not(None))
        .order_by(Transaction.transaction_time.nulls_first(), Transaction.id)
    )
    for account_id, day, kind, flow, balance in rows:
        flows.daily[account_id][day] += flow
        if date_from <= day <= date_to:
            flows.net_flow[account_id] += flow
            if kind != "report":
                flows.extra[kind] += flow
        key = (account_id, day)
        if balance is not None:
            flows.day_end[key] = balance
        elif key in flows.day_end:
            flows.day_end[key] += flow
    return flows


class Known(NamedTuple):
    """Each account's statements and point balances."""

    periods: dict[int, list[Period]]
    anchors: dict[int, list[Anchor]]


async def _known(
    session: AsyncSession,
    account_ids: list[int],
    day_end: dict[tuple[int, datetime.date], Decimal],
) -> Known:
    periods: dict[int, list[Period]] = defaultdict(list)
    anchors: dict[int, list[Anchor]] = defaultdict(list)
    uploads = (
        await session.execute(
            select(BankStatementUpload)
            .where(BankStatementUpload.account_id.in_(account_ids))
            .order_by(BankStatementUpload.id.desc())
        )
    ).scalars()
    for upload in uploads:
        ends = _statement_anchors(upload)
        anchors[upload.account_id].extend(ends)
        if len(ends) == 2:
            periods[upload.account_id].append(
                Period(ends[0].day, ends[1].day, ends[0].value, ends[1].value)
            )

    snapshots = await session.execute(
        select(
            BalanceSnapshot.account_id,
            BalanceSnapshot.as_of_date,
            BalanceSnapshot.value,
        ).where(
            BalanceSnapshot.account_id.in_(account_ids),
            BalanceSnapshot.category == SnapshotCategory.bank_balance.value,
        )
    )
    for account_id, as_of, value in snapshots:
        anchors[account_id].append(Anchor(as_of, value, "snapshot"))
    for (account_id, day), balance in day_end.items():
        anchors[account_id].append(Anchor(day, balance, "transaction"))
    return Known(periods, anchors)


async def cash_bridge(session: AsyncSession, summary: CashflowSummary) -> CashBridge:
    """Bridge the bank balance at the start of the range to the balance at its end.

    Args:
        session: The request session.
        summary: The bank-scope cashflow summary for the range. Its figures are
            the flow terms of the bridge.

    Returns:
        The bridge, with one entry per bank account that has a balance or a row.
    """
    date_from, date_to = summary.date_from, summary.date_to
    accounts = (
        (
            await session.execute(
                select(Account)
                .where(Account.type.in_(BANK_ACCOUNT_TYPES))
                .order_by(Account.id)
            )
        )
        .scalars()
        .all()
    )
    daily, net_flow, extra, day_end = await _bank_flows(session, date_from, date_to)
    periods, anchors = await _known(session, [a.id for a in accounts], day_end)
    # The first date has no day before it. No row can exist there.
    opening_day = max(date_from, datetime.date.min + ONE_DAY) - ONE_DAY
    lines: list[BridgeAccount] = []
    warnings: list[str] = []
    for account in accounts:
        if not anchors[account.id] and account.id not in daily:
            continue
        bounds = [
            (
                day,
                balance_at(
                    day, periods[account.id], anchors[account.id], daily[account.id]
                ),
            )
            for day in (opening_day, date_to)
        ]
        opening, closing = (found.balance for _, found in bounds)
        flow = net_flow[account.id]
        gap = None
        if opening.amount is not None and closing.amount is not None:
            gap = closing.amount - opening.amount - flow
        lines.append(
            BridgeAccount(
                account_id=account.id,
                label=account.label,
                bank=account.bank,
                opening=opening,
                closing=closing,
                net_flow=flow,
                gap=gap,
            )
        )
        if gap is None:
            warnings.append(
                f"{account.label}: no balance found. The totals do not include it."
            )
        warnings.extend(
            f"{account.label}: statements disagree on {day}: "
            f"{', '.join(map(str, found.implied))}. The newest is used."
            for day, found in bounds
            if len(set(found.implied)) > 1
        )

    other = [
        BridgeLine(
            key="internal",
            label="Transfers between your accounts",
            amount=summary.footnotes.internal_net,
        ),
        BridgeLine(key="family", label="Family", amount=extra["family"]),
        BridgeLine(
            key="excluded", label="Rows left out of cashflow", amount=extra["excluded"]
        ),
        BridgeLine(
            key="uncategorized",
            label="No category",
            amount=summary.uncategorized.total,
        ),
        BridgeLine(key="non_inr", label="Other currencies", amount=extra["non_inr"]),
        BridgeLine(
            key="no_balance",
            label="Accounts with no balance",
            amount=-sum((a.net_flow for a in lines if a.gap is None), Decimal(0)),
        ),
    ]
    if extra["non_inr"]:
        warnings.append("Amounts in other currencies are added as rupees.")

    known = [a for a in lines if a.gap is not None]
    opening_total = sum((a.opening.amount or Decimal(0) for a in known), Decimal(0))
    actual = sum((a.closing.amount or Decimal(0) for a in known), Decimal(0))
    expected = (
        opening_total
        + summary.net_cash_retained
        + sum((line.amount for line in other), Decimal(0))
    )
    return CashBridge(
        date_from=date_from,
        date_to=date_to,
        opening=opening_total,
        earned=summary.income.total,
        spent=summary.expense.total,
        net_invested=summary.investment.net,
        transfers_in=summary.transfers_in.total,
        other=other,
        expected_closing=expected,
        actual_closing=actual,
        gap=actual - expected,
        accounts=lines,
        warnings=warnings,
    )
