"""Server reports that the assistant sends to the user as Telegram HTML.

The model chooses only the inputs of a report. The server reads the figures
and renders the message. The figures never go back to the model. The first
line of every report is its heading and must not hold a figure, because model
history keeps that line.
"""

import datetime
import html
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import IST
from financial_dashboard.core.links import page_link
from financial_dashboard.core.templating import format_inr_exact
from financial_dashboard.db.models import Account, StatementUpload, Transaction
from financial_dashboard.services.cashflow.buckets import INVESTMENT_BUCKET
from financial_dashboard.services.cashflow.report import (
    BLANK_CHARS,
    INCLUDED,
    INR_OR_NULL,
    cashflow_summary,
)
from financial_dashboard.services.cashflow.scope import BANK_SCOPE
from financial_dashboard.services.investments import get_current_valuations
from financial_dashboard.services.networth import current_networth
from financial_dashboard.services.assistant.contracts import (
    Answer,
    Error,
    GetCardDues,
    GetCashflowSummary,
    GetHoldings,
    GetInvestmentFlows,
    GetNetworth,
    ReportCall,
)

# Telegram allows 4,096 characters. The Ref footer needs some of them.
MAX_REPORT_CHARS = 3800

_GROUP_LABELS = {
    "bank_balance": "Bank balances",
    "cc_outstanding": "Card outstanding",
    "investment": "Investments",
    "manual_asset": "Other assets",
    "manual_liability": "Other liabilities",
}


async def render_report(session: AsyncSession, call: ReportCall) -> Answer | Error:
    """Render the report that the model asked for.

    Args:
        session: The turn session.
        call: The report call with its inputs.

    Returns:
        The answer with the report as Telegram HTML, or an error for inputs
        that are not valid.
    """
    from financial_dashboard.services.settings import get_app_base_url

    base_url = get_app_base_url()
    match call:
        case GetCashflowSummary() | GetInvestmentFlows():
            try:
                date_from = datetime.date.fromisoformat(call.date_from)
                date_to = datetime.date.fromisoformat(call.date_to)
            except ValueError:
                date_from = date_to = None
            if date_from is None or date_to is None or date_from > date_to:
                return Error(
                    outcome="error",
                    message="I couldn't work out that date range. Please give the dates.",
                    code="invalid_range",
                )
            match call:
                case GetCashflowSummary():
                    text = await _cashflow(session, date_from, date_to, base_url)
                case GetInvestmentFlows():
                    text = await _investment_flows(
                        session, date_from, date_to, base_url
                    )
        case GetHoldings():
            text = await _holdings(session, call, base_url)
        case GetNetworth():
            text = await _networth(session, base_url)
        case GetCardDues():
            text = await _card_dues(session, base_url)
    return Answer(outcome="answer", text=text)


def _fit(head: Sequence[str], rows: Sequence[str], tail: Sequence[str] = ()) -> str:
    """Join the lines and drop the last rows that do not fit in one message."""
    kept = list(rows)
    while True:
        more = len(rows) - len(kept)
        lines = [*head, *kept]
        if more:
            lines.append(f"… and {more} more")
        text = "\n".join([*lines, *tail])
        if len(text) <= MAX_REPORT_CHARS or not kept:
            return text
        kept.pop()


def _money(value: Decimal | None) -> str:
    """Format a rupee figure for HTML. A figure that the source omits shows as "?"."""
    return "?" if value is None else html.escape(format_inr_exact(value))


def _period(date_from: datetime.date, date_to: datetime.date) -> str:
    """Name an inclusive date range for a report heading."""
    return f"{date_from:%d %b %Y} to {date_to:%d %b %Y}"


def _range_query(date_from: datetime.date, date_to: datetime.date) -> dict[str, str]:
    """Return the query parameters that select an inclusive date range."""
    return {"date_from": date_from.isoformat(), "date_to": date_to.isoformat()}


async def _cashflow(
    session: AsyncSession,
    date_from: datetime.date,
    date_to: datetime.date,
    base_url: str,
) -> str:
    """Render the bank-scope cashflow totals for one range, as /cashflow shows them."""
    summary = await cashflow_summary(session, date_from, date_to)
    investment = summary.investment
    period = _range_query(date_from, date_to)
    heading = html.escape(f"Cashflow, {_period(date_from, date_to)}")
    lines = [
        f"<b>{heading}</b> (bank accounts)",
        f"Income: {_money(summary.income.total)}",
        f"Spent: {_money(summary.expense.total)}",
        f"Invested: {_money(investment.contributions)}"
        f" (taken out {_money(investment.redemptions)},"
        f" net {_money(investment.net)})",
        f"Transfers in: {_money(summary.transfers_in.total)}",
        f"Net retained: {_money(summary.net_cash_retained)}",
    ]
    if summary.uncategorized.count:
        query = urlencode(
            {"uncategorized": "1", **period, "scope": "bank", "excluded": "0"}
        )
        label = f"{summary.uncategorized.count} uncategorized rows"
        lines.append(
            f"{page_link(label, f'/transactions?{query}', base_url)},"
            " not in the figures above."
        )
    lines.append(
        page_link(
            "Open the cashflow report", f"/cashflow?{urlencode(period)}", base_url
        )
    )
    return "\n".join(lines)


async def _investment_flows(
    session: AsyncSession,
    date_from: datetime.date,
    date_to: datetime.date,
    base_url: str,
) -> str:
    """Render investment money in and out of one range, grouped by counterparty.

    The rows are the investment rows that the cashflow report counts, so the
    totals agree with its invested figures.
    """
    counterparty = func.coalesce(
        func.nullif(func.trim(Transaction.counterparty, BLANK_CHARS), ""), ""
    )
    grouped = (
        await session.execute(
            select(
                Transaction.direction,
                Transaction.category,
                counterparty,
                func.sum(Transaction.amount),
                func.count(),
            )
            .where(
                Transaction.transaction_date.between(date_from, date_to),
                Transaction.category.in_(INVESTMENT_BUCKET),
                INR_OR_NULL,
                BANK_SCOPE,
                INCLUDED,
            )
            .group_by(Transaction.direction, Transaction.category, counterparty)
            .order_by(func.sum(Transaction.amount).desc())
        )
    ).all()
    period = _range_query(date_from, date_to)
    totals = {"debit": Decimal(0), "credit": Decimal(0)}
    rows: dict[str, list[str]] = {"debit": [], "credit": []}
    for direction, category, name, amount, count in grouped:
        side = "debit" if direction == "debit" else "credit"
        totals[side] += amount or Decimal(0)
        query = urlencode(
            {
                "category": category,
                "direction": side,
                "counterparty": name,
                **period,
                "non_inr": "0",
                "scope": "bank",
                "excluded": "0",
            }
        )
        label = name or "(no counterparty)"
        rows[side].append(
            f"• {page_link(label, f'/transactions?{query}', base_url)}:"
            f" {_money(amount)} ({count})"
        )
    heading = html.escape(f"Investments, {_period(date_from, date_to)}")
    head = [
        f"<b>{heading}</b> (bank accounts)",
        f"Put in: {_money(totals['debit'])} · Taken out: {_money(totals['credit'])}"
        f" · Net: {_money(totals['debit'] - totals['credit'])}",
    ]
    if not grouped:
        head.append("No investment rows in this range.")
    body = [
        *(["", "Put in:", *rows["debit"]] if rows["debit"] else []),
        *(["", "Taken out:", *rows["credit"]] if rows["credit"] else []),
    ]
    tail = [
        "",
        page_link(
            "Open the cashflow report", f"/cashflow?{urlencode(period)}", base_url
        ),
    ]
    return _fit(head, body, tail)


async def _holdings(session: AsyncSession, call: GetHoldings, base_url: str) -> str:
    """Render the current holdings from the latest CAS, largest value first."""
    search, asset_class = call.search, call.asset_class
    needle = (search or "").strip().casefold()
    holdings = [
        row
        for row in await get_current_valuations(session)
        if needle in row.instrument_name.casefold()
        and asset_class in (None, row.asset_class)
    ]
    holdings.sort(key=lambda row: (row.value is None, -(row.value or 0)))
    total = sum((row.value for row in holdings if row.value is not None), Decimal(0))
    filters = ", ".join(
        f"{label} {value!r}"
        for label, value in (("matching", search), ("class", asset_class))
        if value
    )
    heading = "Holdings" + (f" ({filters})" if filters else "")
    dates = sorted({row.statement_date for row in holdings})
    head = [f"<b>{html.escape(heading)}</b>"]
    if not holdings:
        head.append("No holdings match.")
    else:
        as_of = (
            f"{dates[0]:%d %b %Y}"
            if len(dates) == 1
            else f"{dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y}"
        )
        head.append(
            f"Total: {_money(total)} across {len(holdings)} holdings"
            f" (CAS as of {as_of})"
        )
    rows = [
        f"• {html.escape(row.instrument_name)}: {_money(row.value)}"
        + (f" ({format(row.quantity.normalize(), 'f')} units)" if row.quantity else "")
        for row in holdings
    ]
    return _fit(head, rows, ["", page_link("Open net worth", "/networth", base_url)])


async def _networth(session: AsyncSession, base_url: str) -> str:
    """Render net worth and the latest balance of every source."""
    today = datetime.datetime.now(IST).date()
    summary = await current_networth(session, today=today)
    head = [
        f"<b>Net worth</b> (as of {today:%d %b %Y})",
        f"Net worth: {_money(summary.net_worth)}",
        f"Assets: {_money(summary.total_assets)}"
        f" · Liabilities: {_money(summary.total_liabilities)}",
    ]
    if summary.has_stale:
        head.append("Some balances are stale (marked ⚠).")
    rows: list[str] = []
    for group in summary.groups:
        label = _GROUP_LABELS.get(group.category, group.category)
        rows.extend(["", f"<b>{html.escape(label)}</b>: {_money(group.total)}"])
        rows.extend(
            f"• {html.escape(row.label)}: {_money(row.value)}"
            f" ({row.as_of_date:%d %b %Y}{' ⚠' if row.stale else ''})"
            for row in group.rows
        )
    return _fit(head, rows, ["", page_link("Open net worth", "/networth", base_url)])


async def _card_dues(session: AsyncSession, base_url: str) -> str:
    """Render the latest bill and payment state of each active credit card."""
    statements = (
        (
            await session.execute(
                select(StatementUpload)
                .join(Account, StatementUpload.account_id == Account.id)
                .where(
                    StatementUpload.is_active.is_(True),
                    Account.type == "credit_card",
                    Account.active.is_not(False),
                )
                .order_by(StatementUpload.due_date, StatementUpload.id)
            )
        )
        .unique()
        .scalars()
        .all()
    )
    head = ["<b>Card bills</b> (latest statement of each card)"]
    if not statements:
        head.append("No card statements.")
    rows = []
    for statement in statements:
        label = statement.account.label if statement.account else statement.bank
        status = (statement.payment_status or "unknown").replace("_", " ")
        paid = statement.payment_paid_amount or Decimal(0)
        total, minimum = (
            _stored_amount(value)
            for value in (statement.total_amount_due, statement.minimum_amount_due)
        )
        rows.append(
            f"• {page_link(label, f'/statements/{statement.id}', base_url)}:"
            f" {total} due {html.escape(statement.due_date or '?')}"
            f" (min {minimum}), {html.escape(status)}, paid {_money(paid)}"
        )
    return _fit(head, rows, ["", page_link("Open statements", "/statements", base_url)])


def _stored_amount(value: str | None) -> str:
    """Format a stored statement amount. Show the stored text when it is not a number."""
    if not value:
        return "?"
    try:
        return _money(Decimal(value.replace(",", "").strip()))
    except InvalidOperation:
        return html.escape(value)
