"""Read-only reconciliation of bank accounts against their statements.

``build_report`` is pure, so an offline script can feed it the JSON that
``/api/transactions`` returns. ``reconcile`` loads the same rows from the DB.
"""

import datetime
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import month_key
from financial_dashboard.db.models import Account, BankStatementUpload, Transaction
from financial_dashboard.schemas.reconcile import (
    DuplicateCandidate,
    MissingStatementMonths,
    ReconcileReport,
    StatementBalance,
    UnpairedSelfTransfers,
)
from financial_dashboard.schemas.transactions import TransactionRead
from financial_dashboard.services.cashflow.buckets import SELF_TRANSFER_SLUG
from financial_dashboard.services.statements.bank import _parse_amount, _parse_date
from financial_dashboard.services.transaction_reads import _transaction_read
from financial_dashboard.services.txn_merge import (
    _normalized_currency,
    _shortened_reference_match,
)

PAIR_WINDOW = datetime.timedelta(days=3)
SELF_TRANSFER_AMOUNT_TOLERANCE = Decimal(1)


class _Period(NamedTuple):
    start: datetime.date
    end: datetime.date


def _day(row: TransactionRead) -> datetime.date:
    """Return the transaction date. build_report drops undated rows first."""
    assert row.transaction_date is not None
    return row.transaction_date


def _signed(row: TransactionRead) -> Decimal:
    """Return the amount with credits positive and debits negative."""
    return row.amount if row.direction == "credit" else -row.amount


def _is_inr(row: TransactionRead) -> bool:
    """Return whether the row is in INR. A missing currency counts as INR."""
    return _normalized_currency(row.currency) == "INR"


def _balance(text: str | None) -> Decimal | None:
    """Parse a stored statement balance. Return None when it is not a number."""
    try:
        return _parse_amount(text or "")
    except InvalidOperation:
        return None


def _period(statement: BankStatementUpload) -> _Period | None:
    """Return the statement period.

    Return None when a date does not parse or the end is before the start.
    """
    try:
        period = _Period(
            _parse_date(statement.statement_period_start or ""),
            _parse_date(statement.statement_period_end or ""),
        )
    except ValueError, OverflowError:
        return None
    return period if period.start <= period.end else None


def _distinct_references(first: str | None, second: str | None) -> bool:
    """Tell whether two references prove two separate events.

    A short or shortened reference proves nothing, so the rows stay a candidate.
    """
    first, second = (
        "".join(ch for ch in (ref or "").upper() if ch.isalnum())
        for ref in (first, second)
    )
    # A reference under four characters is too short to prove a difference.
    if min(len(first), len(second)) < 4 or first == second:
        return False
    return not _shortened_reference_match(first, second)


def _statement_balance(
    statement: BankStatementUpload, period: _Period, rows: Sequence[TransactionRead]
) -> StatementBalance:
    """Compare one statement with the DB rows dated inside its period.

    The gap is db_net minus statement_delta. Only INR rows count.
    """
    opening = _balance(statement.opening_balance)
    closing = _balance(statement.closing_balance)
    delta = None if opening is None or closing is None else closing - opening
    db_net = sum(
        (
            _signed(r)
            for r in rows
            if r.account_id == statement.account_id
            and period.start <= _day(r) <= period.end
            and _is_inr(r)
        ),
        Decimal(0),
    )
    return StatementBalance(
        statement_id=statement.id,
        account_id=statement.account_id,
        period_start=period.start,
        period_end=period.end,
        opening_balance=opening,
        closing_balance=closing,
        statement_delta=delta,
        db_net=db_net,
        gap=None if delta is None else db_net - delta,
    )


def _missing_statements(
    date_from: datetime.date,
    date_to: datetime.date,
    periods: Mapping[int, list[_Period]],
    rows: Sequence[TransactionRead],
) -> list[MissingStatementMonths]:
    """List the months of in-range rows that no statement period covers.

    Coverage is per date, so a mid-month statement covers only its own days.
    """
    uncovered: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        if r.account_id is None or not date_from <= _day(r) <= date_to:
            continue
        if not any(p.start <= _day(r) <= p.end for p in periods.get(r.account_id, [])):
            uncovered[r.account_id].add(month_key(_day(r)))
    return [
        MissingStatementMonths(account_id=account_id, months=sorted(months))
        for account_id, months in sorted(uncovered.items())
    ]


def _unpaired_self_transfers(
    date_from: datetime.date,
    date_to: datetime.date,
    rows: Sequence[TransactionRead],
) -> UnpairedSelfTransfers:
    """Pair self-transfer legs and return the in-range legs left over.

    A partner is an opposite leg on another account, within ₹1 and 3 days.
    The first pass pairs legs with the same reference. The second pass pairs
    the nearest partner. Each row pairs once.
    """
    legs = sorted(
        (r for r in rows if r.category == SELF_TRANSFER_SLUG),
        key=lambda r: (_day(r), r.id),
    )
    paired: set[int] = set()

    def partners(leg: TransactionRead) -> list[TransactionRead]:
        """Return unpaired opposite legs on another account that fit the window."""
        return [
            r
            for r in legs
            if r.id not in paired
            and r.account_id != leg.account_id
            and r.direction != leg.direction
            and _normalized_currency(r.currency) == _normalized_currency(leg.currency)
            and abs(r.amount - leg.amount) <= SELF_TRANSFER_AMOUNT_TOLERANCE
            and abs(_day(r) - _day(leg)) <= PAIR_WINDOW
        ]

    def shares_reference(leg: TransactionRead, rival: TransactionRead) -> bool:
        """Return whether both legs carry the same non-empty reference."""
        return bool(leg.reference_number) and (
            rival.reference_number == leg.reference_number
        )

    # A shared reference is the strongest evidence, so those pairs go first. A
    # duplicate leg without the reference then stays unpaired.
    # ponytail: O(n^2) scan; a range holds a few hundred self-transfers at most.
    for by_reference in (True, False):
        for leg in legs:
            if leg.id in paired:
                continue
            rivals = [
                r for r in partners(leg) if not by_reference or shares_reference(leg, r)
            ]
            if rivals:
                best = min(
                    rivals,
                    key=lambda r: (
                        abs(_day(r) - _day(leg)),
                        abs(r.amount - leg.amount),
                        r.id,
                    ),
                )
                paired.update((leg.id, best.id))
    orphans = [
        r for r in legs if r.id not in paired and date_from <= _day(r) <= date_to
    ]
    return UnpairedSelfTransfers(
        net=sum((_signed(r) for r in orphans if _is_inr(r)), Decimal(0)),
        items=orphans,
    )


def _duplicate_candidates(
    date_from: datetime.date,
    date_to: datetime.date,
    rows: Sequence[TransactionRead],
) -> list[DuplicateCandidate]:
    """Pair statement rows with other rows that may be the same event.

    Both rows have the same account, direction, currency and amount, within
    3 days. Distinct references remove a pair. Short or shortened ones do not.
    """

    def key(r: TransactionRead) -> tuple:
        """Group rows that could be the same event: account, direction, currency, amount."""
        return (r.account_id, r.direction, _normalized_currency(r.currency), r.amount)

    alerts: dict[tuple, list[TransactionRead]] = defaultdict(list)
    for r in rows:
        if r.email_type != "bank_statement":
            alerts[key(r)].append(r)
    pairs = []
    for stmt in sorted(rows, key=lambda r: (_day(r), r.id)):
        if stmt.email_type != "bank_statement":
            continue
        for alert in alerts[key(stmt)]:
            if (
                abs(_day(alert) - _day(stmt)) <= PAIR_WINDOW
                and (
                    date_from <= _day(stmt) <= date_to
                    or date_from <= _day(alert) <= date_to
                )
                and not _distinct_references(
                    stmt.reference_number, alert.reference_number
                )
            ):
                pairs.append(DuplicateCandidate(statement_txn=stmt, alert_txn=alert))
    return pairs


def build_report(
    date_from: datetime.date,
    date_to: datetime.date,
    accounts: Mapping[int, str],
    statements: Sequence[BankStatementUpload],
    rows: Sequence[TransactionRead],
) -> ReconcileReport:
    """Build the reconciliation report from rows already loaded.

    Args:
        date_from: First day of the range, inclusive.
        date_to: Last day of the range, inclusive.
        accounts: Label of every bank account, keyed by account id.
        statements: Every statement on those accounts, as ORM rows or any
            object with the same fields. Statements outside the range still
            count toward coverage.
        rows: Transactions on those accounts, from ``date_from`` minus
            ``PAIR_WINDOW`` to ``date_to`` plus ``PAIR_WINDOW``. The extra days
            let a leg near the range edge find its partner. Undated rows are
            ignored.

    Returns:
        The report: statement gaps, months without a statement, unpaired
        self-transfers and duplicate candidates.
    """
    rows = [r for r in rows if r.transaction_date is not None]
    periods: dict[int, list[_Period]] = defaultdict(list)
    checked = []
    for statement in statements:
        period = _period(statement)
        if period is None:
            continue
        periods[statement.account_id].append(period)
        if date_from <= period.start and period.end <= date_to:
            checked.append(_statement_balance(statement, period, rows))
    checked.sort(key=lambda s: (s.account_id, s.period_start, s.statement_id))
    return ReconcileReport(
        date_from=date_from,
        date_to=date_to,
        accounts=dict(accounts),
        statements=checked,
        missing_statements=_missing_statements(date_from, date_to, periods, rows),
        unpaired_self_transfers=_unpaired_self_transfers(date_from, date_to, rows),
        duplicate_candidates=_duplicate_candidates(date_from, date_to, rows),
    )


async def reconcile(
    session: AsyncSession, date_from: datetime.date, date_to: datetime.date
) -> ReconcileReport:
    """Load the bank-account rows for a range and reconcile them.

    Args:
        session: The request session.
        date_from: First day of the range, inclusive.
        date_to: Last day of the range, inclusive.

    Returns:
        The reconciliation report for the range.
    """
    accounts = (
        await session.scalars(select(Account).where(Account.type == "bank_account"))
    ).all()
    labels = {a.id: a.label for a in accounts}
    statements = (
        await session.scalars(
            select(BankStatementUpload).where(
                BankStatementUpload.account_id.in_(labels)
            )
        )
    ).all()
    txns = (
        await session.scalars(
            select(Transaction).where(
                Transaction.account_id.in_(labels),
                Transaction.transaction_date.between(
                    date_from - PAIR_WINDOW, date_to + PAIR_WINDOW
                ),
            )
        )
    ).all()
    rows = [_transaction_read(t) for t in txns]
    return build_report(date_from, date_to, labels, statements, rows)
