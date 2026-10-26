"""Shared CC billing-cycle window.

A bill-payment credit belongs to the statement cycle it clears. A payment dated
``D`` clears the bill whose due date is the earliest due date on/after ``D`` —
you pay the bill that is next coming due. So the cycle for a statement due on
``this_due`` is the range ``(prev_due, this_due]``: after the prior bill was due,
up to and including this bill's due date. The latest cycle keeps an open upper
edge so a late payment still counts.

This is expressed as the half-open range ``[prev_due + 1 day, this_due + 1 day)``
so one predicate can place every row. The boundary between two consecutive
statements is a single value derived once (``boundary``), so adjacent cycles
never overlap or leave a gap — even when a due date is unreadable or two
statements share a due date (a duplicate re-send, or an email summary and its
PDF: both share one cycle, and only the newer one is acted on).

Only statements that carry a parseable due date anchor a boundary. A
password-required or parse-error upload has no due date; it must not act as a
cycle edge, or it would strand a payment (and recreate the very bug this fixes).

An earlier version anchored the window on ``created_at`` (the ingestion time, a
proxy for the generation date). A bill paid in the gap between generation and a
late ingestion fell outside its own cycle, so an already-paid statement showed
unpaid and reminded.

Known limitation: ``payment_paid_amount`` is a stored value, recomputed only
when triggered. When a new statement moves an old cycle's upper edge, a LATE
payment that was applied to the old (then-latest) cycle now belongs to the new
one, but the old cycle is not auto-recomputed, so its stored paid amount can go
stale. This only touches a superseded statement's own display and historical
snapshot — the latest cycle (which drives reminders and net-worth) is correct.
A safe auto-fix needs a "manually paid" marker to avoid reverting a manual Mark
as Paid, so it is deferred.
"""

from datetime import date, datetime, timedelta, timezone
from decimal import InvalidOperation
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import StatementUpload, Transaction
from financial_dashboard.services.statements.cc import parse_cc_date


class CycleWindow(NamedTuple):
    """A statement cycle as a half-open date range ``[start, end)``."""

    start: date
    end: date | None  # exclusive upper bound; None = open (latest cycle)


def _due(due_text: str | None) -> date | None:
    if not due_text:
        return None
    try:
        return parse_cc_date(due_text)
    except ValueError, InvalidOperation:
        return None


def _boundary(earlier_due: date | None, later_created: datetime | None) -> date | None:
    """The single date that separates the earlier statement's cycle from the
    later one's. It is the day AFTER the earlier bill was due, so a payment made
    on the earlier due date still clears the earlier bill. When the earlier due
    date is unreadable, fall back to the later statement's ingestion date."""
    if earlier_due is not None:
        return earlier_due + timedelta(days=1)
    if later_created is not None:
        return later_created.date()
    return None


async def cc_cycle_window(
    session: AsyncSession, upload: StatementUpload
) -> CycleWindow:
    """Return ``upload``'s billing-cycle window.

    ``start`` = the boundary with the previous anchor cycle (day after its due
    date), or this statement's ``created_at`` date when there is no earlier
    anchor. ``end`` = the boundary with the next anchor cycle (day after THIS
    due date), or None when this is the latest anchored cycle.

    Neighbours are ordered by DUE DATE, not by ``created_at``: a statement can
    be ingested out of due order (a backfilled old cycle), and its true
    neighbour is the adjacent bill, not the adjacent ingestion. Only anchors (a
    parseable due date) take part; statements that share a due date share
    one window.
    """
    created_start = (upload.created_at or datetime.now(timezone.utc)).date()
    this_due = _due(upload.due_date)
    if this_due is None:
        # Not an anchor (e.g. a password-required upload): it owns no due-date
        # cycle. Fall back to the created_at-bounded window so its panel still
        # spans only its own ingestion range, not everything after it.
        later = (
            (
                await session.execute(
                    select(StatementUpload.created_at)
                    .where(
                        StatementUpload.account_id == upload.account_id,
                        StatementUpload.created_at > upload.created_at,
                    )
                    .order_by(StatementUpload.created_at.asc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        return CycleWindow(
            start=created_start, end=later.date() if later is not None else None
        )

    rows = (
        await session.execute(
            select(
                StatementUpload.id,
                StatementUpload.created_at,
                StatementUpload.due_date,
            ).where(StatementUpload.account_id == upload.account_id)
        )
    ).all()

    anchors: list[tuple[date, int, datetime | None]] = []
    for row in rows:
        due = _due(row.due_date)
        if due is not None:
            anchors.append((due, row.id, row.created_at))
    anchors.sort(key=lambda a: (a[0], a[1]))

    # A statement that shares this due date (a re-send, or an email summary
    # and its PDF) is the same cycle, not a neighbour.
    earlier = [a for a in anchors if a[0] < this_due]
    later = [a for a in anchors if a[0] > this_due]
    prev_due = earlier[-1][0] if earlier else None
    has_next = bool(later)
    next_created = later[0][2] if has_next else None

    # With no earlier bill, a tied group starts at its first ingestion, so a
    # payment between the summary and the later PDF still counts.
    group_created = min(
        (a[2] for a in anchors if a[0] == this_due and a[2] is not None),
        default=upload.created_at,
    )
    start = _boundary(prev_due, group_created) or created_start
    end = _boundary(this_due, next_created) if has_next else None
    return CycleWindow(start=start, end=end)


def transactions_in_cycle(window: CycleWindow):
    """SQLAlchemy predicate: a Transaction row falls in ``window``.

    A dated row is placed by its ``transaction_date``. A date-less row (unusual;
    digital alerts get a date filled from received_at upstream) is placed by its
    ``created_at``. Both edges are half-open so a row lands in exactly one cycle.
    """
    start_dt = datetime.combine(window.start, datetime.min.time(), tzinfo=timezone.utc)
    dated = Transaction.transaction_date >= window.start
    dateless = Transaction.transaction_date.is_(None) & (
        Transaction.created_at >= start_dt
    )
    if window.end is not None:
        end_dt = datetime.combine(window.end, datetime.min.time(), tzinfo=timezone.utc)
        dated = dated & (Transaction.transaction_date < window.end)
        dateless = dateless & (Transaction.created_at < end_dt)
    return dated | dateless
