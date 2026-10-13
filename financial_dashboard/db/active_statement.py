"""Keep exactly one active CC statement per account.

The active statement is the latest by due date. A due-date tie goes to the
newer upload. A statement without a parseable due date is never active.
"""

from dateutil import parser
from sqlalchemy import Connection, select, update

from financial_dashboard.db.models import StatementUpload


def set_active_statement(conn: Connection, account_id: int) -> int | None:
    """Mark the latest statement of ``account_id`` active and clear the rest.

    Args:
        conn: A sync connection inside the caller's transaction.
        account_id: The account to refresh.

    Returns:
        The id of the active statement, or None when no statement has a
        parseable due date.
    """
    ranked = []
    for row in conn.execute(
        select(StatementUpload.id, StatementUpload.due_date).where(
            StatementUpload.account_id == account_id
        )
    ):
        if not row.due_date:
            continue
        # dateutil directly: core.dates would import the db package (a cycle).
        try:
            due = parser.parse(row.due_date, dayfirst=True).date()
        except ValueError, OverflowError:
            continue
        ranked.append((due, row.id))
    active_id = max(ranked)[1] if ranked else None

    # Clear first: the unique index allows one active row at any moment.
    conn.execute(
        update(StatementUpload)
        .where(
            StatementUpload.account_id == account_id,
            StatementUpload.is_active.is_(True),
            StatementUpload.id != active_id,
        )
        .values(is_active=False)
    )
    if active_id is not None:
        conn.execute(
            update(StatementUpload)
            .where(StatementUpload.id == active_id)
            .values(is_active=True)
        )
    return active_id
