"""Guarded delete of phantom statement rows."""

import json
from collections import Counter
from typing import NamedTuple

from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import (
    AuditAction,
    SmsMessage,
    Transaction,
)
from financial_dashboard.exceptions import StatementPreviewError
from financial_dashboard.schemas.transactions import (
    TransactionDeleteReport,
    TransactionDeleteResponse,
)
from financial_dashboard.services.duplicate_merge import (
    BatchRefused,
    move_references,
)
from financial_dashboard.services.statement_previews import (
    STATEMENT_UPLOADS,
    StatementRef,
    could_be_statement_candidate,
    parse_shortfall,
    reconcile_stored_statement,
    statement_candidate_index,
    statement_of,
)
from financial_dashboard.services.transaction_reads import transaction_read

_COLUMNS = tuple(attr.key for attr in inspect(Transaction).column_attrs)


def _snapshot(row: Transaction) -> dict[str, object]:
    """Return every column value of the row."""
    return {name: getattr(row, name) for name in _COLUMNS}


class _StatementCheck(NamedTuple):
    """What today's parse of one statement proves."""

    unheld: frozenset[int] = frozenset()
    error: str | None = None


class _Checks(NamedTuple):
    """Each statement's check, and each row as seen before the write lock."""

    statements: dict[StatementRef, _StatementCheck]
    rows: dict[int, dict[str, object]]


async def _unheld_by(
    session: AsyncSession, statement: StatementRef, requested: int
) -> _StatementCheck:
    """Return the rows today's parse of the statement no longer holds.

    A parse that finds no lines, or fewer lines than the stored parse less the
    requested rows, may have lost real lines. It maps to its error text, as
    does a failed parse.
    """
    try:
        stored = await reconcile_stored_statement(session, statement.kind, statement.id)
    except StatementPreviewError as exc:
        return _StatementCheck(error=str(exc))
    if stored is None:
        return _StatementCheck(error="statement not found")
    if shortfall := await parse_shortfall(
        session, statement, stored.preview, requested
    ):
        return _StatementCheck(error=shortfall)
    if stored.candidate_ids is None:
        return _StatementCheck(error="a statement line has too many candidate rows")
    return _StatementCheck(frozenset(stored.extra_ids) - stored.candidate_ids)


async def _check_statements(session: AsyncSession, ids: list[int]) -> _Checks:
    """Reparse each statement. Return what each proves, and each row as seen.

    A reparse imports every statement line that matches no row. Thus a row
    that today's parse holds must stay, or the next reparse imports it again.
    """
    rows = (
        await session.scalars(
            select(Transaction)
            .where(Transaction.id.in_(ids))
            .execution_options(populate_existing=True)
        )
    ).all()
    seen = {row.id: _snapshot(row) for row in rows}
    requested = Counter(s for row in rows if (s := statement_of(row)))
    statements = {s: await _unheld_by(session, s, n) for s, n in requested.items()}
    return _Checks(statements, seen)


def _names(entry: dict, row_id: int) -> bool:
    """Tell if a stored reconciliation entry could claim the row.

    An entry with a truncated candidate list could name any row.
    """
    return bool(
        entry.get("candidate_ids_truncated")
        or row_id in (entry.get("db_txn_id"), entry.get("imported_txn_id"))
        or row_id in (entry.get("candidate_transaction_ids") or [])
    )


async def _named_elsewhere(
    session: AsyncSession, row: Transaction, own: StatementRef
) -> list[str]:
    """Return each other statement whose stored reconciliation could claim the row.

    Its reparse would import a claimed line again once the row is gone. An
    unimported line names no row, so its date, amount and direction count.
    """
    names = []
    for kind, model in STATEMENT_UPLOADS.items():
        uploads = await session.scalars(
            select(model).where(
                model.account_id == row.account_id,
                model.reconciliation_data.is_not(None),
            )
        )
        for upload in uploads:
            recon = json.loads(upload.reconciliation_data or "{}")
            entries = recon.get("matched", []) + recon.get("missing", [])
            unresolved = [e for e in recon.get("missing", []) if not e.get("imported")]
            if (kind, upload.id) != own and (
                any(_names(entry, row.id) for entry in entries)
                or (
                    unresolved
                    and could_be_statement_candidate(
                        row, *statement_candidate_index(unresolved)
                    )
                )
            ):
                names.append(f"{kind} statement {upload.id}")
    return names


async def _refusal_reasons(
    session: AsyncSession,
    row: Transaction,
    checks: _Checks,
) -> list[str]:
    """Return every reason the row must stay. An empty list allows the delete."""
    if (statement := statement_of(row)) is None:
        return ["the row is not a statement import"]
    if checks.rows.get(row.id) != _snapshot(row):
        # SQLite can give a deleted id to a new row.
        return ["the row changed during the check; try again"]
    reasons = []
    if (
        row.sms_message_id
        or row.email_id
        or await session.scalar(
            select(SmsMessage.id).where(SmsMessage.transaction_id == row.id).limit(1)
        )
    ):
        reasons.append("the row owns an sms or email; a reparse would create it again")
    if row.attachment_path or row.note:
        reasons.append("the row carries an attachment or a note; check it by hand")
    reasons.extend(
        f"{name} could claim the row; its reparse would import it again"
        for name in await _named_elsewhere(session, row, statement)
    )
    check = checks.statements.get(
        statement, _StatementCheck(error="statement not checked")
    )
    name = f"{statement.kind} statement {statement.id}"
    if check.error:
        reasons.append(f"{name} cannot be checked: {check.error}")
    elif row.id not in check.unheld:
        reasons.append(f"today's parse of {name} still holds the row")
    return reasons


async def _delete_row(
    session: AsyncSession, row: Transaction, reason: str
) -> TransactionDeleteReport:
    """Clean the row's references, record it in the audit log, and delete it."""
    before = transaction_read(row)
    columns = _snapshot(row)
    cleaned = await move_references(session, row.id, None)
    session.add(
        AuditAction(
            action_type="delete_transaction",
            target_type="transaction",
            target_id=row.id,
            arguments_json=json.dumps({"reason": reason}),
            before_json=json.dumps(columns, default=str, sort_keys=True),
            status="applied",
        )
    )
    await session.delete(row)
    await session.flush()
    return TransactionDeleteReport(transaction=before, cleaned_references=cleaned)


async def delete_phantoms(
    session: AsyncSession, ids: list[int], *, reason: str, dry_run: bool
) -> TransactionDeleteResponse:
    """Delete phantom statement rows in one DB transaction.

    A phantom is a statement row that today's parse of its statement no
    longer holds. The function cleans every row that names it, writes one
    audit record per row, and deletes it. A card payment never qualifies: a
    cycle's paid sum counts alert rows only, and an alert row is refused.

    Args:
        session: A session with no open transaction.
        ids: The rows to delete.
        reason: Why the rows go. The audit record keeps it.
        dry_run: True rolls back every change after the report is built.

    Returns:
        The report of each delete.

    Raises:
        BatchRefused: A row fails a check. The function writes nothing.
    """
    # The parse runs before the write lock. A PDF parse can take seconds.
    checks = await _check_statements(session, ids)
    await session.rollback()
    reports = []
    refusals = []
    async with session.begin() as txn:
        if session.get_bind().dialect.name == "sqlite":
            await session.execute(text("BEGIN IMMEDIATE"))
        for txn_id in ids:
            if (row := await session.get(Transaction, txn_id)) is None:
                refusals.append({"id": txn_id, "reasons": ["transaction not found"]})
                continue
            if reasons := await _refusal_reasons(session, row, checks):
                refusals.append({"id": txn_id, "reasons": reasons})
                continue
            reports.append(await _delete_row(session, row, reason))
        if dry_run or refusals:
            await txn.rollback()
    if refusals:
        raise BatchRefused(refusals)
    return TransactionDeleteResponse(dry_run=dry_run, deletions=reports)
