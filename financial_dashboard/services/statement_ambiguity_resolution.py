"""Atomic resolution of a held-back CC statement row.

A card states one amount when it authorises and another when it settles, so
the same purchase reaches the DB and the statement as two amounts. The
reconciler cannot tell that from two separate purchases of a similar size, so
it holds the statement row back rather than guess (see ``_reachable`` in
``services.statements.cc``). This module applies the answer a person gives.

``merge`` writes the statement's amount onto the stored row, because the
statement is the bank stating what it really billed. ``create_new`` imports
the statement row as its own transaction, for when the two really were
different purchases.

The decision is read back from the upload's stored ``reconciliation_data``,
so a button stays valid after the process that sent it has gone.
"""

import datetime
import logging
from decimal import Decimal, InvalidOperation
from typing import Literal, NamedTuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Account, StatementUpload, Transaction
from financial_dashboard.services.statements.cc import (
    import_missing_cc_txns,
    parse_cc_amount,
    reconciliation_from_json,
    reconciliation_to_json,
    settles_within_band,
)

logger = logging.getLogger(__name__)

StatementAmbiguityAction = Literal["merge", "create_new"]
ResolutionStatus = Literal["merged", "created", "already_resolved"]


class StatementAmbiguityError(Exception):
    """A stale or invalid statement-ambiguity request."""


class StatementAmbiguityResult(NamedTuple):
    status: ResolutionStatus
    transaction_id: int
    upload_id: int
    stmt_idx: int


async def _lock_upload(session: AsyncSession, upload_id: int) -> StatementUpload | None:
    """Take the upload write lock before reading resolution state."""
    if session.get_bind().dialect.name == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))
    return (
        await session.scalars(
            select(StatementUpload)
            .where(StatementUpload.id == upload_id)
            .with_for_update()
        )
    ).one_or_none()


def _find_entry(recon: dict, stmt_idx: int) -> dict:
    """The held-back entry this decision answers."""
    for entry in recon.get("missing", []):
        if entry.get("stmt_idx") == stmt_idx:
            return entry
    raise StatementAmbiguityError("Statement row no longer awaits a decision")


async def _resolve_merge(
    session: AsyncSession,
    upload: StatementUpload,
    entry: dict,
    transaction_id: int,
) -> int:
    """Write the statement's amount onto the row the person chose."""
    if transaction_id not in entry.get("candidate_transaction_ids", []):
        raise StatementAmbiguityError("Selected transaction is not a candidate")

    target = (
        await session.scalars(
            select(Transaction)
            .where(Transaction.id == transaction_id)
            .with_for_update()
        )
    ).one_or_none()
    if target is None:
        raise StatementAmbiguityError("Merge target no longer exists")
    if target.account_id != upload.account_id:
        raise StatementAmbiguityError("Merge target belongs to another account")
    if target.statement_upload_id is not None:
        raise StatementAmbiguityError("Merge target already came from a statement")

    try:
        settled = parse_cc_amount(entry["amount"])
    except ValueError, InvalidOperation, KeyError:
        raise StatementAmbiguityError("Statement amount is unreadable") from None

    stored = Decimal(str(target.amount))
    if target.direction != entry.get("direction"):
        raise StatementAmbiguityError("Merge target runs the other way")
    # The band is re-read here, not trusted from the prompt. A row that moved
    # since the prompt was sent is no longer the one the person was shown.
    if not settles_within_band(settled, stored):
        raise StatementAmbiguityError("Amounts are too far apart to be one purchase")

    target.amount = settled
    if narration := (entry.get("narration") or "").strip():
        target.raw_description = target.raw_description or narration
    target.statement_upload_id = upload.id
    target.enriched_at = datetime.datetime.now(datetime.UTC)

    entry["imported"] = True
    entry["imported_txn_id"] = target.id
    entry["ambiguous"] = False
    entry["import_error"] = None
    entry["resolution"] = "merged"
    return target.id


async def _resolve_create_new(
    session: AsyncSession,
    upload: StatementUpload,
    account: Account,
    recon: dict,
    entry: dict,
) -> int:
    """Import the held-back row as its own transaction.

    Clears the hold so the real importer runs, rather than rebuilding the row
    here. Only this entry is offered to it, so nothing else can import as a
    side effect of the answer.
    """
    entry["ambiguous"] = False
    entry["import_error"] = None
    one_row = {**recon, "missing": [entry]}

    parsed = _ParsedBank(bank=upload.bank)
    imported = await import_missing_cc_txns(session, upload, parsed, account, one_row)
    if not imported:
        raise StatementAmbiguityError(
            entry.get("import_error") or "Could not create a transaction"
        )
    entry["resolution"] = "created"
    return imported[0].id


class _ParsedBank(NamedTuple):
    """The one field ``import_missing_cc_txns`` reads off a parsed statement.

    Its docstring states the contract: ``parsed`` is used for ``parsed.bank``
    and nothing else. The PDF is not re-parsed to answer a button, because
    every other value the row needs is already in the stored entry. If the
    importer ever reads a second field, this breaks loudly rather than
    importing a row with a missing value.
    """

    bank: str


async def resolve_statement_ambiguity(
    session: AsyncSession,
    upload_id: int,
    stmt_idx: int,
    action: StatementAmbiguityAction,
    transaction_id: int | None = None,
) -> StatementAmbiguityResult:
    """Lock, revalidate, mutate, and commit one held-back statement row."""
    async with session.begin():
        upload = await _lock_upload(session, upload_id)
        if upload is None:
            raise StatementAmbiguityError("Statement not found")
        if not upload.reconciliation_data:
            raise StatementAmbiguityError("Statement has no reconciliation to resolve")

        recon = reconciliation_from_json(upload.reconciliation_data)
        entry = _find_entry(recon, stmt_idx)

        # A second tap, or a resolution made on the web, must not write twice.
        if entry.get("imported"):
            resolved_id = entry.get("imported_txn_id")
            if resolved_id is None:
                raise StatementAmbiguityError("Statement row is already resolved")
            return StatementAmbiguityResult(
                "already_resolved", int(resolved_id), upload_id, stmt_idx
            )
        if not entry.get("ambiguous"):
            raise StatementAmbiguityError("Statement row no longer awaits a decision")

        account = await session.get(Account, upload.account_id)
        if account is None:
            raise StatementAmbiguityError("Statement account no longer exists")

        if action == "merge":
            if transaction_id is None:
                raise StatementAmbiguityError("Merge target is required")
            txn_id = await _resolve_merge(session, upload, entry, transaction_id)
            status: ResolutionStatus = "merged"
        else:
            txn_id = await _resolve_create_new(session, upload, account, recon, entry)
            status = "created"

        upload.missing_count = sum(
            1 for row in recon["missing"] if not row.get("imported")
        )
        upload.imported_count = sum(
            1 for row in recon["missing"] if row.get("imported")
        )
        if upload.missing_count == 0:
            upload.status = "imported"
        upload.reconciliation_data = reconciliation_to_json(recon)

        from financial_dashboard.services.statements.shared import emit_cc_snapshot

        await emit_cc_snapshot(session, upload)

    return StatementAmbiguityResult(status, txn_id, upload_id, stmt_idx)
