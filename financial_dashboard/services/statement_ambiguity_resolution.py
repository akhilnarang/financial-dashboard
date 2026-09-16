"""Atomic resolution of a held-back CC statement row.

A card states two amounts for one purchase. It states the first amount when
it authorises the purchase. It states the second amount when the purchase
settles. The database holds one amount, and the statement holds the other.

The import cannot tell this from two purchases of a similar size. It
therefore holds the statement row back. See ``_reachable`` in
``services.statements.cc``. This module applies the answer of a person.

``merge`` writes the statement amount to the stored row. The statement states
what the bank billed. ``create_new`` imports the statement row as a new
transaction. Use it when the two rows are different purchases.

The resolver reads the decision from the stored ``reconciliation_data`` of
the upload. A button therefore stays valid after the process that sent it
stops.
"""

import datetime
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, NamedTuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Account, StatementUpload, Transaction
from financial_dashboard.services.statements.cc import (
    import_missing_cc_txns,
    parse_cc_amount,
    parse_cc_date,
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
    """Write the statement amount to the row that the person chose."""
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
    if target.direction != entry.get("direction"):
        raise StatementAmbiguityError("Merge target runs the other way")
    # A statement states rupees. A row in another currency holds a different
    # unit, so its amount and the statement amount are not comparable.
    if (target.currency or "INR") != "INR":
        raise StatementAmbiguityError("Merge target is in another currency")

    try:
        settled = parse_cc_amount(entry["amount"])
        stmt_date = parse_cc_date(entry["date"])
    except ValueError, InvalidOperation, KeyError:
        raise StatementAmbiguityError("Statement row is unreadable") from None

    stored = Decimal(str(target.amount))

    # The resolver reads the date and the amount again. It does not trust the
    # prompt. A row that changed after the prompt is not the row that the
    # person saw.
    if (
        target.transaction_date is None
        or abs((target.transaction_date - stmt_date).days) > 1
    ):
        raise StatementAmbiguityError("Merge target is outside the statement window")
    if not settles_within_band(settled, stored):
        raise StatementAmbiguityError("Amounts are too far apart to be one purchase")

    # Keeps the authorised amount. An alert for this purchase can arrive after
    # the statement, and that alert states the authorised amount.
    if target.authorised_amount is None:
        target.authorised_amount = stored

    target.amount = settled
    target.statement_upload_id = upload.id
    target.enriched_at = datetime.datetime.now(datetime.UTC)

    if narration := (entry.get("narration") or "").strip():
        target.raw_description = target.raw_description or narration

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
    """Import the held-back row as a new transaction.

    This function clears the hold and calls the real importer. It does not
    build the row again. The importer gets this entry only. No other row can
    therefore import as an effect of the answer.
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


@dataclass(frozen=True)
class _ParsedBank:
    """The one field that ``import_missing_cc_txns`` reads from a statement.

    Its docstring gives the contract: the importer uses ``parsed.bank`` only.
    A button does not parse the PDF again. The stored entry holds every other
    value that the row needs. If the importer reads a second field, this class
    fails immediately. It does not import a row with a missing value.
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

        # A second tap must not write a second time.
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

        # Adds to the count. It does not calculate the count again. A
        # reprocess moves an imported row from ``missing`` to ``matched``. A
        # count of ``missing`` alone would forget the earlier imported rows.
        if status == "created":
            upload.imported_count = (upload.imported_count or 0) + 1

        if upload.missing_count == 0:
            upload.status = "imported"
        elif upload.imported_count:
            upload.status = "partial_import"

        upload.reconciliation_data = reconciliation_to_json(recon)

        from financial_dashboard.services.statements.shared import emit_cc_snapshot

        await emit_cc_snapshot(session, upload)

    return StatementAmbiguityResult(status, txn_id, upload_id, stmt_idx)
