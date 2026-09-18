"""Answer a statement row whose amount is near a stored amount.

A card states two amounts for one purchase: the amount it authorises at the
swipe, and the amount it settles days later. A fuel surcharge is added at
settlement, and a foreign charge converts at the settlement-day rate. So a
statement row near a stored amount is one purchase billed once, or two
purchases of a similar size. Nothing in the rows tells them apart.

The reconciler holds such a row back. This module asks about it, and applies
the answer:

- **Merge** writes the settled amount onto the stored row.
- **Skip** leaves both rows alone. The statement row stays held.

The question needs no stored state. The callback carries what it is about, and
a digest of the row as it was shown, so a row that changed since is refused.
Everything else is read again at the write.
"""

import datetime
import hashlib
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Literal, NamedTuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import StatementUpload, Transaction
from financial_dashboard.services.statements.cc import (
    parse_cc_amount,
    parse_cc_date,
    reconciliation_from_json,
    reconciliation_to_json,
    settles_within_band,
)

logger = logging.getLogger(__name__)

SettlementAction = Literal["merge", "skip"]
SettlementOutcome = Literal["merged", "skipped", "stale", "done"]

ROW_DIGEST_CHARS = 12
"""Enough of the digest to name one row of one statement in a callback."""


class SettlementResult(NamedTuple):
    outcome: SettlementOutcome
    transaction_id: int | None


class SettlementError(Exception):
    """A request that names no answerable row."""


def row_digest(entry: dict) -> str:
    """A short digest of what a statement row states.

    The callback carries this, so a row that changed after the prompt is
    refused rather than answered. sha256, not ``hash()``: the digest travels to
    Telegram and comes back after a restart, and ``hash()`` is salted per
    process.
    """
    payload = json.dumps(
        [
            entry.get("date"),
            entry.get("amount"),
            entry.get("direction"),
            entry.get("narration") or "",
            entry.get("card_number") or "",
        ],
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:ROW_DIGEST_CHARS]


def held_rows(recon: dict) -> list[dict]:
    """The rows of this reconciliation that await an answer."""
    return [
        entry
        for entry in recon.get("missing", [])
        if entry.get("ambiguous") and not entry.get("imported")
    ]


def _find_row(recon: dict, stmt_idx: int, digest: str) -> dict:
    """The held row this answer is about, or raise.

    The digest must match: a reparse can move a row to this position, and that
    row is not the one the prompt showed.
    """
    for entry in held_rows(recon):
        if entry.get("stmt_idx") == stmt_idx:
            if row_digest(entry) != digest:
                raise SettlementError("The statement row changed")
            return entry
    raise SettlementError("That row no longer awaits an answer")


async def _lock_upload(session: AsyncSession, upload_id: int) -> StatementUpload | None:
    """Take the write lock before reading what the answer depends on.

    A reparse and an answer can run at once, so a read outside the lock can
    return state the other writer has replaced.
    """
    if session.get_bind().dialect.name == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))
    return (
        await session.scalars(
            select(StatementUpload)
            .where(StatementUpload.id == upload_id)
            .with_for_update()
        )
    ).one_or_none()


def _claimed(recon: dict) -> set[int]:
    """The transactions this statement already speaks for.

    A transaction answers one statement row. A matched row holds the one it
    won, and an answered row holds the one it took.
    """
    claimed = {
        row["db_txn_id"]
        for row in recon.get("matched", [])
        if row.get("db_txn_id") is not None
    }
    claimed.update(
        row["imported_txn_id"]
        for row in recon.get("missing", [])
        if row.get("imported_txn_id") is not None
    )
    return claimed


def _merge(entry: dict, target: Transaction) -> None:
    """Write the settled amount onto the stored row.

    Every rule is read again here. The prompt is a question, and a row that
    changed after it was sent is not the row the person saw.
    """
    try:
        settled = parse_cc_amount(entry["amount"])
        row_date = parse_cc_date(entry["date"])
    except ValueError, InvalidOperation, KeyError:
        raise SettlementError("The statement row is unreadable") from None

    stored = Decimal(str(target.amount))

    if target.direction != entry.get("direction"):
        raise SettlementError("That row runs the other way")
    if (target.currency or "INR") != "INR":
        raise SettlementError("That row is in another currency")
    if target.statement_upload_id is not None:
        raise SettlementError("That row already came from a statement")
    if (
        target.transaction_date is None
        or abs((target.transaction_date - row_date).days) > 1
    ):
        raise SettlementError("That row is outside the statement window")
    if not settles_within_band(settled, stored):
        raise SettlementError("The amounts are too far apart")

    target.amount = settled
    target.enriched_at = datetime.datetime.now(datetime.UTC)


async def answer(
    session: AsyncSession,
    upload_id: int,
    stmt_idx: int,
    digest: str,
    action: SettlementAction,
    transaction_id: int | None = None,
) -> SettlementResult:
    """Apply one answer, under the write lock.

    Returns ``stale`` when the row no longer asks, so a second tap of an old
    prompt reports that rather than writing again.
    """
    async with session.begin():
        upload = await _lock_upload(session, upload_id)
        if upload is None or not upload.reconciliation_data:
            raise SettlementError("That statement is gone")

        recon = reconciliation_from_json(upload.reconciliation_data)
        try:
            entry = _find_row(recon, stmt_idx, digest)
        except SettlementError:
            return SettlementResult("stale", None)

        if action == "skip":
            entry["ambiguous"] = False
            entry["import_error"] = "a separate purchase, left alone"
            upload.reconciliation_data = reconciliation_to_json(recon)
            return SettlementResult("skipped", None)

        if transaction_id is None:
            raise SettlementError("No row was chosen")
        if transaction_id in _claimed(recon):
            raise SettlementError("Another row already uses that transaction")
        if transaction_id not in (entry.get("candidate_transaction_ids") or []):
            raise SettlementError("That row was not offered")

        target = (
            await session.scalars(
                select(Transaction)
                .where(Transaction.id == transaction_id)
                .with_for_update()
            )
        ).one_or_none()
        if target is None:
            raise SettlementError("That row is gone")
        if target.account_id != upload.account_id:
            raise SettlementError("That row belongs to another account")

        _merge(entry, target)

        entry["imported"] = True
        entry["imported_txn_id"] = target.id
        entry["ambiguous"] = False
        entry["import_error"] = None
        upload.missing_count = sum(
            1 for row in recon.get("missing", []) if not row.get("imported")
        )
        if upload.missing_count == 0:
            upload.status = "imported"
        upload.reconciliation_data = reconciliation_to_json(recon)

        return SettlementResult("merged", target.id)


async def ask_about_held_rows(upload_id: int) -> None:
    """Send one prompt for each row of this statement that awaits an answer.

    Best effort: the import is already committed and the rows are on the
    statement page, so a prompt that fails must not fail the import.
    """
    try:
        await _send_prompts(upload_id)
    except Exception as exc:
        logger.warning("Settlement prompt failed: %s", exc)


async def _send_prompts(upload_id: int) -> None:
    from financial_dashboard.db import async_session
    from financial_dashboard.services.settings import get_telegram_chat_id
    from financial_dashboard.services.telegram import send_settlement_prompt

    chat_id = get_telegram_chat_id()
    if not chat_id:
        return

    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        if upload is None or not upload.reconciliation_data:
            return

        recon = reconciliation_from_json(upload.reconciliation_data)
        for entry in held_rows(recon):
            candidate_ids = entry.get("candidate_transaction_ids") or []
            rows = (
                await session.scalars(
                    select(Transaction).where(Transaction.id.in_(candidate_ids))
                )
            ).all()

            await send_settlement_prompt(
                {
                    "upload_id": upload_id,
                    "stmt_idx": entry["stmt_idx"],
                    "digest": row_digest(entry),
                    "bank": upload.bank,
                    "amount": entry.get("amount"),
                    "narration": entry.get("narration"),
                    "date": entry.get("date"),
                    "candidates": [
                        {
                            "id": row.id,
                            "amount": f"{Decimal(str(row.amount)):,.2f}",
                            "counterparty": row.counterparty,
                            "date": (
                                row.transaction_date.strftime("%d/%m/%Y")
                                if row.transaction_date
                                else None
                            ),
                            "card_mask": row.card_mask,
                        }
                        for row in rows
                    ],
                },
                chat_id,
            )
