"""This module resolves statement rows that are near stored transaction amounts.

It asks users whether to merge the row into a stored purchase. It then updates the
database records with the chosen action.
"""

import datetime
import hashlib
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Literal, NamedTuple, NotRequired, TypedDict, cast

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

SettlementOutcome = Literal["merged", "stale"]

# The digest length that identifies a statement row in a callback.
ROW_DIGEST_CHARS = 12


class HeldRow(TypedDict):
    """A statement row in a reconciliation, as the settlement reads it."""

    stmt_idx: int
    date: str
    amount: str
    direction: str
    narration: str | None
    card_number: str | None
    imported: bool
    imported_txn_id: int | None
    ambiguous: NotRequired[bool]
    candidate_transaction_ids: NotRequired[list[int]]


class MatchedRow(TypedDict):
    """A statement row that the reconciler paired with a stored row."""

    stmt_idx: int
    db_txn_id: int


class Reconciliation(TypedDict):
    """The stored result of one statement's reconciliation."""

    matched: list[MatchedRow]
    missing: list[HeldRow]


class SettlementResult(NamedTuple):
    outcome: SettlementOutcome
    transaction_id: int | None


class SettlementError(Exception):
    """Raised when an answer names no row it can act on."""


def row_digest(entry: HeldRow) -> str:
    """Returns a stable hash of key statement row fields.

    Callbacks carry this value to detect changes to the row.
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


def held_rows(recon: Reconciliation) -> list[HeldRow]:
    """Returns the recorded statement rows that still wait for an answer."""
    return [
        entry
        for entry in recon.get("missing", [])
        if entry.get("ambiguous") and entry.get("imported_txn_id") is not None
    ]


def _find_row(recon: Reconciliation, stmt_idx: int, digest: str) -> HeldRow:
    """Finds the held statement row with the specified index and digest.

    Raises ``SettlementError`` if the row is missing or the digest changed.
    """
    for entry in held_rows(recon):
        if entry.get("stmt_idx") == stmt_idx:
            if row_digest(entry) != digest:
                raise SettlementError("The statement row changed")
            return entry
    raise SettlementError("That row no longer awaits an answer")


async def _lock_upload(session: AsyncSession, upload_id: int) -> StatementUpload | None:
    """Locks the statement upload record for update.

    This lock prevents concurrent tasks from modifying reconciliation data.
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


def _claimed(recon: Reconciliation) -> set[int]:
    """Returns the set of transaction IDs already used in this reconciliation.

    Settlement checks this set to prevent assigning the same transaction twice.
    """
    claimed = {
        txn_id
        for row in recon.get("matched", [])
        if (txn_id := row.get("db_txn_id")) is not None
    }
    claimed.update(
        txn_id
        for row in recon.get("missing", [])
        if (txn_id := row.get("imported_txn_id")) is not None
    )
    return claimed


def _merge(entry: HeldRow, target: Transaction, upload_id: int) -> None:
    """Updates a stored transaction with settled statement values after verifying rules.

    Raises ``SettlementError`` when the row is unreadable, or when the target fails a
    direction, currency, origin, date or amount check.
    Preserves user aliases when updating the counterparty name.
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

    narration = entry.get("narration")
    if narration and target.counterparty_source != "user_alias":
        target.counterparty = narration

    target.amount = settled
    target.transaction_date = row_date
    target.statement_upload_id = upload_id
    target.enriched_at = datetime.datetime.now(datetime.UTC)


async def answer(
    session: AsyncSession,
    upload_id: int,
    stmt_idx: int,
    digest: str,
    transaction_id: int,
) -> SettlementResult:
    """Folds a held statement row into the stored transaction the person chose.

    Deletes the statement's recorded copy of the row. Returns ``stale`` when the row
    changed or has an answer already.
    """
    async with session.begin():
        upload = await _lock_upload(session, upload_id)
        if upload is None or not upload.reconciliation_data:
            raise SettlementError("That statement is gone")

        recon = cast(
            Reconciliation, reconciliation_from_json(upload.reconciliation_data)
        )
        try:
            entry = _find_row(recon, stmt_idx, digest)
        except SettlementError:
            return SettlementResult("stale", None)

        recorded_id = entry.get("imported_txn_id")
        if recorded_id is None:
            return SettlementResult("stale", None)

        if transaction_id == recorded_id:
            raise SettlementError("That is the statement row itself")
        if transaction_id in _claimed(recon):
            raise SettlementError("Another row already uses that transaction")
        if transaction_id not in (entry.get("candidate_transaction_ids") or []):
            raise SettlementError("That row was not offered")

        recorded = await session.get(Transaction, recorded_id)
        if recorded is None:
            return SettlementResult("stale", None)

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

        _merge(entry, target, upload.id)

        await session.delete(recorded)
        entry["imported_txn_id"] = target.id
        entry["ambiguous"] = False
        upload.reconciliation_data = reconciliation_to_json(recon)

        if target.direction == "credit":
            await _resync_payment_state(session, upload)

        return SettlementResult("merged", target.id)


async def _resync_payment_state(session: AsyncSession, upload: StatementUpload) -> None:
    """Updates the credit card payment state after a merge changes a credit amount.

    Catches and logs any errors to avoid failing the settlement.
    """
    from financial_dashboard.services.reminders import resync_tracked_cc_payment_state

    try:
        await resync_tracked_cc_payment_state(session, upload)
    except Exception as exc:
        logger.warning("Settlement payment resync failed: %s", exc)


async def ask_about_held_rows(upload_id: int) -> None:
    """Sends resolution prompts for all held statement rows in an upload.

    Suppresses prompt errors because the statement import is already committed.
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

        recon = cast(
            Reconciliation, reconciliation_from_json(upload.reconciliation_data)
        )
        claimed = _claimed(recon)
        for entry in held_rows(recon):
            candidate_ids = [
                txn_id
                for txn_id in entry.get("candidate_transaction_ids") or []
                if txn_id not in claimed
            ]
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
                    "amount": entry["amount"],
                    "narration": entry["narration"],
                    "date": entry["date"],
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
