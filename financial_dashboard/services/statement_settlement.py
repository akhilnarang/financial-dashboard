"""This module resolves statement rows that are near stored transaction amounts.

It asks users whether to merge the row into a stored purchase. It then updates the
database records with the chosen action.
"""

import hashlib
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Literal, NamedTuple, NotRequired, TypedDict, cast

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import StatementUpload, Transaction
from financial_dashboard.services.assistant.message_context import (
    move_transaction_references,
)
from financial_dashboard.services.categorization.decision_lifecycle import (
    supersede_active_decisions,
)
from financial_dashboard.services.categorization.engine import requeue_after_enrichment
from financial_dashboard.services.statements.cc import (
    parse_cc_amount,
    parse_cc_date,
    reconciliation_from_json,
    reconciliation_to_json,
    statement_card_holder,
)

logger = logging.getLogger(__name__)


class HeldRow(TypedDict):
    """A statement row in a reconciliation, as the settlement reads it."""

    stmt_idx: int
    date: str
    amount: str
    direction: str
    narration: str | None
    card_number: str | None
    person: NotRequired[str | None]
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
    outcome: Literal["merged", "stale"]
    transaction_id: int | None


class SettlementError(Exception):
    """Raised when an answer names no row it can act on."""


def row_digest(entry: HeldRow) -> str:
    """Returns a stable hash of key statement row fields.

    Callbacks carry this value to detect changes to the row.
    """
    payload = json.dumps(
        [
            entry["date"],
            entry["amount"],
            entry["direction"],
            entry["narration"] or "",
            entry["card_number"] or "",
            entry.get("person") or "",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


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
        if entry["stmt_idx"] == stmt_idx:
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


def _merge(
    entry: HeldRow, target: Transaction, upload_id: int, holder: str | None
) -> None:
    """Writes the statement's amount, date, merchant and cardholder onto the stored row.

    The reconciler applied every pairing rule when it offered this row, so the
    fold does not check them again.

    Raises ``SettlementError`` when the statement row is unreadable.
    """
    try:
        target.amount = parse_cc_amount(entry["amount"])
        target.transaction_date = parse_cc_date(entry["date"])
    except ValueError, InvalidOperation:
        raise SettlementError("The statement row is unreadable") from None

    if entry["narration"]:
        target.counterparty = entry["narration"]
    if holder:
        target.card_holder = holder
    target.statement_upload_id = upload_id


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

        if transaction_id in _claimed(recon):
            raise SettlementError("Another row already uses that transaction")
        if transaction_id not in (entry.get("candidate_transaction_ids") or []):
            raise SettlementError("That row was not offered")

        recorded = await session.get(Transaction, entry["imported_txn_id"])
        target = await session.get(Transaction, transaction_id)
        if recorded is None or target is None:
            return SettlementResult("stale", None)

        holder = await statement_card_holder(
            session, target.account_id, entry.get("person")
        )
        _merge(entry, target, upload.id, holder)
        if target.attachment_path is None:
            target.attachment_path = recorded.attachment_path
        elif recorded.attachment_path is not None:
            logger.warning(
                "Settlement dropped receipt %s of transaction %s",
                recorded.attachment_path,
                recorded.id,
            )
        await supersede_active_decisions(session, recorded.id)
        await move_transaction_references(session, recorded.id, target.id)
        await requeue_after_enrichment(session, target)

        await session.delete(recorded)
        entry["imported_txn_id"] = target.id
        entry["ambiguous"] = False
        upload.reconciliation_data = reconciliation_to_json(recon)
        return SettlementResult("merged", target.id)


async def ask_about_held_rows(upload_id: int) -> None:
    """Sends one Telegram prompt for each held statement row in an upload.

    The import is already committed, so a failed prompt is logged, not raised.
    """
    from financial_dashboard.db import async_session
    from financial_dashboard.services.settings import get_telegram_chat_id
    from financial_dashboard.services.telegram import send_settlement_prompt

    chat_id = get_telegram_chat_id()
    if not chat_id:
        return

    try:
        async with async_session() as session:
            upload = await session.get(StatementUpload, upload_id)
            if upload is None or not upload.reconciliation_data:
                return

            recon = cast(
                Reconciliation, reconciliation_from_json(upload.reconciliation_data)
            )
            for entry in held_rows(recon):
                rows = (
                    await session.scalars(
                        select(Transaction).where(
                            Transaction.id.in_(
                                entry.get("candidate_transaction_ids") or []
                            )
                        )
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
    except Exception as exc:
        logger.warning("Settlement prompt failed: %s", exc)
