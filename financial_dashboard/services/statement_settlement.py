"""Resolution of a statement row whose amount is near a stored amount.

Only one answer is offered: the row is a separate purchase, so it is imported.
Answering "this settles that stored row" needs a record of which transaction
each statement already holds, and one upload's reconciliation is not that
record. Until there is one, a settlement is corrected on the statement page.

A card states two amounts for one purchase: the amount it authorises at the
purchase, and the amount it settles days later. A fuel surcharge of about 1% is
added at settlement, and a foreign purchase converts at the settlement-day
rate. So a statement row can be one purchase that settled, or a second purchase
of a similar size. Nothing in the two rows tells them apart, and a wrong answer
either loses a purchase or stores one twice. A person decides.

This module holds the answer and applies it. It writes nothing on its own.

The decision outlives its evidence. A reparse replaces the reconciliation of an
upload, and deleting an upload removes it. So the row is copied into
``StatementRowDecision`` when the question is asked, and the answer is written
there when it is given.
"""

import datetime
import hashlib
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Literal, NamedTuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import (
    Account,
    StatementRowDecision,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.statements.cc import (
    parse_cc_amount,
    parse_cc_date,
)

logger = logging.getLogger(__name__)

SettlementAction = Literal["create_new"]
DecisionStatus = Literal["pending", "merged", "created", "superseded"]


class SettlementError(Exception):
    """A stale or invalid settlement request."""


class SettlementResult(NamedTuple):
    status: DecisionStatus
    decision_id: int
    transaction_id: int | None


def parse_revision(recon: dict) -> str:
    """A token for one parse of one statement.

    ``stmt_idx`` is a position within a parse, so an answer that names one must
    also name its parse. Equal rows in equal order give an equal token.

    sha256, not ``hash()``: the token is stored and read back after a restart,
    and ``hash()`` is salted per process.
    """
    rows = [
        (entry.get("stmt_idx"), *_row_identity(entry))
        for entry in [*recon.get("matched", []), *recon.get("missing", [])]
    ]
    rows.sort(key=lambda r: (r[0] is None, r[0]))
    payload = json.dumps(rows, default=str, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


async def _lock_upload(session: AsyncSession, upload_id: int) -> StatementUpload | None:
    """Take the write lock before reading what a decision depends on.

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


def _row_identity(entry: dict) -> tuple:
    """What a row states, for the revision token."""
    return (
        entry.get("date"),
        entry.get("amount"),
        entry.get("direction"),
        entry.get("narration") or "",
        entry.get("card_number") or "",
    )


async def _carry_answers(
    session: AsyncSession,
    upload: StatementUpload,
    recon: dict,
    revision: str,
) -> None:
    """Copy this parse's answers onto a reconciliation about to replace them.

    A writer computes its reconciliation outside the lock, so a person can
    answer while it runs. Writing it would return the row to unanswered while
    its transaction remains.

    The answers come from the decision table, scoped to this revision, so an
    ordinal means the same row. An answer whose transaction is gone is skipped:
    the row asks again, which is how a wrong answer is undone.
    """
    answered = (
        await session.scalars(
            select(StatementRowDecision).where(
                StatementRowDecision.statement_upload_id == upload.id,
                StatementRowDecision.parse_revision == revision,
                StatementRowDecision.status.in_(("merged", "created")),
                StatementRowDecision.transaction_id.is_not(None),
            )
        )
    ).all()
    if not answered:
        return

    live = set(
        (
            await session.scalars(
                select(Transaction.id).where(
                    Transaction.id.in_([d.transaction_id for d in answered])
                )
            )
        ).all()
    )
    by_idx = {d.stmt_idx: d for d in answered if d.transaction_id in live}

    for entry in recon.get("missing", []):
        if prior := by_idx.get(entry.get("stmt_idx")):
            entry["imported"] = True
            entry["imported_txn_id"] = prior.transaction_id
            entry["ambiguous"] = False
            entry["import_error"] = None


async def record_pending_decisions(
    session: AsyncSession,
    upload: StatementUpload,
    recon: dict,
) -> list[StatementRowDecision]:
    """Write a pending decision for each held row, and retire the old ones.

    An answered decision keeps its answer. A pending one from an earlier parse
    is superseded: its ordinal no longer names the same row.
    """
    revision = parse_revision(recon)
    await _carry_answers(session, upload, recon, revision)

    stale = (
        await session.scalars(
            select(StatementRowDecision).where(
                StatementRowDecision.statement_upload_id == upload.id,
                StatementRowDecision.parse_revision != revision,
                StatementRowDecision.status == "pending",
            )
        )
    ).all()
    for decision in stale:
        decision.status = "superseded"

    existing = {
        d.stmt_idx: d
        for d in (
            await session.scalars(
                select(StatementRowDecision).where(
                    StatementRowDecision.statement_upload_id == upload.id,
                    StatementRowDecision.parse_revision == revision,
                )
            )
        ).all()
    }

    groups = {
        idx: group
        for group in recon.get("settlement_groups", [])
        for idx in group["held_idxs"]
    }

    # A row can stop asking without the parse changing. Its question retires,
    # or a tap on the old prompt stores the purchase again.
    for idx, decision in existing.items():
        if decision.status == "pending" and idx not in groups:
            decision.status = "superseded"

    created: list[StatementRowDecision] = []
    for entry in recon.get("missing", []):
        idx = entry.get("stmt_idx")
        if idx not in groups or entry.get("imported"):
            continue
        if (prior := existing.get(idx)) is not None:
            # The row is held and unanswered, so its question is live again. A
            # resolved decision reaching here means its transaction is gone,
            # which is how a wrong answer is undone.
            if prior.status != "pending":
                prior.status = "pending"
                prior.transaction_id = None
            # Only the candidates can change without changing the token.
            prior.candidate_txn_ids = json.dumps(groups[idx]["txn_ids"])
            created.append(prior)
            continue

        decision = StatementRowDecision(
            statement_upload_id=upload.id,
            parse_revision=revision,
            stmt_idx=idx,
            row_date=entry.get("date"),
            row_amount=str(entry.get("amount") or ""),
            row_direction=entry.get("direction"),
            row_narration=entry.get("narration"),
            row_card_number=entry.get("card_number"),
            candidate_txn_ids=json.dumps(groups[idx]["txn_ids"]),
            status="pending",
        )
        session.add(decision)
        created.append(decision)

    # The caller counted before the carry ran, so count again.
    upload.missing_count = sum(
        1 for entry in recon.get("missing", []) if not entry.get("imported")
    )
    if upload.missing_count == 0 and upload.status == "partial_import":
        upload.status = "imported"

    await session.flush()
    return created


def _record_answer_on_upload(
    upload: StatementUpload,
    decision: StatementRowDecision,
    txn_id: int,
) -> None:
    """Mark the answered row resolved in the upload's reconciliation.

    The statement page reads that JSON.
    """
    from financial_dashboard.services.statements.cc import (
        reconciliation_from_json,
        reconciliation_to_json,
    )

    if not upload.reconciliation_data:
        return

    recon = reconciliation_from_json(upload.reconciliation_data)
    for entry in recon.get("missing", []):
        if entry.get("stmt_idx") == decision.stmt_idx:
            entry["imported"] = True
            entry["imported_txn_id"] = txn_id
            entry["ambiguous"] = False
            entry["import_error"] = None
            break

    upload.missing_count = sum(
        1 for entry in recon.get("missing", []) if not entry.get("imported")
    )
    if decision.status == "created":
        upload.imported_count = (upload.imported_count or 0) + 1
    if upload.missing_count == 0:
        upload.status = "imported"
    elif upload.imported_count:
        upload.status = "partial_import"
    upload.reconciliation_data = reconciliation_to_json(recon)


def _claimed_transaction_ids(
    upload: StatementUpload,
    decision: StatementRowDecision,
) -> set[int] | None:
    """The transactions this statement already claims.

    A transaction answers one statement row, so a claimed one is not offered
    again.

    ``None`` means the question is dead: the parse changed, or the row is
    imported or matched now. The caller retires the decision.
    """
    from financial_dashboard.services.statements.cc import reconciliation_from_json

    if not upload.reconciliation_data:
        return None

    recon = reconciliation_from_json(upload.reconciliation_data)
    if parse_revision(recon) != decision.parse_revision:
        return None

    entry = next(
        (
            row
            for row in recon.get("missing", [])
            if row.get("stmt_idx") == decision.stmt_idx
        ),
        None,
    )
    if entry is None or entry.get("imported") or not entry.get("ambiguous"):
        return None

    # A matched row holds the transaction it won, and an answered row holds the
    # one it was given. Offering either again puts two purchases on one row.
    claimed = {
        row["db_txn_id"]
        for row in recon.get("matched", [])
        if row.get("db_txn_id") is not None
    }
    claimed.update(
        row["imported_txn_id"]
        for row in recon.get("missing", [])
        if row.get("stmt_idx") != decision.stmt_idx
        and row.get("imported_txn_id") is not None
    )
    return claimed


async def resolve_settlement(
    session: AsyncSession,
    decision_id: int,
    action: SettlementAction = "create_new",
) -> SettlementResult:
    """Lock, read, revalidate, apply and commit one answer."""
    async with session.begin():
        decision = await session.get(StatementRowDecision, decision_id)
        if decision is None:
            raise SettlementError("Decision not found")
        if decision.statement_upload_id is None:
            raise SettlementError("The statement of this decision is gone")

        upload = await _lock_upload(session, decision.statement_upload_id)
        if upload is None:
            raise SettlementError("Statement not found")

        # Re-read after the lock: another writer may have answered this.
        await session.refresh(decision)
        if decision.status != "pending":
            status: DecisionStatus = (
                "merged"
                if decision.status == "merged"
                else ("created" if decision.status == "created" else "superseded")
            )
            return SettlementResult(status, decision.id, decision.transaction_id)

        account = await session.get(Account, upload.account_id)
        if account is None:
            raise SettlementError("Statement account no longer exists")

        # Read the stored reconciliation again: the row can stop asking while
        # the prompt waits, and answering then stores the purchase twice.
        if _claimed_transaction_ids(upload, decision) is None:
            decision.status = "superseded"
            return SettlementResult("superseded", decision.id, None)

        txn_id = await _apply_create_new(session, upload, account, decision)
        _record_answer_on_upload(upload, decision, txn_id)
        return SettlementResult("created", decision.id, txn_id)


async def _apply_create_new(
    session: AsyncSession,
    upload: StatementUpload,
    account: Account,
    decision: StatementRowDecision,
) -> int:
    """Store the statement row as a transaction of its own."""
    from financial_dashboard.services.linker import (
        build_link_context,
        link_transaction,
    )
    from financial_dashboard.services.statements.cc import resolve_cc_card_mask

    try:
        amount = parse_cc_amount(decision.row_amount or "")
        txn_date = parse_cc_date(decision.row_date or "")
    except ValueError, InvalidOperation:
        raise SettlementError("Statement row is unreadable") from None

    txn = Transaction(
        statement_upload_id=upload.id,
        account_id=upload.account_id,
        bank=upload.bank,
        email_type="cc_statement",
        direction=decision.row_direction or "debit",
        amount=amount,
        currency="INR",
        transaction_date=txn_date,
        counterparty=decision.row_narration,
        # The row's own card. A statement can bill several cards.
        card_mask=await resolve_cc_card_mask(
            session, account, decision.row_card_number or upload.card_number
        ),
        channel="cc_statement",
        raw_description=decision.row_narration,
    )
    session.add(txn)
    await session.flush()
    link_transaction(await build_link_context(session), txn)
    await session.flush()

    decision.status = "created"
    decision.transaction_id = txn.id
    decision.resulting_amount = amount
    decision.reason = "a separate purchase, not a settlement"
    decision.resolved_at = datetime.datetime.now(datetime.UTC)
    return txn.id


async def notify_pending_decisions(upload_id: int) -> None:
    """Ask every question of this upload that still waits.

    Reads the table, not a caller's copy, so an answered question is not asked
    again. Best effort: the import is committed and a failed prompt must not
    fail it.
    """
    try:
        await _send_pending_prompts(upload_id)
    except Exception as exc:
        logger.warning("Settlement prompt failed: %s", exc)


async def _send_pending_prompts(upload_id: int) -> None:
    """Send one prompt for each pending question."""
    from financial_dashboard.db import async_session
    from financial_dashboard.services.settings import get_telegram_chat_id
    from financial_dashboard.services.telegram import send_settlement_prompt

    chat_id = get_telegram_chat_id()
    if not chat_id:
        return

    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        if upload is None:
            return

        pending = (
            await session.scalars(
                select(StatementRowDecision).where(
                    StatementRowDecision.statement_upload_id == upload_id,
                    StatementRowDecision.status == "pending",
                )
            )
        ).all()
        if not pending:
            return

        for decision in pending:
            # A claimed transaction is refused at the write.
            claimed = _claimed_transaction_ids(upload, decision) or set()
            candidate_ids = [
                txn_id
                for txn_id in json.loads(decision.candidate_txn_ids or "[]")
                if txn_id not in claimed
            ]
            rows = (
                await session.scalars(
                    select(Transaction).where(Transaction.id.in_(candidate_ids))
                )
            ).all()

            payload = {
                "decision_id": decision.id,
                "bank": upload.bank,
                "amount": decision.row_amount,
                "narration": decision.row_narration,
                "date": decision.row_date,
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
            }

            try:
                await send_settlement_prompt(payload, chat_id)
            except Exception as exc:
                logger.warning("Settlement prompt failed: %s", exc)
