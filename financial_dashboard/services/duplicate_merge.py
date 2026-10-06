"""Explicit merge of a duplicate transaction into the row that stays."""

import json
from datetime import timedelta
from typing import cast

from sqlalchemy import CursorResult, inspect, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import (
    AuditAction,
    AuditInteraction,
    BankStatementUpload,
    CategoryReviewDecision,
    PaymentStatus,
    SmsMessage,
    StatementUpload,
    Transaction,
)
from financial_dashboard.schemas.transactions import (
    CcPaymentStateChange,
    TransactionMergePair,
    TransactionMergeReport,
    TransactionMergeResponse,
)
from financial_dashboard.services.assistant.message_context import (
    move_transaction_references,
)
from financial_dashboard.services.categorization.decision_lifecycle import (
    supersede_active_decisions,
)
from financial_dashboard.services.cc_cycle import cc_cycle_window
from financial_dashboard.services.cc_disambiguation import (
    is_cc_payment_received_email,
)
from financial_dashboard.services.reminders import (
    _latest_active_cycle,
    resync_tracked_cc_payment_state,
)
from financial_dashboard.services.transaction_reads import _transaction_read
from financial_dashboard.services.txn_merge import (
    _normalized_currency,
    _quantize_balance,
    _shortened_reference_match,
    apply_transaction_enrichment,
)

# The statement reconcilers match rows up to one day apart. A wider window
# lets a statement reprocess import the merged duplicate again.
_DATE_WINDOW = timedelta(days=1)
_COLUMNS = tuple(attr.key for attr in inspect(Transaction).column_attrs)
_REFERRING_MODELS = (
    SmsMessage,
    AuditInteraction,
    AuditAction,
    CategoryReviewDecision,
)
_LINK_FIELDS = (
    "email_id",
    "statement_upload_id",
    "bank_statement_upload_id",
    "card_id",
    "attachment_path",
)
_CATEGORY_FIELDS = (
    "category",
    "category_method",
    "category_confidence",
    "category_model",
    "category_input_hash",
    "category_vocab_version",
    "categorized_at",
    "review_status",
    "review_reason",
)


class PairRefused(Exception):
    """One pair fails a check that needs the DB."""


class MergeRefused(Exception):
    """One or more pairs fail the merge checks. Nothing is written."""

    def __init__(self, refusals: list[dict]) -> None:
        super().__init__("merge refused")
        self.refusals = refusals


def _normalized_ref(ref: str | None) -> str:
    """Return the reference without spaces or leading zeros."""
    return (ref or "").strip().lstrip("0")


def _refs_prove_distinct(first: str | None, second: str | None) -> bool:
    """Tell if two references prove two different events.

    Both must be set and differ. A shortened copy of the other proves nothing.
    """
    a, b = _normalized_ref(first), _normalized_ref(second)
    return bool(a and b) and a != b and not _shortened_reference_match(a, b)


def _is_cc_payment(row: Transaction) -> bool:
    """Tell if the row counts toward a card bill's paid sum."""
    return row.direction == "credit" and is_cc_payment_received_email(row.email_type)


def _from_statement(row: Transaction | dict) -> bool:
    if isinstance(row, dict):
        return bool(row["statement_upload_id"] or row["bank_statement_upload_id"])
    return bool(row.statement_upload_id or row.bank_statement_upload_id)


def _surviving_reference(keep: Transaction, data: dict) -> str | None:
    """Return the reference the keeper keeps.

    A statement reprocess pairs rows a day apart only on an equal reference.
    Thus a statement row's reference must survive the merge. Otherwise the
    longer form of the reference survives.
    """
    ref = data["reference_number"]
    if not ref or _from_statement(keep):
        return keep.reference_number
    if _from_statement(data) or len(_normalized_ref(ref)) > len(
        _normalized_ref(keep.reference_number)
    ):
        return ref
    return keep.reference_number


def _owns_other(keep: Transaction, dup: Transaction, slot: str) -> bool:
    """Tell if each row owns a different source in ``slot``.

    The merge keeps one source per slot. A reparse of the lost source would
    insert the row again.
    """
    owners = {getattr(keep, slot), getattr(dup, slot)}
    return None not in owners and len(owners) == 2


def _refusal_reasons(keep: Transaction, dup: Transaction) -> list[str]:
    """Return every reason the pair must not merge. An empty list allows it."""
    days_apart = (
        abs(keep.transaction_date - dup.transaction_date)
        if keep.transaction_date and dup.transaction_date
        else None
    )
    balances = {_quantize_balance(keep.balance), _quantize_balance(dup.balance)}
    checks = [
        (
            keep.account_id is None or keep.account_id != dup.account_id,
            "rows are not on the same account",
        ),
        (keep.direction != dup.direction, "direction differs"),
        (
            _normalized_currency(keep.currency) != _normalized_currency(dup.currency),
            "currency differs",
        ),
        (keep.amount != dup.amount, "amount differs"),
        (
            days_apart is None or days_apart > _DATE_WINDOW,
            "dates are more than 1 day apart",
        ),
        (None not in balances and len(balances) == 2, "balances differ"),
        (
            _refs_prove_distinct(keep.reference_number, dup.reference_number),
            "references prove distinct events",
        ),
        (
            _is_cc_payment(dup) and not _is_cc_payment(keep),
            "the duplicate counts as a card payment; keep it instead",
        ),
        (_owns_other(keep, dup, "sms_message_id"), "both rows own a different sms"),
        (_owns_other(keep, dup, "email_id"), "both rows own a different email"),
        (
            _from_statement(keep)
            and _from_statement(dup)
            and None not in (keep.reference_number, dup.reference_number)
            and keep.reference_number != dup.reference_number,
            "both statement rows carry a different reference",
        ),
    ]
    return [reason for failed, reason in checks if failed]


def _repoint_reconciliation(recon: dict, old_id: int, new_id: int) -> bool:
    """Replace ``old_id`` with ``new_id`` in stored statement reconciliation data.

    Return whether anything changed.
    """
    changed = False
    for entry in recon.get("matched", []) + recon.get("missing", []):
        for key in ("db_txn_id", "imported_txn_id"):
            if entry.get(key) == old_id:
                entry[key] = new_id
                changed = True
        candidates = entry.get("candidate_transaction_ids") or []
        if old_id in candidates:
            entry["candidate_transaction_ids"] = list(
                dict.fromkeys(new_id if i == old_id else i for i in candidates)
            )
            changed = True
    return changed


async def _move_references(
    session: AsyncSession, old_id: int, new_id: int
) -> dict[str, int]:
    """Point every row that names ``old_id`` at ``new_id``.

    Return the moved row count per table. Telegram rows move too but are not
    counted.
    """
    moved = {}
    await supersede_active_decisions(session, old_id)
    await move_transaction_references(session, old_id, new_id)
    for model in _REFERRING_MODELS:
        result = await session.execute(
            update(model)
            .where(model.transaction_id == old_id)
            .values(transaction_id=new_id)
        )
        if rowcount := cast(CursorResult, result).rowcount:
            moved[model.__tablename__] = rowcount

    await session.execute(
        update(AuditAction)
        .where(
            AuditAction.target_type == "transaction", AuditAction.target_id == old_id
        )
        .values(target_id=new_id)
    )

    for model in (StatementUpload, BankStatementUpload):
        uploads = await session.scalars(
            select(model).where(model.reconciliation_data.contains(str(old_id)))
        )
        for upload in uploads:
            assert upload.reconciliation_data is not None
            recon = json.loads(upload.reconciliation_data)
            if _repoint_reconciliation(recon, old_id, new_id):
                upload.reconciliation_data = json.dumps(recon)
                key = f"{model.__tablename__}.reconciliation_data"
                moved[key] = moved.get(key, 0) + 1
    return moved


async def _fold_manual_fields(
    session: AsyncSession, keep: Transaction, data: dict
) -> list[str]:
    """Copy a manual category and note the keeper lacks. Return the conflicts."""
    conflicts = []
    if data["category_method"] == "manual":
        if keep.category_method != "manual":
            for name in _CATEGORY_FIELDS:
                setattr(keep, name, data[name])
            await supersede_active_decisions(session, keep.id)
        elif keep.category != data["category"]:
            conflicts.append("category")
    if not keep.note:
        keep.note = data["note"]
    elif data["note"] and data["note"] != keep.note:
        conflicts.append("note")
    if keep.exclude_from_cashflow != data["exclude_from_cashflow"]:
        conflicts.append("exclude_from_cashflow")
    return conflicts


async def _cc_cycle(
    session: AsyncSession, keep: Transaction, dup: Transaction
) -> tuple[StatementUpload | None, list[str]]:
    """Return the card cycle to recompute, or a conflict that says why not."""
    if not _is_cc_payment(dup):
        return None, []
    assert keep.account_id is not None and dup.transaction_date is not None
    cycle = await _latest_active_cycle(session, keep.account_id, include_paid=True)
    if cycle is None:
        return None, []
    window = await cc_cycle_window(session, cycle)
    if dup.transaction_date < window.start or (
        window.end is not None and dup.transaction_date >= window.end
    ):
        return None, [
            f"cc cycle {cycle.id} does not hold the duplicate; not recomputed"
        ]
    if cycle.payment_status == PaymentStatus.PAID:
        # No marker tells a manual "Mark as Paid" from a paid sum, so neither a
        # recompute nor a skip is safe.
        raise PairRefused("the duplicate is a payment in a paid card cycle")
    return cycle, []


async def _merge_pair(
    session: AsyncSession, keep: Transaction, dup: Transaction
) -> TransactionMergeReport:
    """Fold ``dup`` into ``keep`` and delete ``dup``.

    The keeper gains the duplicate's links and empty fields. Every reference
    moves to the keeper. The card cycle that holds the duplicate is recomputed
    unless it has no total due. The caller owns the commit or rollback.

    Raises:
        PairRefused: The duplicate is a payment in a paid card cycle.
    """
    before = _transaction_read(keep)
    data = {name: getattr(dup, name) for name in _COLUMNS}
    if data["channel"] == "bank_statement":
        # A statement import writes this label when the row names no channel.
        data["channel"] = None

    # ponytail: only the latest cycle is recomputed. A duplicate in an older
    # cycle is reported as a conflict.
    cycle, cc_conflicts = await _cc_cycle(session, keep, dup)
    cycle_before = (cycle.payment_paid_amount, cycle.payment_status) if cycle else None

    moved = await _move_references(session, dup.id, keep.id)
    await session.delete(dup)
    # The duplicate holds the unique reference until the delete flushes.
    await session.flush()

    keep.reference_number = _surviving_reference(keep, data)
    for name in _LINK_FIELDS:
        if getattr(keep, name) is None:
            setattr(keep, name, data[name])
    sources = {
        part
        for value in (keep.source, data["source"])
        if value
        for part in value.split("+")
    }
    source = (
        "sms+email" if sources == {"sms", "email"} else keep.source or data["source"]
    )
    conflicts = cc_conflicts + await _fold_manual_fields(session, keep, data)
    # The enrichment runs the self-transfer rule. Thus it must see the final
    # reference and category.
    # The "sms" channel fills empty fields only, so the keeper wins a conflict.
    await apply_transaction_enrichment(
        session, keep, data, "sms", sms_message_id=data["sms_message_id"]
    )
    keep.source = source

    await session.flush()

    cc_state = None
    if cycle is not None and not await resync_tracked_cc_payment_state(session, cycle):
        conflicts.append(f"cc cycle {cycle.id} has no total due; not recomputed")
    elif cycle is not None and cycle_before is not None:
        cc_state = CcPaymentStateChange(
            statement_upload_id=cycle.id,
            paid_before=cycle_before[0],
            paid_after=cycle.payment_paid_amount,
            status_before=cycle_before[1],
            status_after=cycle.payment_status,
        )

    return TransactionMergeReport(
        keep_id=keep.id,
        duplicate_id=data["id"],
        keeper_before=before,
        keeper_after=_transaction_read(keep),
        moved_references=moved,
        conflicts=conflicts,
        cc_payment_state=cc_state,
    )


async def merge_duplicates(
    session: AsyncSession, pairs: list[TransactionMergePair], *, dry_run: bool
) -> TransactionMergeResponse:
    """Merge each duplicate into its keeper in one DB transaction.

    The keeper keeps its own values. It gains the links and the empty fields of
    the duplicate, and a manual category or note that it does not have. The
    function re-points every row that names the duplicate, deletes the
    duplicate, and recomputes the card payment state that the duplicate fed.

    Args:
        session: A session with no open transaction.
        pairs: The (keep_id, duplicate_id) pairs, applied in order.
        dry_run: True rolls back every change after the report is built.

    Returns:
        The report of each merge.

    Raises:
        MergeRefused: A pair fails a check. The function writes nothing.
    """
    duplicate_ids = [pair.duplicate_id for pair in pairs]
    keep_ids = {pair.keep_id for pair in pairs}
    reports = []
    refusals = []
    async with session.begin() as txn:
        if session.get_bind().dialect.name == "sqlite":
            await session.execute(text("BEGIN IMMEDIATE"))
        for pair in pairs:
            keep = await session.get(Transaction, pair.keep_id)
            dup = await session.get(Transaction, pair.duplicate_id)
            if keep is None or dup is None:
                reasons = ["transaction not found"]
            elif keep.id == dup.id:
                reasons = ["a row cannot merge into itself"]
            elif duplicate_ids.count(dup.id) > 1 or dup.id in keep_ids:
                reasons = ["the duplicate appears in another pair"]
            else:
                reasons = _refusal_reasons(keep, dup)
            if reasons:
                refusals.append(pair.model_dump() | {"reasons": reasons})
                continue
            assert keep is not None and dup is not None
            try:
                reports.append(await _merge_pair(session, keep, dup))
            except PairRefused as exc:
                refusals.append(pair.model_dump() | {"reasons": [str(exc)]})
        if dry_run or refusals:
            await txn.rollback()
    if refusals:
        raise MergeRefused(refusals)
    return TransactionMergeResponse(dry_run=dry_run, merges=reports)
