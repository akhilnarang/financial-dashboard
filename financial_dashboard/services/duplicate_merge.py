"""Explicit merge of a duplicate transaction into the row that stays."""

import json
from datetime import timedelta
from typing import NamedTuple, cast

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
from financial_dashboard.exceptions import StatementPreviewError
from financial_dashboard.schemas.transactions import (
    CcPaymentStateChange,
    MergeOverride,
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
from financial_dashboard.services.statement_previews import (
    ParsedStoredStatement,
    StatementRef,
    could_be_statement_candidate,
    parse_shortfall,
    parse_stored_statement,
    reconcile_parsed_statement,
    statement_candidate_index,
    statement_of,
)
from financial_dashboard.services.transaction_reads import transaction_read
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
# A person may lift these refusals for a pair they checked by hand.
_OVERRIDABLE: dict[str, MergeOverride] = {
    "dates are more than 1 day apart": "dates",
    "references prove distinct events": "references",
    "currency differs": "currency",
}


class PairChecks(NamedTuple):
    """The refusals that stand and the overrides that lifted the others."""

    reasons: list[str]
    applied: list[MergeOverride]


class PairRefused(Exception):
    """One pair fails a check that needs the DB."""


class BatchRefused(Exception):
    """One or more batch items fail their checks. Nothing is written."""

    def __init__(self, refusals: list[dict]) -> None:
        super().__init__("batch refused")
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


def _refusal_reasons(
    keep: Transaction, dup: Transaction, override: list[MergeOverride]
) -> PairChecks:
    """Return every reason the pair must not merge. No reason allows it.

    An override lifts only its own refusal. Every other refusal stands.
    """
    days_apart = (
        abs(keep.transaction_date - dup.transaction_date)
        if keep.transaction_date and dup.transaction_date
        else None
    )
    balances = {_quantize_balance(keep.balance), _quantize_balance(dup.balance)}
    keep_currency = _normalized_currency(keep.currency)
    same_currency = keep_currency == _normalized_currency(dup.currency)
    checks = [
        (
            keep.account_id is None or keep.account_id != dup.account_id,
            "rows are not on the same account",
        ),
        (keep.direction != dup.direction, "direction differs"),
        (not same_currency, "currency differs"),
        (same_currency and keep.amount != dup.amount, "amount differs"),
        (
            not same_currency and not (keep_currency == "INR" and statement_of(keep)),
            "a foreign-currency row folds only into a rupee statement row",
        ),
        (days_apart is None, "a row has no date"),
        (
            days_apart is not None and days_apart > _DATE_WINDOW,
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
    failed = [reason for refused, reason in checks if refused]
    return PairChecks(
        reasons=[r for r in failed if _OVERRIDABLE.get(r) not in override],
        applied=[_OVERRIDABLE[r] for r in failed if _OVERRIDABLE.get(r) in override],
    )


def _repoint_reconciliation(recon: dict, old_id: int, new_id: int | None) -> bool:
    """Replace ``old_id`` with ``new_id`` in stored statement reconciliation data.

    ``None`` drops each entry that names ``old_id`` as its row. Return whether
    anything changed.
    """
    changed = False
    for group in ("matched", "missing"):
        kept = []
        for entry in recon.get(group, []):
            named = False
            for key in ("db_txn_id", "imported_txn_id"):
                if entry.get(key) == old_id:
                    entry[key] = new_id
                    changed = named = True
            candidates = entry.get("candidate_transaction_ids") or []
            if old_id in candidates:
                replaced = (new_id if i == old_id else i for i in candidates)
                entry["candidate_transaction_ids"] = list(
                    dict.fromkeys(i for i in replaced if i is not None)
                )
                changed = True
            if new_id is not None or not named:
                kept.append(entry)
        if group in recon:
            recon[group] = kept
    return changed


async def move_references(
    session: AsyncSession, old_id: int, new_id: int | None
) -> dict[str, int]:
    """Point every row that names ``old_id`` at ``new_id``.

    ``None`` cleans the references of a deleted row. A category review row
    only expires, because its link cannot be empty. An audit row keeps its
    target id as history. A stored statement reconciliation forgets the row.

    Args:
        session: The session that deletes ``old_id``.
        old_id: The row that goes away.
        new_id: The row that keeps the data, or ``None`` for a plain delete.

    Returns:
        The changed row count per table. Telegram rows change too but are not
        counted.
    """
    moved = {}
    await supersede_active_decisions(session, old_id)
    await move_transaction_references(session, old_id, new_id)
    for model in _REFERRING_MODELS:
        if new_id is None and model is CategoryReviewDecision:
            continue
        result = await session.execute(
            update(model)
            .where(model.transaction_id == old_id)
            .values(transaction_id=new_id)
        )
        if rowcount := cast(CursorResult, result).rowcount:
            moved[model.__tablename__] = rowcount

    if new_id is not None:
        await session.execute(
            update(AuditAction)
            .where(
                AuditAction.target_type == "transaction",
                AuditAction.target_id == old_id,
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


def _checked_statement(keep: Transaction, dup: Transaction) -> StatementRef | None:
    """Return the statement whose reparse must back an override of the pair.

    A card statement can post a purchase days after its alert. The keeper can
    then be the statement row. A duplicate that a statement already matched
    must be that statement's own row: the merge keeps one statement link.
    """
    if (statement := statement_of(dup)) or _from_statement(dup):
        return statement
    return statement_of(keep)


async def _parse_statements(
    session: AsyncSession, pairs: list[TransactionMergePair]
) -> dict[int, ParsedStoredStatement | str]:
    """Parse the statement of each pair that an override lifts.

    The parse runs before the write lock. A PDF parse can take seconds.
    Each duplicate id maps to today's parse, or to why it cannot be checked.
    """
    parses: dict[int, ParsedStoredStatement | str] = {}
    for pair in pairs:
        if not pair.override:
            continue
        keep = await session.get(Transaction, pair.keep_id)
        dup = await session.get(Transaction, pair.duplicate_id)
        if keep is None or dup is None:
            continue
        if (statement := _checked_statement(keep, dup)) is None:
            parses[pair.duplicate_id] = (
                "an override needs a statement row as the duplicate, "
                "or as the keeper of a row no statement matched"
            )
            continue
        name = f"{statement.kind} statement {statement.id}"
        try:
            stored = await parse_stored_statement(session, *statement)
        except StatementPreviewError as exc:
            stored = f"{name} cannot be checked: {exc}"
        parses[pair.duplicate_id] = stored or f"{name} cannot be checked: not found"
    await session.rollback()
    return parses


async def _reparse_refusal(
    session: AsyncSession,
    keep: Transaction,
    dup: Transaction,
    stored: ParsedStoredStatement | str,
) -> str | None:
    """Return why a reparse of the pair's statement would import a row again.

    The merge must be flushed. Today's parse must match a line to the keeper.
    It must leave no line unmatched that could be the keeper or the duplicate
    by date, amount and direction, or by reference.
    """
    if isinstance(stored, str):
        return stored
    statement = StatementRef(stored.kind, stored.statement_id)
    name = f"{statement.kind} statement {statement.id}"
    if _checked_statement(keep, dup) != statement:
        return "the pair changed during the check; try again"
    try:
        recon = await reconcile_parsed_statement(session, stored, as_reparse=True)
    except StatementPreviewError as exc:
        return f"{name} cannot be checked: {exc}"
    preview = recon.preview
    if shortfall := await parse_shortfall(session, statement, preview, 0):
        return f"{name} cannot be checked: {shortfall}"
    if keep.id not in recon.matched_ids:
        return f"today's parse of {name} matches no line to the keeper"
    if preview.missing_truncated or preview.ambiguous_truncated:
        return f"{name} cannot be checked: too many unmatched lines"
    unresolved = [e.model_dump() for e in (*preview.missing, *preview.ambiguous)]
    index = statement_candidate_index(unresolved)
    refs = {_normalized_ref(r) for r in (keep.reference_number, dup.reference_number)}
    if unresolved and (
        any(could_be_statement_candidate(r, *index) for r in (keep, dup))
        or any(
            _normalized_ref(e["reference_number"]) in refs - {""} for e in unresolved
        )
    ):
        return (
            f"today's parse of {name} leaves the row unmatched; "
            "a reparse would import it again"
        )
    return None


def _pair_checks(
    pair: TransactionMergePair,
    keep: Transaction | None,
    dup: Transaction | None,
    duplicate_ids: list[int],
    keep_ids: set[int],
) -> PairChecks:
    """Return the refusals of one batch pair and the overrides it applies."""
    if keep is None or dup is None:
        return PairChecks(["transaction not found"], [])
    if keep.id == dup.id:
        return PairChecks(["a row cannot merge into itself"], [])
    if duplicate_ids.count(dup.id) > 1 or dup.id in keep_ids:
        return PairChecks(["the duplicate appears in another pair"], [])
    return _refusal_reasons(keep, dup, pair.override)


class _Proof(NamedTuple):
    """An overridden merge that today's parse must back."""

    ids: dict[str, int]
    keep: Transaction
    dup: Transaction
    statement: ParsedStoredStatement | str


async def _merge_pair(
    session: AsyncSession,
    keep: Transaction,
    dup: Transaction,
    applied: list[MergeOverride],
    reason: str,
) -> TransactionMergeReport:
    """Fold ``dup`` into ``keep`` and delete ``dup``.

    The keeper gains the duplicate's links and empty fields. Every reference
    moves to the keeper. The card cycle that holds the duplicate is recomputed
    unless it has no total due. An applied override writes an audit record
    with the reason. The caller owns the commit or rollback.

    Raises:
        PairRefused: The duplicate is a payment in a paid card cycle.
    """
    before = transaction_read(keep)
    data = {name: getattr(dup, name) for name in _COLUMNS}
    original = json.dumps(data, default=str, sort_keys=True)
    if data["channel"] == "bank_statement":
        # A statement import writes this label when the row names no channel.
        data["channel"] = None

    # ponytail: only the latest cycle is recomputed. A duplicate in an older
    # cycle is reported as a conflict.
    cycle, cc_conflicts = await _cc_cycle(session, keep, dup)
    cycle_before = (cycle.payment_paid_amount, cycle.payment_status) if cycle else None

    moved = await move_references(session, dup.id, keep.id)
    await session.delete(dup)
    # The duplicate holds the unique reference until the delete flushes.
    await session.flush()

    # An override keeps the keeper's reference, even an empty one, not the
    # statement reference. A person found the duplicate's parse wrong: an old
    # parse misdated it or took another line's reference. The reparse check
    # below proves that today's parse still pairs a line with the keeper.
    if applied:
        data["reference_number"] = keep.reference_number
    else:
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

    if applied:
        session.add(
            AuditAction(
                transaction_id=keep.id,
                action_type="merge_transaction",
                target_type="transaction",
                target_id=data["id"],
                arguments_json=json.dumps(
                    {"keep_id": keep.id, "override": applied, "reason": reason}
                ),
                before_json=original,
                status="applied",
            )
        )
        await session.flush()

    return TransactionMergeReport(
        keep_id=keep.id,
        duplicate_id=data["id"],
        keeper_before=before,
        keeper_after=transaction_read(keep),
        moved_references=moved,
        conflicts=conflicts,
        cc_payment_state=cc_state,
        overrides=applied,
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
        pairs: The (keep_id, duplicate_id) pairs, applied in order. A pair
            can lift the date or the reference refusal with a reason.
        dry_run: True rolls back every change after the report is built.

    Returns:
        The report of each merge.

    Raises:
        BatchRefused: A pair fails a check. The function writes nothing.
    """
    duplicate_ids = [pair.duplicate_id for pair in pairs]
    keep_ids = {pair.keep_id for pair in pairs}
    reports = []
    refusals = []
    proofs: list[_Proof] = []
    parses = await _parse_statements(session, pairs)
    async with session.begin() as txn:
        if session.get_bind().dialect.name == "sqlite":
            await session.execute(text("BEGIN IMMEDIATE"))
        for pair in pairs:
            keep = await session.get(Transaction, pair.keep_id)
            dup = await session.get(Transaction, pair.duplicate_id)
            reasons, applied = _pair_checks(pair, keep, dup, duplicate_ids, keep_ids)
            ids = pair.model_dump(include={"keep_id", "duplicate_id"})
            if reasons:
                refusals.append(ids | {"reasons": reasons})
                continue
            assert keep is not None and dup is not None
            try:
                reports.append(
                    await _merge_pair(session, keep, dup, applied, pair.reason)
                )
            except PairRefused as exc:
                refusals.append(ids | {"reasons": [str(exc)]})
                continue
            if applied:
                statement = parses.get(dup.id, "statement not checked")
                proofs.append(_Proof(ids, keep, dup, statement))
        # A person checked each overridden pair. Today's parse must agree on
        # the batch's final state: a later pair can change a keeper.
        await session.flush()
        refusals += [
            proof.ids | {"reasons": [refusal]}
            for proof in proofs
            if (refusal := await _reparse_refusal(session, *proof[1:]))
        ]
        if dry_run or refusals:
            await txn.rollback()
    if refusals:
        raise BatchRefused(refusals)
    return TransactionMergeResponse(dry_run=dry_run, merges=reports)
