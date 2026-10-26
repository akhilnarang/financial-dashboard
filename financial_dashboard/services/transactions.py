"""Transaction-domain service helpers."""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Account, Card, Transaction
from financial_dashboard.db.models import ReceivableLink
from financial_dashboard.schemas import transactions as transaction_schemas
from financial_dashboard.services.cc_disambiguation import (
    should_auto_reconcile_statement,
)

logger = logging.getLogger(__name__)


class ApplySnapshot(NamedTuple):
    """Stored fields that the apply endpoint can change, directly or as a side effect.

    It leaves out categorized_at. Each category write stamps it again.
    """

    category: str | None
    category_method: str | None
    category_confidence: float | None
    category_model: str | None
    category_vocab_version: int | None
    review_status: str | None
    note: str | None
    exclude_from_cashflow: bool
    voided: bool
    amount: Decimal
    currency: str | None
    amount_source: str


def _apply_snapshot(txn: Transaction) -> ApplySnapshot:
    return ApplySnapshot(
        category=txn.category,
        category_method=txn.category_method,
        category_confidence=txn.category_confidence,
        category_model=txn.category_model,
        category_vocab_version=txn.category_vocab_version,
        review_status=txn.review_status,
        note=txn.note,
        exclude_from_cashflow=txn.exclude_from_cashflow,
        voided=txn.voided,
        amount=txn.amount,
        currency=txn.currency,
        amount_source=txn.amount_source,
    )


class NoteResult(NamedTuple):
    ok: bool
    note: str | None


class RelinkError(Exception):
    """Raised when a manual relink request can't be honored."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RelinkResult:
    account_id: int | None
    card_id: int | None
    account_label: str | None
    card_label: str | None
    statement_marked_paid: bool


async def update_transaction_note(
    session: AsyncSession,
    txn_id: int,
    note: str,
) -> NoteResult:
    result = await update_transaction_note_no_commit(session, txn_id, note)
    if result[0]:
        await session.commit()
        # Preserve the public API's historical clear response (empty string)
        # while the stored value and no-commit mutation snapshot remain NULL.
        return NoteResult(ok=True, note=note.strip())
    return result


async def update_transaction_note_no_commit(
    session: AsyncSession, txn_id: int, note: str
) -> NoteResult:
    txn = await session.get(Transaction, txn_id)
    if not txn:
        return NoteResult(ok=False, note=None)
    cleaned = note.strip()
    txn.note = cleaned or None
    return NoteResult(ok=True, note=cleaned or None)


class ExcludeResult(NamedTuple):
    ok: bool
    exclude_from_cashflow: bool


async def set_transaction_excluded(
    session: AsyncSession,
    txn_id: int,
    excluded: bool,
) -> ExcludeResult:
    """Set whether one transaction is dropped from the cashflow.

    Set the state to the given value. A repeated request lands the same value. It
    does not toggle. The row keeps its category and every other view. Only the
    cashflow report reads this flag. ``ok`` is False when no such row exists.
    """
    result = await set_transaction_excluded_no_commit(session, txn_id, excluded)
    if result.ok:
        await session.commit()
    return result


async def set_transaction_excluded_no_commit(
    session: AsyncSession, txn_id: int, excluded: bool
) -> ExcludeResult:
    txn = await session.get(Transaction, txn_id)
    if txn is None:
        return ExcludeResult(ok=False, exclude_from_cashflow=False)
    txn.exclude_from_cashflow = excluded
    return ExcludeResult(ok=True, exclude_from_cashflow=excluded)


class VoidResult(NamedTuple):
    ok: bool
    voided: bool


def check_voidable(txn: Transaction, voided: bool) -> None:
    """Refuse to void a credit.

    Args:
        txn: The row to change.
        voided: The new state.

    Raises:
        ValueError: The row is a credit. Only a charge can be voided.
    """
    # A credit can be a card bill payment, and its paid state would go stale.
    if voided and txn.direction != "debit":
        raise ValueError("Only a debit can be voided")


async def set_transaction_voided(
    session: AsyncSession, txn_id: int, voided: bool
) -> VoidResult:
    """Void a transaction, or undo the void, and commit.

    A voided charge never settled. The row then counts as no money, and no
    statement row matches it. A repeated request lands the same value.

    Args:
        session: The request session.
        txn_id: The transaction to change.
        voided: The new state.

    Returns:
        The stored state. ``ok`` is False when no such row exists.

    Raises:
        ValueError: The row is a credit. Only a charge can be voided.
    """
    result = await set_transaction_voided_no_commit(session, txn_id, voided)
    if result.ok:
        await session.commit()
    return result


async def set_transaction_voided_no_commit(
    session: AsyncSession, txn_id: int, voided: bool
) -> VoidResult:
    """Void a transaction, or undo the void, and leave the commit to the caller.

    Args:
        session: The request session.
        txn_id: The transaction to change.
        voided: The new state.

    Returns:
        The stored state. ``ok`` is False when no such row exists.

    Raises:
        ValueError: The row is a credit. Only a charge can be voided.
    """
    txn = await session.get(Transaction, txn_id)
    if txn is None:
        return VoidResult(ok=False, voided=False)
    check_voidable(txn, voided)
    txn.voided = voided
    return VoidResult(ok=True, voided=voided)


async def update_transaction_category(
    session: AsyncSession,
    txn_id: int,
    category: str,
) -> tuple[bool, str | None]:
    """Returns (ok, slug). Raises ValueError if the transaction exists but the
    category is invalid (so callers can return 400, not 404)."""
    from financial_dashboard.services.categorization.manual import (
        assign_category_manual,
    )

    txn = await session.get(Transaction, txn_id)
    if txn is None:
        return False, None
    ok, slug = await assign_category_manual(session, txn_id, category)
    if not ok:
        raise ValueError(f"Invalid category: {category!r}")
    return True, slug


class ApplyResult(NamedTuple):
    changed: list[transaction_schemas.TransactionApplyRow]
    unchanged: list[int]


async def _validate_change(
    session: AsyncSession, change: transaction_schemas.TransactionApplyChange
) -> str | None:
    """Check one change against the single-endpoint rules. Do not write.

    Args:
        session: Request session.
        change: The fields to set.

    Returns:
        The reason the change is invalid, or None when it is valid.
    """
    from financial_dashboard.services.categorization.manual import (
        find_manual_category,
    )
    from financial_dashboard.services.statement_previews import statement_of

    txn = await session.get(Transaction, change.id)
    if txn is None:
        return "Transaction not found"
    category = change.category
    if category and category.strip():
        if await find_manual_category(session, category) is None:
            return f"Invalid category: {category!r}"
    if change.voided is not None:
        try:
            check_voidable(txn, change.voided)
        except ValueError as exc:
            return str(exc)
    if {"amount", "currency"} & change.model_fields_set:
        if should_auto_reconcile_statement(txn):
            return "A card payment row keeps its amount; apply does not recompute the statement paid sum"
        if statement_of(txn) is not None:
            return "A statement row keeps its amount; a reparse matches the line by it"
        if change.amount is None and change.currency not in (None, txn.currency):
            return "A currency change must set the amount in that currency too"
        if change.currency not in (None, "INR") and await session.scalar(
            select(ReceivableLink.id).where(ReceivableLink.transaction_id == txn.id)
        ):
            return "A row linked to a receivable stays in INR"
    return None


async def _apply_change_no_commit(
    session: AsyncSession,
    txn_id: int,
    change: transaction_schemas.TransactionApplyChange,
) -> None:
    """Apply one change through the single-field functions.

    Args:
        session: Request session. This function does not commit.
        txn_id: The row to change. It must exist.
        change: The fields to set. ``_validate_change`` must accept it.
    """
    from financial_dashboard.services.categorization.manual import (
        assign_category_no_commit,
    )

    if "note" in change.model_fields_set:
        await update_transaction_note_no_commit(session, txn_id, change.note or "")
    if change.category is not None:
        await assign_category_no_commit(session, txn_id, change.category)
    if change.exclude_from_cashflow is not None:
        await set_transaction_excluded_no_commit(
            session, txn_id, change.exclude_from_cashflow
        )
    if change.voided is not None:
        await set_transaction_voided_no_commit(session, txn_id, change.voided)
    if change.amount is not None or change.currency is not None:
        txn = await session.get(Transaction, txn_id)
        assert txn is not None
        txn.amount_source = "manual"
        if change.amount is not None:
            txn.amount = change.amount
        if change.currency is not None:
            txn.currency = change.currency


async def apply_transaction_changes(
    session: AsyncSession,
    changes: list[transaction_schemas.TransactionApplyChange],
    *,
    dry_run: bool,
) -> ApplyResult:
    """Apply note, category, exclusion, void, amount and currency changes as one unit.

    The function checks every change before it changes a row, so a rejected
    batch takes no write lock. Each change then goes through the same per-row
    function as the single endpoints, so the side effects are the same. The
    function commits only when every change is valid and ``dry_run`` is False.
    Otherwise it rolls back, so nothing is written. A dry run writes and then
    rolls back, so it holds the write lock for the whole batch.

    Args:
        session: Request session. This function commits or rolls it back.
        changes: Row changes with unique ids. An absent field stays unchanged.
        dry_run: Roll back after computing the diff.

    Returns:
        The before and after value of each stored field that changes, per row,
        and the ids of rows where no tracked field changes.

    Raises:
        ValueError: One or more changes are invalid. ``args[0]`` holds one
            ``{"id", "error"}`` dict per invalid change.
    """
    errors = [
        {"id": change.id, "error": error}
        for change in changes
        if (error := await _validate_change(session, change)) is not None
    ]
    if errors:
        await session.rollback()
        raise ValueError(errors)

    changed: list[transaction_schemas.TransactionApplyRow] = []
    unchanged: list[int] = []
    for change in changes:
        txn = await session.get(Transaction, change.id)
        assert txn is not None
        before = _apply_snapshot(txn)
        await _apply_change_no_commit(session, txn.id, change)
        after = _apply_snapshot(txn)
        diff = {
            field: transaction_schemas.TransactionFieldDiff(before=old, after=new)
            for field, old, new in zip(ApplySnapshot._fields, before, after)
            if old != new
        }
        if diff:
            changed.append(
                transaction_schemas.TransactionApplyRow(id=txn.id, changes=diff)
            )
        else:
            unchanged.append(txn.id)

    if dry_run:
        await session.rollback()
    else:
        await session.commit()
    return ApplyResult(changed=changed, unchanged=unchanged)


async def relink_transaction(
    session: AsyncSession,
    txn_id: int,
    *,
    account_id: int | None,
    card_id: int | None,
) -> RelinkResult | None:
    """Manually relink a Transaction to an account (and optional card).

    Returns ``None`` when the txn doesn't exist.
    Raises ``RelinkError`` with a code on validation failures:

      - ``card_not_found`` / ``account_not_found``
      - ``card_account_mismatch`` — card.account_id != account_id
      - ``bank_mismatch`` — account.bank != txn.bank

    Both fields may be null to clear the link.
    If only ``card_id`` is given, ``account_id`` is derived from the
    card's owning account.

    Fires ``check_payment_received`` post-commit when the txn was
    previously orphaned (account_id == None) AND the new state passes
    ``should_auto_reconcile_statement``. This mirrors the upsert-path
    rule in /emails/{id}/reparse: only credit the statement once,
    when the link is first established.

    Race note: ``was_orphaned`` is captured before mutation but two
    concurrent relink calls on the same orphan row could both observe
    ``account_id is None`` and both fire ``check_payment_received``,
    cumulatively double-crediting ``payment_paid_amount``. Acceptable
    for the operator-only workflow this endpoint serves (single
    person clicking once per orphan); not race-proof against
    automation.
    """
    from financial_dashboard.services.reminders import check_payment_received

    txn = await session.get(Transaction, txn_id)
    if not txn:
        return None

    card_obj: Card | None = None
    account_obj: Account | None = None

    if card_id is not None:
        card_obj = await session.get(Card, card_id)
        if card_obj is None:
            raise RelinkError("card_not_found", f"Card #{card_id} not found.")
        if account_id is None:
            account_id = card_obj.account_id
        elif card_obj.account_id != account_id:
            raise RelinkError(
                "card_account_mismatch",
                f"Card #{card_id} belongs to account "
                f"#{card_obj.account_id}, not #{account_id}.",
            )

    if account_id is not None:
        account_obj = await session.get(Account, account_id)
        if account_obj is None:
            raise RelinkError("account_not_found", f"Account #{account_id} not found.")
        if account_obj.bank != txn.bank:
            raise RelinkError(
                "bank_mismatch",
                f"Account #{account_id} is bank {account_obj.bank!r}; "
                f"transaction is bank {txn.bank!r}.",
            )

    was_orphaned = txn.account_id is None

    txn.account_id = account_id
    txn.card_id = card_id

    statement_marked_paid = False
    await session.commit()

    # Fire check_payment_received outside the relink transaction so
    # the statement update commits in its own scope (the existing
    # reminders helper opens its own async_session). Same shape as
    # the email-reparse path.
    #
    # `should_auto_reconcile_statement` guarantees `account_id is not
    # None`, but ty can't narrow through a helper call — re-check
    # locally so the downstream `int` argument types cleanly.
    if (
        was_orphaned
        and should_auto_reconcile_statement(txn)
        and txn.account_id is not None
    ):
        resolved_account_id = txn.account_id
        try:
            statement_marked_paid = await check_payment_received(
                txn.id, resolved_account_id, txn.amount
            )
        except Exception:  # noqa: BLE001
            # Don't fail the relink just because the statement-marking
            # hiccupped — the link itself is the user's primary intent.
            # Log so a silent failure is debuggable.
            logger.exception(
                "check_payment_received failed during relink of txn %s "
                "to account %s; the link was set but no statement was "
                "auto-marked.",
                txn.id,
                resolved_account_id,
            )
            statement_marked_paid = False

    return RelinkResult(
        account_id=txn.account_id,
        card_id=txn.card_id,
        account_label=account_obj.label if account_obj else None,
        card_label=(card_obj.label or card_obj.card_mask) if card_obj else None,
        statement_marked_paid=statement_marked_paid,
    )
