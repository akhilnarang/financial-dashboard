"""Transaction-domain service helpers."""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Account, Card, Transaction
from financial_dashboard.services.cc_disambiguation import (
    should_auto_reconcile_statement,
)

logger = logging.getLogger(__name__)


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
) -> tuple[bool, str | None]:
    cleaned = note.strip()
    txn = await session.get(Transaction, txn_id)
    if not txn:
        return False, None
    txn.note = cleaned or None
    await session.commit()
    return True, cleaned


async def update_transaction_category(
    session: AsyncSession,
    txn_id: int,
    category: str,
) -> tuple[bool, str | None]:
    cleaned = category.strip()
    txn = await session.get(Transaction, txn_id)
    if not txn:
        return False, None
    txn.category = cleaned or None
    await session.commit()
    return True, cleaned


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
