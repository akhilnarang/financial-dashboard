"""Atomic resolution of deferred SMS duplicate decisions."""

import datetime
from decimal import Decimal
from typing import Literal, NamedTuple

from bank_sms_parser import parse_sms
from bank_sms_parser.exceptions import ParseError, UnsupportedSmsTypeError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import SmsMessage, Transaction
from financial_dashboard.services.linker import build_link_context, link_transaction
from financial_dashboard.services.sms_pipeline import (
    ProcessSmsOutcome,
    parsed_sms_to_txn_data,
    process_sms_row,
)
from financial_dashboard.services.txn_merge import (
    DUP_DEFER_PREFIX,
    apply_transaction_enrichment,
    find_match,
    qualifies_as_explicit_match,
)

SmsDuplicateAction = Literal["merge", "create_new"]
ResolutionStatus = Literal["merged", "created", "already_resolved"]


class SmsDuplicateResolutionError(Exception):
    """A stale or invalid duplicate-resolution request."""


class SmsDuplicateResolutionResult(NamedTuple):
    status: ResolutionStatus
    transaction_id: int
    pending_payment_check: tuple[int, int, Decimal] | None = None
    pending_disambiguation: dict | None = None
    notification: dict | None = None


async def _lock_sms(session: AsyncSession, sms_id: int) -> SmsMessage | None:
    """Take the SMS write lock before reading resolution state."""
    if session.get_bind().dialect.name == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))
    return (
        await session.scalars(
            select(SmsMessage).where(SmsMessage.id == sms_id).with_for_update()
        )
    ).one_or_none()


def _parse_stored_sms(sms: SmsMessage):
    received_at = sms.received_at
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=datetime.UTC)
    try:
        parsed = parse_sms(
            sms.bank,
            sms.body,
            sender=sms.sender,
            received_at=received_at,
        )
    except ParseError, UnsupportedSmsTypeError:
        raise SmsDuplicateResolutionError(
            "Stored SMS no longer parses as a transaction"
        ) from None
    txn_data = parsed_sms_to_txn_data(parsed, sms)
    if txn_data is None:
        raise SmsDuplicateResolutionError(
            "Stored SMS no longer parses as a transaction"
        )
    return parsed, txn_data


async def _linked_transaction_id(session: AsyncSession, sms: SmsMessage) -> int | None:
    if sms.transaction_id is not None:
        return sms.transaction_id
    return await session.scalar(
        select(Transaction.id)
        .where(Transaction.sms_message_id == sms.id)
        .limit(1)
        .with_for_update()
    )


async def _resolve_merge(
    session: AsyncSession,
    sms: SmsMessage,
    txn_data: dict,
    transaction_id: int | None,
) -> SmsDuplicateResolutionResult:
    if transaction_id is None:
        raise SmsDuplicateResolutionError("Merge target is required")
    target = (
        await session.scalars(
            select(Transaction)
            .where(Transaction.id == transaction_id)
            .with_for_update()
        )
    ).one_or_none()
    if target is None:
        raise SmsDuplicateResolutionError("Merge target no longer exists")

    decision = await find_match(session, txn_data, "sms")
    current_candidate_ids = decision.resolution_candidate_ids
    if decision.action == "match" and decision.transaction is not None:
        current_candidate_ids = (decision.transaction.id,)
    if (
        transaction_id not in current_candidate_ids
        or not await qualifies_as_explicit_match(session, target, txn_data)
    ):
        raise SmsDuplicateResolutionError(
            "Selected transaction is no longer a compatible duplicate"
        )

    prior_account_id = target.account_id
    diff = await apply_transaction_enrichment(session, target, txn_data, "sms")
    sms.transaction_id = target.id
    sms.status = "enriched"
    sms.parse_error = None
    sms.parsed_at = datetime.datetime.now(datetime.UTC)

    if target.account_id is None and (
        diff.filled.get("card_mask") or diff.filled.get("account_mask")
    ):
        link_context = await build_link_context(session)
        link_transaction(link_context, target)
        await session.flush()

    from financial_dashboard.services.cc_disambiguation import (
        resolve_cc_payment_account,
        should_auto_reconcile_statement,
    )

    pending_disambiguation = None
    if target.account_id is None:
        pending_disambiguation = await resolve_cc_payment_account(session, target)
    pending_payment_check = None
    if prior_account_id is None and should_auto_reconcile_statement(target):
        assert target.account_id is not None
        pending_payment_check = (target.id, target.account_id, target.amount)

    await session.flush()
    return SmsDuplicateResolutionResult(
        "merged",
        target.id,
        pending_payment_check,
        pending_disambiguation,
    )


async def _resolve_create_new(
    session: AsyncSession,
    sms: SmsMessage,
    parsed,
    txn_data: dict,
) -> SmsDuplicateResolutionResult:
    decision = await find_match(session, txn_data, "sms")
    if decision.action != "defer":
        raise SmsDuplicateResolutionError("Duplicate decision is no longer deferred")
    reason = decision.deferral_reason or ""
    if reason.startswith("reference_") or reason == "multiple_reference_candidates":
        raise SmsDuplicateResolutionError(
            "A reference mismatch cannot create a second transaction"
        )

    link_context = await build_link_context(session)
    outcome: ProcessSmsOutcome = await process_sms_row(
        session,
        sms,
        link_context,
        force_new=True,
        settle_provisional=parsed.ledger_role == "provisional",
    )
    if outcome.transaction_id is None:
        raise SmsDuplicateResolutionError("Could not create a transaction")
    return SmsDuplicateResolutionResult(
        "created",
        outcome.transaction_id,
        outcome.pending_payment_check,
        outcome.pending_disambiguation,
        outcome.primary_notification,
    )


async def resolve_sms_duplicate(
    session: AsyncSession,
    sms_id: int,
    action: SmsDuplicateAction,
    transaction_id: int | None = None,
) -> SmsDuplicateResolutionResult:
    """Lock, revalidate, mutate, and commit one deferred SMS decision."""
    async with session.begin():
        sms = await _lock_sms(session, sms_id)
        if sms is None:
            raise SmsDuplicateResolutionError("SMS not found")

        linked_id = await _linked_transaction_id(session, sms)
        if linked_id is not None:
            return SmsDuplicateResolutionResult("already_resolved", linked_id)
        if (
            sms.status != "skipped"
            or not sms.parse_error
            or not sms.parse_error.startswith(DUP_DEFER_PREFIX)
        ):
            raise SmsDuplicateResolutionError(
                "SMS is not a deferred possible duplicate"
            )

        parsed, txn_data = _parse_stored_sms(sms)
        if action == "merge":
            return await _resolve_merge(session, sms, txn_data, transaction_id)
        return await _resolve_create_new(session, sms, parsed, txn_data)
