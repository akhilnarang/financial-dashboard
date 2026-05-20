"""SMS pipeline: from `sms_messages` row → Transaction + notifications.

Entry point: ``process_sms_row(session, sms_row, link_context)``. Called
by the ``POST /api/sms`` endpoint after the raw row is inserted, and by
the reparse endpoints. Caller owns the outer transaction — this module
does NOT commit and does NOT fire Telegram.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo

from bank_sms_parser import parse_sms
from bank_sms_parser.exceptions import ParseError, UnsupportedSmsTypeError
from bank_sms_parser.models import ParsedSms

from financial_dashboard.db import SmsMessage
from financial_dashboard.services.linker import LinkContext, link_transaction
from financial_dashboard.services.txn_merge import merge_transaction

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")


@dataclass
class ProcessSmsOutcome:
    status: Literal["parsed", "enriched", "error", "skipped"]
    transaction_id: int | None
    primary_notification: dict | None = None
    enrichment_notification: tuple[int, object, dict] | None = None
    """(txn_id, EnrichmentDiff, txn_info_payload) — txn_info matches the
    shape of primary_notification so the Telegram renderer can produce a
    concise inline message with bank/amount/counterparty context."""
    # The second tuple member is an EnrichmentDiff (typed as object here to
    # avoid a circular import with txn_merge).
    pending_payment_check: tuple[int, int, Decimal] | None = None
    """(txn_id, account_id, amount) — fires check_payment_received post-commit
    when the SMS is a CC payment-received alert with a resolved account_id."""
    pending_disambiguation: dict | None = None
    """{txn_id, candidate_account_ids, amount, bank} — fires the Telegram
    inline-keyboard prompt when account_id couldn't be resolved (the user
    has more than one CC for this bank registered)."""


def parsed_sms_to_txn_data(parsed: ParsedSms, sms_row: SmsMessage) -> dict | None:
    """Map ``ParsedSms.transaction`` → ``txn_data`` dict shape used by
    ``merge_transaction``. Returns ``None`` if ``parsed.transaction`` is
    ``None`` (non-transaction SMS shape, e.g. OneCard statement-ready).

    Falls back to ``sms_row.received_at`` (converted IST) for
    ``transaction_date`` and ``transaction_time`` when the parsed body
    didn't extract them — matches bank-sms-parser §10 date fallback.
    """
    txn = parsed.transaction
    if txn is None:
        return None

    transaction_date = txn.transaction_date
    transaction_time = txn.transaction_time
    if transaction_date is None or transaction_time is None:
        # Convert UTC received_at → IST then extract date/time.
        # SQLite drops tzinfo on round-trip, but `schemas/sms.py` normalises
        # incoming payloads to UTC at insert time, so re-attaching UTC here
        # is correct. Without this guard `.astimezone()` would interpret
        # the naive datetime in the system's local tz.
        received_utc = sms_row.received_at
        if received_utc.tzinfo is None:
            received_utc = received_utc.replace(tzinfo=datetime.UTC)
        ist = received_utc.astimezone(_IST)
        if transaction_date is None:
            transaction_date = ist.date()
        if transaction_time is None:
            transaction_time = ist.time().replace(microsecond=0)

    return {
        "bank": parsed.bank,
        "email_type": parsed.email_type,
        "direction": txn.direction,
        "amount": Decimal(str(txn.amount.amount)),
        "currency": txn.amount.currency,
        "transaction_date": transaction_date,
        "transaction_time": transaction_time,
        "counterparty": txn.counterparty,
        "card_mask": txn.card_mask,
        "account_mask": txn.account_mask,
        "reference_number": txn.reference_number,
        "channel": txn.channel,
        "balance": Decimal(str(txn.balance.amount)) if txn.balance else None,
        "raw_description": txn.raw_description,
    }


_DECLINED_DIRECTION = "declined"

_HDFC_PAYMENT_RECEIVED_EMAIL_TYPE = "hdfc_cc_payment_received_alert"


def _is_hdfc_provisional_payment(txn_data: dict) -> bool:
    """True for HDFC's provisional CC payment-received SMS — the variant
    with NO reference number.

    HDFC fires two SMS per CC bill payment: a provisional "PAYMENT ...
    RECEIVED TOWARDS ..." (no ref) and a settlement "Online Payment ...
    vide Ref# ... was credited ..." (has ref). Only the settlement should
    become a ledger row; the provisional is notify-only. Gate on the
    parser's email_type plus an empty reference_number — robust even if the
    bank-string casing ever shifts (``parsed.bank`` is lowercase ``hdfc``).

    Also require ``direction == "credit"``: this branch runs ahead of the
    declined pre-gate, so the guard ensures a (hypothetical) non-credit
    payment-received shape can't be silently swallowed as notify-only and
    bypass the declined/merge paths. The current parser only ever emits
    ``credit`` for this email_type, so this is defense-in-depth.
    """
    if (txn_data.get("bank") or "").lower() != "hdfc":
        return False
    if txn_data.get("email_type") != _HDFC_PAYMENT_RECEIVED_EMAIL_TYPE:
        return False
    if txn_data.get("direction") != "credit":
        return False
    return not (txn_data.get("reference_number") or "").strip()


async def process_sms_row(
    session,
    sms_row: SmsMessage,
    link_context: LinkContext,
) -> ProcessSmsOutcome:
    """Parse one ``SmsMessage`` and merge into Transaction.

    Caller owns the outer transaction; this function never commits.
    Returns a ``ProcessSmsOutcome`` carrying status + pending notification
    payloads, which the caller dispatches after commit.
    """
    now = datetime.datetime.now(datetime.UTC)
    sms_row.parsed_at = now

    # 1. Parse.
    # SQLite drops tzinfo on round-trip, but schemas/sms.py normalises
    # incoming payloads to UTC at insert time. bank-sms-parser rejects
    # naive datetimes for the IST date-fallback path, so re-attach UTC
    # before handing the value over.
    received_at_for_parser = sms_row.received_at
    if received_at_for_parser is not None and received_at_for_parser.tzinfo is None:
        received_at_for_parser = received_at_for_parser.replace(tzinfo=datetime.UTC)
    try:
        parsed = parse_sms(
            sms_row.bank,
            sms_row.body,
            sender=sms_row.sender,
            received_at=received_at_for_parser,
        )
    except UnsupportedSmsTypeError as e:
        sms_row.status = "skipped"
        sms_row.parse_error = str(e)
        return ProcessSmsOutcome(status="skipped", transaction_id=None)
    except ParseError as e:
        # Known-stub shapes (e.g. onecard_cc_statement_notice_stub) are
        # intentional non-transaction outcomes from the parser — disposition
        # them as `skipped`, not `error`, so the dashboard's failed-rows
        # view doesn't fill with them. `bank-sms-parser` raises these with
        # a `_stub` suffix in the message text.
        msg = str(e)
        if "_stub" in msg.lower():
            sms_row.status = "skipped"
            sms_row.parse_error = msg
            return ProcessSmsOutcome(status="skipped", transaction_id=None)
        sms_row.status = "error"
        sms_row.parse_error = msg
        return ProcessSmsOutcome(status="error", transaction_id=None)

    txn_data = parsed_sms_to_txn_data(parsed, sms_row)
    if txn_data is None:
        sms_row.status = "skipped"
        sms_row.parse_error = "non-transaction SMS shape"
        return ProcessSmsOutcome(status="skipped", transaction_id=None)

    # HDFC provisional payment-received pre-gate. HDFC sends two SMS per CC
    # bill payment; the no-reference "RECEIVED TOWARDS" variant is a
    # provisional notice. Notify only — do NOT create a Transaction, do NOT
    # touch outstanding, and do NOT fire the payment-check / disambiguation
    # hooks (leaving them None here is a correctness requirement, not an
    # incidental default). The settlement SMS (with a ref) proceeds normally.
    if _is_hdfc_provisional_payment(txn_data):
        sms_row.status = "parsed"
        sms_row.transaction_id = None
        sms_row.parse_error = None  # clear any stale error from a prior parse
        primary = {**txn_data, "_provisional": True}
        return ProcessSmsOutcome(
            status="parsed",
            transaction_id=None,
            primary_notification=primary,
        )

    # 2. Declined-event pre-gate. Declined events never enter merge —
    #    they take the existing declined-notification path. The match
    #    key includes `direction`, so a correctly-parsed declined event
    #    wouldn't pair with a debit anyway; this is defense-in-depth.
    if txn_data["direction"] == _DECLINED_DIRECTION:
        sms_row.status = "parsed"
        primary = {**txn_data, "_declined": True}
        return ProcessSmsOutcome(
            status="parsed",
            transaction_id=None,
            primary_notification=primary,
        )

    # 3. Merge.
    outcome, txn_row, diff = await merge_transaction(
        session, "sms", txn_data, sms_message_id=sms_row.id
    )

    # 4. Link.
    if outcome == "created":
        link_transaction(link_context, txn_row)
        await session.flush()
    elif outcome == "enriched":
        if txn_row.account_id is None and (
            diff.filled.get("card_mask") or diff.filled.get("account_mask")
        ):
            link_transaction(link_context, txn_row)
            await session.flush()

    # 5. CC bill-payment account resolution (runs BEFORE the notification
    # payload so the primary message includes the resolved account label).
    # The helper handles single-CC short-circuit + amount-match against
    # open statements; returns the prompt payload only when both fail.
    pending_payment_check = None
    pending_disambiguation = None
    if outcome == "created":
        from financial_dashboard.services.cc_disambiguation import (
            resolve_cc_payment_account,
            should_auto_reconcile_statement,
        )

        pending_disambiguation = await resolve_cc_payment_account(session, txn_row)
        # The redundant `account_id is not None` check is for ty —
        # should_auto_reconcile_statement already guarantees it at
        # runtime but ty can't narrow through helper calls.
        if should_auto_reconcile_statement(txn_row) and txn_row.account_id is not None:
            pending_payment_check = (txn_row.id, txn_row.account_id, txn_row.amount)

    # 6. Record row state and notification payload.
    # Use a literal-typed local so both `sms_row.status` (a plain str
    # column in the DB) and `ProcessSmsOutcome.status` (a Literal in
    # the dataclass) are assigned from the same narrowed value. This
    # also makes future control-flow edits visible at the type-check
    # layer — if a new code path sets a different status, ty will
    # flag it.
    final_status: Literal["parsed", "enriched"] = (
        "parsed" if outcome == "created" else "enriched"
    )
    sms_row.transaction_id = txn_row.id
    sms_row.status = final_status

    primary_notification = None
    enrichment_notification = None
    if outcome == "created":
        primary_notification = await _notification_payload(txn_row, session)
    elif outcome == "enriched" and diff.changed_fields:
        enrichment_notification = (
            txn_row.id,
            diff,
            await _notification_payload(txn_row, session),
        )

    return ProcessSmsOutcome(
        status=final_status,
        transaction_id=txn_row.id,
        primary_notification=primary_notification,
        enrichment_notification=enrichment_notification,
        pending_payment_check=pending_payment_check,
        pending_disambiguation=pending_disambiguation,
    )


async def _notification_payload(txn_row, session) -> dict:
    """Build the dict shape that send_transaction_notification expects."""
    from financial_dashboard.db import Account, Card
    from financial_dashboard.services.telegram import build_account_label

    account_obj = (
        await session.get(Account, txn_row.account_id) if txn_row.account_id else None
    )
    card_obj = await session.get(Card, txn_row.card_id) if txn_row.card_id else None
    return {
        "bank": txn_row.bank,
        "direction": txn_row.direction,
        "amount": txn_row.amount,
        "counterparty": txn_row.counterparty,
        "transaction_date": txn_row.transaction_date,
        "transaction_time": txn_row.transaction_time,
        "card_mask": txn_row.card_mask,
        "account_label": build_account_label(account_obj, card_obj),
        "channel": txn_row.channel,
    }
