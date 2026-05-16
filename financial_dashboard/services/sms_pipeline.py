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


def parsed_sms_to_txn_data(
    parsed: ParsedSms, sms_row: SmsMessage
) -> dict | None:
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
    if (
        received_at_for_parser is not None
        and received_at_for_parser.tzinfo is None
    ):
        received_at_for_parser = received_at_for_parser.replace(
            tzinfo=datetime.UTC
        )
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

    # 5. Record row state and notification payload.
    sms_row.transaction_id = txn_row.id
    sms_row.status = "parsed" if outcome == "created" else "enriched"

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

    pending_payment_check = None
    pending_disambiguation = None
    if (
        outcome == "created"
        and txn_row.direction == "credit"
        and txn_row.email_type.endswith("_cc_payment_received_alert")
    ):
        if txn_row.account_id is not None:
            pending_payment_check = (
                txn_row.id, txn_row.account_id, txn_row.amount
            )
        else:
            # account_id couldn't be resolved → try CC-account lookup.
            disambiguation = await _build_disambiguation(session, txn_row)
            if disambiguation is not None and len(disambiguation["candidate_account_ids"]) == 1:
                # Exactly one CC account for this bank → auto-resolve.
                pending_payment_check = (
                    txn_row.id,
                    disambiguation["candidate_account_ids"][0],
                    txn_row.amount,
                )
            else:
                # Multiple CC candidates for this bank — ask the user
                # which card the payment hit via Telegram inline keyboard.
                pending_disambiguation = disambiguation

    return ProcessSmsOutcome(
        status=sms_row.status,
        transaction_id=txn_row.id,
        primary_notification=primary_notification,
        enrichment_notification=enrichment_notification,
        pending_payment_check=pending_payment_check,
        pending_disambiguation=pending_disambiguation,
    )


async def _build_disambiguation(session, txn_row) -> dict | None:
    """Build payload for the Telegram inline-keyboard prompt.

    Returns ``None`` if there are no candidate accounts (in which case
    the caller falls back to no-op — the txn row exists but no
    statement-marking happens).
    """
    from financial_dashboard.db import Account
    from sqlalchemy import select

    candidates = (
        await session.execute(
            select(Account).where(
                Account.bank == txn_row.bank,
                Account.type == "credit_card",
            )
        )
    ).scalars().all()
    if not candidates:
        return None
    return {
        "txn_id": txn_row.id,
        "candidate_account_ids": [a.id for a in candidates],
        "candidate_labels": {a.id: a.label for a in candidates},
        "amount": txn_row.amount,
        "bank": txn_row.bank,
    }


async def _notification_payload(txn_row, session) -> dict:
    """Build the dict shape that send_transaction_notification expects."""
    from financial_dashboard.db import Account, Card
    from financial_dashboard.services.telegram import build_account_label

    account_obj = (
        await session.get(Account, txn_row.account_id) if txn_row.account_id else None
    )
    card_obj = (
        await session.get(Card, txn_row.card_id) if txn_row.card_id else None
    )
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
