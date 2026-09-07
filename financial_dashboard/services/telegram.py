"""Telegram bot for transaction notifications and note replies.

Sends a notification for each new transaction (post-backfill only).
If the user replies to a notification, the reply text is saved as the
transaction's note. Only the configured chat_id is authorized.
"""

import asyncio
import html
import logging
import re
from decimal import Decimal
from typing import Literal

from datetime import timedelta
from typing import cast

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError, RetryAfter
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)
from sqlalchemy.exc import OperationalError

from financial_dashboard.db import Transaction, async_session
from financial_dashboard.services.settings import (
    get_telegram_chat_id,
    is_telegram_assistant_enabled,
)

logger = logging.getLogger(__name__)

tg_app: Application | None = None

_SMS_DUPLICATE_HEADER = re.compile(
    r"^\s*⚠️\s+(?:<b>)?[^<\n]+?(?:</b>)?\s+"
    r"(?:DEBIT|CREDIT)\s+SMS\s+#\d+\s*$",
    re.IGNORECASE,
)


def is_sms_duplicate_prompt(text: str) -> bool:
    """Identify the exact deferred-SMS prompt in Bot API or rendered form."""
    first_line = text.splitlines()[0] if text else ""
    return bool(_SMS_DUPLICATE_HEADER.fullmatch(first_line))


def _telegram_file_base_url(base_url: str) -> str:
    """Derive PTB's file endpoint from the configured Bot API endpoint."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/bot"):
        return f"{normalized[:-4]}/file/bot"
    return f"{normalized}/file/bot"


async def init_telegram(token: str, base_url: str | None = None):
    """Initialize the Telegram bot application.

    ``base_url`` overrides the Bot API server (e.g. a self-hosted local Bot
    API server). When None, PTB defaults to https://api.telegram.org/bot.
    """
    global tg_app
    builder = Application.builder().token(token)
    if base_url:
        builder = builder.base_url(base_url).base_file_url(
            _telegram_file_base_url(base_url)
        )
    app = builder.build()
    # Application.updater is Optional in the PTB type stubs because some
    # builders (webhook mode, custom updater=None) intentionally produce
    # an app without one. The standard builder we use here always sets it;
    # assert so downstream attribute access types cleanly and we get a
    # loud failure if PTB ever changes the default.
    updater = app.updater
    assert updater is not None, "Application.builder().build() returned no updater"
    app.add_handler(CommandHandler("ask", _handle_ask))
    app.add_handler(
        MessageHandler(filters.TEXT & filters.REPLY & ~filters.COMMAND, _handle_reply)
    )
    app.add_handler(
        MessageHandler(
            (filters.PHOTO | filters.Document.ALL) & filters.REPLY,
            _handle_attachment_reply,
        )
    )
    app.add_handler(CallbackQueryHandler(_handle_callback))
    try:
        await app.initialize()
        await app.start()
        await updater.start_polling(
            drop_pending_updates=False, allowed_updates=["message", "callback_query"]
        )
    except Exception:
        try:
            if updater.running:
                await updater.stop()
            if app.running:
                await app.stop()
            await app.shutdown()
        except Exception:
            pass
        raise
    tg_app = app
    async with async_session() as session:
        from financial_dashboard.services.assistant.delivery import (
            recover_assistant_work,
        )

        await recover_assistant_work(session)
        await session.commit()
    if is_telegram_assistant_enabled():
        from financial_dashboard.services.assistant.orchestrator import (
            resume_claimed_interactions,
        )

        try:
            await resume_claimed_interactions(bot=app.bot)
        except Exception:
            logger.exception(
                "Assistant interaction recovery failed at Telegram startup"
            )
    if is_telegram_assistant_enabled():
        try:
            await dispatch_pending_deliveries()
        except Exception:
            logger.exception("Assistant delivery recovery failed at Telegram startup")
    logger.info("Telegram bot started")


async def shutdown_telegram():
    """Shutdown the Telegram bot."""
    global tg_app
    if tg_app:
        app = tg_app
        tg_app = None
        try:
            updater = app.updater
            if updater is not None and updater.running:
                await updater.stop()
            if app.running:
                await app.stop()
            await app.shutdown()
        except Exception as e:
            logger.warning("Error during Telegram shutdown: %s", e)
        logger.info("Telegram bot stopped")


def build_account_label(account, card) -> str:
    """Render the "Account: …" label used in Telegram notifications.

    Pure function — callers pass already-loaded Account / Card ORM rows (both
    relationships are ``lazy="joined"``) so the notification path doesn't open
    a DB session per send.
    """
    if card:
        card_label = card.label or card.card_mask
        if account:
            return f"{account.label} - {card_label}"
        return card_label
    if account:
        return account.label
    return ""


async def _send_with_retry(
    app,
    *,
    chat_id,
    text,
    parse_mode="HTML",
    reply_markup=None,
    attempts=3,
):
    """Send a message with retries on transient network errors.

    - Retries up to `attempts` total tries on `telegram.error.NetworkError`
      (which includes `TimedOut`), with exponential backoff (1s, 2s, 4s, ...).
    - On `RetryAfter`, sleeps the bot's recommended duration plus a small
      buffer before retrying. RetryAfter does NOT consume an attempt —
      it's a server-issued rate-limit, not a transient network failure.
    - Re-raises the last `NetworkError` if all attempts fail.
    """
    network_attempt = 0
    while True:
        try:
            kwargs = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
            if reply_markup is not None:
                kwargs["reply_markup"] = reply_markup
            return await app.bot.send_message(**kwargs)
        except RetryAfter as e:
            retry_after = e.retry_after
            # PTB v22.2+ will switch retry_after from float seconds to
            # timedelta. The stub already widens the field to `object`,
            # so handle both shapes explicitly with isinstance.
            if isinstance(retry_after, timedelta):
                delay = retry_after.total_seconds()
            else:
                # Runtime is a float/int today (pre-v22.2). Cast keeps
                # ty quiet without silencing.
                delay = float(cast(float, retry_after))
            await asyncio.sleep(delay + 0.5)
            continue
        except NetworkError as e:
            network_attempt += 1
            if network_attempt >= attempts:
                raise
            backoff = 2 ** (network_attempt - 1)  # 1s, 2s, 4s, ...
            logger.warning(
                "Telegram send attempt %d/%d failed (%s); retrying in %ds",
                network_attempt,
                attempts,
                e,
                backoff,
            )
            await asyncio.sleep(backoff)


def format_money(amount, currency: str | None) -> str:
    """Render an amount with its currency: "\u20b91,234.00" for INR, "USD 12.34"
    for any other currency. A foreign amount must never wear the rupee sign."""
    amount_str = f"{amount:,.2f}"
    code = (currency or "INR").strip().upper()
    if code == "INR":
        return f"\u20b9{amount_str}"
    return f"{html.escape(code)} {amount_str}"


async def send_transaction_notification(
    txn_id: int,
    txn_info: dict,
    chat_id: int,
    *,
    source: Literal["sms", "email"] | None = None,
) -> None:
    """Send a transaction notification. Includes #txn_id for reply matching."""
    app = tg_app
    if not app:
        return
    try:
        is_declined = txn_info.get("_declined", False)
        # A notify-only message carries no ledger row; its parser role picks the
        # label. "declined" is a separate axis (a transaction outcome) and keeps
        # its own flag.
        ledger_role = txn_info.get("_ledger_role")
        direction = txn_info.get("direction", "")
        if is_declined:
            direction_emoji = "\U0001f6ab"
            direction_label = "DECLINED"
        elif ledger_role == "provisional":
            # A payment seen but not yet settled — no transaction row exists
            # yet; the settlement message that follows becomes the row.
            direction_emoji = "⏳"  # hourglass
            direction_label = "PAYMENT RECEIVED — NOT YET SETTLED"
        elif ledger_role == "restatement":
            # A message restating a payment already on the ledger from an
            # earlier one — the gate that sets this makes no row for it.
            direction_emoji = "\U0001f501"  # repeat
            direction_label = "PAYMENT CONFIRMED — ALREADY RECORDED"
        elif direction == "debit":
            direction_emoji = "\U0001f534"
            direction_label = "DEBIT"
        else:
            direction_emoji = "\U0001f7e2"
            direction_label = "CREDIT"
        sign = "-" if direction == "debit" else "+"
        money = format_money(txn_info.get("amount", 0), txn_info.get("currency"))
        bank = html.escape(str(txn_info.get("bank", "")).upper())
        counterparty = html.escape(str(txn_info.get("counterparty", "") or ""))
        card_mask = html.escape(str(txn_info.get("card_mask", "") or ""))
        txn_date = txn_info.get("transaction_date", "")
        txn_time = txn_info.get("transaction_time", "")
        channel = txn_info.get("channel", "")
        account_label = txn_info.get("account_label", "") or ""

        # Build notification text
        id_suffix = f"  #{txn_id}" if txn_id else ""
        lines = [
            f"{direction_emoji} <b>{bank}</b> {direction_label}"
            f"{' · via SMS' if source == 'sms' else ' · via Email' if source == 'email' else ''}"
            f"{id_suffix}",
            f"<b>{sign}{money}</b>",
        ]
        if counterparty:
            # Add channel badge if present
            if channel:
                lines.append(f"{counterparty} \u00b7 <code>{channel}</code>")
            else:
                lines.append(counterparty)

        # Date line with account/card info
        details_parts = []
        if txn_date:
            date_str = html.escape(str(txn_date))
            if txn_time:
                date_str += f" {html.escape(str(txn_time)[:5])}"
            details_parts.append(date_str)
        if account_label:
            details_parts.append(f"Account: {html.escape(account_label)}")
        elif card_mask:
            details_parts.append(f"Card: {card_mask}")
        if details_parts:
            lines.append(" \u00b7 ".join(details_parts))

        text = "\n".join(lines)

        sent = await _send_with_retry(app, chat_id=chat_id, text=text)
        from financial_dashboard.services.assistant.message_context import (
            record_physical_message,
        )

        async with async_session() as session:
            await record_physical_message(
                session,
                chat_id=chat_id,
                message_id=int(sent.message_id),
                transaction_id=txn_id or None,
                context_kind="transaction_notification",
            )
            await session.commit()
    except Exception as e:
        logger.warning(
            "Failed to send Telegram notification for txn #%s: %s", txn_id, e
        )


async def send_bulk_summary(
    count: int,
    chat_id: int,
    *,
    account_label: str | None = None,
    source: str | None = None,
    txns: list[tuple[int, dict]] | None = None,
) -> None:
    """Send a single summary when too many transactions arrive at once."""
    app = tg_app
    if not app:
        return
    try:
        lines = [f"\U0001f4e5 Imported <b>{count}</b> transactions"]

        detail_parts = []
        if account_label:
            detail_parts.append(html.escape(account_label))
        if source:
            _source_display = {"cc_statement": "CC statement", "email": "Email"}
            detail_parts.append(_source_display.get(source, source))
        if detail_parts:
            lines.append(" \u00b7 ".join(detail_parts))

        if txns:
            debits = [t for _, t in txns if t.get("direction") == "debit"]
            credits = [t for _, t in txns if t.get("direction") == "credit"]
            parts = []
            if debits:
                total = sum(float(t.get("amount", 0)) for t in debits)
                parts.append(f"{len(debits)} debits (\u20b9{total:,.2f})")
            if credits:
                total = sum(float(t.get("amount", 0)) for t in credits)
                parts.append(f"{len(credits)} credits (\u20b9{total:,.2f})")
            if parts:
                lines.append(" \u00b7 ".join(parts))

        text = "\n".join(lines)
        await _send_with_retry(app, chat_id=chat_id, text=text)
    except Exception as e:
        logger.warning("Failed to send Telegram bulk summary: %s", e)


def _reply_markup_from_json(raw: str | None) -> InlineKeyboardMarkup | None:
    if not raw:
        return None
    import json

    rows = json.loads(raw)
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    str(button["text"]), callback_data=str(button["callback_data"])
                )
                for button in row
            ]
            for row in rows
        ]
    )


async def dispatch_saved_delivery(delivery_id: int) -> bool:
    """Send one already committed outbox row without repeating its owner work."""
    app = tg_app
    if app is None or not is_telegram_assistant_enabled():
        return False
    from financial_dashboard.db import (
        AuditInteraction,
        CategoryReviewDecision,
        TelegramOutboundDelivery,
    )
    from financial_dashboard.db.models import utc_now
    from sqlalchemy import func, select, update
    from financial_dashboard.services.assistant.audit import mark_authorization_changed
    from financial_dashboard.services.assistant.delivery import (
        abandon_exhausted,
        claim_delivery,
        mark_delivery_delivered,
        mark_delivery_unknown,
        refresh_interaction_delivery_status,
    )
    from financial_dashboard.services.assistant.message_context import (
        record_physical_message,
    )

    async with async_session() as session:
        saved = await session.get(TelegramOutboundDelivery, delivery_id)
        if saved is None or saved.status in {"delivered", "cancelled", "abandoned"}:
            return saved is not None and saved.status == "delivered"
        if saved.recipient_chat_id != get_telegram_chat_id():
            saved.status = "cancelled"
            if saved.interaction_id is not None:
                await mark_authorization_changed(session, saved.interaction_id)
            await session.commit()
            return False
        delivery, worker_token = await claim_delivery(session, delivery_id)
        if delivery is None:
            await abandon_exhausted(session)
            if saved.interaction_id is not None:
                await refresh_interaction_delivery_status(session, saved.interaction_id)
            await session.commit()
            return False
        recipient_chat_id = delivery.recipient_chat_id
        text = delivery.text
        parse_mode = delivery.parse_mode
        reply_markup = _reply_markup_from_json(delivery.reply_markup_json)
        await session.commit()

    try:
        sent = await _send_with_retry(
            app,
            chat_id=recipient_chat_id,
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        async with async_session() as session:
            await mark_delivery_unknown(
                session, delivery_id, worker_token, error=str(exc)[:500]
            )
            await abandon_exhausted(session)
            delivery = await session.get(TelegramOutboundDelivery, delivery_id)
            if delivery is not None and delivery.interaction_id is not None:
                await refresh_interaction_delivery_status(
                    session, delivery.interaction_id
                )
            await session.commit()
        return False

    async with async_session() as session:
        delivery = await session.get(TelegramOutboundDelivery, delivery_id)
        if delivery is None:
            return False
        if not await mark_delivery_delivered(session, delivery_id, worker_token):
            await session.rollback()
            settled = await session.get(
                TelegramOutboundDelivery, delivery_id, populate_existing=True
            )
            return settled is not None and settled.status == "delivered"
        conversation_id = None
        context_kind = "category_review"
        if delivery.interaction_id is not None:
            interaction = await session.get(AuditInteraction, delivery.interaction_id)
            conversation_id = interaction.conversation_id if interaction else None
            context_kind = (
                "query_result"
                if delivery.transaction_id is not None
                else "assistant_response"
            )
        elif delivery.category_review_decision_id is not None:
            if delivery.ordinal == 0:
                active_decision = select(CategoryReviewDecision.id).where(
                    CategoryReviewDecision.id == delivery.category_review_decision_id,
                    CategoryReviewDecision.status == "active",
                    CategoryReviewDecision.source_interaction_id.is_(None),
                )
                await session.execute(
                    update(Transaction)
                    .where(
                        Transaction.id == delivery.transaction_id,
                        Transaction.review_status == "pending",
                        active_decision.exists(),
                    )
                    .values(
                        review_status="notified",
                        last_notified_at=utc_now(),
                        notify_attempts=func.coalesce(Transaction.notify_attempts, 0)
                        + 1,
                    )
                    .execution_options(synchronize_session=False)
                )
        await record_physical_message(
            session,
            chat_id=recipient_chat_id,
            message_id=int(sent.message_id),
            context_kind=context_kind,
            conversation_id=conversation_id,
            transaction_id=delivery.transaction_id,
            interaction_id=delivery.interaction_id,
            outbound_delivery_id=delivery.id,
        )
        if delivery.interaction_id is not None:
            await refresh_interaction_delivery_status(session, delivery.interaction_id)
        await session.commit()
    return True


async def dispatch_pending_deliveries(*, limit: int = 50) -> int:
    """Replay pending outbox rows through the same idempotent dispatcher."""
    from sqlalchemy import select

    from financial_dashboard.db import TelegramOutboundDelivery

    async with async_session() as session:
        ids = list(
            (
                await session.scalars(
                    select(TelegramOutboundDelivery.id)
                    .where(
                        TelegramOutboundDelivery.status.in_(
                            ["pending", "delivery_unknown"]
                        )
                    )
                    .order_by(TelegramOutboundDelivery.id)
                    .limit(limit)
                )
            ).all()
        )
    sent = 0
    for delivery_id in ids:
        sent += int(await dispatch_saved_delivery(delivery_id))
    return sent


async def _handle_callback(update: Update, context) -> None:
    """Route callback queries to appropriate handlers."""
    query = update.callback_query
    if not query or not query.data:
        return

    if query.data.startswith(("cat:v1:", "undo:v1:")):
        if not query.message or query.message.chat.id != get_telegram_chat_id():
            await query.answer("Unauthorized")
            return
        if not is_telegram_assistant_enabled():
            await query.answer("Assistant is disabled")
            return
        await _call_assistant(
            update,
            context,
            trigger="category_button" if query.data.startswith("cat:") else "undo",
        )
        return

    if query.data.startswith("paid:"):
        # function-local: breaks cycle with services.reminders (reminders imports telegram at top)
        from financial_dashboard.services.reminders import handle_mark_paid_callback

        await handle_mark_paid_callback(update, context)
        return
    if query.data.startswith("smsdup:v1:"):
        await _handle_sms_duplicate_callback(update, context)
        return
    if query.data.startswith("cc_pay_pick:"):
        await _handle_cc_pay_pick_callback(update, context)
        return
    await query.answer("Unknown action")


async def _call_assistant(update: Update, context, *, trigger: str) -> bool:
    """Invoke the assistant through its narrow Telegram-facing entrypoint."""
    if not is_telegram_assistant_enabled():
        return False
    try:
        from financial_dashboard.services.assistant.orchestrator import (
            handle_telegram_update,
        )
    except ImportError:
        logger.warning("Telegram assistant orchestrator is unavailable")
        return False
    await handle_telegram_update(update, context, trigger=trigger)
    return True


async def _handle_ask(update: Update, context) -> None:
    msg = update.message
    if not msg or msg.chat_id != get_telegram_chat_id():
        return
    if not is_telegram_assistant_enabled():
        await msg.reply_text("/ask is disabled")
        return
    await _call_assistant(update, context, trigger="ask")


async def _handle_attachment_reply(update: Update, context) -> None:
    msg = update.message
    if not msg or msg.chat_id != get_telegram_chat_id():
        return
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        return
    if msg.reply_to_message.from_user.id != context.bot.id:
        return
    if not is_telegram_assistant_enabled():
        await msg.reply_text("Assistant is disabled")
        return
    await _call_assistant(update, context, trigger="attachment")


async def _handle_reply(update: Update, context) -> None:
    """Handle reply messages — save as transaction note. Only authorized chat."""
    msg = update.message
    if not msg or not msg.text:
        return
    # Only accept from configured chat
    if msg.chat_id != get_telegram_chat_id():
        return
    if not msg.reply_to_message or not msg.reply_to_message.text:
        return
    # Only accept replies to messages sent by this bot
    if (
        not msg.reply_to_message.from_user
        or msg.reply_to_message.from_user.id != context.bot.id
    ):
        return

    if is_telegram_assistant_enabled():
        await _call_assistant(update, context, trigger="reply")
        return

    # Parse transaction ID from the first line of the notification (e.g., "#1234")
    original_text = msg.reply_to_message.text
    first_line = original_text.splitlines()[0] if original_text else ""
    # SMS duplicate notifications use ``SMS #<sms-id>`` and never identify a
    # transaction.  Keep the compatibility parser deliberately narrow.
    if is_sms_duplicate_prompt(original_text):
        return
    match = re.search(r"#(\d+)\s*$", first_line)
    if not match:
        return
    txn_id = int(match.group(1))

    reply_text = msg.text.strip()
    if not reply_text:
        return

    # Reply format: note on line 1, optional category on line 2+.
    # If the reply has no second line, category is left untouched.
    first_line, sep, rest = reply_text.partition("\n")
    note_text = first_line.strip() or None
    category_text = rest.strip() if sep else None

    async with async_session() as session:
        txn = await session.get(Transaction, txn_id)
        if txn:
            txn.note = note_text
            await session.commit()
            category_saved = False
            if category_text is not None:
                from financial_dashboard.services.categorization.manual import (
                    assign_category_manual,
                )

                category_saved, _ = await assign_category_manual(
                    session, txn_id, category_text or ""
                )
            saved = "note" + (" + category" if category_saved else "")
            reply = f"Saved {saved} for #{txn_id}"
            if category_text and not category_saved:
                reply += f" (category {category_text!r} was invalid, not saved)"
            await msg.reply_text(reply)
        else:
            await msg.reply_text(f"Transaction #{txn_id} not found")


def _format_diff_value(key: str, value) -> str:
    """Render a diff value compactly. Time values keep HH:MM:SS so a
    diff like 12:55:35→12:55:20 doesn't render as 12:55→12:55 (which
    looks like a no-op). Everything else is HTML-escaped."""
    if key == "transaction_time" and value is not None:
        return html.escape(str(value)[:8])
    return html.escape(str(value))


async def send_enrichment_notification(
    txn_id: int,
    diff,  # EnrichmentDiff (typed loosely to avoid circular import)
    chat_id: int,
    *,
    source: Literal["sms", "email"],
    txn_info: dict | None = None,
) -> None:
    """Fire a follow-up Telegram message describing what changed when
    a second source enriched an existing Transaction.

    Caller MUST gate this on ``diff.changed_fields`` being non-empty.
    When ``txn_info`` is provided (bank/direction/amount/counterparty),
    the notification renders as a single inline line:

        🔄 HDFC #1234 -₹500.00 Zomato — filled transaction_time

    Otherwise it falls back to the txn-id-only form.
    """
    app = tg_app
    if not app:
        return
    badge = "via SMS" if source == "sms" else "via Email"
    try:
        # Build the diff fragment: ignore raw_description (debug-only).
        filled = [
            f"{k}={_format_diff_value(k, v)}"
            for k, v in diff.filled.items()
            if k != "raw_description"
        ]
        overwritten = [
            f"{k}: {_format_diff_value(k, old)}→{_format_diff_value(k, new)}"
            for k, (old, new) in diff.overwritten.items()
            if k != "raw_description"
        ]
        diff_parts = []
        if filled:
            diff_parts.append(f"filled {', '.join(filled)}")
        if overwritten:
            diff_parts.append(f"updated {', '.join(overwritten)}")
        if not diff_parts:
            # Only raw_description changed — stay silent.
            return
        diff_text = " · ".join(diff_parts)

        if txn_info:
            direction = txn_info.get("direction", "")
            sign = "-" if direction == "debit" else "+"
            money = format_money(txn_info.get("amount", 0), txn_info.get("currency"))
            bank = html.escape(str(txn_info.get("bank", "")).upper())
            counterparty = html.escape(str(txn_info.get("counterparty", "") or ""))
            header = f"\U0001f504 <b>{bank}</b> {sign}{money}"
            if counterparty:
                header += f" {counterparty}"
            header += f" — {diff_text} ({badge}) #{txn_id}"
            text = header
        else:
            text = f"\U0001f504 enriched {badge} — {diff_text} #{txn_id}"
        sent = await _send_with_retry(app, chat_id=chat_id, text=text)
        # Enrichment messages are replyable assistant context even though they
        # are emitted by legacy source paths. Persist the physical mapping after
        # send; the trailing transaction id remains a guarded recovery fallback
        # for the crash window between Telegram accepting the message and this
        # write completing.
        async with async_session() as session:
            from financial_dashboard.services.assistant.message_context import (
                record_physical_message,
            )

            await record_physical_message(
                session,
                chat_id=chat_id,
                message_id=int(sent.message_id),
                context_kind="enrichment",
                transaction_id=txn_id,
            )
            await session.commit()
    except Exception as e:
        logger.warning(
            "Failed to send enrichment notification for txn #%s: %s", txn_id, e
        )


def _parse_sms_duplicate_callback(
    data: str,
) -> tuple[Literal["merge", "create_new"], int, int | None] | None:
    if len(data.encode()) > 64:
        return None
    parts = data.split(":")
    if len(parts) == 5 and parts[:3] == ["smsdup", "v1", "m"]:
        action: Literal["merge", "create_new"] = "merge"
        raw_ids = parts[3:]
    elif len(parts) == 4 and parts[:3] == ["smsdup", "v1", "n"]:
        action = "create_new"
        raw_ids = parts[3:]
    else:
        return None
    try:
        ids = [int(value) for value in raw_ids]
    except ValueError:
        return None
    if any(value <= 0 for value in ids):
        return None
    transaction_id = ids[1] if action == "merge" else None
    return action, ids[0], transaction_id


async def send_sms_duplicate_disambiguation_prompt(payload: dict, chat_id: int) -> None:
    """Send the deferred SMS duplicate decision keyboard."""
    app = tg_app
    if not app:
        return

    sms_id = int(payload["sms_id"])
    candidate_ids = [
        int(value) for value in payload["resolution_candidate_ids"] if int(value) > 0
    ]
    bank = html.escape(str(payload.get("bank", "")).upper())
    direction = html.escape(str(payload.get("direction", "")).upper())
    amount = html.escape(f"{Decimal(str(payload.get('amount', 0))):,.2f}")
    counterparty = html.escape(str(payload.get("counterparty") or ""))
    transaction_date = html.escape(str(payload.get("transaction_date") or ""))
    lines = [
        f"⚠️ <b>{bank}</b> {direction} SMS #{sms_id}",
        f"₹{amount}" + (f" · {counterparty}" if counterparty else ""),
    ]
    if transaction_date:
        lines.append(transaction_date)

    buttons = [
        [
            InlineKeyboardButton(
                f"Merge into #{transaction_id}",
                callback_data=f"smsdup:v1:m:{sms_id}:{transaction_id}",
            )
        ]
        for transaction_id in candidate_ids
    ]
    reason = str(payload.get("reason") or "")
    # A reference mismatch cannot create a second row: the unique reference
    # index forbids it. Offer no Create-new button for those reasons.
    ref_conflict = reason.startswith("reference_") or reason == (
        "multiple_reference_candidates"
    )
    if not ref_conflict:
        buttons.append(
            [InlineKeyboardButton("Create new", callback_data=f"smsdup:v1:n:{sms_id}")]
        )
    if buttons:
        lines.append("Possible duplicate. Choose an action below.")
    else:
        lines.append("Possible duplicate. Resolve it on the web.")
    # TODO: Persist duplicate prompts in a transactional outbox before dispatch.
    await app.bot.send_message(
        chat_id=chat_id,
        text="\n".join(lines),
        reply_markup=InlineKeyboardMarkup(buttons) if buttons else None,
        parse_mode="HTML",
    )


async def _handle_sms_duplicate_callback(update: Update, context) -> None:
    """Resolve an authorized deferred SMS duplicate callback."""
    query = update.callback_query
    if not query or not query.data:
        return
    if not query.message:
        await query.answer("Message no longer available")
        return
    if query.message.chat.id != get_telegram_chat_id():
        await query.answer("Unauthorized")
        return

    parsed = _parse_sms_duplicate_callback(query.data)
    if parsed is None:
        await query.answer("Invalid callback")
        return
    action, sms_id, transaction_id = parsed

    from financial_dashboard.services.sms_duplicate_resolution import (
        SmsDuplicateResolutionError,
        resolve_sms_duplicate,
    )

    try:
        async with async_session() as session:
            result = await resolve_sms_duplicate(
                session, sms_id, action, transaction_id
            )
    except SmsDuplicateResolutionError as exc:
        await query.answer(str(exc))
        return
    except OperationalError:
        # A rival tap holds the SMS write lock past the busy timeout.
        await query.answer("Busy, try again")
        return

    await query.answer()
    if result.status == "already_resolved":
        text = f"Already resolved as #{result.transaction_id}"
    elif result.status == "merged":
        text = f"SMS #{sms_id} merged into #{result.transaction_id}"
    else:
        text = f"SMS #{sms_id} created transaction #{result.transaction_id}"
    try:
        await query.edit_message_text(text)
    except Exception as exc:
        logger.warning("SMS duplicate callback edit failed: %s", exc)

    if result.pending_payment_check is not None:
        from financial_dashboard.services.reminders import check_payment_received

        try:
            await check_payment_received(*result.pending_payment_check)
        except Exception as exc:
            logger.warning("SMS duplicate payment check failed: %s", exc)
    if result.pending_disambiguation is not None:
        try:
            await send_disambiguation_prompt(
                result.pending_disambiguation, get_telegram_chat_id()
            )
        except Exception as exc:
            logger.warning("SMS duplicate account picker failed: %s", exc)


async def send_disambiguation_prompt(payload: dict, chat_id: int) -> None:
    """Telegram inline-keyboard prompt for CC payment account picker.

    payload shape (built by sms_pipeline._build_disambiguation):
      {txn_id, candidate_account_ids, candidate_labels, amount, bank}
    """
    app = tg_app
    if not app:
        return
    txn_id = payload["txn_id"]
    amount = payload["amount"]
    bank = payload["bank"]
    text = (
        f"\U0001f4b3 #{txn_id} — couldn't auto-match this payment to a card\n"
        f"+₹{Decimal(str(amount)):,.2f} · {html.escape(bank)}\n"
        f"Which card did you pay?"
    )
    buttons = []
    for acct_id in payload["candidate_account_ids"]:
        label = payload["candidate_labels"].get(acct_id, f"Account #{acct_id}")
        buttons.append(
            [
                InlineKeyboardButton(
                    label, callback_data=f"cc_pay_pick:{txn_id}:{acct_id}"
                )
            ]
        )
    buttons.append(
        [InlineKeyboardButton("Skip", callback_data=f"cc_pay_pick:{txn_id}:skip")]
    )
    await app.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup(buttons),
        parse_mode="HTML",
    )


async def _handle_cc_pay_pick_callback(update, context) -> None:
    """User picked which CC account a maskless payment-received SMS hit."""
    from financial_dashboard.db import Transaction, async_session
    from financial_dashboard.services.reminders import check_payment_received

    query = update.callback_query
    await query.answer()
    # Format: cc_pay_pick:{txn_id}:{account_id|skip}
    parts = query.data.split(":", 2)
    if len(parts) != 3:
        return
    _, txn_id_str, choice = parts
    try:
        txn_id = int(txn_id_str)
    except ValueError:
        return

    if choice == "skip":
        await query.edit_message_text(f"#{txn_id}: skipped (no statement marked paid)")
        return

    try:
        account_id = int(choice)
    except ValueError:
        return

    async with async_session() as session:
        async with session.begin():
            txn = await session.get(Transaction, txn_id)
            if txn is None:
                await query.edit_message_text(f"#{txn_id}: transaction not found")
                return
            txn.account_id = account_id
            amount = txn.amount

    try:
        await check_payment_received(txn_id, account_id, amount)
    except Exception as exc:
        logger.warning("check_payment_received failed for txn %s: %s", txn_id, exc)

    await query.edit_message_text(f"#{txn_id}: applied to account #{account_id}")
