# ty: ignore
"""Telegram bot for transaction notifications and note replies.

Sends a notification for each new transaction (post-backfill only).
If the user replies to a notification, the reply text is saved as the
transaction's note. Only the configured chat_id is authorized.
"""

import asyncio
import html
import logging
import re

from telegram import Update
from telegram.error import NetworkError, RetryAfter
from telegram.ext import Application, CallbackQueryHandler, MessageHandler, filters

from financial_dashboard.db import Transaction, async_session
from financial_dashboard.services.settings import get_telegram_chat_id

logger = logging.getLogger(__name__)

tg_app: Application | None = None


async def init_telegram(token: str):
    """Initialize the Telegram bot application."""
    global tg_app
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & filters.REPLY, _handle_reply))
    app.add_handler(CallbackQueryHandler(_handle_callback))
    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(
            drop_pending_updates=True, allowed_updates=["message", "callback_query"]
        )
    except Exception:
        try:
            if app.updater.running:
                await app.updater.stop()
            if app.running:
                await app.stop()
            await app.shutdown()
        except Exception:
            pass
        raise
    tg_app = app
    logger.info("Telegram bot started")


async def shutdown_telegram():
    """Shutdown the Telegram bot."""
    global tg_app
    if tg_app:
        app = tg_app
        tg_app = None
        try:
            if app.updater.running:
                await app.updater.stop()
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


async def _send_with_retry(app, *, chat_id, text, parse_mode="HTML", attempts=3):
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
            return await app.bot.send_message(
                chat_id=chat_id, text=text, parse_mode=parse_mode
            )
        except RetryAfter as e:
            retry_after = e.retry_after
            delay = (
                retry_after.total_seconds()
                if hasattr(retry_after, "total_seconds")
                else float(retry_after)
            )
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


async def send_transaction_notification(
    txn_id: int, txn_info: dict, chat_id: int
) -> None:
    """Send a transaction notification. Includes #txn_id for reply matching."""
    app = tg_app
    if not app:
        return
    try:
        is_declined = txn_info.get("_declined", False)
        direction = txn_info.get("direction", "")
        if is_declined:
            direction_emoji = "\U0001f6ab"
            direction_label = "DECLINED"
        elif direction == "debit":
            direction_emoji = "\U0001f534"
            direction_label = "DEBIT"
        else:
            direction_emoji = "\U0001f7e2"
            direction_label = "CREDIT"
        sign = "-" if direction == "debit" else "+"
        amount = txn_info.get("amount", 0)
        amount_str = f"{amount:,.2f}"
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
            f"{direction_emoji} <b>{bank}</b> {direction_label}{id_suffix}",
            f"<b>{sign}\u20b9{amount_str}</b>",
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

        await _send_with_retry(app, chat_id=chat_id, text=text)
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


async def _handle_callback(update: Update, context) -> None:
    """Route callback queries to appropriate handlers."""
    query = update.callback_query
    if not query or not query.data:
        return

    if query.data.startswith("paid:"):
        # function-local: breaks cycle with services.reminders (reminders imports telegram at top)
        from financial_dashboard.services.reminders import handle_mark_paid_callback

        await handle_mark_paid_callback(update, context)
    else:
        await query.answer("Unknown action")


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

    # Parse transaction ID from the first line of the notification (e.g., "#1234")
    original_text = msg.reply_to_message.text
    first_line = original_text.splitlines()[0] if original_text else ""
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
            if category_text is not None:
                txn.category = category_text or None
            await session.commit()
            saved = "note" + (" + category" if category_text is not None else "")
            await msg.reply_text(f"Saved {saved} for #{txn_id}")
        else:
            await msg.reply_text(f"Transaction #{txn_id} not found")
