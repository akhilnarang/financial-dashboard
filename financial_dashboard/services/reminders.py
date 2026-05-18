# ty: ignore
"""Payment reminder logic: scheduling, mark-as-paid, and auto-detection.

Payment tracking fields live on StatementUpload directly:
- payment_status: PaymentStatus enum (NULL = no due date)
- payment_sent_offsets: JSON list of reminder day-offsets already sent
- payment_paid_amount: cumulative payment amount
- payment_paid_at: when fully paid

Provides:
- init_payment_tracking(): sets payment_status on a StatementUpload with a due date
- check_and_send_reminders(): poll-loop hook that sends due-date reminders
- handle_mark_paid_callback(): Telegram inline button handler
- check_payment_received(): auto-marks statements paid on incoming credits
"""

import html
import json
import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from financial_dashboard.db import (
    Account,
    PaymentStatus,
    StatementUpload,
    async_session,
)
from financial_dashboard.services import telegram as telegram_service
from financial_dashboard.services.settings import (
    get_setting_bool,
    get_setting_json,
    get_telegram_chat_id,
    is_telegram_configured,
)
from financial_dashboard.services.statements.cc import parse_cc_amount, parse_cc_date

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = (
    PaymentStatus.UNPAID,
    PaymentStatus.PARTIALLY_PAID,
    PaymentStatus.LATE,
)
STALE_DAYS = 7  # stop reminding after this many days past due


def latest_per_account(uploads: list[StatementUpload]) -> list[StatementUpload]:
    """Collapse a list of StatementUploads to at most one per account — the latest by due_date.

    Why: a newer statement supersedes all older cycles (any unpaid balance rolls into it),
    so reminders and payment auto-match should only ever act on the most recent cycle per
    account. Uploads with unparseable due_date are skipped entirely since we can't rank them.
    """
    latest: dict[int, tuple[date, StatementUpload]] = {}
    for upload in uploads:
        if not upload.due_date:
            continue
        try:
            due = parse_cc_date(upload.due_date)
        except ValueError, InvalidOperation:
            continue
        current = latest.get(upload.account_id)
        if current is None or due > current[0]:
            latest[upload.account_id] = (due, upload)
    return [pair[1] for pair in latest.values()]


async def init_payment_tracking(statement_upload_id: int) -> bool:
    """Set payment_status on a StatementUpload if it has a valid due date.

    Skips if: no due_date, unparseable, more than 1 day past, or already tracked.
    Never raises — logs errors internally.
    """
    try:
        async with async_session() as session:
            if not (upload := await session.get(StatementUpload, statement_upload_id)):
                return False
            if (
                not upload.due_date
                or not upload.total_amount_due
                or upload.payment_status is not None
            ):
                return False

            try:
                due = parse_cc_date(upload.due_date)
                amount_due = parse_cc_amount(upload.total_amount_due)
            except ValueError, InvalidOperation:
                return False

            today = date.today()
            if due < today.replace(day=1):
                return False

            if amount_due <= 0:
                upload.payment_status = PaymentStatus.PAID
                upload.payment_paid_at = datetime.now(timezone.utc)
                upload.payment_paid_amount = amount_due
            else:
                upload.payment_status = PaymentStatus.UNPAID

            await session.commit()
            logger.info(
                "Payment tracking initialized for statement #%s: due=%s status=%s",
                upload.id,
                due,
                upload.payment_status,
            )
            return True
    except Exception:
        logger.exception(
            "Failed to init payment tracking for statement %s", statement_upload_id
        )
        return False


async def check_and_send_reminders() -> int:
    """Check all active statements and send Telegram reminder notifications.

    Called from the poll loop after poll_all(). Returns count of messages sent.
    """
    if not is_telegram_configured() or not get_setting_bool(
        "telegram.notify_reminders"
    ):
        return 0

    offsets = get_setting_json("telegram.reminder_days_before", [7, 3, 1, 0])
    if not offsets:
        return 0

    chat_id = get_telegram_chat_id()
    today = date.today()
    sent_count = 0

    async with async_session() as session:
        all_active = (
            (
                await session.execute(
                    select(StatementUpload).where(
                        StatementUpload.payment_status.in_(ACTIVE_STATUSES),
                    )
                )
            )
            .scalars()
            .all()
        )
        # Why: only remind on the latest cycle per account; older superseded statements
        # are no longer actionable (any outstanding rolls into the new statement).
        uploads = latest_per_account(list(all_active))

        for upload in uploads:
            if not upload.due_date:
                continue
            try:
                due = parse_cc_date(upload.due_date)
            except ValueError, InvalidOperation, AttributeError:
                continue

            days_past = (today - due).days

            # Auto-transition to late (before stale check so status is persisted)
            if (
                upload.payment_status
                in (PaymentStatus.UNPAID, PaymentStatus.PARTIALLY_PAID)
                and today > due
            ):
                upload.payment_status = PaymentStatus.LATE

            if days_past > STALE_DAYS:
                continue

            already_sent = set(json.loads(upload.payment_sent_offsets or "[]"))

            try:
                amount_due = parse_cc_amount(upload.total_amount_due)
            except ValueError, InvalidOperation:
                continue

            # Nothing owed — auto-mark as paid and skip
            if amount_due <= 0:
                upload.payment_status = PaymentStatus.PAID
                continue

            days_until_due = (due - today).days  # positive = future, negative = overdue
            changed = False
            sent_this_cycle = False

            for offset in offsets:
                if offset in already_sent:
                    continue
                # Skip advance reminders for already-overdue statements
                if upload.payment_status == PaymentStatus.LATE and offset > 0:
                    already_sent.add(offset)
                    changed = True
                    continue
                trigger_date = due - timedelta(days=offset)
                if today >= trigger_date:
                    already_sent.add(offset)
                    changed = True
                    # Send at most one notification per statement per poll cycle
                    if not sent_this_cycle:
                        await _send_reminder_notification(
                            upload,
                            due,
                            amount_due,
                            days_until_due,
                            chat_id,
                        )
                        upload.payment_last_reminded_at = datetime.now(timezone.utc)
                        sent_count += 1
                        sent_this_cycle = True

            if changed:
                upload.payment_sent_offsets = json.dumps(sorted(already_sent))

        await session.commit()

    return sent_count


async def _send_reminder_notification(upload, due, amount_due, days_until_due, chat_id):
    """Send a single reminder notification with a Mark as Paid button.

    days_until_due: positive = days remaining, 0 = today, negative = overdue.
    """
    app = telegram_service.tg_app
    if not app:
        return

    try:
        account_label = html.escape(
            upload.account.label if upload.account else "Unknown"
        )
        bank = html.escape((upload.account.bank if upload.account else "").upper())
        amount_str = f"{amount_due:,.2f}"
        due_str = due.strftime("%d %b %Y")

        if days_until_due > 0:
            time_str = f"in {days_until_due} day{'s' if days_until_due != 1 else ''}"
        elif days_until_due == 0:
            time_str = "today"
        else:
            days_late = abs(days_until_due)
            time_str = f"{days_late} day{'s' if days_late != 1 else ''} overdue"

        if days_until_due < 0:
            emoji = "\U0001f534"
            header = "Payment Overdue"
            date_line = f"\u20b9{amount_str} was due {due_str} ({time_str})"
        else:
            emoji = "\u23f0"
            header = "Payment Due"
            date_line = f"\u20b9{amount_str} due {due_str} ({time_str})"

        if bank and account_label.upper().startswith(f"{bank} "):
            title_suffix = account_label
        else:
            title_suffix = f"{bank} {account_label}".strip()

        lines = [
            f"{emoji} <b>{header}</b> \u2014 {title_suffix}",
            date_line,
        ]

        if upload.payment_paid_amount and upload.payment_paid_amount > 0:
            paid = f"{Decimal(str(upload.payment_paid_amount)):,.2f}"
            remaining = f"{amount_due - Decimal(str(upload.payment_paid_amount)):,.2f}"
            lines.append(f"\u20b9{paid} paid so far \u2014 \u20b9{remaining} remaining")

        text = "\n".join(lines)
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Mark as Paid", callback_data=f"paid:{upload.id}")]]
        )
        await app.bot.send_message(
            chat_id=chat_id, text=text, parse_mode="HTML", reply_markup=keyboard
        )
    except Exception as e:
        logger.warning("Failed to send reminder for statement #%s: %s", upload.id, e)


async def handle_mark_paid_callback(update, context) -> None:
    """Handle the 'Mark as Paid' inline button callback."""
    query = update.callback_query
    if not query or not query.data:
        return

    if not query.message:
        await query.answer("Message no longer available")
        return

    if query.message.chat_id != get_telegram_chat_id():
        await query.answer("Unauthorized")
        return

    try:
        upload_id = int(query.data.split(":")[1])
    except IndexError, ValueError:
        await query.answer("Invalid callback")
        return

    async with async_session() as session:
        if not (upload := await session.get(StatementUpload, upload_id)):
            await query.answer("Statement not found")
            return

        if upload.payment_status == PaymentStatus.PAID:
            await query.answer("Already marked as paid")
            return

        upload.payment_status = PaymentStatus.PAID
        upload.payment_paid_at = datetime.now(timezone.utc)
        if upload.total_amount_due:
            try:
                upload.payment_paid_amount = parse_cc_amount(upload.total_amount_due)
            except ValueError, InvalidOperation:
                pass
        await session.commit()

    await query.answer("Marked as paid!")

    try:
        original_text = query.message.text or ""
        await query.edit_message_text(
            text=original_text + "\n\n\u2705 Marked as paid",
        )
    except Exception:
        pass


async def check_payment_received(txn_id: int, account_id: int, amount) -> bool:
    """Check if a credit transaction satisfies a pending statement payment.

    Called after a credit transaction is committed. Updates payment status
    and sends a Telegram notification if enabled.
    """
    amount = Decimal(str(amount))

    async with async_session() as session:
        if (
            not (account := await session.get(Account, account_id))
            or account.type != "credit_card"
        ):
            return False

        candidates = (
            (
                await session.execute(
                    select(StatementUpload).where(
                        StatementUpload.account_id == account_id,
                        StatementUpload.payment_status.in_(ACTIVE_STATUSES),
                        StatementUpload.due_date.isnot(None),
                    )
                )
            )
            .scalars()
            .all()
        )

        # Why: always act on the latest cycle per account — any unpaid balance from older
        # cycles rolls into the new statement, so applying a payment to an older row would
        # double-count.
        latest = latest_per_account(list(candidates))
        if not latest:
            return False
        upload = latest[0]

        try:
            amount_due = parse_cc_amount(upload.total_amount_due)
        except ValueError, InvalidOperation:
            return False

        new_paid = Decimal(str(upload.payment_paid_amount or 0)) + amount
        upload.payment_paid_amount = new_paid
        fully_paid = new_paid >= amount_due

        if fully_paid:
            upload.payment_status = PaymentStatus.PAID
            upload.payment_paid_at = datetime.now(timezone.utc)
        else:
            upload.payment_status = PaymentStatus.PARTIALLY_PAID

        logger.info(
            "Statement #%s %s: received=%s total_paid=%s due=%s",
            upload.id,
            "fully paid" if fully_paid else "partially paid",
            amount,
            new_paid,
            amount_due,
        )

        if is_telegram_configured() and get_setting_bool(
            "telegram.notify_payment_received"
        ):
            await _send_payment_received_notification(
                upload,
                account,
                amount,
                new_paid,
                amount_due,
                get_telegram_chat_id(),
            )

        await session.commit()

    return fully_paid


async def _send_payment_received_notification(
    upload,
    account,
    credit_amount,
    total_paid,
    amount_due,
    chat_id,
) -> None:
    """Send a Telegram notification about a detected payment."""
    app = telegram_service.tg_app
    if not app:
        return

    try:
        account_label = html.escape(account.label)
        bank = html.escape(account.bank.upper())
        credit_str = f"{credit_amount:,.2f}"

        if total_paid >= amount_due:
            lines = [
                f"\u2705 <b>Payment Received</b> \u2014 {bank} {account_label}",
                f"\u20b9{credit_str} received \u2014 fully paid",
            ]
        else:
            remaining = f"{amount_due - total_paid:,.2f}"
            due_str = f"{amount_due:,.2f}"
            lines = [
                f"\U0001f4b3 <b>Partial Payment</b> \u2014 {bank} {account_label}",
                f"\u20b9{credit_str} received (\u20b9{remaining} remaining of \u20b9{due_str} due)",
            ]

        text = "\n".join(lines)
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        logger.warning("Failed to send payment-received notification: %s", e)
