"""Provisional transactions from UPI screenshots sent to the Telegram bot.

The vision model reads the screenshot. Code checks every mandatory field.
Nothing is saved without a Yes tap. A bank alert or statement with the same
UTR later confirms the row. A row that no bank source confirms in 60 days is
flagged once on Telegram. The owner replies with what happened, and the
assistant proposes the fix.
"""

import datetime
import logging
import re
from decimal import Decimal
from typing import NamedTuple

from pydantic import ValidationError
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import IST, parse_datetime
from financial_dashboard.core.masks import mask_last4
from financial_dashboard.core.templating import format_inr_exact
from financial_dashboard.db import async_session
from financial_dashboard.db.models import (
    Account,
    AuditInteraction,
    Transaction,
    utc_now,
)
from financial_dashboard.services.assistant.contracts import (
    ScreenshotTransaction,
    ScreenshotTransactionConfirmation,
)
from financial_dashboard.services.assistant.mutations import MutationRejected
from financial_dashboard.services.settings import (
    get_telegram_chat_id,
    is_telegram_assistant_enabled,
    is_telegram_configured,
)
from financial_dashboard.services.transaction_attachments import (
    StoredAttachment,
    attach_downloaded_attachment,
    resolve_attachment_path,
)
from financial_dashboard.services.transactions import set_transaction_voided_no_commit

logger = logging.getLogger(__name__)

FLAG_AGE = datetime.timedelta(days=60)

_UTR = re.compile(r"\d{12}")
_TRAILING_LAST4 = re.compile(r"(\d{4})[\s).]*$")
_KEYED_AMOUNT = re.compile(
    r"(?:(?:is|was)\s+)?(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d{1,2})?)"
    r"\s*(?:rupees|rs\.?|inr|/-)?\.?",
    re.IGNORECASE,
)
_RUPEES = re.compile(
    r"(?:₹|\brs\.?|\binr)\s*(\d[\d,]*(?:\.\d{1,2})?)(?![.,]?\d)", re.IGNORECASE
)


class ScreenshotCheck(NamedTuple):
    """The resolved account, the failed mandatory fields and a stored duplicate."""

    account: Account | None
    missing: list[str]
    duplicate: Transaction | None


class ScreenshotReview(NamedTuple):
    """The reply to a screenshot, and the action a Yes applies, if any."""

    text: str
    action: ScreenshotTransactionConfirmation | None


async def _check(
    session: AsyncSession, read: ScreenshotTransaction, not_duplicates: list[int]
) -> ScreenshotCheck:
    """Run the mandatory-field and duplicate checks on one read.

    The account resolves only by an exact last-4 match to exactly one active
    bank account. The hint must end with those 4 digits. A duplicate is a row
    with the same UTR and direction. Otherwise it is a live row with the same
    account, direction and amount within one day whose reference is not
    another UTR, and that the owner did not call another payment.
    """
    account = None
    if (hint := _TRAILING_LAST4.search(read.account_hint or "")) is not None:
        last4 = hint.group(1)
        accounts = [
            row
            for row in await session.scalars(
                select(Account).where(
                    Account.type == "bank_account", Account.active.is_(True)
                )
            )
            if mask_last4(row.account_number) == last4
        ]
        account = accounts[0] if len(accounts) == 1 else None
    now = datetime.datetime.now(IST).replace(tzinfo=None)
    missing = [
        field
        for field, ok in (
            ("amount", read.amount is not None and read.amount > 0),
            ("name or VPA", bool(read.counterparty or read.vpa)),
            ("UTR (12 digits)", _UTR.fullmatch(read.utr or "") is not None),
            (
                "date and time",
                read.occurred_at is not None and read.occurred_at <= now,
            ),
            ("account (last 4 digits)", account is not None),
        )
        if not ok
    ]
    utr = read.utr if read.utr and _UTR.fullmatch(read.utr) else None
    duplicate = (
        await session.scalar(
            select(Transaction)
            .where(
                Transaction.reference_number == utr,
                Transaction.direction == read.direction,
            )
            .order_by(Transaction.id)
            .limit(1)
        )
        if utr
        else None
    )
    if duplicate is None and account is not None and read.amount and read.occurred_at:
        day = read.occurred_at.date()
        duplicate = await session.scalar(
            select(Transaction)
            .where(
                Transaction.account_id == account.id,
                Transaction.direction == read.direction,
                Transaction.amount == read.amount,
                Transaction.transaction_date.between(
                    day - datetime.timedelta(days=1), day + datetime.timedelta(days=1)
                ),
                Transaction.id.not_in(not_duplicates),
                Transaction.voided.is_(False),
                or_(
                    Transaction.reference_number.is_(None),
                    ~Transaction.reference_number.regexp_match(r"^\d{12}$"),
                ),
            )
            .order_by(Transaction.id)
            .limit(1)
        )
    return ScreenshotCheck(account, missing, duplicate)


async def review_screenshot(
    session: AsyncSession,
    read: ScreenshotTransaction,
    *,
    attachment_path: str,
    media_type: str,
    not_duplicates: list[int],
) -> ScreenshotReview:
    """Check one screenshot read and word the question for the owner.

    A screenshot that does not show a success is refused. A duplicate offers
    to attach the screenshot to the stored row. A read with a missing field
    names each field and asks for it. Otherwise the question asks to save.

    Args:
        session: The turn session.
        read: The fields read off the screenshot, with any corrections.
        attachment_path: The stored screenshot file.
        media_type: Its MIME type.

    Returns:
        The reply text, and the action that a Yes applies. The action is
        ``None`` for a refused screenshot.
    """
    if read.status != "success":
        return ScreenshotReview(
            f"This screenshot shows a {read.status} payment, not a success. "
            "I saved nothing.",
            None,
        )
    account, missing, duplicate = await _check(session, read, not_duplicates)
    credit = read.direction == "credit"
    party = " ".join(
        part
        for part in (read.counterparty, f"({read.vpa})" if read.vpa else "")
        if part
    )
    lines = [
        f"UPI screenshot ({read.app}): "
        f"{'received' if credit else 'paid'} "
        f"{format_inr_exact(read.amount) if read.amount is not None else '?'} "
        f"{'from' if credit else 'to'} {party[:200] or '?'}",
        f"UTR {read.utr or '?'} · "
        f"{read.occurred_at.strftime('%Y-%m-%d %H:%M') if read.occurred_at else '?'}"
        f" · {account.label if account else read.account_hint or '?'}",
    ]
    if duplicate is not None and read.utr and duplicate.reference_number == read.utr:
        lines.append(
            f"This is recorded as #{duplicate.id}, which is voided. Attach the "
            "screenshot and un-void it? The screenshot shows a success."
            if duplicate.voided
            else f"This is already recorded as #{duplicate.id}. "
            "Attach the screenshot to it instead?"
        )
    elif duplicate is not None:
        lines.append(
            f"This may already be recorded as #{duplicate.id} "
            f"({(duplicate.counterparty or 'unknown')[:200]}, "
            f"{duplicate.transaction_date}). Attach the screenshot to it, or "
            'reply "new" if it is another payment.'
        )
    elif missing:
        lines.append(f"Missing or not valid: {', '.join(missing)}.")
        lines.append(
            'Reply with the values, for example "utr 123456789012, account 1234", '
            "or tap Skip."
        )
    else:
        lines.append("Save this as a provisional transaction?")
    return ScreenshotReview(
        "\n".join(lines),
        ScreenshotTransactionConfirmation(
            kind="screenshot_transaction",
            read=read,
            attachment_path=attachment_path,
            media_type=media_type,
            missing=missing,
            duplicate_of=duplicate.id if duplicate is not None else None,
            duplicate_voided=duplicate is not None and duplicate.voided,
            not_duplicates=not_duplicates,
        ),
    )


def apply_corrections(read: ScreenshotTransaction, text: str) -> ScreenshotTransaction:
    """Apply the owner's reply, such as "utr 123456789012, account 1234".

    Each comma, semicolon or line holds one ``key value`` pair. A comma before
    a digit groups the digits of an amount instead. A part with no known key
    can still give a rupee amount, such as "it was ₹950". A value that does
    not parse is ignored, so the checks name its field again.

    Args:
        read: The current read.
        text: The reply.

    Returns:
        The read with the corrected fields, validated again.
    """
    updates: dict[str, str | Decimal | datetime.datetime | None] = {}
    for part in re.split(r"[;\n]|,(?!\d)", text):
        key, _, value = part.strip().partition(" ")
        value = value.strip()
        match key.lower().rstrip(":"):
            case "utr" | "ref" | "rrn":
                updates["utr"] = value
            case "account" | "acct" | "a/c":
                updates["account_hint"] = value
            case "name" | "to" | "from":
                updates["counterparty"] = value or None
            case "vpa" | "upi":
                updates["vpa"] = value or None
            case "amount":
                if (found := _KEYED_AMOUNT.fullmatch(value)) is not None:
                    updates["amount"] = Decimal(found.group(1).replace(",", ""))
            case "date" | "time" | "on" | "at":
                try:
                    updates["occurred_at"] = parse_datetime(
                        value,
                        dayfirst=True,
                        default=read.occurred_at
                        or datetime.datetime.now(IST).replace(
                            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
                        ),
                    )
                except ValueError, OverflowError:
                    logger.info("Screenshot date correction did not parse")
            case _ if (found := _RUPEES.search(part)) is not None:
                updates["amount"] = Decimal(found.group(1).replace(",", ""))
    for field, value in updates.items():
        try:
            read = ScreenshotTransaction.model_validate(
                read.model_dump() | {field: value}
            )
        except ValidationError:
            logger.info("Screenshot %s correction did not validate", field)
    return read


async def save_screenshot_transaction(
    session: AsyncSession,
    action: ScreenshotTransactionConfirmation,
    *,
    interaction_id: int | None,
) -> str:
    """Save the row a screenshot shows, or attach the screenshot to its duplicate.

    The checks run again here, because a row can arrive or change after the
    question.

    Args:
        session: The confirmation session, which holds the write lock.
        action: The confirmed screenshot action.
        interaction_id: The turn that owns the audit action.

    Returns:
        The reply text.

    Raises:
        MutationRejected: A check fails now.
    """
    stored = StoredAttachment(
        action.attachment_path,
        resolve_attachment_path(action.attachment_path),
        action.media_type,
    )
    read = action.read
    account, missing, duplicate = await _check(session, read, action.not_duplicates)
    if action.duplicate_of is not None:
        if (
            (target := duplicate) is None
            or target.id != action.duplicate_of
            or target.voided != action.duplicate_voided
        ):
            raise MutationRejected(
                f"#{action.duplicate_of} no longer matches this screenshot"
            )
        if target.attachment_path:
            raise MutationRejected(f"#{target.id} already has an attachment")
        await attach_downloaded_attachment(
            session, target.id, stored, interaction_id=interaction_id
        )
        if target.voided:
            await set_transaction_voided_no_commit(session, target.id, False)
            return f"Attached the screenshot to #{target.id} and un-voided it."
        return f"Attached the screenshot to #{target.id}."
    if duplicate is not None:
        raise MutationRejected(f"already recorded as #{duplicate.id}")
    if (
        missing
        or account is None
        or read.amount is None
        or read.occurred_at is None
        or read.status != "success"
    ):
        raise MutationRejected(f"missing or not valid: {', '.join(missing)}")
    txn = Transaction(
        bank=account.bank.strip().lower(),
        account_id=account.id,
        account_mask=f"XX{mask_last4(account.account_number)}",
        source="screenshot",
        email_type="upi_screenshot",
        channel="upi",
        direction=read.direction,
        amount=read.amount,
        currency="INR",
        reference_number=read.utr,
        counterparty=read.counterparty or read.vpa,
        raw_description=read.vpa,
        transaction_date=read.occurred_at.date(),
        transaction_time=read.occurred_at.time(),
    )
    session.add(txn)
    await session.flush()
    await attach_downloaded_attachment(
        session, txn.id, stored, interaction_id=interaction_id
    )
    return (
        f"Saved #{txn.id} with the screenshot. It stays provisional until a bank "
        "alert or statement shows the same UTR."
    )


async def run_screenshot_flag_sweep() -> int:
    """Ask once about each screenshot row that no bank source matched in 60 days.

    The prompt goes through the Telegram outbox. A reply to it reaches the
    assistant with the row as its target.

    Returns:
        The number of prompts sent. 0 when Telegram or the assistant is off.
    """
    from financial_dashboard.services import telegram
    from financial_dashboard.services.assistant.delivery import make_delivery

    if (
        not is_telegram_configured()
        or not is_telegram_assistant_enabled()
        or telegram.tg_app is None
    ):
        return 0
    chat_id = get_telegram_chat_id()
    delivery_ids = []
    async with async_session() as session:
        rows = await session.scalars(
            select(Transaction)
            .where(
                Transaction.source == "screenshot",
                Transaction.voided.is_(False),
                Transaction.screenshot_flagged_at.is_(None),
                Transaction.transaction_date
                <= datetime.datetime.now(IST).date() - FLAG_AGE,
            )
            .order_by(Transaction.id)
        )
        for txn in rows.all():
            credit = txn.direction == "credit"
            text = (
                f"#{txn.id} {format_inr_exact(txn.amount)} "
                f"{'from' if credit else 'to'} {(txn.counterparty or 'unknown')[:200]} "
                f"on {txn.transaction_date} came from a screenshot and no bank "
                "record ever matched it. What happened? Reply to this message."
            )
            owner = AuditInteraction(
                inbound_chat_id=chat_id,
                trigger="screenshot_flag",
                transaction_id=txn.id,
                status="ready_to_send",
                outcome="screenshot_flag",
                assistant_text=text,
            )
            session.add(owner)
            await session.flush()
            delivery = make_delivery(
                recipient_chat_id=chat_id,
                text=text,
                ordinal=0,
                interaction_id=owner.id,
                transaction_id=txn.id,
            )
            # The token lets a reply find this row if the send outlives a crash.
            delivery.text = f"{text}\n\nRef: {delivery.delivery_token}"
            session.add(delivery)
            await session.flush()
            txn.screenshot_flagged_at = utc_now()
            delivery_ids.append(delivery.id)
        await session.commit()
    for delivery_id in delivery_ids:
        await telegram.dispatch_saved_delivery(delivery_id)
    return len(delivery_ids)
