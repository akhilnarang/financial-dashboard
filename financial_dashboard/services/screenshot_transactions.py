"""Provisional transactions from screenshots sent to the Telegram bot.

The screenshot shows a UPI payment or a card notification. A card or UPI
screenshot of a refund or a reversal also becomes a provisional row.
The vision model reads the screenshot. Code checks every mandatory field.
Nothing is saved without a Yes tap. A bank alert or statement for the same
payment later confirms the row. A row that no bank source confirms in 60 days
is flagged once on Telegram. The owner replies with what happened, and the
assistant proposes the fix.
"""

import datetime
import logging
import re
from decimal import Decimal
from typing import NamedTuple

from pydantic import ValidationError
from sqlalchemy import ColumnElement, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import IST, parse_datetime
from financial_dashboard.core.masks import mask_last4
from financial_dashboard.core.templating import format_inr_exact
from financial_dashboard.db import async_session
from financial_dashboard.db.models import (
    Account,
    AuditInteraction,
    Card,
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

DUPLICATE_WINDOW = datetime.timedelta(days=1)

_UTR = re.compile(r"\d{12}")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]")
_TRAILING_LAST4 = re.compile(r"(\d{4})[\s).]*$")
_KEYED_AMOUNT = re.compile(
    r"(?:(?:is|was)\s+)?(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d{1,2})?)"
    r"\s*(?:rupees|rs\.?|inr|/-)?\.?",
    re.IGNORECASE,
)
_RUPEES = re.compile(
    r"(?:₹|\brs\.?|\binr)\s*(\d[\d,]*(?:\.\d{1,2})?)(?![.,]?\d)", re.IGNORECASE
)


def _saved_status(read: ScreenshotTransaction) -> bool:
    """Tell if the read shows a payment that a provisional row can hold.

    A card screenshot saves only a refund. A card bill payment or a card
    spend has its own alert, and that alert must create the row.
    """
    return read.status == "reversed" or (
        read.status == "success" and read.channel == "upi"
    )


class ScreenshotTarget(NamedTuple):
    """The account and card that a read points at, or the cards it may mean.

    ``choices`` lists the cards that an ambiguous card hint fits, as
    "label 1234".
    """

    account: Account | None
    card: Card | None = None
    choices: tuple[str, ...] = ()


class ScreenshotCheck(NamedTuple):
    """The resolved target, the failed mandatory fields and a stored duplicate."""

    target: ScreenshotTarget
    missing: list[str]
    duplicate: Transaction | None


class ScreenshotReview(NamedTuple):
    """The reply to a screenshot, and the action a Yes applies, if any."""

    text: str
    action: ScreenshotTransactionConfirmation | None


def _normalize_hint(text: str) -> str:
    """Reduce a card hint to its lowercase letters and digits.

    Args:
        text: The card or bank text that the screen shows.

    Returns:
        The letters and digits only, in lowercase.
    """
    return _NON_ALPHANUMERIC.sub("", text.lower())


async def _resolve_credit_card_target(
    session: AsyncSession, hint: str
) -> ScreenshotTarget:
    """Resolve a card read to one credit card account and its card.

    A last-4 run at the end of the hint selects the card with that mask.
    Without one, the bank name or the label of one credit card account must appear in the
    hint. An account with one card returns that card. An ambiguous hint
    returns an empty target, so the reply asks for the card.

    Args:
        session: The turn session.
        hint: The card or bank text that the screen shows.

    Returns:
        The account and its card, or an empty target when no unique card
        matches. An ambiguous name lists the cards that it fits.
    """
    accounts = list(
        await session.scalars(
            select(Account).where(
                Account.type == "credit_card", Account.active.is_(True)
            )
        )
    )
    cards = list(
        await session.scalars(
            select(Card)
            .where(Card.active.is_(True), Card.account_id.in_([a.id for a in accounts]))
            .order_by(Card.id)
        )
    )
    if (found := _TRAILING_LAST4.search(hint)) is not None:
        matched = [c for c in cards if mask_last4(c.card_mask) == found.group(1)]
        if len(matched) != 1:
            return ScreenshotTarget(None)
        card = matched[0]
        return ScreenshotTarget(
            next(a for a in accounts if a.id == card.account_id), card
        )
    normalized = _normalize_hint(hint)
    if not normalized:
        return ScreenshotTarget(None)
    selected: set[int] = set()
    for account in accounts:
        names = (
            _normalize_hint(account.bank or ""),
            _normalize_hint(account.label or ""),
        )
        if any(name and name in normalized for name in names):
            selected.add(account.id)
    owned = [card for card in cards if card.account_id in selected]
    if len(selected) == 1 and len(owned) <= 1:
        account = next(a for a in accounts if a.id in selected)
        return ScreenshotTarget(account, owned[0] if owned else None)
    labels = {a.id: a.label for a in accounts}
    return ScreenshotTarget(
        None,
        None,
        tuple(
            f"{labels[card.account_id]} {last4}"
            for card in owned
            if (last4 := mask_last4(card.card_mask))
        ),
    )


async def _resolve_target(
    session: AsyncSession, read: ScreenshotTransaction
) -> ScreenshotTarget:
    """Resolve the account and card that one screenshot read points at.

    A UPI read resolves by an exact last-4 match to exactly one active bank
    account. The hint must end with those 4 digits. A card read resolves by
    the card digits, else by the bank name or the account label.

    Args:
        session: The turn session.
        read: The fields read off the screenshot.

    Returns:
        The account and card of the read. A field is None when it does not
        resolve to one row.
    """
    hint = read.account_hint or ""
    if read.channel == "card":
        return await _resolve_credit_card_target(session, hint)
    if (found := _TRAILING_LAST4.search(hint)) is None:
        return ScreenshotTarget(None)
    last4 = found.group(1)
    accounts = [
        row
        for row in await session.scalars(
            select(Account).where(
                Account.type == "bank_account", Account.active.is_(True)
            )
        )
        if mask_last4(row.account_number) == last4
    ]
    return ScreenshotTarget(accounts[0] if len(accounts) == 1 else None, None)


def _owner_filter(target: ScreenshotTarget) -> ColumnElement[bool] | None:
    """Return the filter that names the card or the account of one target.

    A card target matches rows on that card, and also rows on its account that
    record no card yet.

    Args:
        target: The account and card that a read resolves to.

    Returns:
        The card filter when the target has a card, else the account filter.
        None when the target has no account.
    """
    if target.account is None:
        return None
    if target.card is not None:
        return or_(
            Transaction.card_id == target.card.id,
            (Transaction.account_id == target.account.id)
            & Transaction.card_id.is_(None),
        )
    return Transaction.account_id == target.account.id


async def _find_duplicate(
    session: AsyncSession,
    read: ScreenshotTransaction,
    target: ScreenshotTarget,
    not_duplicates: list[int],
) -> Transaction | None:
    """Find the stored row that holds the same payment as the read.

    A row with the same UTR and direction holds the payment. Otherwise a live
    row for the same target, direction and amount within one day holds it,
    when its reference is not another UTR.

    Args:
        session: The turn session.
        read: The fields read off the screenshot.
        target: The account and card that the read resolves to.
        not_duplicates: Rows that the owner called another payment.

    Returns:
        The stored row, or None when no row holds the payment.
    """
    utr = read.utr if read.utr and _UTR.fullmatch(read.utr) else None
    if utr:
        found = await session.scalar(
            select(Transaction)
            .where(
                Transaction.reference_number == utr,
                Transaction.direction == read.direction,
            )
            .order_by(Transaction.id)
            .limit(1)
        )
        if found is not None:
            return found
    owner = _owner_filter(target)
    if owner is None or not read.amount or read.occurred_at is None:
        return None
    day = read.occurred_at.date()
    stmt = (
        select(Transaction)
        .where(
            owner,
            Transaction.direction == read.direction,
            Transaction.amount == read.amount,
            Transaction.transaction_date.between(
                day - DUPLICATE_WINDOW, day + DUPLICATE_WINDOW
            ),
            Transaction.id.not_in(not_duplicates),
            Transaction.voided.is_(False),
        )
        .order_by(Transaction.id)
        .limit(1)
    )
    if read.channel == "upi":
        stmt = stmt.where(
            or_(
                Transaction.reference_number.is_(None),
                ~Transaction.reference_number.regexp_match(r"^\d{12}$"),
            )
        )
    return await session.scalar(stmt)


async def _check(
    session: AsyncSession, read: ScreenshotTransaction, not_duplicates: list[int]
) -> ScreenshotCheck:
    """Run the mandatory-field and duplicate checks on one read.

    Args:
        session: The turn session.
        read: The fields read off the screenshot.
        not_duplicates: Rows that the owner called another payment.

    Returns:
        The resolved target, the failed mandatory fields and the stored
        duplicate if one exists.
    """
    target = await _resolve_target(session, read)
    now = datetime.datetime.now(IST).replace(tzinfo=None)
    checks = [
        ("amount", read.amount is not None and read.amount > 0),
        ("name or VPA", bool(read.counterparty or read.vpa)),
        ("date and time", read.occurred_at is not None and read.occurred_at <= now),
    ]
    if read.channel == "upi":
        checks.append(("UTR (12 digits)", _UTR.fullmatch(read.utr or "") is not None))
    if read.channel == "card":
        checks.append(("card", target.account is not None))
    else:
        checks.append(("account (last 4 digits)", target.account is not None))
    missing = [field for field, ok in checks if not ok]
    duplicate = await _find_duplicate(session, read, target, not_duplicates)
    return ScreenshotCheck(target, missing, duplicate)


def _headline(read: ScreenshotTransaction) -> str:
    """Word the first line of the question for one read.

    Args:
        read: The fields read off the screenshot.

    Returns:
        The line that names the source, the amount and the other party.
    """
    credit = read.direction == "credit"
    amount_text = format_inr_exact(read.amount) if read.amount is not None else "?"
    party = (
        " ".join(
            part
            for part in (read.counterparty, f"({read.vpa})" if read.vpa else "")
            if part
        )[:200]
        or "?"
    )
    if read.channel == "card":
        # A card screenshot saves only a refund, so no other card read gets here.
        return f"Card refund: {amount_text} from {party}"
    return (
        f"UPI screenshot ({read.app}): {'received' if credit else 'paid'} "
        f"{amount_text} {'from' if credit else 'to'} {party}"
    )


def _detail_line(read: ScreenshotTransaction, target: ScreenshotTarget) -> str:
    """Word the second line of the question for one read.

    The line holds the UTR when a bank source can confirm the row with it, the
    date and time, and the account or card that the read resolves to.

    Args:
        read: The fields read off the screenshot.
        target: The account and card that the read resolves to.

    Returns:
        The detail line.
    """
    parts = []
    if read.utr:
        parts.append(f"UTR {read.utr}")
    parts.append(
        read.occurred_at.strftime("%Y-%m-%d %H:%M") if read.occurred_at else "?"
    )
    parts.append(target.account.label if target.account else read.account_hint or "?")
    return " · ".join(parts)


async def review_screenshot(
    session: AsyncSession,
    read: ScreenshotTransaction,
    *,
    attachment_path: str,
    media_type: str,
    not_duplicates: list[int],
) -> ScreenshotReview:
    """Check one screenshot read and word the question for the owner.

    A screenshot that does not show a success or a reversal is refused. A
    duplicate offers to attach the screenshot to the stored row. A read with
    a missing field names each field and asks for it. Otherwise the question
    asks to save.

    Args:
        session: The turn session.
        read: The fields read off the screenshot, with any corrections.
        attachment_path: The stored screenshot file.
        media_type: Its MIME type.

    Returns:
        The reply text, and the action that a Yes applies. The action is
        ``None`` for a refused screenshot.
    """
    if not _saved_status(read):
        return ScreenshotReview(
            "This card screenshot is not a refund. I save only refunds from "
            "card screenshots. I saved nothing."
            if read.channel == "card"
            else f"This screenshot shows a {read.status} payment, not a success. "
            "I saved nothing.",
            None,
        )
    target, missing, duplicate = await _check(session, read, not_duplicates)
    lines = [
        _headline(read),
        _detail_line(read, target),
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
        if target.choices:
            lines.append(f"Which card: {', '.join(target.choices)}?")
        example = (
            '"card 1234, amount 100"'
            if read.channel == "card"
            else '"utr 123456789012, account 1234"'
        )
        lines.append(f"Reply with the values, for example {example}, or tap Skip.")
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
            case "card":
                updates["account_hint"] = value
                updates["channel"] = "card"
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


def _row_fields(read: ScreenshotTransaction, target: ScreenshotTarget) -> dict:
    """Build the row columns that a confirmed screenshot saves.

    Args:
        read: The fields read off the screenshot.
        target: The account and card that the read resolves to.

    Returns:
        The column values of the new row.
    """
    account, card = target.account, target.card
    assert account is not None
    suffix = "refund_screenshot" if read.status == "reversed" else "screenshot"
    return {
        "bank": account.bank.strip().lower(),
        "account_id": account.id,
        "account_mask": f"XX{last4}"
        if (last4 := mask_last4(account.account_number))
        else None,
        "card_id": card.id if card is not None else None,
        "card_mask": card.card_mask if card is not None else None,
        "source": "screenshot",
        "email_type": f"{read.channel}_{suffix}",
        "channel": read.channel,
        "direction": read.direction,
        "amount": read.amount,
        "currency": "INR",
        "reference_number": read.utr,
        "counterparty": read.counterparty or read.vpa,
        "raw_description": read.vpa,
    }


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
    target, missing, duplicate = await _check(session, read, action.not_duplicates)
    if action.duplicate_of is not None:
        if (
            (found := duplicate) is None
            or found.id != action.duplicate_of
            or found.voided != action.duplicate_voided
        ):
            raise MutationRejected(
                f"#{action.duplicate_of} no longer matches this screenshot"
            )
        if found.attachment_path:
            raise MutationRejected(f"#{found.id} already has an attachment")
        await attach_downloaded_attachment(
            session, found.id, stored, interaction_id=interaction_id
        )
        if found.voided:
            await set_transaction_voided_no_commit(session, found.id, False)
            return f"Attached the screenshot to #{found.id} and un-voided it."
        return f"Attached the screenshot to #{found.id}."
    if duplicate is not None:
        raise MutationRejected(f"already recorded as #{duplicate.id}")
    if (
        missing
        or target.account is None
        or read.amount is None
        or read.occurred_at is None
        or not _saved_status(read)
    ):
        raise MutationRejected(f"missing or not valid: {', '.join(missing)}")
    txn = Transaction(
        **_row_fields(read, target),
        transaction_date=read.occurred_at.date(),
        transaction_time=read.occurred_at.time(),
    )
    session.add(txn)
    await session.flush()
    await attach_downloaded_attachment(
        session, txn.id, stored, interaction_id=interaction_id
    )
    if read.utr is None:
        return (
            f"Saved #{txn.id} with the screenshot. It stays provisional until a "
            "bank alert or statement confirms it."
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
