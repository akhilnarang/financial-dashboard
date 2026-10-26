"""Offer to link a new row to the open receivable it pays.

The rules are fixed. The amount must equal the outstanding exactly. Nothing
links without a Telegram tap, and a tap never changes the status.
"""

import datetime
import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.templating import format_inr_exact
from financial_dashboard.db import async_session
from financial_dashboard.db.enums import (
    PersonIdentifierKind,
    ReceivableKind,
    ReceivableRole,
    ReceivableStatus,
)
from financial_dashboard.db.models import (
    AuditInteraction,
    PersonIdentifier,
    Receivable,
    ReceivableLink,
    ReceivableMatchPrompt,
    Transaction,
    utc_now,
)
from financial_dashboard.exceptions import (
    ConflictException,
    UnprocessableEntityException,
)
from financial_dashboard.schemas.receivables import ReceivableRead
from financial_dashboard.services.categorization.normalize import (
    normalize_counterparty,
    normalize_text,
)
from financial_dashboard.services.assistant.message_context import resolve_reply
from financial_dashboard.services.person_identifiers import (
    canonical_vpa,
    normalize_person,
    raw_identifiers,
)
from financial_dashboard.services.receivables import (
    ROLE_DIRECTION,
    attach_link,
    commit_links,
    list_receivables,
    receivable_read,
    write_lock,
)
from financial_dashboard.services.settings import (
    get_telegram_chat_id,
    is_telegram_assistant_enabled,
    is_telegram_configured,
)

logger = logging.getLogger(__name__)

LOOKBACK = datetime.timedelta(days=7)
MAX_CANDIDATES = 3

Identifiers = Mapping[str, Sequence[tuple[str, str]]]


class Match(NamedTuple):
    receivable: ReceivableRead
    strong: bool
    identifier: str


class MatchTap(NamedTuple):
    text: str
    resolved: bool


def _identity(
    txn: Transaction, receivable: ReceivableRead, identifiers: Identifiers
) -> Match | None:
    """Decide whether a row names the person of a receivable.

    A stored identifier in the row is a strong match. A person with no stored
    identifiers matches weakly when each name word of 4+ letters is in the
    squashed counterparty.

    Args:
        txn: The row.
        receivable: The receivable whose person to look for.
        identifiers: The stored ``(kind, value)`` pairs of each person key.

    Returns:
        The match, with the longest identifier that hit, or ``None``.
    """
    squashed = normalize_counterparty(txn.counterparty)
    known = identifiers.get(normalize_person(receivable.person), ())
    in_raw = raw_identifiers(txn.raw_description)
    hits = []
    for kind, value in known:
        match kind:
            case PersonIdentifierKind.counterparty:
                hit = value in squashed
            case PersonIdentifierKind.vpa:
                hit = canonical_vpa(value) in in_raw.vpas
            case PersonIdentifierKind.phone:
                hit = value in in_raw.phones
        if hit:
            hits.append(value)
    if hits:
        return Match(receivable, True, max(hits, key=len))
    words = [
        word for word in normalize_text(receivable.person).split() if len(word) >= 4
    ]
    if not known and words and all(word in squashed for word in words):
        return Match(receivable, False, "".join(words))
    return None


def match_receivables(
    txn: Transaction, receivables: Sequence[ReceivableRead], identifiers: Identifiers
) -> list[ReceivableRead]:
    """Return the open receivables that a row can pay, best first.

    Args:
        txn: The new row.
        receivables: The open receivables.
        identifiers: The stored ``(kind, value)`` pairs of each person key.

    Returns:
        At most three receivables. None when no rule holds, or when more than
        three people match.
    """
    matches = [
        match
        for receivable in receivables
        if receivable.status == ReceivableStatus.open
        and txn.direction == ROLE_DIRECTION[receivable.kind][ReceivableRole.payment]
        and txn.amount == receivable.outstanding
        and (match := _identity(txn, receivable, identifiers)) is not None
    ]
    # A row that names a full name is not also offered to a person whose
    # name sits inside it. Two names side by side are both offered.
    kept = [
        match
        for match in matches
        if not any(
            len(other.identifier) > len(match.identifier)
            and match.identifier in other.identifier
            for other in matches
        )
    ]
    kept.sort(
        key=lambda match: (
            not match.strong,
            -len(match.identifier),
            match.receivable.id,
        )
    )
    return [match.receivable for match in kept] if len(kept) <= MAX_CANDIDATES else []


async def _queue_prompt(
    session: AsyncSession,
    txn: Transaction,
    matches: Sequence[ReceivableRead],
    chat_id: int,
) -> int:
    """Save the prompt of one row, its owner interaction and its outbox message.

    The message has one Link button per match and a Not this button. The
    text comes from stored data only. Nothing is sent here.

    Args:
        session: The sweep session.
        txn: The row to ask about.
        matches: The receivables to offer, best first.
        chat_id: The authorized Telegram chat.

    Returns:
        The id of the outbox message.
    """
    from financial_dashboard.services.assistant.delivery import make_delivery

    # The outbox needs an owner. A background interaction owns this message.
    owner = AuditInteraction(
        inbound_chat_id=chat_id,
        trigger="receivable_match",
        transaction_id=txn.id,
        status="ready_to_send",
        outcome="receivable_match",
    )
    prompt = ReceivableMatchPrompt(
        transaction_id=txn.id, candidate_ids=[match.id for match in matches]
    )
    session.add_all([owner, prompt])
    await session.flush()
    buttons = [
        [
            {
                "text": f"Link to {match.person[:30]} #{match.id}",
                "callback_data": f"rcv:v1:l:{prompt.id}:{match.id}",
            }
        ]
        for match in matches
    ]
    buttons.append([{"text": "Not this", "callback_data": f"rcv:v1:x:{prompt.id}"}])
    credit = txn.direction == "credit"
    lines = [
        f"\U0001f4b8 Possible {'repayment' if credit else 'payment'}: #{txn.id} "
        f"{txn.direction} {format_inr_exact(txn.amount)} "
        f"{'from' if credit else 'to'} {(txn.counterparty or 'unknown')[:200]} "
        f"on {txn.transaction_date or 'unknown date'}"
    ]
    lines += [
        f"  {'Owed to you' if match.kind == ReceivableKind.receivable else 'You owe'}"
        f": {match.person} — outstanding {format_inr_exact(match.outstanding)}"
        for match in matches
    ]
    owner.assistant_text = text = "\n".join(lines)
    delivery = make_delivery(
        recipient_chat_id=chat_id,
        text=text,
        ordinal=0,
        interaction_id=owner.id,
        transaction_id=txn.id,
        reply_markup_json=json.dumps(buttons, separators=(",", ":")),
    )
    session.add(delivery)
    await session.flush()
    prompt.delivery_id = delivery.id
    return delivery.id


async def run_receivable_match_sweep() -> int:
    """Prompt once for each recent row that can pay an open receivable.

    A row qualifies when it is new in the last 7 days, in INR, not voided,
    not linked to a receivable, and has no prompt yet. The prompt goes
    through the Telegram outbox, so a failed send is sent again later.

    Returns:
        The number of prompts sent. 0 when Telegram or the assistant is off,
        or when no receivable is open.
    """
    from financial_dashboard.services import telegram

    if (
        not is_telegram_configured()
        or not is_telegram_assistant_enabled()
        or telegram.tg_app is None
    ):
        return 0
    chat_id = get_telegram_chat_id()
    async with async_session() as session:
        receivables = [
            receivable_read(row)
            for row in await list_receivables(session, ReceivableStatus.open)
        ]
        if not receivables:
            return 0
        identifiers: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in await session.scalars(select(PersonIdentifier)):
            identifiers[row.person].append((row.kind, row.value))
        linked = select(ReceivableLink.id).where(
            ReceivableLink.transaction_id == Transaction.id
        )
        prompted = select(ReceivableMatchPrompt.id).where(
            ReceivableMatchPrompt.transaction_id == Transaction.id
        )
        rows = await session.scalars(
            select(Transaction)
            .where(
                Transaction.created_at >= utc_now() - LOOKBACK,
                func.coalesce(Transaction.currency, "INR") == "INR",
                Transaction.voided.is_(False),
                ~linked.exists(),
                ~prompted.exists(),
            )
            .order_by(Transaction.id)
        )
        delivery_ids = []
        for txn in rows.all():
            if not (matches := match_receivables(txn, receivables, identifiers)):
                continue
            # One bad row, or a row that another sweep prompted, must not stop the rest.
            try:
                async with session.begin_nested():
                    delivery_ids.append(
                        await _queue_prompt(session, txn, matches, chat_id)
                    )
            except IntegrityError:
                logger.info("Receivable match prompt for #%s exists", txn.id)
            except Exception:
                logger.exception("Receivable match prompt failed for #%s", txn.id)
        await session.commit()
    for delivery_id in delivery_ids:
        await telegram.dispatch_saved_delivery(delivery_id)
    return len(delivery_ids)


async def resolve_match_tap(
    session: AsyncSession,
    prompt_id: int,
    receivable_id: int | None,
    *,
    chat_id: int,
    message_id: int,
) -> MatchTap:
    """Link the row of a prompt to the tapped receivable, or dismiss the prompt.

    The link goes through the checks of a manual link. The row must still
    fit the receivable: not voided, the right direction, the amount equal to
    the outstanding, and the same person. So a stale prompt cannot pay a
    receivable twice. The tapped message must be the one sent for this
    prompt, because SQLite can give a deleted prompt id to a new prompt.
    The status of the receivable does not change.

    Args:
        session: A session that holds no write yet.
        prompt_id: The prompt that the button belongs to.
        receivable_id: The tapped receivable, or ``None`` for "Not this".
        chat_id: The chat of the tapped message.
        message_id: The tapped message.

    Returns:
        The reply text, and whether the prompt is now resolved.
    """
    await write_lock(session)
    prompt = await session.get(ReceivableMatchPrompt, prompt_id)
    if prompt is None or prompt.status != "pending":
        await session.rollback()
        return MatchTap("Already handled", False)
    tapped = await resolve_reply(session, chat_id=chat_id, message_id=message_id)
    if tapped is None or tapped.outbound_delivery_id != prompt.delivery_id:
        await session.rollback()
        return MatchTap("This button is stale", False)
    if receivable_id is None:
        prompt.status, prompt.resolved_at = "dismissed", utc_now()
        await session.commit()
        return MatchTap(f"Not linked: #{prompt.transaction_id}", True)
    receivable = (
        await session.get(Receivable, receivable_id)
        if receivable_id in prompt.candidate_ids
        else None
    )
    txn = await session.get(Transaction, prompt.transaction_id)
    person = normalize_person(receivable.person) if receivable else ""
    known = await session.execute(
        select(PersonIdentifier.kind, PersonIdentifier.value).where(
            PersonIdentifier.person == person
        )
    )
    identifiers = {person: [(kind, value) for kind, value in known]}
    if (
        receivable is None
        or txn is None
        or receivable.status != ReceivableStatus.open
        or txn.voided
        or txn.direction
        != ROLE_DIRECTION[ReceivableKind(receivable.kind)][ReceivableRole.payment]
        or receivable_read(receivable).outstanding != txn.amount
        or _identity(txn, receivable_read(receivable), identifiers) is None
    ):
        await session.rollback()
        return MatchTap("This receivable changed. Link the row on the page.", False)
    try:
        await attach_link(session, receivable, txn.id, ReceivableRole.payment)
        prompt.status, prompt.resolved_at = "linked", utc_now()
        await commit_links(session)
    except (UnprocessableEntityException, ConflictException) as exc:
        await session.rollback()
        return MatchTap(str(exc.detail), False)
    return MatchTap(
        f"Linked #{txn.id} to {receivable.person} #{receivable.id}. It is fully "
        "paid. Settle it on the page or ask the assistant.",
        True,
    )
