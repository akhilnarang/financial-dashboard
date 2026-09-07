"""Telegram message to application-record lookup and recovery helpers."""

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    AuditInteraction,
    CategoryReviewDecision,
    TelegramMessageContext,
    TelegramOutboundDelivery,
)

# Only the application-appended footer is authoritative.  Anchoring the end
# with ``\Z`` prevents a quoted Ref from selecting an earlier delivery token.
_REF_RE = re.compile(r"(?:^|\n)Ref:\s*([A-Za-z0-9_-]{8,100})\s*\Z")


async def resolve_reply(
    session: AsyncSession, *, chat_id: int, message_id: int
) -> TelegramMessageContext | None:
    return await session.scalar(
        select(TelegramMessageContext).where(
            TelegramMessageContext.chat_id == chat_id,
            TelegramMessageContext.message_id == message_id,
        )
    )


async def resolve_delivery_token(
    session: AsyncSession, token: str
) -> TelegramOutboundDelivery | None:
    return await session.scalar(
        select(TelegramOutboundDelivery).where(
            TelegramOutboundDelivery.delivery_token == token
        )
    )


async def recover_ref_context(
    session: AsyncSession,
    *,
    chat_id: int,
    message_id: int,
    message_text: str,
) -> TelegramMessageContext | None:
    """Repair a missing mapping using the stored outbound Ref token.

    The caller must first verify that the replied-to message was authored by
    this bot.  A token alone is never accepted as a free-text mutation target.
    """
    match = _REF_RE.search(message_text)
    if match is None:
        return None
    delivery = await resolve_delivery_token(session, match.group(1))
    if delivery is None or delivery.recipient_chat_id != chat_id:
        return None
    conversation_id: int | None = None
    context_kind = "assistant_response"
    if delivery.interaction_id is not None:
        interaction = await session.get(AuditInteraction, delivery.interaction_id)
        if interaction is None or interaction.inbound_chat_id != chat_id:
            return None
        conversation_id = interaction.conversation_id
        context_kind = (
            "query_result" if delivery.transaction_id is not None else context_kind
        )
    elif delivery.category_review_decision_id is not None:
        decision = await session.get(
            CategoryReviewDecision, delivery.category_review_decision_id
        )
        if decision is None:
            return None
        if decision.source_interaction_id is not None:
            interaction = await session.get(
                AuditInteraction, decision.source_interaction_id
            )
            if interaction is None or interaction.inbound_chat_id != chat_id:
                return None
            conversation_id = interaction.conversation_id
        context_kind = "category_review"
    # The Ref token is only present in a message Telegram accepted.  Settle a
    # retryable outbox row before exposing its context so restart recovery does
    # not send it a second time.  ``settle_delivery_from_callback`` also
    # accepts an already delivered row and refreshes its owning interaction.
    from financial_dashboard.services.assistant.delivery import (
        settle_delivery_from_callback,
        validate_delivery_proof,
    )

    if not await validate_delivery_proof(
        session, delivery.id, recipient_chat_id=chat_id
    ):
        return None
    if not await settle_delivery_from_callback(session, delivery.id):
        return None
    context = TelegramMessageContext(
        chat_id=chat_id,
        message_id=message_id,
        conversation_id=conversation_id,
        transaction_id=delivery.transaction_id,
        interaction_id=delivery.interaction_id,
        outbound_delivery_id=delivery.id,
        context_kind=context_kind,
    )
    session.add(context)
    await session.flush()
    return context


async def record_physical_message(
    session: AsyncSession,
    *,
    chat_id: int,
    message_id: int,
    context_kind: str,
    conversation_id: int | None = None,
    transaction_id: int | None = None,
    interaction_id: int | None = None,
    outbound_delivery_id: int | None = None,
) -> TelegramMessageContext:
    row = TelegramMessageContext(
        chat_id=chat_id,
        message_id=message_id,
        context_kind=context_kind,
        conversation_id=conversation_id,
        transaction_id=transaction_id,
        interaction_id=interaction_id,
        outbound_delivery_id=outbound_delivery_id,
    )
    session.add(row)
    await session.flush()
    return row
