"""Bounded read models for the conversational-assistant audit pages."""

import datetime
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import (
    AuditAction,
    AuditInteraction,
    TelegramConversation,
    TelegramMessageContext,
    TelegramOutboundDelivery,
)

AUDIT_PAGE_SIZE = 50


class AuditDetail(NamedTuple):
    interaction: AuditInteraction
    actions: list[AuditAction]
    deliveries: list[TelegramOutboundDelivery]
    message_contexts: list[TelegramMessageContext]
    conversation: TelegramConversation | None


async def list_audit_interactions(
    session: AsyncSession,
    *,
    offset: int = 0,
    status: str | None = None,
    trigger: str | None = None,
    transaction_id: int | None = None,
    action_type: str | None = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
) -> list[AuditInteraction]:
    """Return the newest matching accepted interactions with a fixed bound."""
    clauses = []
    if status:
        clauses.append(AuditInteraction.status == status)
    if trigger:
        clauses.append(AuditInteraction.trigger == trigger)
    if transaction_id is not None:
        clauses.append(AuditInteraction.transaction_id == transaction_id)
    if date_from is not None:
        clauses.append(
            AuditInteraction.created_at
            >= datetime.datetime.combine(date_from, datetime.time.min)
        )
    if date_to is not None:
        clauses.append(
            AuditInteraction.created_at
            < datetime.datetime.combine(
                date_to + datetime.timedelta(days=1), datetime.time.min
            )
        )
    statement = select(AuditInteraction).where(*clauses)
    if action_type:
        statement = statement.join(
            AuditAction, AuditAction.interaction_id == AuditInteraction.id
        ).where(AuditAction.action_type == action_type)
    rows = await session.execute(
        statement.distinct()
        .order_by(AuditInteraction.created_at.desc(), AuditInteraction.id.desc())
        .offset(max(offset, 0))
        .limit(AUDIT_PAGE_SIZE)
    )
    return list(rows.scalars().all())


async def list_terminal_proactive_deliveries(
    session: AsyncSession, *, limit: int = 50
) -> list[TelegramOutboundDelivery]:
    """Expose abandoned/cancelled decision notifications without interactions."""
    rows = await session.scalars(
        select(TelegramOutboundDelivery)
        .where(
            TelegramOutboundDelivery.category_review_decision_id.is_not(None),
            TelegramOutboundDelivery.status.in_(["abandoned", "cancelled"]),
        )
        .order_by(TelegramOutboundDelivery.created_at.desc())
        .limit(limit)
    )
    return list(rows.all())


async def get_audit_detail(
    session: AsyncSession, interaction_id: int
) -> AuditDetail | None:
    interaction = await session.get(AuditInteraction, interaction_id)
    if interaction is None:
        return None
    actions = list(
        (
            await session.execute(
                select(AuditAction)
                .where(AuditAction.interaction_id == interaction_id)
                .order_by(AuditAction.id)
            )
        )
        .scalars()
        .all()
    )
    deliveries = list(
        (
            await session.execute(
                select(TelegramOutboundDelivery)
                .where(TelegramOutboundDelivery.interaction_id == interaction_id)
                .order_by(TelegramOutboundDelivery.ordinal)
            )
        )
        .scalars()
        .all()
    )
    contexts = list(
        (
            await session.execute(
                select(TelegramMessageContext)
                .where(TelegramMessageContext.interaction_id == interaction_id)
                .order_by(TelegramMessageContext.id)
            )
        )
        .scalars()
        .all()
    )
    conversation = (
        await session.get(TelegramConversation, interaction.conversation_id)
        if interaction.conversation_id is not None
        else None
    )
    return AuditDetail(interaction, actions, deliveries, contexts, conversation)
