"""Reply-thread and pending-confirmation persistence."""

import datetime
import json
from typing import cast

from sqlalchemy import select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import TelegramConversation, utc_now

CONVERSATION_TTL = datetime.timedelta(hours=24)


async def start_conversation(
    session: AsyncSession,
    *,
    chat_id: int,
    started_by: str,
    transaction_id: int | None = None,
) -> TelegramConversation:
    now = utc_now()
    row = TelegramConversation(
        chat_id=chat_id,
        started_by=started_by,
        transaction_id=transaction_id,
        started_at=now,
        last_activity_at=now,
        expires_at=now + CONVERSATION_TTL,
        status="active",
    )
    session.add(row)
    await session.flush()
    return row


async def active_conversation(
    session: AsyncSession, *, chat_id: int, transaction_id: int | None = None
) -> TelegramConversation | None:
    now = utc_now()
    query = select(TelegramConversation).where(
        TelegramConversation.chat_id == chat_id,
        TelegramConversation.status == "active",
        TelegramConversation.expires_at > now,
    )
    if transaction_id is not None:
        query = query.where(TelegramConversation.transaction_id == transaction_id)
    return await session.scalar(
        query.order_by(TelegramConversation.last_activity_at.desc()).limit(1)
    )


async def touch_conversation(
    session: AsyncSession, conversation: TelegramConversation
) -> None:
    now = utc_now()
    conversation.last_activity_at = now
    conversation.expires_at = now + CONVERSATION_TTL
    await session.flush()


async def expire_conversations(session: AsyncSession) -> int:
    from sqlalchemy import update

    result = cast(
        CursorResult,
        await session.execute(
            update(TelegramConversation)
            .where(
                TelegramConversation.status == "active",
                TelegramConversation.expires_at <= utc_now(),
            )
            .values(status="expired")
            .execution_options(synchronize_session="fetch")
        ),
    )
    return result.rowcount


def set_pending_confirmation(
    conversation: TelegramConversation,
    payload: dict[str, object],
    *,
    kind: str,
    source_interaction_id: int | None,
    expires_at: datetime.datetime,
    state_hash: str,
) -> None:
    conversation.pending_confirmation_json = json.dumps(
        payload, separators=(",", ":"), sort_keys=True
    )
    conversation.pending_confirmation_kind = kind
    conversation.pending_confirmation_state_hash = state_hash
    conversation.pending_confirmation_source_interaction_id = source_interaction_id
    conversation.pending_confirmation_expires_at = expires_at


def clear_pending_confirmation(conversation: TelegramConversation) -> None:
    conversation.pending_confirmation_json = None
    conversation.pending_confirmation_kind = None
    conversation.pending_confirmation_state_hash = None
    conversation.pending_confirmation_source_interaction_id = None
    conversation.pending_confirmation_expires_at = None
