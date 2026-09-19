"""Reply-thread and pending-confirmation persistence."""

import json
import datetime

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


async def touch_conversation(
    session: AsyncSession, conversation: TelegramConversation
) -> None:
    now = utc_now()
    conversation.last_activity_at = now
    conversation.expires_at = now + CONVERSATION_TTL
    await session.flush()


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
