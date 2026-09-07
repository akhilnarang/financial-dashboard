import datetime
from types import SimpleNamespace

import pytest

from financial_dashboard.services.assistant.conversations import (
    active_conversation,
    expire_conversations,
    start_conversation,
)
from financial_dashboard.services.assistant.orchestrator import (
    _conversation_for_message,
)


@pytest.mark.anyio
async def test_conversation_expires_after_24_hours(session):
    conversation = await start_conversation(session, chat_id=9, started_by="ask")
    conversation.expires_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
        seconds=1
    )
    await session.flush()
    assert await active_conversation(session, chat_id=9) is None
    assert await expire_conversations(session) == 1


@pytest.mark.anyio
async def test_conversation_timestamp_survives_sqlite_round_trip(session):
    conversation = await start_conversation(session, chat_id=10, started_by="ask")
    await session.commit()
    conversation_id = conversation.id
    session.expunge_all()

    reloaded = await session.get(type(conversation), conversation_id)

    assert reloaded is not None
    assert reloaded.expires_at is not None
    resumed = await _conversation_for_message(
        session,
        chat_id=10,
        trigger="reply",
        mapped=SimpleNamespace(conversation_id=conversation_id, transaction_id=None),
    )
    assert resumed.id == conversation_id
