from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.db as db_package
from financial_dashboard.db import (
    AuditInteraction,
    TelegramConversation,
    TelegramMessageContext,
    Transaction,
)
from financial_dashboard.services import settings as settings_service
from financial_dashboard.services import telegram
from financial_dashboard.services.assistant.contracts import Answer
from financial_dashboard.services.assistant.provider import StructuredResult


class FakeProvider:
    async def complete(self, context):
        response = Answer(
            outcome="answer",
            text="It was held for review because the merchant metadata was ambiguous.",
        )
        return StructuredResult(
            response,
            "bounded prompt",
            "json_schema",
            "fake",
            "test-model",
            2,
            response.model_dump(),
        )


class FakeMessage:
    def __init__(self, *, message_id, chat_id, text, reply_to_message=None):
        self.message_id = message_id
        self.chat_id = chat_id
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.caption = None
        self.reply_to_message = reply_to_message
        self.document = None
        self.photo = []
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


@pytest.mark.anyio
async def test_mapped_reply_runs_once_and_persists_deliverable_answer(
    session, monkeypatch
):
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        counterparty="PUREBERRYSMUMBAI",
        review_status="pending",
        review_reason="merchant could be dining or groceries",
    )
    session.add(transaction)
    await session.flush()
    session.add(
        TelegramMessageContext(
            chat_id=77,
            message_id=10,
            transaction_id=transaction.id,
            context_kind="category_review",
        )
    )
    await session.commit()

    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)
    monkeypatch.setattr(telegram, "async_session", maker)
    monkeypatch.setattr(
        "financial_dashboard.services.assistant.orchestrator._provider_from_application_settings",
        lambda: FakeProvider(),
    )
    settings_service._cache["telegram.chat_id"] = "77"
    settings_service._cache["telegram.assistant_enabled"] = "true"

    original = FakeMessage(message_id=10, chat_id=77, text="🔍 Needs a category: #1")
    original.from_user = SimpleNamespace(id=900)
    message = FakeMessage(
        message_id=11,
        chat_id=77,
        text="why?",
        reply_to_message=original,
    )
    update = SimpleNamespace(update_id=500, message=message, callback_query=None)
    context = SimpleNamespace(bot=SimpleNamespace(id=900), args=[])

    await telegram._handle_reply(update, context)
    await telegram._handle_reply(update, context)

    async with maker() as verification:
        count = await verification.scalar(select(func.count(AuditInteraction.id)))
        interaction = await verification.scalar(select(AuditInteraction))
    assert count == 1
    assert interaction.status == "ready_to_send"
    assert interaction.outcome == "answer"
    assert "ambiguous" in interaction.assistant_text
    assert interaction.transaction_id == transaction.id


@pytest.mark.anyio
async def test_disabled_ask_never_calls_assistant(monkeypatch):
    settings_service._cache["telegram.chat_id"] = "77"
    settings_service._cache["telegram.assistant_enabled"] = "false"
    message = FakeMessage(message_id=20, chat_id=77, text="/ask recent groceries")
    update = SimpleNamespace(update_id=501, message=message)

    await telegram._handle_ask(update, SimpleNamespace(args=["recent", "groceries"]))

    assert message.replies == ["/ask is disabled"]


def test_legacy_parser_rejects_sms_ids():
    from financial_dashboard.services.assistant.orchestrator import (
        _legacy_transaction_id,
    )

    assert _legacy_transaction_id("⚠️ HDFC DEBIT SMS #8507") is None
    assert _legacy_transaction_id("🔍 Needs a category: #8507") == 8507


def test_sms_duplicate_guard_is_narrow_and_preserves_transaction_sms_headers():
    from financial_dashboard.services.assistant.orchestrator import (
        _legacy_transaction_id,
    )
    from financial_dashboard.services.telegram import is_sms_duplicate_prompt

    duplicate = "⚠️ <b>HDFC</b> DEBIT SMS #8507"
    rendered_duplicate = "⚠️ HDFC DEBIT SMS #8507"
    normal = "🔴 <b>HDFC</b> DEBIT · via SMS  #8507"

    assert is_sms_duplicate_prompt(duplicate)
    assert is_sms_duplicate_prompt(rendered_duplicate)
    assert not is_sms_duplicate_prompt(normal)
    assert _legacy_transaction_id(duplicate) is None
    assert _legacy_transaction_id(rendered_duplicate) is None
    assert _legacy_transaction_id(normal) == 8507


@pytest.mark.anyio
async def test_disabled_legacy_reply_accepts_transaction_sms_header(
    session, monkeypatch
):
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    session.add(transaction)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(telegram, "async_session", maker)
    settings_service._cache["telegram.chat_id"] = "77"
    settings_service._cache["telegram.assistant_enabled"] = "false"
    original = FakeMessage(
        message_id=30,
        chat_id=77,
        text=f"🔴 <b>HDFC</b> DEBIT · via SMS  #{transaction.id}",
    )
    original.from_user = SimpleNamespace(id=900)
    message = FakeMessage(
        message_id=31,
        chat_id=77,
        text="fresh basket",
        reply_to_message=original,
    )

    await telegram._handle_reply(
        SimpleNamespace(update_id=502, message=message),
        SimpleNamespace(bot=SimpleNamespace(id=900), args=[]),
    )

    async with maker() as verification:
        saved = await verification.get(Transaction, transaction.id)
    assert saved.note == "fresh basket"
    assert message.replies == [f"Saved note for #{transaction.id}"]


def test_custom_bot_api_derives_matching_file_endpoint():
    assert (
        telegram._telegram_file_base_url("http://telegram-api:8081/bot")
        == "http://telegram-api:8081/file/bot"
    )


@pytest.mark.anyio
async def test_claimed_turn_resumes_once_after_restart(session, monkeypatch):
    conversation = TelegramConversation(
        chat_id=77,
        started_by="ask",
        status="active",
    )
    session.add(conversation)
    await session.flush()
    interaction = AuditInteraction(
        telegram_update_id="restart-1",
        inbound_chat_id=77,
        inbound_message_id=40,
        trigger="ask",
        conversation_id=conversation.id,
        user_text="why was the last transaction held?",
        status="claimed",
    )
    session.add(interaction)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)
    monkeypatch.setattr(telegram, "async_session", maker)
    monkeypatch.setattr(
        "financial_dashboard.services.assistant.orchestrator._provider_from_application_settings",
        lambda: FakeProvider(),
    )
    settings_service._cache["telegram.chat_id"] = "77"

    from financial_dashboard.services.assistant.orchestrator import (
        resume_claimed_interactions,
    )

    assert await resume_claimed_interactions() == 1
    assert await resume_claimed_interactions() == 0

    async with maker() as verification:
        saved = await verification.get(AuditInteraction, interaction.id)
    assert saved.status == "ready_to_send"
    assert saved.outcome == "answer"
    assert "ambiguous" in saved.assistant_text
