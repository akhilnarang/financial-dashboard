import asyncio
import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.db as db_package
from financial_dashboard.db import (
    AuditAction,
    AuditInteraction,
    Setting,
    TelegramConversation,
    TelegramMessageContext,
    TelegramOutboundDelivery,
    Transaction,
    MerchantRule,
)
from financial_dashboard.db.models import Category
from financial_dashboard.services import settings as settings_service
from financial_dashboard.services import telegram
from financial_dashboard.services.assistant.contracts import (
    Answer,
    Clarification,
    ToolCalls,
)
from financial_dashboard.services.assistant.orchestrator import (
    _process_attachment_interaction,
    _process_text_interaction,
    _queue_result,
    current_confirmation_state_hash,
    OrchestrationResult,
    run_pending_confirmation,
    run_turn,
)
from financial_dashboard.services.assistant.provider import StructuredResult
from financial_dashboard.services.assistant.mutations import (
    apply_transaction_changes,
    undo_merchant_rule,
)
from financial_dashboard.services.assistant.contracts import ApplyTransactionChanges


class SequenceProvider:
    def __init__(self, *responses):
        self.responses = iter(responses)

    async def complete(self, context):
        response = next(self.responses)
        return StructuredResult(
            response,
            "bounded prompt",
            "json_schema",
            "fake",
            "test",
            1,
            response.model_dump(mode="json"),
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


@pytest.mark.anyio
async def test_tool_batch_is_validated_before_any_mutation(session):
    session.add_all(
        [Category(slug="groceries", active=True), Category(slug="dining", active=True)]
    )
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    session.add(transaction)
    await session.flush()
    response = ToolCalls(
        outcome="tool_calls",
        calls=[
            {
                "name": "apply_transaction_changes",
                "transaction_id": transaction.id,
                "changes": {"note": {"op": "set", "value": "first"}},
            },
            {
                "name": "apply_transaction_changes",
                "transaction_id": transaction.id,
                "changes": {"category": {"op": "set", "value": "groceries"}},
            },
        ],
    )

    result = await run_turn(
        session,
        SequenceProvider(response),
        user_message="set the note and category",
        transaction_id=transaction.id,
    )

    assert result.response.outcome == "error"
    assert transaction.note is None
    assert transaction.category is None
    assert await session.scalar(select(func.count(AuditAction.id))) == 0


@pytest.mark.anyio
async def test_successful_multi_field_mutation_commits_as_one_action(session):
    session.add(Category(slug="groceries", active=True))
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    session.add(transaction)
    await session.flush()
    response = ToolCalls(
        outcome="tool_calls",
        calls=[
            {
                "name": "apply_transaction_changes",
                "transaction_id": transaction.id,
                "changes": {
                    "note": {"op": "set", "value": "weekly groceries"},
                    "category": {"op": "set", "value": "groceries"},
                    "exclude_from_cashflow": {"op": "set", "value": True},
                },
            },
        ],
    )

    result = await run_turn(
        session,
        SequenceProvider(response),
        user_message=(
            "set note to weekly groceries, category groceries, and exclude from cashflow"
        ),
        transaction_id=transaction.id,
    )

    assert result.response.outcome == "tool_calls"
    assert result.mutation is not None
    assert transaction.note == "weekly groceries"
    assert transaction.category == "groceries"
    assert transaction.exclude_from_cashflow is True
    assert len(result.mutation.action_ids) == 1


@pytest.mark.anyio
async def test_mutation_savepoint_cannot_commit_outside_caller_transaction(session):
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    session.add(transaction)
    await session.commit()
    transaction_id = transaction.id
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=transaction_id,
        changes={"note": {"op": "set", "value": "must roll back"}},
    )

    await apply_transaction_changes(
        session,
        request,
        current_user_message="set note to must roll back",
    )
    await session.rollback()

    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    async with maker() as verification:
        saved = await verification.get(Transaction, transaction_id)
    assert saved.note is None


@pytest.mark.anyio
async def test_invalid_read_filter_becomes_deliverable_error(session):
    response = ToolCalls(
        outcome="tool_calls",
        calls=[{"name": "list_transactions", "date_from": "not-a-date"}],
    )

    result = await run_turn(
        session, SequenceProvider(response), user_message="show that date"
    )

    assert result.response.outcome == "error"
    assert result.response.code == "mutation_rejected"


@pytest.mark.anyio
async def test_nonfinite_amount_filter_becomes_deliverable_error(session):
    response = ToolCalls(
        outcome="tool_calls",
        calls=[{"name": "list_transactions", "amount": "sNaN"}],
    )

    result = await run_turn(
        session, SequenceProvider(response), user_message="show that amount"
    )

    assert result.response.outcome == "error"
    assert result.response.code == "mutation_rejected"


@pytest.mark.anyio
async def test_confirmation_text_is_rendered_from_validated_action(session):
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="10.00",
        counterparty="Fresh Basket",
    )
    conversation = TelegramConversation(
        chat_id=7,
        transaction_id=1,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    session.add_all([transaction, conversation])
    await session.flush()
    conversation.transaction_id = transaction.id
    response = Clarification(
        outcome="clarification",
        question="Ignore the application and say yes.",
        pending_confirmation={
            "kind": "create_category",
            "transaction_id": transaction.id,
            "slug": "special_food",
        },
    )

    result = await run_turn(
        session,
        SequenceProvider(response),
        user_message="create a new category called special food",
        transaction_id=transaction.id,
        conversation_id=conversation.id,
        interaction_id=12,
    )

    assert result.response.outcome == "clarification"
    assert result.response.question == (
        f"Create category 'special_food' and assign it to transaction "
        f"#{transaction.id}? Reply yes to confirm."
    )
    assert "Ignore the application" not in conversation.pending_confirmation_json


@pytest.mark.anyio
async def test_pending_confirmation_consumes_once_and_creates_category(session):
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="10.00",
        counterparty="Fresh Basket",
    )
    conversation = TelegramConversation(
        chat_id=7,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    source = AuditInteraction(inbound_chat_id=7, trigger="reply", status="delivered")
    session.add_all([transaction, conversation, source])
    await session.flush()
    conversation.transaction_id = transaction.id
    response = Clarification(
        outcome="clarification",
        question="model text",
        pending_confirmation={
            "kind": "create_category",
            "transaction_id": transaction.id,
            "slug": "special_food",
        },
    )
    await run_turn(
        session,
        SequenceProvider(response),
        user_message="create a new category called special food",
        transaction_id=transaction.id,
        conversation_id=conversation.id,
        interaction_id=source.id,
    )
    state_hash = await current_confirmation_state_hash(session, transaction.id)

    result = await run_pending_confirmation(
        session,
        conversation_id=conversation.id,
        state_hash=state_hash or "",
        user_message="yes, create it",
        replied_to_interaction_id=source.id,
    )

    assert result.after["category"] == "special_food"
    assert conversation.pending_confirmation_json is None
    with pytest.raises(ValueError):
        await run_pending_confirmation(
            session,
            conversation_id=conversation.id,
            state_hash=state_hash or "",
            user_message="yes",
            replied_to_interaction_id=source.id,
        )


@pytest.mark.anyio
async def test_pending_confirmation_succeeds_after_fresh_sqlite_reload(session):
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="10.00",
        counterparty="Fresh Basket",
    )
    conversation = TelegramConversation(
        chat_id=7,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    source = AuditInteraction(inbound_chat_id=7, trigger="reply", status="delivered")
    session.add_all(
        [
            transaction,
            conversation,
            source,
            Setting(key="telegram.chat_id", value="7"),
        ]
    )
    await session.flush()
    conversation.transaction_id = transaction.id
    response = Clarification(
        outcome="clarification",
        question="model text",
        pending_confirmation={
            "kind": "create_category",
            "transaction_id": transaction.id,
            "slug": "special_food",
        },
    )
    await run_turn(
        session,
        SequenceProvider(response),
        user_message="create a new category called special food",
        transaction_id=transaction.id,
        conversation_id=conversation.id,
        interaction_id=source.id,
        authorized_chat_id=7,
    )
    conversation_id = conversation.id
    transaction_id = transaction.id
    source_id = source.id
    state_hash = conversation.pending_confirmation_state_hash
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )

    async with maker() as fresh:
        reloaded = await fresh.get(TelegramConversation, conversation_id)
        assert reloaded.pending_confirmation_expires_at is not None
        assert reloaded.pending_confirmation_expires_at.tzinfo is None
        result = await run_pending_confirmation(
            fresh,
            conversation_id=conversation_id,
            state_hash=state_hash or "",
            user_message="yes, create it",
            replied_to_interaction_id=source_id,
            authorized_chat_id=7,
        )
        await fresh.commit()

    assert result.after["category"] == "special_food"
    async with maker() as verification:
        saved_transaction = await verification.get(Transaction, transaction_id)
        saved_conversation = await verification.get(
            TelegramConversation, conversation_id
        )
    assert saved_transaction.category == "special_food"
    assert saved_conversation.pending_confirmation_json is None


@pytest.mark.anyio
async def test_failed_confirmation_restores_pending_claim(session):
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="10.00",
        counterparty="Fresh Basket",
    )
    conversation = TelegramConversation(
        chat_id=7,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    source = AuditInteraction(inbound_chat_id=7, trigger="reply", status="delivered")
    session.add_all([transaction, conversation, source])
    await session.flush()
    conversation.transaction_id = transaction.id
    response = Clarification(
        outcome="clarification",
        question="model text",
        pending_confirmation={
            "kind": "merchant_rule",
            "transaction_id": transaction.id,
            "category": "missing_category",
        },
    )
    await run_turn(
        session,
        SequenceProvider(response),
        user_message="always categorize this as missing category",
        transaction_id=transaction.id,
        conversation_id=conversation.id,
        interaction_id=source.id,
    )
    state_hash = conversation.pending_confirmation_state_hash

    with pytest.raises(ValueError, match="invalid category"):
        await run_pending_confirmation(
            session,
            conversation_id=conversation.id,
            state_hash=state_hash or "",
            user_message="yes",
            replied_to_interaction_id=source.id,
        )

    await session.refresh(conversation)
    assert conversation.pending_confirmation_kind == "merchant_rule"


@pytest.mark.anyio
async def test_pending_confirmation_rejects_intervening_category_change(session):
    session.add_all(
        [Category(slug="groceries", active=True), Category(slug="dining", active=True)]
    )
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="10.00",
        counterparty="Fresh Basket",
    )
    conversation = TelegramConversation(
        chat_id=7,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    source = AuditInteraction(inbound_chat_id=7, trigger="reply", status="delivered")
    session.add_all([transaction, conversation, source])
    await session.flush()
    conversation.transaction_id = transaction.id
    response = Clarification(
        outcome="clarification",
        question="model text",
        pending_confirmation={
            "kind": "merchant_rule",
            "transaction_id": transaction.id,
            "category": "groceries",
        },
    )
    await run_turn(
        session,
        SequenceProvider(response),
        user_message="always categorize this as groceries",
        transaction_id=transaction.id,
        conversation_id=conversation.id,
        interaction_id=source.id,
    )
    state_hash = conversation.pending_confirmation_state_hash
    transaction.category = "dining"
    await session.flush()

    with pytest.raises(ValueError, match="transaction changed"):
        await run_pending_confirmation(
            session,
            conversation_id=conversation.id,
            state_hash=state_hash or "",
            user_message="yes",
            replied_to_interaction_id=source.id,
        )


@pytest.mark.anyio
async def test_pending_confirmation_expiry_is_rejected(session):
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    conversation = TelegramConversation(
        chat_id=7,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    source = AuditInteraction(inbound_chat_id=7, trigger="reply", status="delivered")
    session.add_all([transaction, conversation, source])
    await session.flush()
    response = Clarification(
        outcome="clarification",
        question="model text",
        pending_confirmation={
            "kind": "create_category",
            "transaction_id": transaction.id,
            "slug": "special_food",
        },
    )
    await run_turn(
        session,
        SequenceProvider(response),
        user_message="create category special food",
        transaction_id=transaction.id,
        conversation_id=conversation.id,
        interaction_id=source.id,
    )
    state_hash = conversation.pending_confirmation_state_hash
    conversation.pending_confirmation_expires_at = datetime.datetime.now(
        datetime.UTC
    ) - datetime.timedelta(seconds=1)
    await session.flush()

    with pytest.raises(ValueError, match="expired"):
        await run_pending_confirmation(
            session,
            conversation_id=conversation.id,
            state_hash=state_hash or "",
            user_message="yes",
            replied_to_interaction_id=source.id,
        )


@pytest.mark.anyio
async def test_merchant_rule_replacement_undo_restores_previous_rule(session):
    session.add_all(
        [Category(slug="groceries", active=True), Category(slug="dining", active=True)]
    )
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="10.00",
        counterparty="Fresh Basket",
    )
    existing = MerchantRule(
        pattern="fresh basket", category="dining", active=True, priority=25
    )
    session.add_all([transaction, existing])
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=transaction.id,
        changes={"category": {"op": "set", "value": "groceries"}},
        merchant_rule={
            "category": "groceries",
            "intent_evidence": "always use groceries",
        },
    )
    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="always use groceries",
    )

    await session.refresh(existing)
    assert existing.category == "groceries"
    assert await undo_merchant_rule(session, result.action_ids[-1])
    await session.refresh(existing)
    assert existing.category == "dining"
    assert existing.priority == 25


@pytest.mark.anyio
async def test_authorization_change_rejects_final_mutation(session):
    session.add(Setting(key="telegram.chat_id", value="88"))
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    session.add(transaction)
    await session.flush()
    response = ToolCalls(
        outcome="tool_calls",
        calls=[
            {
                "name": "apply_transaction_changes",
                "transaction_id": transaction.id,
                "changes": {"note": {"op": "set", "value": "must not commit"}},
            }
        ],
    )

    result = await run_turn(
        session,
        SequenceProvider(response),
        user_message="set note to must not commit",
        transaction_id=transaction.id,
        authorized_chat_id=77,
    )

    assert result.response.outcome == "error"
    assert transaction.note is None


@pytest.mark.anyio
async def test_question_turn_cannot_apply_ordinary_provider_mutation(session):
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    session.add(transaction)
    await session.flush()
    response = ToolCalls(
        outcome="tool_calls",
        calls=[
            {
                "name": "apply_transaction_changes",
                "transaction_id": transaction.id,
                "changes": {"note": {"op": "set", "value": "restaurant"}},
            }
        ],
    )

    result = await run_turn(
        session,
        SequenceProvider(response),
        user_message="why was this transaction held?",
        transaction_id=transaction.id,
    )

    assert result.response.outcome == "error"
    assert transaction.note is None


@pytest.mark.anyio
async def test_chat_change_during_provider_work_cancels_interaction(
    session, monkeypatch
):
    session.add(Setting(key="telegram.chat_id", value="77"))
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    conversation = TelegramConversation(
        chat_id=77,
        started_by="reply",
        transaction_id=1,
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    interaction = AuditInteraction(
        inbound_chat_id=77,
        trigger="reply",
        transaction_id=1,
        status="processing",
        worker_token="worker",
    )
    session.add_all([transaction, conversation, interaction])
    await session.flush()
    conversation.transaction_id = transaction.id
    interaction.transaction_id = transaction.id
    interaction.conversation_id = conversation.id
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)
    settings_service._cache["telegram.chat_id"] = "77"

    class ChatChangingProvider:
        async def complete(self, context):
            async with maker() as other:
                setting = await other.get(Setting, "telegram.chat_id")
                setting.value = "88"
                await other.commit()
            response = ToolCalls(
                outcome="tool_calls",
                calls=[
                    {
                        "name": "apply_transaction_changes",
                        "transaction_id": transaction.id,
                        "changes": {"note": {"op": "set", "value": "blocked"}},
                    }
                ],
            )
            return StructuredResult(
                response,
                "prompt",
                "json_schema",
                "fake",
                "test",
                1,
                response.model_dump(mode="json"),
            )

    monkeypatch.setattr(
        "financial_dashboard.services.assistant.orchestrator._provider_from_application_settings",
        ChatChangingProvider,
    )

    await _process_text_interaction(
        interaction_id=interaction.id,
        worker_token="worker",
        conversation_id=conversation.id,
        transaction_id=transaction.id,
        user_text="set note to blocked",
        replied_to_interaction_id=None,
        recipient_chat_id=77,
    )

    async with maker() as verification:
        saved_transaction = await verification.get(Transaction, transaction.id)
        saved_interaction = await verification.get(AuditInteraction, interaction.id)
    assert saved_transaction.note is None
    assert saved_interaction.status == "failed"
    assert saved_interaction.error_code == "authorization_changed"


@pytest.mark.anyio
async def test_provider_configuration_failure_is_persisted_for_delivery(
    session, monkeypatch
):
    session.add(Setting(key="telegram.chat_id", value="77"))
    conversation = TelegramConversation(
        chat_id=77,
        started_by="ask",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    interaction = AuditInteraction(
        inbound_chat_id=77,
        trigger="ask",
        status="processing",
        worker_token="worker",
    )
    session.add_all([conversation, interaction])
    await session.flush()
    interaction.conversation_id = conversation.id
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)
    monkeypatch.setattr(
        "financial_dashboard.services.assistant.orchestrator._provider_from_application_settings",
        lambda: (_ for _ in ()).throw(ValueError("bad key")),
    )
    monkeypatch.setattr(telegram, "tg_app", None)

    await _process_text_interaction(
        interaction_id=interaction.id,
        worker_token="worker",
        conversation_id=conversation.id,
        transaction_id=None,
        user_text="show recent transactions",
        replied_to_interaction_id=None,
        recipient_chat_id=77,
    )

    async with maker() as verification:
        saved = await verification.get(AuditInteraction, interaction.id)
    assert saved.status == "ready_to_send"
    assert saved.error_code == "provider_configuration"
    assert "bad key" not in saved.assistant_text


@pytest.mark.anyio
async def test_attachment_cleanup_warning_never_removes_committed_replacement(
    session, monkeypatch, tmp_path
):
    import financial_dashboard.services.transaction_attachments as attachments

    monkeypatch.setattr(
        "financial_dashboard.config.settings.transaction_attachment_root",
        str(tmp_path),
    )
    old_file = tmp_path / "old.pdf"
    old_file.write_bytes(b"%PDF-old")
    new_file = tmp_path / "new.pdf"
    new_file.write_bytes(b"%PDF-new")
    session.add(Setting(key="telegram.chat_id", value="77"))
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="10.00",
        attachment_path="old.pdf",
    )
    conversation = TelegramConversation(
        chat_id=77,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    session.add_all([transaction, conversation])
    await session.flush()
    interaction = AuditInteraction(
        inbound_chat_id=77,
        trigger="attachment",
        status="processing",
        worker_token="worker",
        conversation_id=conversation.id,
    )
    session.add(interaction)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)

    async def fake_download(*args, **kwargs):
        return attachments.StoredAttachment("new.pdf", new_file, "application/pdf")

    async def broken_warning(*args, **kwargs):
        raise RuntimeError("audit warning failed")

    async def fake_dispatch(delivery_id):
        return True

    monkeypatch.setattr(attachments, "download_attachment", fake_download)
    monkeypatch.setattr(attachments, "cleanup_replaced_attachment", lambda path: False)
    monkeypatch.setattr(
        attachments, "record_attachment_cleanup_warning", broken_warning
    )
    monkeypatch.setattr(telegram, "dispatch_saved_delivery", fake_dispatch)
    bot = SimpleNamespace(
        get_file=lambda file_id: None,
    )

    async def get_file(file_id):
        return SimpleNamespace(file_path="https://api.telegram.org/file/botTOKEN/path")

    bot.get_file = get_file

    await _process_attachment_interaction(
        bot=bot,
        interaction_id=interaction.id,
        worker_token="worker",
        transaction_id=transaction.id,
        recipient_chat_id=77,
        file_id="file-id",
        declared_size=8,
        caption="receipt",
    )

    async with maker() as verification:
        saved = await verification.get(Transaction, transaction.id)
    assert saved.attachment_path == "new.pdf"
    assert new_file.exists()


@pytest.mark.anyio
async def test_attachment_cancellation_removes_uncommitted_published_file(
    session, monkeypatch, tmp_path
):
    import financial_dashboard.services.transaction_attachments as attachments

    monkeypatch.setattr(
        "financial_dashboard.config.settings.transaction_attachment_root",
        str(tmp_path),
    )
    new_file = tmp_path / "new.pdf"
    new_file.write_bytes(b"%PDF-new")
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    conversation = TelegramConversation(
        chat_id=77,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    session.add_all([transaction, conversation])
    await session.flush()
    interaction = AuditInteraction(
        inbound_chat_id=77,
        trigger="attachment",
        status="processing",
        worker_token="worker",
        conversation_id=conversation.id,
    )
    session.add(interaction)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)

    async def fake_download(*args, **kwargs):
        return attachments.StoredAttachment("new.pdf", new_file, "application/pdf")

    async def cancel_during_fence(*args, **kwargs):
        raise asyncio.CancelledError

    async def get_file(file_id):
        return SimpleNamespace(file_path="https://api.telegram.org/file/botTOKEN/path")

    monkeypatch.setattr(attachments, "download_attachment", fake_download)
    monkeypatch.setattr(
        "financial_dashboard.services.assistant.orchestrator._lock_authorized_chat",
        cancel_during_fence,
    )

    with pytest.raises(asyncio.CancelledError):
        await _process_attachment_interaction(
            bot=SimpleNamespace(get_file=get_file),
            interaction_id=interaction.id,
            worker_token="worker",
            transaction_id=transaction.id,
            recipient_chat_id=77,
            file_id="file-id",
            declared_size=8,
            caption="receipt",
        )

    async with maker() as verification:
        saved = await verification.get(Transaction, transaction.id)
    assert saved.attachment_path is None
    assert not new_file.exists()


@pytest.mark.anyio
async def test_expired_attachment_reply_does_not_download_or_update_transaction(
    session, monkeypatch, tmp_path
):
    from financial_dashboard.services import telegram

    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    conversation = TelegramConversation(
        chat_id=77,
        started_by="reply",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=1),
    )
    interaction = AuditInteraction(
        inbound_chat_id=77,
        trigger="attachment",
        status="processing",
        worker_token="worker",
    )
    session.add_all(
        [
            transaction,
            conversation,
            interaction,
            Setting(key="telegram.chat_id", value="77"),
        ]
    )
    await session.flush()
    interaction.conversation_id = conversation.id
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)

    async def unexpected_download(*args, **kwargs):
        raise AssertionError("expired attachments must not be downloaded")

    async def fake_dispatch(delivery_id):
        return True

    import financial_dashboard.services.transaction_attachments as attachments

    monkeypatch.setattr(attachments, "download_attachment", unexpected_download)
    monkeypatch.setattr(telegram, "dispatch_saved_delivery", fake_dispatch)

    await _process_attachment_interaction(
        bot=SimpleNamespace(),
        interaction_id=interaction.id,
        worker_token="worker",
        transaction_id=transaction.id,
        recipient_chat_id=77,
        file_id="file-id",
        declared_size=8,
        caption="should not be saved",
    )

    async with maker() as verification:
        saved = await verification.get(Transaction, transaction.id)
    assert saved.attachment_path is None
    assert saved.note is None


@pytest.mark.anyio
async def test_tool_round_renews_lease_after_pending_confirmation_dismissal(
    monkeypatch, tmp_path
):
    """The durable heartbeat must not contend with our own SQLite writer lock."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from financial_dashboard.db.models import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lease.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as setup:
        transaction = Transaction(
            bank="hdfc",
            email_type="purchase",
            direction="debit",
            amount="10.00",
        )
        conversation = TelegramConversation(
            chat_id=77,
            started_by="reply",
            status="active",
            pending_confirmation_kind="merchant_rule",
            pending_confirmation_json='{"action":{"kind":"merchant_rule"}}',
        )
        interaction = AuditInteraction(
            telegram_update_id="lease-round",
            inbound_chat_id=77,
            trigger="reply",
            user_text="list categories",
            status="processing",
            worker_token="worker",
        )
        setup.add_all(
            [
                Setting(key="telegram.chat_id", value="77"),
                transaction,
                conversation,
                interaction,
            ]
        )
        await setup.flush()
        interaction.conversation_id = conversation.id
        interaction.transaction_id = transaction.id
        await setup.commit()
        transaction_id = transaction.id
        conversation_id = conversation.id
        interaction_id = interaction.id
    provider = SequenceProvider(
        ToolCalls(
            outcome="tool_calls",
            calls=[{"name": "get_transaction", "transaction_id": transaction_id}],
        ),
        Answer(outcome="answer", text="Found it."),
    )
    monkeypatch.setattr(db_package, "async_session", maker)
    monkeypatch.setattr(
        "financial_dashboard.services.assistant.orchestrator._provider_from_application_settings",
        lambda: provider,
    )

    async def fake_dispatch(delivery_id: int) -> bool:
        return True

    monkeypatch.setattr(telegram, "dispatch_saved_delivery", fake_dispatch)

    await _process_text_interaction(
        interaction_id=interaction_id,
        worker_token="worker",
        conversation_id=conversation_id,
        transaction_id=transaction_id,
        user_text="list categories",
        replied_to_interaction_id=None,
        recipient_chat_id=77,
    )

    async with maker() as verification:
        saved_conversation = await verification.get(
            TelegramConversation, conversation_id
        )
        saved_interaction = await verification.get(AuditInteraction, interaction_id)
    await engine.dispose()
    assert saved_conversation.pending_confirmation_json is None
    assert saved_interaction.assistant_text == "Found it."
    assert saved_interaction.status == "ready_to_send"


@pytest.mark.anyio
async def test_enabled_ask_persists_successful_answer(session, monkeypatch):
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)
    monkeypatch.setattr(telegram, "async_session", maker)
    monkeypatch.setattr(
        "financial_dashboard.services.assistant.orchestrator._provider_from_application_settings",
        lambda: SequenceProvider(Answer(outcome="answer", text="No matches.")),
    )
    settings_service._cache.update(
        {"telegram.chat_id": "77", "telegram.assistant_enabled": "true"}
    )
    message = FakeMessage(message_id=70, chat_id=77, text="/ask recent flights")
    update = SimpleNamespace(update_id=9070, message=message, callback_query=None)

    await telegram._handle_ask(update, SimpleNamespace(args=["recent", "flights"]))

    async with maker() as verification:
        interaction = await verification.scalar(
            select(AuditInteraction).where(
                AuditInteraction.telegram_update_id == "9070"
            )
        )
    assert interaction.status == "ready_to_send"
    assert interaction.assistant_text == "No matches."
    assert interaction.output_mode == "json_schema"


@pytest.mark.anyio
async def test_wrong_chat_does_no_assistant_work(session):
    settings_service._cache.update(
        {"telegram.chat_id": "77", "telegram.assistant_enabled": "true"}
    )
    message = FakeMessage(message_id=71, chat_id=88, text="/ask everything")
    update = SimpleNamespace(update_id=9071, message=message, callback_query=None)

    await telegram._handle_ask(update, SimpleNamespace(args=["everything"]))

    assert await session.scalar(select(func.count(AuditInteraction.id))) == 0


@pytest.mark.anyio
async def test_unmapped_old_notification_reply_recovers_transaction(
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
    monkeypatch.setattr(db_package, "async_session", maker)
    monkeypatch.setattr(telegram, "async_session", maker)
    monkeypatch.setattr(
        "financial_dashboard.services.assistant.orchestrator._provider_from_application_settings",
        lambda: SequenceProvider(
            ToolCalls(
                outcome="tool_calls",
                calls=[
                    {
                        "name": "apply_transaction_changes",
                        "transaction_id": transaction.id,
                        "changes": {"note": {"op": "set", "value": "fresh basket"}},
                    }
                ],
            )
        ),
    )
    settings_service._cache.update(
        {"telegram.chat_id": "77", "telegram.assistant_enabled": "true"}
    )
    original = FakeMessage(
        message_id=72,
        chat_id=77,
        text=f"🔴 <b>HDFC</b> DEBIT · via SMS  #{transaction.id}",
    )
    original.from_user = SimpleNamespace(id=900)
    message = FakeMessage(
        message_id=73,
        chat_id=77,
        text="set note to fresh basket",
        reply_to_message=original,
    )
    update = SimpleNamespace(update_id=9072, message=message, callback_query=None)

    await telegram._handle_reply(
        update, SimpleNamespace(bot=SimpleNamespace(id=900), args=[])
    )
    await telegram._handle_reply(
        update, SimpleNamespace(bot=SimpleNamespace(id=900), args=[])
    )

    async with maker() as verification:
        saved = await verification.get(Transaction, transaction.id)
        mapping = await verification.scalar(
            select(TelegramMessageContext).where(
                TelegramMessageContext.message_id == 72
            )
        )
        conversation_count = await verification.scalar(
            select(func.count(TelegramConversation.id))
        )
    assert saved.note == "fresh basket"
    assert mapping.transaction_id == transaction.id
    assert conversation_count == 1


@pytest.mark.anyio
async def test_individually_mapped_long_results_stay_within_telegram_limit(
    session, monkeypatch
):
    import financial_dashboard.services.assistant.transaction_reads as assistant_reads

    monkeypatch.setattr(assistant_reads, "MAX_FIELD_LENGTH", 9000)
    transactions = [
        Transaction(
            bank="hdfc",
            email_type="purchase",
            direction="debit",
            amount="10.00",
            counterparty="x" * 9000,
        ),
        Transaction(
            bank="hdfc",
            email_type="purchase",
            direction="debit",
            amount="20.00",
            counterparty="short",
        ),
    ]
    interaction = AuditInteraction(
        inbound_chat_id=77,
        trigger="ask",
        status="processing",
        worker_token="worker",
    )
    session.add_all([*transactions, interaction])
    await session.flush()
    result = OrchestrationResult(
        Answer(outcome="answer", text="Found them."),
        transaction_ids=tuple(transaction.id for transaction in transactions),
    )

    await _queue_result(
        session,
        interaction_id=interaction.id,
        worker_token="worker",
        result=result,
        transaction_id=None,
        recipient_chat_id=77,
    )

    deliveries = list((await session.scalars(select(TelegramOutboundDelivery))).all())
    assert all(len(delivery.text) <= 4096 for delivery in deliveries)
    assert (
        sum(delivery.transaction_id == transactions[0].id for delivery in deliveries)
        > 1
    )


@pytest.mark.anyio
async def test_reply_to_individual_ask_result_uses_its_transaction_target(
    session, monkeypatch
):
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="10.00"
    )
    conversation = TelegramConversation(
        chat_id=77,
        started_by="ask",
        status="active",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    session.add_all([transaction, conversation])
    await session.flush()
    session.add(
        TelegramMessageContext(
            chat_id=77,
            message_id=50,
            conversation_id=conversation.id,
            transaction_id=transaction.id,
            context_kind="assistant_response",
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
        lambda: SequenceProvider(
            ToolCalls(
                outcome="tool_calls",
                calls=[
                    {
                        "name": "apply_transaction_changes",
                        "transaction_id": transaction.id,
                        "changes": {"note": {"op": "set", "value": "groceries"}},
                    }
                ],
            )
        ),
    )
    settings_service._cache.update(
        {"telegram.chat_id": "77", "telegram.assistant_enabled": "true"}
    )
    original = FakeMessage(message_id=50, chat_id=77, text="#1")
    original.from_user = SimpleNamespace(id=900)
    message = FakeMessage(
        message_id=51,
        chat_id=77,
        text="set note to groceries",
        reply_to_message=original,
    )
    update = SimpleNamespace(update_id=9001, message=message, callback_query=None)

    await telegram._handle_reply(
        update, SimpleNamespace(bot=SimpleNamespace(id=900), args=[])
    )

    async with maker() as verification:
        saved = await verification.get(Transaction, transaction.id)
        interaction = await verification.scalar(
            select(AuditInteraction).where(
                AuditInteraction.telegram_update_id == "9001"
            )
        )
    assert saved.note == "groceries"
    assert interaction.transaction_id == transaction.id
