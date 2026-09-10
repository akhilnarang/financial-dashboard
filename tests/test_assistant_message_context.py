import pytest

from financial_dashboard.db.models import AuditInteraction, TelegramOutboundDelivery
from financial_dashboard.services.assistant.message_context import recover_ref_context
from financial_dashboard.services.assistant.delivery import validate_delivery_proof


@pytest.mark.anyio
async def test_recovered_delivery_inherits_interaction_conversation(session):
    interaction = AuditInteraction(
        inbound_chat_id=77,
        trigger="reply",
        conversation_id=42,
        assistant_text="answer",
    )
    session.add(interaction)
    await session.flush()
    delivery = TelegramOutboundDelivery(
        interaction_id=interaction.id,
        recipient_chat_id=77,
        ordinal=0,
        transaction_id=9,
        text="answer",
        delivery_token="token-12345678",
    )
    session.add(delivery)
    await session.flush()

    context = await recover_ref_context(
        session,
        chat_id=77,
        message_id=100,
        message_text="answer\n\nRef: token-12345678",
    )

    assert context is not None
    assert context.conversation_id == 42
    assert context.interaction_id == interaction.id
    assert context.transaction_id == 9
    assert delivery.status == "delivered"
    assert interaction.status == "delivered"


@pytest.mark.anyio
async def test_ref_recovery_proves_abandoned_delivery_and_reopens_context(session):
    interaction = AuditInteraction(
        inbound_chat_id=77,
        trigger="reply",
        conversation_id=42,
        assistant_text="answer",
        status="delivery_failed",
    )
    session.add(interaction)
    await session.flush()
    delivery = TelegramOutboundDelivery(
        interaction_id=interaction.id,
        recipient_chat_id=77,
        ordinal=0,
        text="answer",
        delivery_token="token-abandoned",
        status="abandoned",
    )
    session.add(delivery)
    await session.flush()

    context = await recover_ref_context(
        session,
        chat_id=77,
        message_id=101,
        message_text="answer\n\nRef: token-abandoned",
    )

    assert context is not None
    assert delivery.status == "delivered"
    assert interaction.status == "delivered"


@pytest.mark.anyio
async def test_recovered_delivery_rejects_wrong_recipient(session):
    interaction = AuditInteraction(inbound_chat_id=88, trigger="reply")
    session.add(interaction)
    await session.flush()
    delivery = TelegramOutboundDelivery(
        interaction_id=interaction.id,
        recipient_chat_id=88,
        ordinal=0,
        text="answer",
        delivery_token="token-87654321",
    )
    session.add(delivery)
    await session.flush()

    context = await recover_ref_context(
        session,
        chat_id=77,
        message_id=100,
        message_text="answer\n\nRef: token-87654321",
    )

    assert context is None


@pytest.mark.anyio
async def test_ref_recovery_uses_final_footer_when_message_quotes_another_ref(session):
    first = AuditInteraction(inbound_chat_id=77, trigger="reply", conversation_id=1)
    final = AuditInteraction(inbound_chat_id=77, trigger="reply", conversation_id=2)
    session.add_all([first, final])
    await session.flush()
    session.add_all(
        [
            TelegramOutboundDelivery(
                interaction_id=first.id,
                recipient_chat_id=77,
                ordinal=0,
                transaction_id=1,
                text="old",
                delivery_token="token-old-123456",
            ),
            TelegramOutboundDelivery(
                interaction_id=final.id,
                recipient_chat_id=77,
                ordinal=0,
                transaction_id=2,
                text="new",
                delivery_token="token-new-123456",
            ),
        ]
    )
    await session.flush()

    context = await recover_ref_context(
        session,
        chat_id=77,
        message_id=102,
        message_text="Quoted text\nRef: token-old-123456\n\nnew answer\nRef: token-new-123456",
    )

    assert context is not None
    assert context.outbound_delivery_id is not None
    assert context.transaction_id == 2


@pytest.mark.anyio
async def test_delivery_proof_rejects_unrelated_transaction_without_settling(session):
    interaction = AuditInteraction(inbound_chat_id=77, trigger="reply")
    session.add(interaction)
    await session.flush()
    delivery = TelegramOutboundDelivery(
        interaction_id=interaction.id,
        recipient_chat_id=77,
        ordinal=0,
        transaction_id=9,
        text="choice",
        delivery_token="token-proof-123",
        status="pending",
    )
    session.add(delivery)
    await session.flush()

    assert not await validate_delivery_proof(
        session,
        delivery.id,
        recipient_chat_id=77,
        transaction_id=10,
    )
    assert delivery.status == "pending"
