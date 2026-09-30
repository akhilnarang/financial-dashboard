import datetime

import pytest

from financial_dashboard.db.models import AuditInteraction, TelegramOutboundDelivery
from financial_dashboard.services.assistant.audit import (
    claim_processing,
    finalize_interaction,
    mark_authorization_changed,
    renew_processing_lease,
)
from financial_dashboard.services.assistant.delivery import (
    MAX_DELIVERY_ATTEMPTS,
    make_delivery,
    recover_assistant_work,
    refresh_interaction_delivery_status,
    settle_delivery_from_callback,
    validate_delivery_proof,
)
from financial_dashboard.services.assistant.evals import iter_audit_eval_rows
from financial_dashboard.services.assistant.message_context import recover_ref_context
from financial_dashboard.services.assistant.rendering import split_plain_text


@pytest.mark.anyio
async def test_processing_lease_is_fenced_to_owning_worker(session):
    row = AuditInteraction(inbound_chat_id=1, trigger="reply", status="claimed")
    session.add(row)
    await session.flush()
    claimed, token = await claim_processing(
        session, row.id, lease=datetime.timedelta(seconds=1)
    )
    assert claimed is not None
    await session.refresh(claimed)
    original_expiry = claimed.processing_lease_until
    assert original_expiry is not None

    assert not await finalize_interaction(
        session, row.id, "wrong", status="ready_to_send", outcome="answer"
    )
    assert not await renew_processing_lease(session, row.id, "wrong-worker")
    assert await renew_processing_lease(
        session, row.id, token, lease=datetime.timedelta(minutes=5)
    )
    await session.refresh(claimed)
    assert claimed.processing_lease_until > original_expiry

    assert await mark_authorization_changed(session, row.id)
    await session.flush()
    await session.refresh(claimed)
    assert claimed.status == "failed"


@pytest.mark.anyio
async def test_callback_settlement_finishes_source_interaction_audit(session):
    source = AuditInteraction(
        inbound_chat_id=1,
        trigger="reply",
        status="ready_to_send",
    )
    session.add(source)
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=1,
        text="choose",
        ordinal=0,
        interaction_id=source.id,
    )
    delivery.status = "delivering"
    delivery.worker_token = "sender-worker"
    session.add(delivery)
    await session.flush()

    assert await settle_delivery_from_callback(session, delivery.id)

    await session.refresh(source)
    assert delivery.status == "delivered"
    assert source.status == "delivered"


@pytest.mark.anyio
async def test_delivery_proof_cannot_revive_authorization_cancelled_output(session):
    row = AuditInteraction(
        inbound_chat_id=1,
        trigger="reply",
        status="delivery_partial",
        outcome="mutation",
        assistant_text="Updated category.",
    )
    session.add(row)
    await session.flush()
    delivered = make_delivery(
        recipient_chat_id=1,
        text="Updated category.",
        ordinal=0,
        interaction_id=row.id,
    )
    delivered.status = "delivered"
    unsent = make_delivery(
        recipient_chat_id=1,
        text="More details.",
        ordinal=1,
        interaction_id=row.id,
    )
    session.add_all([delivered, unsent])
    await session.flush()

    assert await mark_authorization_changed(session, row.id)
    assert (
        await refresh_interaction_delivery_status(session, row.id) == "delivery_failed"
    )
    await session.refresh(row)

    assert unsent.status == "cancelled"
    assert row.status == "delivery_failed"
    assert row.outcome == "mutation"
    assert row.error_code == "authorization_changed"
    assert row.completed_at is not None


@pytest.mark.anyio
async def test_recovery_marks_crashed_final_attempt_delivery_failed(session):
    row = AuditInteraction(inbound_chat_id=1, trigger="ask", status="delivery_partial")
    session.add(row)
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=1,
        text="result",
        ordinal=0,
        interaction_id=row.id,
    )
    delivery.status = "delivering"
    delivery.delivery_attempts = MAX_DELIVERY_ATTEMPTS
    delivery.delivery_lease_until = datetime.datetime.now(
        datetime.UTC
    ) - datetime.timedelta(minutes=1)
    session.add(delivery)
    await session.flush()

    await recover_assistant_work(session)

    assert delivery.status == "abandoned"
    assert row.status == "delivery_failed"


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
        transaction_id=9,
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
    assert context.conversation_id == 42
    assert context.interaction_id == interaction.id
    assert context.transaction_id == 9
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


def test_split_plain_text_reserves_footer_on_every_chunk():
    chunks = split_plain_text("one two three four", footer="Ref: token", limit=18)
    assert len(chunks) > 1
    assert all(len(chunk) <= 18 for chunk in chunks)
    assert all(chunk.endswith("Ref: token") for chunk in chunks)
    with pytest.raises(ValueError):
        split_plain_text("hello", footer="x" * 20, limit=10)


@pytest.mark.anyio
async def test_eval_export_has_stable_ids_and_no_attachment_bytes(session):
    session.add(
        AuditInteraction(
            telegram_update_id="eval-1",
            inbound_chat_id=2,
            trigger="attachment",
            user_text="receipt",
            inbound_payload_json='{"file_id":"telegram-id","bytes":"omitted"}',
            model_input_json='{"transaction":{"id":9}}',
            status="delivered",
            outcome="attachment",
            output_mode="json_schema",
        )
    )
    await session.commit()

    rows = [row async for row in iter_audit_eval_rows(session)]

    assert rows[0]["interaction_id"] == 1
    assert rows[0]["model_input"] == {"transaction": {"id": 9}}
    assert rows[0]["output_mode"] == "json_schema"
    assert "inbound_payload" not in rows[0]
