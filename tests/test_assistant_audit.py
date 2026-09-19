import pytest
import datetime

from financial_dashboard.db.models import AuditInteraction
from financial_dashboard.services.assistant.audit import (
    claim_processing,
    finalize_interaction,
    mark_authorization_changed,
    renew_processing_lease,
)
from financial_dashboard.services.assistant.delivery import (
    MAX_DELIVERY_ATTEMPTS,
    abandon_exhausted,
    make_delivery,
    refresh_interaction_delivery_status,
    recover_assistant_work,
    settle_delivery_from_callback,
)


@pytest.mark.anyio
async def test_processing_lease_is_fenced_and_authorization_transitions(session):
    row = AuditInteraction(
        telegram_update_id="audit-1",
        inbound_chat_id=1,
        trigger="reply",
        status="claimed",
    )
    session.add(row)
    await session.flush()
    claimed, token = await claim_processing(session, row.id)
    assert claimed is not None
    assert not await finalize_interaction(
        session, row.id, "wrong", status="ready_to_send", outcome="answer"
    )
    assert await mark_authorization_changed(session, row.id)
    await session.flush()
    assert (await session.get(AuditInteraction, row.id)).status == "failed"


@pytest.mark.anyio
async def test_processing_lease_renews_only_for_owning_worker(session):
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

    assert not await renew_processing_lease(session, row.id, "wrong-worker")
    assert await renew_processing_lease(
        session, row.id, token, lease=datetime.timedelta(minutes=5)
    )
    await session.refresh(claimed)
    assert claimed.processing_lease_until is not None
    assert claimed.processing_lease_until > original_expiry


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
async def test_ready_output_authorization_change_is_delivery_failed(session):
    row = AuditInteraction(
        inbound_chat_id=1,
        trigger="reply",
        status="ready_to_send",
    )
    session.add(row)
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=1,
        text="private financial result",
        ordinal=0,
        interaction_id=row.id,
    )
    session.add(delivery)
    await session.flush()
    assert await mark_authorization_changed(session, row.id)
    assert (await session.get(AuditInteraction, row.id)).status == "delivery_failed"
    assert delivery.status == "cancelled"


@pytest.mark.anyio
async def test_authorization_change_preserves_committed_outcome(session):
    row = AuditInteraction(
        inbound_chat_id=1,
        trigger="ask",
        status="delivery_partial",
        outcome="query_result",
        assistant_text="Found two transactions.",
    )
    session.add(row)
    await session.flush()

    assert await mark_authorization_changed(session, row.id)
    await session.refresh(row)
    assert row.status == "delivery_failed"
    assert row.outcome == "query_result"
    assert row.error_code == "authorization_changed"


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
async def test_exhausted_delivery_marks_interaction_failed(session):
    row = AuditInteraction(inbound_chat_id=1, trigger="ask", status="ready_to_send")
    session.add(row)
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=1,
        text="result",
        ordinal=0,
        interaction_id=row.id,
    )
    delivery.delivery_attempts = MAX_DELIVERY_ATTEMPTS
    session.add(delivery)
    await session.flush()

    assert await abandon_exhausted(session) == 1
    assert (
        await refresh_interaction_delivery_status(session, row.id) == "delivery_failed"
    )
    assert delivery.status == "abandoned"
    assert row.status == "delivery_failed"


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
