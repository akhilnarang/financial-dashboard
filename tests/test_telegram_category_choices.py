import datetime
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.db as db_package
from financial_dashboard.db import (
    AuditInteraction,
    CategoryReviewDecision,
    Setting,
    TelegramMessageContext,
    TelegramOutboundDelivery,
    Transaction,
)
from financial_dashboard.db.models import Category
from financial_dashboard.services.assistant.delivery import make_delivery
from financial_dashboard.services.assistant.orchestrator import _lock_authorized_chat
from financial_dashboard.services.assistant.orchestrator import (
    _process_callback_interaction,
)
from financial_dashboard.services.categorization.hashing import (
    build_input_payload,
    compute_input_hash,
)
from financial_dashboard.services.categorization.review_decisions import (
    consume_decision,
)
from financial_dashboard.services import settings as settings_service


@pytest.mark.anyio
async def test_category_choice_consumes_once_with_assignment_and_audit(session):
    session.add_all(
        [Category(slug="groceries", active=True), Category(slug="dining", active=True)]
    )
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        counterparty="PUREBERRYSMUMBAI",
        review_status="pending",
    )
    session.add(transaction)
    await session.flush()
    input_hash = compute_input_hash(build_input_payload(transaction, None))
    transaction.category_input_hash = input_hash
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash=input_hash,
        candidates_json=json.dumps(
            [
                {"category": "groceries", "confidence": 0.55},
                {"category": "dining", "confidence": 0.42},
            ]
        ),
        gate_reason="confidence 0.55 below 0.60",
    )
    session.add(decision)
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=7,
        text="choose",
        ordinal=0,
        category_review_decision_id=decision.id,
        transaction_id=transaction.id,
    )
    session.add(delivery)
    await session.flush()

    action = await consume_decision(
        session,
        decision.id,
        selected_slug="groceries",
        category_input_hash=input_hash,
        delivery_id=delivery.id,
    )
    second = await consume_decision(
        session,
        decision.id,
        selected_slug="dining",
        category_input_hash=input_hash,
        delivery_id=delivery.id,
    )

    assert action is not None
    assert second is None
    assert transaction.category == "groceries"
    assert decision.status == "consumed"


@pytest.mark.anyio
async def test_category_choice_succeeds_after_fresh_sqlite_reload(session):
    session.add(Category(slug="groceries", active=True))
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        counterparty="PUREBERRYSMUMBAI",
        review_status="pending",
    )
    session.add(transaction)
    await session.flush()
    input_hash = compute_input_hash(build_input_payload(transaction, None))
    transaction.category_input_hash = input_hash
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash=input_hash,
        candidates_json=json.dumps([{"category": "groceries", "confidence": 0.55}]),
        gate_reason="low confidence",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    session.add(decision)
    await session.commit()
    decision_id = decision.id
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )

    async with maker() as fresh:
        reloaded = await fresh.get(CategoryReviewDecision, decision_id)
        assert reloaded.expires_at is not None
        assert reloaded.expires_at.tzinfo is None
        await _lock_authorized_chat(fresh, None)
        action = await consume_decision(
            fresh,
            decision_id,
            selected_slug="groceries",
            category_input_hash=input_hash,
        )
        await fresh.commit()

    assert action is not None
    async with maker() as verification:
        saved = await verification.get(Transaction, transaction.id)
    assert saved.category == "groceries"


@pytest.mark.anyio
async def test_delivery_finalizer_does_not_reopen_review_consumed_during_send(
    session, monkeypatch
):
    """A callback may arrive after Telegram accepts but before we record delivery."""
    from financial_dashboard.db import TelegramOutboundDelivery
    from financial_dashboard.services import telegram

    session.add(Category(slug="groceries", active=True))
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        counterparty="PUREBERRYSMUMBAI",
        review_status="pending",
    )
    session.add(transaction)
    await session.flush()
    input_hash = compute_input_hash(build_input_payload(transaction, None))
    transaction.category_input_hash = input_hash
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash=input_hash,
        candidates_json=json.dumps([{"category": "groceries", "confidence": 0.55}]),
        gate_reason="low confidence",
    )
    session.add(decision)
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=7,
        text="choose",
        ordinal=0,
        category_review_decision_id=decision.id,
        transaction_id=transaction.id,
    )
    session.add(delivery)
    await session.commit()
    transaction_id = transaction.id
    decision_id = decision.id
    delivery_id = delivery.id
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(telegram, "async_session", maker)
    monkeypatch.setattr(telegram, "tg_app", object())
    settings_service._cache["telegram.chat_id"] = "7"
    settings_service._cache["telegram.assistant_enabled"] = "true"

    async def callback_arrives_while_send_is_returning(*args, **kwargs):
        async with maker() as callback_session:
            action = await consume_decision(
                callback_session,
                decision_id,
                selected_slug="groceries",
                category_input_hash=input_hash,
                delivery_id=delivery_id,
            )
            assert action is not None
            await callback_session.commit()
        return SimpleNamespace(message_id=700)

    monkeypatch.setattr(
        telegram, "_send_with_retry", callback_arrives_while_send_is_returning
    )

    assert await telegram.dispatch_saved_delivery(delivery_id)

    async with maker() as verification:
        saved_transaction = await verification.get(Transaction, transaction_id)
        saved_decision = await verification.get(CategoryReviewDecision, decision_id)
        saved_delivery = await verification.get(TelegramOutboundDelivery, delivery_id)
    assert saved_transaction.review_status == "resolved"
    assert saved_transaction.category == "groceries"
    assert saved_decision.status == "consumed"
    assert saved_delivery.status == "delivered"


@pytest.mark.anyio
async def test_callback_repairs_delivery_and_physical_message_context(
    session, monkeypatch
):
    from financial_dashboard.services import telegram

    session.add_all(
        [
            Category(slug="groceries", active=True),
            Setting(key="telegram.chat_id", value="7"),
        ]
    )
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        counterparty="PUREBERRYSMUMBAI",
        review_status="pending",
    )
    session.add(transaction)
    await session.flush()
    input_hash = compute_input_hash(build_input_payload(transaction, None))
    transaction.category_input_hash = input_hash
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash=input_hash,
        candidates_json=json.dumps([{"category": "groceries", "confidence": 0.55}]),
        gate_reason="low confidence",
    )
    callback = AuditInteraction(
        telegram_update_id="callback-1",
        inbound_chat_id=7,
        inbound_message_id=700,
        trigger="category_button",
        user_text="pending",
        status="processing",
        worker_token="callback-worker",
    )
    session.add_all([decision, callback])
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=7,
        text="choose",
        ordinal=0,
        category_review_decision_id=decision.id,
        transaction_id=transaction.id,
    )
    delivery.status = "delivering"
    delivery.worker_token = "sender-worker"
    session.add(delivery)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_package, "async_session", maker)

    async def fake_dispatch(delivery_id: int) -> bool:
        return True

    monkeypatch.setattr(telegram, "dispatch_saved_delivery", fake_dispatch)

    await _process_callback_interaction(
        interaction_id=callback.id,
        worker_token="callback-worker",
        trigger="category_button",
        callback_data=f"cat:v1:{decision.id}:{delivery.id}:0",
        recipient_chat_id=7,
        physical_message_id=700,
    )

    async with maker() as verification:
        saved_delivery = await verification.get(TelegramOutboundDelivery, delivery.id)
        physical = await verification.scalar(
            select(TelegramMessageContext).where(
                TelegramMessageContext.chat_id == 7,
                TelegramMessageContext.message_id == 700,
            )
        )
    assert saved_delivery.status == "delivered"
    assert saved_delivery.worker_token is None
    assert physical.transaction_id == transaction.id
    assert physical.outbound_delivery_id == delivery.id
    assert physical.context_kind == "category_review"


@pytest.mark.anyio
async def test_disabled_assistant_does_not_dispatch_persisted_output(
    session, monkeypatch
):
    from financial_dashboard.services import telegram

    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="10.00",
        review_status="pending",
    )
    session.add(transaction)
    await session.flush()
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash="hash",
        candidates_json="[]",
    )
    session.add(decision)
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=7,
        text="persisted assistant output",
        ordinal=0,
        category_review_decision_id=decision.id,
        transaction_id=transaction.id,
    )
    session.add(delivery)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(telegram, "async_session", maker)
    monkeypatch.setattr(telegram, "tg_app", object())
    settings_service._cache["telegram.chat_id"] = "7"
    settings_service._cache["telegram.assistant_enabled"] = "false"
    sent = False

    async def fail_if_sent(*args, **kwargs):
        nonlocal sent
        sent = True

    monkeypatch.setattr(telegram, "_send_with_retry", fail_if_sent)

    assert await telegram.dispatch_saved_delivery(delivery.id) is False
    assert sent is False
    async with maker() as verification:
        saved = await verification.get(TelegramOutboundDelivery, delivery.id)
    assert saved.status == "pending"


@pytest.mark.anyio
async def test_enabled_review_notification_persists_outbox_before_dispatch(
    session, monkeypatch
):
    from financial_dashboard.services.categorization import sweep
    from financial_dashboard.services import telegram
    from financial_dashboard.db import TelegramOutboundDelivery

    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        counterparty="PUREBERRYSMUMBAI",
        review_status="pending",
        review_reason="food or beverage merchant",
        notify_attempts=0,
    )
    session.add(transaction)
    await session.flush()
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash="saved-hash",
        candidates_json=json.dumps(
            [
                {"category": "groceries", "confidence": 0.55},
                {"category": "dining", "confidence": 0.42},
            ]
        ),
        gate_reason="confidence 0.55 below 0.60",
    )
    session.add(decision)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(sweep, "async_session", maker)
    monkeypatch.setattr(telegram, "tg_app", object())
    dispatched = []

    async def fake_dispatch(delivery_id):
        dispatched.append(delivery_id)
        return True

    monkeypatch.setattr(telegram, "dispatch_saved_delivery", fake_dispatch)
    settings_service._cache.update(
        {
            "telegram.enabled": "true",
            "telegram.bot_token": "test",
            "telegram.chat_id": "7",
            "telegram.assistant_enabled": "true",
        }
    )

    assert await sweep.run_review_notify() == 1

    async with maker() as verification:
        delivery = await verification.scalar(select(TelegramOutboundDelivery))
    assert delivery is not None
    assert dispatched == [delivery.id]
    assert delivery.category_review_decision_id == decision.id
    assert "Ref: " in delivery.text
    assert delivery.reply_markup_json is not None
    assert "cat:v1:" in delivery.reply_markup_json


@pytest.mark.anyio
async def test_notifier_does_not_duplicate_active_conversational_decision(
    session, monkeypatch
):
    from financial_dashboard.services import telegram
    from financial_dashboard.services.categorization import sweep

    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        review_status="pending",
        review_reason="ambiguous",
    )
    source = AuditInteraction(
        inbound_chat_id=7,
        trigger="reply",
        status="delivered",
        outcome="category_proposal",
    )
    session.add_all([transaction, source])
    await session.flush()
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        source_interaction_id=source.id,
        category_input_hash="hash",
        candidates_json=json.dumps([{"category": "groceries", "confidence": 0.5}]),
    )
    session.add(decision)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(sweep, "async_session", maker)
    monkeypatch.setattr(telegram, "tg_app", object())
    settings_service._cache.update(
        {
            "telegram.enabled": "true",
            "telegram.bot_token": "test",
            "telegram.chat_id": "7",
            "telegram.assistant_enabled": "true",
        }
    )

    assert await sweep.run_review_notify() == 0

    async with maker() as verification:
        active_count = await verification.scalar(
            select(func.count(CategoryReviewDecision.id)).where(
                CategoryReviewDecision.transaction_id == transaction.id,
                CategoryReviewDecision.status == "active",
            )
        )
    assert active_count == 1


@pytest.mark.anyio
async def test_enabled_long_review_queues_all_chunks_with_buttons_only_first(
    session, monkeypatch
):
    from financial_dashboard.services import telegram
    from financial_dashboard.services.categorization import sweep
    from financial_dashboard.db import TelegramOutboundDelivery

    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        counterparty="MERCHANT " + ("x" * 9000),
        review_status="pending",
        review_reason="ambiguous merchant " + ("reason " * 1200),
        notify_attempts=0,
    )
    session.add(transaction)
    await session.flush()
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash="saved-hash",
        candidates_json=json.dumps(
            [
                {"category": "groceries", "confidence": 0.55},
                {"category": "dining", "confidence": 0.42},
            ]
        ),
        gate_reason="ambiguous evidence",
    )
    session.add(decision)
    await session.commit()
    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(sweep, "async_session", maker)
    monkeypatch.setattr(telegram, "tg_app", object())
    dispatched: list[int] = []

    async def fake_dispatch(delivery_id: int) -> bool:
        dispatched.append(delivery_id)
        return True

    monkeypatch.setattr(telegram, "dispatch_saved_delivery", fake_dispatch)
    settings_service._cache.update(
        {
            "telegram.enabled": "true",
            "telegram.bot_token": "test",
            "telegram.chat_id": "7",
            "telegram.assistant_enabled": "true",
        }
    )

    assert await sweep.run_review_notify() == 1

    async with maker() as verification:
        deliveries = list(
            (
                await verification.scalars(
                    select(TelegramOutboundDelivery)
                    .where(
                        TelegramOutboundDelivery.category_review_decision_id
                        == decision.id
                    )
                    .order_by(TelegramOutboundDelivery.ordinal)
                )
            ).all()
        )
    assert len(deliveries) > 1
    assert dispatched == [delivery.id for delivery in deliveries]
    assert all(len(delivery.text) <= 4096 for delivery in deliveries)
    assert all(delivery.parse_mode is None for delivery in deliveries)
    assert deliveries[0].reply_markup_json is not None
    assert "cat:v1:" in deliveries[0].reply_markup_json
    assert all(delivery.reply_markup_json is None for delivery in deliveries[1:])


@pytest.mark.anyio
async def test_category_choice_rejects_changed_transaction_input(session):
    session.add(Category(slug="groceries", active=True))
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="960.00",
        counterparty="PUREBERRYSMUMBAI",
        review_status="pending",
    )
    session.add(transaction)
    await session.flush()
    original_hash = compute_input_hash(build_input_payload(transaction, None))
    transaction.category_input_hash = original_hash
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash=original_hash,
        candidates_json=json.dumps([{"category": "groceries", "confidence": 0.55}]),
        gate_reason="confidence below threshold",
    )
    session.add(decision)
    await session.flush()
    transaction.counterparty = "A DIFFERENT MERCHANT"
    await session.flush()

    action = await consume_decision(
        session,
        decision.id,
        selected_slug="groceries",
        category_input_hash=original_hash,
    )

    assert action is None
    assert decision.status == "active"
    assert transaction.category is None
