from types import SimpleNamespace

import pytest
from sqlalchemy import select

from financial_dashboard.db.models import (
    AuditInteraction,
    TelegramOutboundDelivery,
    Transaction,
)
from financial_dashboard.services.assistant.contracts import (
    Answer,
    CategoryProposal,
    Error,
    ToolCalls,
)
from financial_dashboard.services.assistant.conversations import start_conversation
from financial_dashboard.services.assistant.orchestrator import (
    OrchestrationResult,
    _conversation_for_message,
    _queue_result,
    run_pending_confirmation,
    run_turn,
)
from financial_dashboard.services.assistant.provider import StructuredResult
from financial_dashboard.services.categorization.vocabulary import ensure_category


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def complete(self, context):
        self.calls += 1
        response = next(self.responses)
        return StructuredResult(
            response, "prompt", "json_schema", "fake", "test", 1, {}
        )


@pytest.mark.anyio
async def test_orchestrator_caps_tool_rounds_and_renews_lease(session):
    txn = Transaction(bank="test", email_type="test", direction="debit", amount=10)
    session.add(txn)
    await session.flush()
    provider = FakeProvider(
        [
            ToolCalls(
                outcome="tool_calls",
                calls=[{"name": "get_transaction", "transaction_id": txn.id}],
            )
        ]
        * 4
    )
    renewals = 0

    async def renew() -> bool:
        nonlocal renewals
        renewals += 1
        return True

    result = await run_turn(
        session,
        provider,
        user_message="look",
        transaction_id=txn.id,
        renew_lease=renew,
    )
    assert result.response.outcome == "error"
    assert provider.calls == 4
    assert renewals == 3


@pytest.mark.anyio
async def test_orchestrator_refuses_unsupported_aggregate_questions(session):
    provider = FakeProvider([])

    result = await run_turn(session, provider, user_message="how much did I spend?")

    assert result.response.outcome == "error"
    assert result.response.code == "unsupported_aggregate"
    assert provider.calls == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("message", "note"),
    [
        ("This was lunch with a friend", "lunch with a friend"),
        # A bare reply is the note itself.
        ("Snickers in train", "Snickers in train"),
        # A category-only guess writes nothing and still offers the button.
        ("Auto rickshaw", None),
    ],
)
async def test_unnamed_category_is_offered_as_a_button(session, message, note):
    await ensure_category(session, "food")
    txn = Transaction(bank="test", email_type="test", direction="debit", amount=10)
    session.add(txn)
    await session.flush()
    changes = {"category": {"op": "set", "value": "food"}}
    if note is not None:
        changes["note"] = {"op": "set", "value": note}
    provider = FakeProvider(
        [
            ToolCalls(
                outcome="tool_calls",
                calls=[
                    {
                        "name": "apply_transaction_changes",
                        "transaction_id": txn.id,
                        "changes": changes,
                    }
                ],
            )
        ]
    )

    result = await run_turn(
        session, provider, user_message=message, transaction_id=txn.id
    )

    assert txn.note == note
    assert txn.category is None
    assert (result.mutation is not None) == (note is not None)
    assert isinstance(result.response, CategoryProposal)
    assert [c.slug for c in result.response.candidates] == ["food"]
    assert result.decision_id is not None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("message", "note"),
    [
        ("food", "lunch"),
        ("food for lunch with a friend", "food for lunch with a friend"),
    ],
)
async def test_category_only_reply_keeps_the_existing_note(session, message, note):
    await ensure_category(session, "food")
    txn = Transaction(
        bank="test", email_type="test", direction="debit", amount=10, note="lunch"
    )
    session.add(txn)
    await session.flush()
    provider = FakeProvider(
        [
            ToolCalls(
                outcome="tool_calls",
                calls=[
                    {
                        "name": "apply_transaction_changes",
                        "transaction_id": txn.id,
                        "changes": {
                            "note": {"op": "set", "value": message},
                            "category": {"op": "set", "value": "food"},
                        },
                    }
                ],
            )
        ]
    )

    await run_turn(session, provider, user_message=message, transaction_id=txn.id)

    assert txn.category == "food"
    assert txn.note == note


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("marked", None),
        # A bare number can be an amount, so the user confirms the row first.
        ("bare", None),
        ("unnamed", "ambiguous_target"),
        ("changed", "transaction_changed"),
    ],
)
async def test_typed_transaction_number_is_a_change_target(session, case, error):
    txn = Transaction(
        bank="test",
        email_type="test",
        direction="debit",
        amount=10,
        counterparty="ZERODHA BROKING",
    )
    conversation = await start_conversation(session, chat_id=7, started_by="ask")
    interaction = AuditInteraction(
        inbound_chat_id=7,
        trigger="ask",
        status="processing",
        conversation_id=conversation.id,
    )
    session.add_all([txn, interaction])
    await session.flush()

    class EditingProvider(FakeProvider):
        async def complete(self, context):
            if case == "changed":
                # Another writer edits the row while the model runs.
                txn.note = "edited elsewhere"
                await session.flush()
            return await super().complete(context)

    provider = EditingProvider(
        [
            ToolCalls(
                outcome="tool_calls",
                calls=[
                    {
                        "name": "apply_transaction_changes",
                        "transaction_id": txn.id,
                        "changes": {
                            "exclude_from_cashflow": {"op": "set", "value": True}
                        },
                    }
                ],
            )
        ]
    )
    # /ask has no bound target. Only a number the user typed can name one.
    message = {
        "unnamed": "exclude it from cashflow",
        "bare": f"exclude {txn.id} from cashflow",
    }.get(case, f"exclude #{txn.id} from cashflow")

    result = await run_turn(
        session,
        provider,
        user_message=message,
        conversation_id=conversation.id,
        interaction_id=interaction.id,
    )

    if case == "bare":
        assert txn.exclude_from_cashflow is False
        assert "ZERODHA BROKING" in result.response.question
        await run_pending_confirmation(
            session,
            conversation_id=conversation.id,
            state_hash=conversation.pending_confirmation_state_hash,
            user_message="yes",
            replied_to_interaction_id=interaction.id,
        )
    assert txn.exclude_from_cashflow is (error is None)
    if error is not None:
        assert result.response.code == error


@pytest.mark.anyio
async def test_turn_does_not_write_after_its_target_moved(session):
    txn = Transaction(bank="test", email_type="test", direction="debit", amount=10)
    session.add(txn)
    await session.flush()
    # A settlement fold moved this turn to another row while it ran.
    interaction = AuditInteraction(
        inbound_chat_id=7,
        trigger="reply",
        status="processing",
        transaction_id=txn.id + 1,
    )
    session.add(interaction)
    await session.flush()
    provider = FakeProvider(
        [
            ToolCalls(
                outcome="tool_calls",
                calls=[
                    {
                        "name": "apply_transaction_changes",
                        "transaction_id": txn.id,
                        "changes": {"note": {"op": "set", "value": "fuel"}},
                    }
                ],
            )
        ]
    )

    result = await run_turn(
        session,
        provider,
        user_message="set note to fuel",
        transaction_id=txn.id,
        interaction_id=interaction.id,
    )

    assert result.response.code == "transaction_changed"
    assert txn.note is None


@pytest.mark.anyio
@pytest.mark.parametrize(("rows", "names_row"), [(1, False), (2, False), (1, True)])
async def test_multi_transaction_query_queues_individually_mapped_results(
    session, rows, names_row
):
    # A single row gets its own card unless the answer names it. The answer
    # text must not be bound to a row that it does not show.
    transactions = [
        Transaction(
            bank="test",
            email_type="test",
            direction="debit",
            amount=amount,
            counterparty=counterparty,
        )
        for amount, counterparty in [(10, "ONE"), (20, "TWO")][:rows]
    ]
    interaction = AuditInteraction(
        inbound_chat_id=7,
        trigger="ask",
        status="processing",
        worker_token="worker",
    )
    session.add_all([*transactions, interaction])
    await session.flush()
    text = (
        f"Transaction {transactions[0].id} is a debit."
        if names_row
        else "I found two transactions."
    )
    result = OrchestrationResult(
        Answer(outcome="answer", text=text),
        transaction_ids=tuple(transaction.id for transaction in transactions),
        model_input_json='[{"role":"user","text":"find"}]',
        model_output_json='[{"outcome":"answer"}]',
        model_explanation="read-only transaction lookup",
        provider="fake",
        model="test-model",
        prompt_version="test-v1",
        output_mode="json_schema",
        input_tokens=3,
        output_tokens=4,
        latency_ms=5,
    )

    await _queue_result(
        session,
        interaction_id=interaction.id,
        worker_token="worker",
        result=result,
        transaction_id=None,
        recipient_chat_id=7,
    )

    deliveries = list(
        (
            await session.scalars(
                select(TelegramOutboundDelivery).order_by(
                    TelegramOutboundDelivery.ordinal
                )
            )
        ).all()
    )
    if names_row:
        assert [delivery.transaction_id for delivery in deliveries] == [
            transactions[0].id
        ]
        return
    assert [delivery.transaction_id for delivery in deliveries] == [
        None,
        *(transaction.id for transaction in transactions),
    ]
    for delivery, transaction in zip(deliveries[1:], transactions, strict=True):
        assert f"#{transaction.id}" in delivery.text
    await session.refresh(interaction)
    assert interaction.outcome == "query_result"
    assert interaction.model_input_json is not None
    assert interaction.model_output_json is not None
    assert interaction.model_explanation == "read-only transaction lookup"
    assert interaction.provider == "fake"
    assert interaction.model == "test-model"
    assert interaction.prompt_version == "test-v1"
    assert interaction.output_mode == "json_schema"
    assert interaction.input_tokens == 3
    assert interaction.output_tokens == 4
    assert interaction.latency_ms == 5


@pytest.mark.anyio
async def test_error_outcome_is_not_overridden_by_reads_or_transport_label(session):
    # A settlement fold moved this turn from row 100 to row 42 while it ran.
    interaction = AuditInteraction(
        inbound_chat_id=7,
        trigger="attachment",
        status="processing",
        worker_token="worker",
        transaction_id=42,
    )
    session.add(interaction)
    await session.flush()

    await _queue_result(
        session,
        interaction_id=interaction.id,
        worker_token="worker",
        result=OrchestrationResult(
            Error(outcome="error", message="download failed", code="attachment_failed"),
            transaction_ids=(42,),
        ),
        transaction_id=100,
        recipient_chat_id=7,
        outcome_override="attachment",
    )

    await session.refresh(interaction)
    assert interaction.outcome == "error"
    assert interaction.error_code == "attachment_failed"
    # SQLite can give id 100 to a new row. A reply must not reach it.
    delivery = await session.scalar(select(TelegramOutboundDelivery))
    assert delivery.transaction_id is None


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
