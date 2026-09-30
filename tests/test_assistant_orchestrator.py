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
from financial_dashboard.services.assistant.orchestrator import (
    OrchestrationResult,
    _queue_result,
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
async def test_orchestrator_caps_tool_rounds(session):
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
    result = await run_turn(
        session, provider, user_message="look", transaction_id=txn.id
    )
    assert result.response.outcome == "error"
    assert provider.calls == 4


@pytest.mark.anyio
async def test_orchestrator_returns_answer(session):
    provider = FakeProvider([Answer(outcome="answer", text="No changes made.")])
    result = await run_turn(session, provider, user_message="why")
    assert result.response.text == "No changes made."


@pytest.mark.anyio
async def test_orchestrator_refuses_unsupported_aggregate_questions(session):
    provider = FakeProvider([])

    result = await run_turn(session, provider, user_message="how much did I spend?")

    assert result.response.outcome == "error"
    assert result.response.code == "unsupported_aggregate"
    assert provider.calls == 0


@pytest.mark.anyio
async def test_shorthand_note_offers_an_unnamed_category_as_a_button(session):
    await ensure_category(session, "food")
    txn = Transaction(bank="test", email_type="test", direction="debit", amount=10)
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
                            "note": {"op": "set", "value": "lunch with a friend"},
                            "category": {"op": "set", "value": "food"},
                        },
                    }
                ],
            )
        ]
    )

    result = await run_turn(
        session,
        provider,
        user_message="This was lunch with a friend",
        transaction_id=txn.id,
    )

    assert result.mutation is not None
    assert result.mutation.after["note"] == "lunch with a friend"
    assert result.mutation.after["category"] is None
    assert isinstance(result.response, CategoryProposal)
    assert [c.slug for c in result.response.candidates] == ["food"]
    assert result.decision_id is not None


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
async def test_orchestrator_renews_lease_between_provider_tool_rounds(session):
    txn = Transaction(bank="test", email_type="test", direction="debit", amount=10)
    session.add(txn)
    await session.flush()
    provider = FakeProvider(
        [
            ToolCalls(
                outcome="tool_calls",
                calls=[{"name": "get_transaction", "transaction_id": txn.id}],
            ),
            Answer(outcome="answer", text="Found it."),
        ]
    )
    renewals = 0

    async def renew() -> bool:
        nonlocal renewals
        renewals += 1
        return True

    result = await run_turn(
        session,
        provider,
        user_message="find it",
        transaction_id=txn.id,
        renew_lease=renew,
    )

    assert result.response.text == "Found it."
    assert renewals == 1


@pytest.mark.anyio
@pytest.mark.parametrize("rows", [1, 2])
async def test_multi_transaction_query_queues_individually_mapped_results(
    session, rows
):
    # A single row must also get its own card. The answer text must not be
    # bound to a row that it does not show.
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
    result = OrchestrationResult(
        Answer(outcome="answer", text="I found two transactions."),
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
async def test_error_outcome_is_not_overridden_by_transport_label(session):
    interaction = AuditInteraction(
        inbound_chat_id=7,
        trigger="attachment",
        status="processing",
        worker_token="worker",
    )
    session.add(interaction)
    await session.flush()

    await _queue_result(
        session,
        interaction_id=interaction.id,
        worker_token="worker",
        result=OrchestrationResult(
            Error(outcome="error", message="download failed", code="attachment_failed")
        ),
        transaction_id=None,
        recipient_chat_id=7,
        outcome_override="attachment",
    )

    await session.refresh(interaction)
    assert interaction.outcome == "error"
    assert interaction.error_code == "attachment_failed"


@pytest.mark.anyio
async def test_error_after_reads_is_not_classified_as_query_result(session):
    interaction = AuditInteraction(
        inbound_chat_id=7,
        trigger="ask",
        status="processing",
        worker_token="worker",
    )
    session.add(interaction)
    await session.flush()

    await _queue_result(
        session,
        interaction_id=interaction.id,
        worker_token="worker",
        result=OrchestrationResult(
            Error(
                outcome="error",
                message="invalid provider output",
                code="invalid_model_output",
            ),
            transaction_ids=(42,),
        ),
        transaction_id=None,
        recipient_chat_id=7,
    )

    await session.refresh(interaction)
    assert interaction.outcome == "error"
    assert interaction.error_code == "invalid_model_output"
