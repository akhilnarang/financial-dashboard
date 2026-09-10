from html import unescape
import json
import re
from urllib.parse import parse_qs, urlparse

import pytest

from financial_dashboard.db import (
    AuditAction,
    AuditInteraction,
    CategoryReviewDecision,
    TelegramConversation,
    TelegramOutboundDelivery,
    Transaction,
)
from financial_dashboard.services.audit_reads import list_terminal_proactive_deliveries


@pytest.mark.anyio
async def test_audit_page_lists_and_filters_interactions(session, client):
    interaction = AuditInteraction(
        telegram_update_id="audit-web-1",
        inbound_chat_id=123,
        trigger="reply",
        transaction_id=None,
        user_text="why?",
        status="delivered",
        outcome="answer",
        assistant_text="The category was ambiguous.",
    )
    session.add(interaction)
    await session.flush()
    session.add(
        AuditAction(
            interaction_id=interaction.id,
            action_type="set_note",
            target_type="transaction",
            target_id=7,
            arguments_json='{"note":"lunch"}',
            before_json='{"note":null}',
            after_json='{"note":"lunch"}',
            status="applied",
        )
    )
    await session.commit()

    response = await client.get("/audit", params={"action_type": "set_note"})

    assert response.status_code == 200
    assert "why?" in response.text
    assert "delivered" in response.text


@pytest.mark.anyio
async def test_audit_detail_shows_model_and_action_evidence(session, client):
    conversation = TelegramConversation(
        chat_id=123,
        started_by="ask",
        status="active",
    )
    session.add(conversation)
    await session.flush()
    interaction = AuditInteraction(
        telegram_update_id="audit-web-2",
        inbound_chat_id=123,
        trigger="reply",
        user_text="set note to lunch",
        status="ready_to_send",
        outcome="mutation",
        assistant_text="Saved note.",
        model_input_json='{"prompt":"bounded"}',
        model_output_json=json.dumps(
            {
                "outcome": "mutation",
                "merchant_search": {
                    "sources": [
                        {
                            "url": "https://example.com/merchant",
                            "title": "Merchant website",
                        },
                        {"url": "javascript:alert(1)", "title": "Invalid source"},
                    ]
                },
            }
        ),
        model_explanation="bounded explanation",
        provider="fake",
        model="test-model",
        prompt_version="assistant-v7",
        output_mode="json_schema",
        conversation_id=conversation.id,
    )
    session.add(interaction)
    await session.commit()

    response = await client.get(f"/audit/{interaction.id}")

    assert response.status_code == 200
    assert "set note to lunch" in response.text
    assert "Saved note." in response.text
    assert "bounded" in response.text
    assert "Conversation context" in response.text
    assert "bounded explanation" in response.text
    assert "assistant-v7" in response.text
    assert "json_schema" in response.text
    assert 'class="app-header"' in response.text
    assert 'href="https://example.com/merchant"' in response.text
    assert 'href="javascript:' not in response.text


@pytest.mark.anyio
async def test_audit_lists_cancelled_proactive_delivery(session):
    transaction = Transaction(
        bank="test",
        email_type="transaction",
        direction="debit",
        amount=10,
    )
    session.add(transaction)
    await session.flush()
    decision = CategoryReviewDecision(
        transaction_id=transaction.id,
        category_input_hash="hash",
        candidates_json="[]",
        gate_reason="uncertain",
    )
    session.add(decision)
    await session.flush()
    delivery = TelegramOutboundDelivery(
        category_review_decision_id=decision.id,
        ordinal=0,
        recipient_chat_id=123,
        transaction_id=transaction.id,
        text="review",
        delivery_token="cancelled-audit-token",
        status="cancelled",
    )
    session.add(delivery)
    await session.commit()

    rows = await list_terminal_proactive_deliveries(session)

    assert [row.id for row in rows] == [delivery.id]


@pytest.mark.anyio
async def test_audit_pagination_preserves_all_filters(session, client):
    transaction = Transaction(
        bank="test",
        email_type="transaction",
        direction="debit",
        amount=10,
    )
    session.add(transaction)
    await session.flush()
    for index in range(51):
        interaction = AuditInteraction(
            telegram_update_id=f"audit-pagination-{index}",
            inbound_chat_id=123,
            trigger="reply",
            transaction_id=transaction.id,
            user_text=f"set note {index}",
            status="delivered",
            outcome="mutation",
            assistant_text="saved",
        )
        session.add(interaction)
        await session.flush()
        session.add(
            AuditAction(
                interaction_id=interaction.id,
                transaction_id=transaction.id,
                action_type="set_note",
                target_type="transaction",
                target_id=transaction.id,
                status="applied",
            )
        )
    await session.commit()

    response = await client.get(
        "/audit",
        params={
            "status": "delivered",
            "trigger": "reply",
            "transaction_id": transaction.id,
            "action_type": "set_note",
            "date_from": "2020-01-01",
            "date_to": "2100-01-01",
        },
    )

    assert response.status_code == 200
    match = re.search(r'href="([^"]+)">Next', response.text)
    assert match is not None
    query = parse_qs(urlparse(unescape(match.group(1))).query)
    assert query == {
        "page": ["2"],
        "status": ["delivered"],
        "trigger": ["reply"],
        "transaction_id": [str(transaction.id)],
        "action_type": ["set_note"],
        "date_from": ["2020-01-01"],
        "date_to": ["2100-01-01"],
    }

    second_page = await client.get(f"/audit?{urlparse(unescape(match.group(1))).query}")
    assert second_page.status_code == 200
    assert "set note 0" in second_page.text
    assert "set note 50" not in second_page.text
