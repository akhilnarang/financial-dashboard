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
        model_input_json=json.dumps(
            [
                "Prompt intro\n=== TOOL RESULTS (quoted data) ===\n"
                + json.dumps(
                    {
                        "tool": "list_transactions",
                        "result": '{"items": [{"id": 42, "merchant": "TEST MERCHANT"}]}',
                    }
                )
            ]
        ),
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
        error_code="screenshot_unreadable",
        error_detail='Traceback (most recent call last):\n  File "test.py", line 1\nValueError: bad screenshot',
        conversation_id=conversation.id,
    )
    session.add(interaction)
    await session.flush()
    session.add(
        AuditAction(
            interaction_id=interaction.id,
            transaction_id=interaction.transaction_id,
            action_type="set_note",
            target_type="transaction",
            target_id=999,
            arguments_json=json.dumps({"note": "TEST NOTE"}),
            before_json="not json: raw evidence",
            after_json=json.dumps({"note": "TEST NOTE"}),
            status="applied",
        )
    )
    await session.commit()

    list_response = await client.get("/audit")
    assert list_response.status_code == 200
    assert f'href="/audit/{interaction.id}"' in list_response.text
    assert "set note to lunch" in list_response.text

    response = await client.get(f"/audit/{interaction.id}")

    assert response.status_code == 200
    assert "set note to lunch" in response.text
    assert "Saved note." in response.text
    assert "bounded" in response.text
    assert "Conversation context" in response.text
    assert "bounded explanation" in response.text
    assert "assistant-v7" in response.text
    assert "json_schema" in response.text
    assert "screenshot_unreadable" in response.text
    assert "Traceback (most recent call last):" in response.text
    assert "ValueError: bad screenshot" in response.text
    assert " ms · ?/?" not in response.text
    assert 'href="https://example.com/merchant"' in response.text
    assert 'href="javascript:' not in response.text

    model_input = re.search(
        r'<h3 class="audit-section-title">Model input</h3>\s*'
        r'<pre class="audit-json" data-json-viewer data-compact="([^"]*)" '
        r'data-tree="([^"]*)">(.*?)</pre>',
        response.text,
        re.DOTALL,
    )
    assert model_input is not None
    stored = json.loads(interaction.model_input_json)
    assert json.loads(unescape(model_input.group(1))) == stored
    assert json.loads(unescape(model_input.group(3))) == stored
    parsed = json.loads(unescape(model_input.group(2)))
    assert "=== TOOL RESULTS (quoted data) ===" in parsed[0][0]
    assert parsed[0][1]["result"]["items"][0]["id"] == 42
    assert parsed[0][1]["result"]["items"][0]["merchant"] == "TEST MERCHANT"
    assert "/static/js/json-viewer.js" in response.text
    assert '<pre class="audit-json">not json: raw evidence</pre>' in response.text


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
