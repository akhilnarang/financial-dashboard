import pytest
from pydantic import ValidationError

from financial_dashboard.services.assistant.contracts import parse_response


def test_response_contract_accepts_typed_tool_calls_and_pending_confirmation():
    response = parse_response(
        {
            "outcome": "clarification",
            "question": "Should I create this merchant rule?",
            "pending_confirmation": {
                "kind": "merchant_rule",
                "transaction_id": 7,
                "category": "groceries",
            },
        }
    )
    assert response.pending_confirmation.category == "groceries"


def test_extra_fields_and_unknown_tools_fail_closed():
    with pytest.raises(ValidationError):
        parse_response({"outcome": "answer", "text": "ok", "tool": "drop table"})
    with pytest.raises(ValidationError):
        parse_response(
            {
                "outcome": "tool_calls",
                "calls": [{"name": "execute_sql", "sql": "select 1"}],
            }
        )
    with pytest.raises(ValidationError):
        parse_response(
            {
                "outcome": "tool_calls",
                "calls": [
                    {
                        "name": "apply_transaction_changes",
                        "transaction_id": 1,
                        "changes": {"category": {"op": "set", "value": "new"}},
                        "create_category": {"slug": "new", "intent_evidence": "new"},
                    }
                ],
            }
        )


def test_category_proposal_requires_two_or_three_unique_candidates():
    with pytest.raises(ValidationError):
        parse_response(
            {
                "outcome": "category_proposal",
                "transaction_id": 1,
                "explanation": "uncertain",
                "candidates": [{"slug": "groceries", "reason": "a", "confidence": 0.5}],
            }
        )
