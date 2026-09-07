import pytest
from pydantic import ValidationError

from financial_dashboard.services.assistant.contracts import parse_response


def test_response_contract_accepts_typed_tool_calls_and_pending_confirmation():
    response = parse_response(
        {
            "outcome": "clarification",
            "question": "Should I create groceries?",
            "pending_confirmation": {
                "kind": "create_category",
                "transaction_id": 7,
                "slug": "groceries",
            },
        }
    )
    assert response.pending_confirmation.slug == "groceries"


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
