import pytest
from pydantic import ValidationError

from financial_dashboard.services.assistant.contracts import parse_response


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


def test_category_proposal_requires_unique_candidates():
    candidate = {"slug": "groceries", "reason": "a", "confidence": 0.5}
    with pytest.raises(ValidationError):
        parse_response(
            {
                "outcome": "category_proposal",
                "transaction_id": 1,
                "explanation": "uncertain",
                "candidates": [candidate, candidate],
            }
        )
