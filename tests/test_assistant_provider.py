import json

import pytest
from pydantic import ValidationError

from financial_dashboard.services.assistant.contracts import (
    parse_response,
    response_json_schema,
)
from financial_dashboard.services.assistant.prompt import PromptContext
from financial_dashboard.services.assistant.provider import (
    GeminiProvider,
    ProviderFailure,
)


def test_invalid_model_output_fails_closed():
    duplicate = {"slug": "groceries", "reason": "a", "confidence": 0.5}
    for payload in (
        {"outcome": "answer", "text": "ok", "tool": "drop table"},
        {
            "outcome": "tool_calls",
            "calls": [{"name": "execute_sql", "sql": "select 1"}],
        },
        {
            "outcome": "category_proposal",
            "transaction_id": 1,
            "explanation": "uncertain",
            "candidates": [duplicate, duplicate],
        },
    ):
        with pytest.raises(ValidationError):
            parse_response(payload)


@pytest.mark.anyio
async def test_gemini_attempts_full_schema_then_records_json_fallback():
    calls = []

    class Models:
        async def generate_content(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("response_schema unsupported by this model")
            return type(
                "Response",
                (),
                {"text": '{"response":{"outcome":"answer","text":"ok"}}'},
            )()

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.model = "test-gemini"
    provider.client = type(
        "Client", (), {"aio": type("Aio", (), {"models": Models()})()}
    )()
    tool_result = {
        "tool": "list_transactions",
        "result": {
            "items": [
                {"id": index, "note": "synthetic transaction context " * 6}
                for index in range(20)
            ]
        },
    }
    result = await provider.complete(
        PromptContext("show transactions", tool_results=[tool_result])
    )
    assert result.output_mode == "validated_json_object"
    assert calls[0]["config"].response_json_schema == response_json_schema()
    assert calls[0]["config"].response_schema is None
    assert calls[1]["config"].response_json_schema is None
    supplied_results = calls[1]["contents"].split(
        "=== TOOL RESULTS (quoted data) ===\n", 1
    )[1]
    assert json.loads(supplied_results) == tool_result


@pytest.mark.anyio
async def test_gemini_complete_rejects_malformed_response_shape():
    class Models:
        async def generate_content(self, **kwargs):
            return type("Response", (), {"text": 42})()

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.model = "test-gemini"
    provider.client = type(
        "Client", (), {"aio": type("Aio", (), {"models": Models()})()}
    )()
    with pytest.raises(ProviderFailure):
        await provider.complete(PromptContext("why"))
