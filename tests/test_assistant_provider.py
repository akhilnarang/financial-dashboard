import json

import pytest
from pydantic import ValidationError

from financial_dashboard.services.assistant.contracts import parse_response
from financial_dashboard.services.assistant.context import ToolResult
from financial_dashboard.services.assistant.packs import pack_response_schema
from financial_dashboard.services.assistant.prompt import PromptContext
from financial_dashboard.services.assistant.transaction_reads import (
    AssistantAccount,
    AssistantCard,
)
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
    accounts = [
        AssistantAccount(
            id=index,
            bank="samplebank",
            label="synthetic account context " * 6,
            type="bank_account",
            active=True,
            # A null field in a row stays in what the model sees.
            cards=[AssistantCard(id=index, label=None, active=True, is_primary=True)],
        )
        for index in range(20)
    ]
    result = await provider.complete(
        PromptContext(
            "show accounts",
            tool_results=[ToolResult(tool="list_accounts", items=accounts)],
            packs=frozenset({"reports"}),
        )
    )
    assert result.output_mode == "validated_json_object"
    schema = calls[0]["config"].response_json_schema
    assert schema == pack_response_schema(frozenset({"reports"}))
    # The schema holds the tools of the loaded packs only.
    assert "get_networth" in json.dumps(schema)
    assert "create_receivable" not in json.dumps(schema)
    assert calls[0]["config"].response_schema is None
    assert calls[1]["config"].response_json_schema is None
    supplied_results = calls[1]["contents"].split(
        "=== TOOL RESULTS (quoted data) ===\n", 1
    )[1]
    assert json.loads(supplied_results) == {
        "tool": "list_accounts",
        "items": [account.model_dump(mode="json") for account in accounts],
    }


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
