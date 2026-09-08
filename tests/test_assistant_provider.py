from financial_dashboard.services.assistant.contracts import response_json_schema
import pytest

from financial_dashboard.services.assistant.provider import (
    GeminiProvider,
    ProviderFailure,
    provider_from_settings,
)
from financial_dashboard.services.assistant.prompt import PromptContext


def test_provider_schema_has_object_root_for_strict_openai_mode():
    schema = response_json_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["response"]
    assert schema["additionalProperties"] is False


def test_provider_configuration_rejects_a_missing_key():
    with pytest.raises(ProviderFailure):
        provider_from_settings(provider="openai", api_key="", model="test")


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
    result = await provider.complete(PromptContext("why"))
    assert result.output_mode == "validated_json_object"
    assert calls[0]["config"].response_json_schema == response_json_schema()
    assert calls[0]["config"].response_schema is None
    assert calls[1]["config"].response_json_schema is None


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
