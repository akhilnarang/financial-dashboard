"""Structured provider adapter for the conversational assistant.

OpenAI-compatible providers are attempted with ``json_schema`` first.  Only
an explicit unsupported-schema response falls back to ``json_object``; the
same strict Pydantic validation is then applied.  There is no prose or regex
recovery path.
"""

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol, cast

from openai import AsyncOpenAI, omit
from google import genai
from google.genai import types

from financial_dashboard.services.assistant.contracts import (
    AssistantResponse,
    parse_response,
    response_json_schema,
)
from financial_dashboard.services.assistant.prompt import (
    PROMPT_VERSION,
    PromptContext,
    build_prompt,
)


class ProviderFailure(RuntimeError):
    """Provider call or strict response validation failed."""


class StructuredResult(NamedTuple):
    response: AssistantResponse
    prompt: str
    output_mode: str
    provider: str
    model: str
    latency_ms: int
    raw_payload: Mapping[str, object]
    prompt_version: str = PROMPT_VERSION
    input_tokens: int | None = None
    output_tokens: int | None = None


class DecodedResponse(NamedTuple):
    response: AssistantResponse
    payload: Mapping[str, object]


class UsageResult(NamedTuple):
    input_tokens: int | None
    output_tokens: int | None


class AssistantProvider(Protocol):
    async def complete(self, context: PromptContext) -> StructuredResult: ...


def _openai_content(response: object) -> str | None:
    """Extract model text while converting malformed SDK responses to our error."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        raise ProviderFailure("provider returned no assistant choices")
    if not choices:
        raise ProviderFailure("provider returned an empty assistant response")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ProviderFailure("provider returned a choice without a message")
    content = getattr(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise ProviderFailure("provider returned non-text assistant content")
    return content


def _gemini_content(response: object) -> str | None:
    content = getattr(response, "text", None)
    if content is not None and not isinstance(content, str):
        raise ProviderFailure("provider returned non-text assistant content")
    return content


def _unsupported_schema(exc: Exception) -> bool:
    text = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status is not None and status not in (400, 404, 422):
        return False
    names_schema_feature = any(
        token in text
        for token in ("json_schema", "response_schema", "structured output")
    )
    explicitly_rejects_it = any(
        token in text
        for token in ("unsupported", "not support", "unknown", "unrecognized")
    )
    return bool(names_schema_feature and explicitly_rejects_it)


def _decode_response(
    content: str | None,
) -> DecodedResponse:
    try:
        payload = json.loads(content or "")
        if isinstance(payload, dict) and set(payload) == {"response"}:
            payload = payload["response"]
        parsed = parse_response(payload)
    except (ValueError, TypeError) as exc:
        raise ProviderFailure("provider returned invalid assistant JSON") from exc
    if not isinstance(payload, Mapping):
        raise ProviderFailure("provider returned a non-object assistant response")
    return DecodedResponse(parsed, payload)


def _usage(response: object) -> UsageResult:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageResult(None, None)
    return UsageResult(
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        reasoning_effort: str = "",
    ) -> None:
        self.model = model
        self.reasoning_effort = (
            reasoning_effort if reasoning_effort in {"low", "medium", "high"} else ""
        )
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url or None, timeout=30.0
        )

    async def complete(self, context: PromptContext) -> StructuredResult:
        prompt = build_prompt(context)
        started = time.monotonic()
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "assistant_response",
                    "strict": True,
                    "schema": response_json_schema(),
                },
            },
            "temperature": omit if self.reasoning_effort else 0.0,
            "reasoning_effort": self.reasoning_effort or omit,
        }
        output_mode = "json_schema"
        try:
            response = await self.client.chat.completions.create(**cast(Any, kwargs))
        except Exception as exc:
            if not _unsupported_schema(exc):
                raise ProviderFailure("structured provider request failed") from exc
            output_mode = "validated_json_object"
            kwargs["response_format"] = {"type": "json_object"}
            try:
                response = await self.client.chat.completions.create(
                    **cast(Any, kwargs)
                )
            except Exception as fallback_exc:
                raise ProviderFailure(
                    "provider does not support a usable JSON response mode"
                ) from fallback_exc
        content = _openai_content(response)
        parsed, payload = _decode_response(content)
        input_tokens, output_tokens = _usage(response)
        return StructuredResult(
            parsed,
            prompt,
            output_mode,
            "openai",
            self.model,
            int((time.monotonic() - started) * 1000),
            payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class GeminiProvider:
    """Gemini transport using its native JSON schema response mode."""

    def __init__(self, *, api_key: str, model: str) -> None:
        self.model = model
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=30_000)
        )

    async def complete(self, context: PromptContext) -> StructuredResult:
        prompt = build_prompt(context)
        started = time.monotonic()
        output_mode = "json_schema"
        config: dict[str, object] = {
            "response_mime_type": "application/json",
            # The native JSON Schema field accepts the full Pydantic schema.
            # ``response_schema`` first coerces through Gemini's reduced Schema
            # model and rejects valid JSON Schema keywords before HTTP dispatch.
            "response_json_schema": response_json_schema(),
            "temperature": 0.0,
        }
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**cast(Any, config)),
            )
        except Exception as exc:
            if not _unsupported_schema(exc):
                raise ProviderFailure("structured Gemini request failed") from exc
            output_mode = "validated_json_object"
            config.pop("response_json_schema", None)
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cast(Any, config)),
                )
            except Exception as fallback_exc:
                raise ProviderFailure(
                    "Gemini does not support a usable JSON response mode"
                ) from fallback_exc
        parsed, payload = _decode_response(_gemini_content(response))
        input_tokens, output_tokens = _usage(response)
        return StructuredResult(
            parsed,
            prompt,
            output_mode,
            "gemini",
            self.model,
            int((time.monotonic() - started) * 1000),
            payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def provider_from_settings(
    *,
    provider: str,
    api_key: str,
    model: str,
    base_url: str = "",
    reasoning_effort: str = "",
) -> AssistantProvider:
    """Construct the assistant adapter from existing categorization settings."""
    if provider not in {"gemini", "openai"}:
        raise ProviderFailure(f"unsupported assistant provider: {provider}")
    if not api_key.strip():
        raise ProviderFailure(f"{provider} API key is missing")
    try:
        if provider == "gemini":
            return GeminiProvider(api_key=api_key, model=model)
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        raise ProviderFailure(f"failed to configure {provider} provider") from exc
