"""Google Gemini structured-output classifier (LLM fallback).

Transport only. The prompt, the result type, and the parser live in llm.py.
"""

import json
from collections.abc import Mapping, Sequence

from google import genai
from google.genai import types

from financial_dashboard.services.categorization.fewshot import FewShotExample
from financial_dashboard.services.categorization.llm import (
    LLM_TIMEOUT_MS,
    LlmResult,
    build_prompt,
    parse_result,
)

MODEL_DEFAULT = "gemini-2.5-flash"

_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "candidates": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["category", "confidence"],
            },
        },
    },
    "required": ["category", "confidence", "reason"],
}


async def classify(
    *,
    fields: Mapping[str, str | None],
    examples: Sequence[FewShotExample],
    active_slugs: list[str],
    api_key: str,
    model: str,
    name_tokens: Sequence[str] = (),
) -> LlmResult:
    prompt = build_prompt(
        fields=fields,
        examples=examples,
        active_slugs=active_slugs,
        name_tokens=name_tokens,
    )
    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=LLM_TIMEOUT_MS)
    )
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.0,
        ),
    )
    data = json.loads(response.text or "{}")
    return parse_result(data, active_slugs)
