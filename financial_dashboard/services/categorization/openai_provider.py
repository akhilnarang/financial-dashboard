"""OpenAI-compatible structured-output classifier (LLM fallback).

Reuses build_prompt and parse_result from gemini.py — the prompt format and
result parsing are provider-agnostic.
"""

import json
from collections.abc import Mapping, Sequence

from openai import AsyncOpenAI

from financial_dashboard.services.categorization.fewshot import FewShotExample
from financial_dashboard.services.categorization.gemini import (
    GeminiResult,
    build_prompt,
    parse_result,
)


async def classify(
    *,
    fields: Mapping[str, str | None],
    examples: Sequence[FewShotExample],
    active_slugs: list[str],
    api_key: str,
    model: str,
    base_url: str,
    name_tokens: Sequence[str] = (),
) -> GeminiResult:
    prompt = build_prompt(
        fields=fields,
        examples=examples,
        active_slugs=active_slugs,
        name_tokens=name_tokens,
    )
    # 30s cap so a slow provider can't stall the poll loop once enabled.
    client = AsyncOpenAI(api_key=api_key, base_url=base_url or None, timeout=30.0)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = response.choices[0].message.content
    data = json.loads(content or "{}")
    return parse_result(data, active_slugs)
