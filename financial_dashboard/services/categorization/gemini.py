"""Google Gemini structured-output classifier (LLM fallback)."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from google import genai
from google.genai import types

from financial_dashboard.services.categorization.fewshot import FewShotExample
from financial_dashboard.services.categorization.normalize import (
    normalize_text,
    redact_names,
    redact_pii,
)

MODEL_DEFAULT = "gemini-2.5-flash"
NEEDS_REVIEW = "needs_review"
LLM_TIMEOUT_MS = 30_000  # cap per-call latency so a slow provider can't stall the poll loop

_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["category", "confidence", "reason"],
}


class GeminiResult(NamedTuple):
    slug: str
    confidence: float
    reason: str


def _sanitize_counterparty(text: str | None, name_tokens: Sequence[str]) -> str:
    """Counterparty: strip numbers (PII) then mask configured name tokens."""
    return redact_names(redact_pii(text), name_tokens)


def _sanitize_description(text: str | None, name_tokens: Sequence[str]) -> str:
    """Description: strip numbers, mask names, then normalize whitespace/case."""
    return normalize_text(redact_names(redact_pii(text), name_tokens))


def build_prompt(
    *,
    fields: Mapping[str, str | None],
    examples: Sequence[FewShotExample],
    active_slugs: list[str],
    name_tokens: Sequence[str] = (),
) -> str:
    lines = [
        "You are a personal-finance transaction categorizer.",
        "Choose exactly ONE category slug from this list:",
        ", ".join(active_slugs),
        f'If none fit, return "{NEEDS_REVIEW}".',
        "Return JSON: {category, confidence (0..1), reason (one short sentence)}.",
        "",
        "IMPORTANT: 'direction: credit' = money RECEIVED — use an income category "
        "(refund, salary, interest, cashback_rewards, repayment, other_income); "
        "NEVER a spending category.",
        "'direction: debit' = money SPENT — use a spending category.",
        "A credit from an individual paying you back = repayment; "
        "a credit from a merchant = refund.",
        "Do NOT use self_transfer (handled separately). For money moved to/from another "
        "person, use 'repayment' for a credit or 'expense'/the specific spending category "
        "for a debit.",
        "",
    ]
    if examples:
        lines.append("Examples of previously categorized transactions:")
        for ex in examples:
            lines.append(
                f"- [{ex.direction}] {_sanitize_counterparty(ex.counterparty, name_tokens)} "
                f"| {_sanitize_description(ex.raw_description, name_tokens)} -> {ex.category}"
            )
        lines.append("")
    lines.append("Transaction to categorize:")
    lines.append(f"direction: {fields.get('direction')}")
    lines.append(f"amount: {fields.get('amount')} {fields.get('currency')}")
    lines.append(f"channel: {fields.get('channel')}")
    lines.append(
        f"counterparty: {_sanitize_counterparty(fields.get('counterparty'), name_tokens)}"
    )
    lines.append(
        f"description: {_sanitize_description(fields.get('raw_description'), name_tokens)}"
    )
    return "\n".join(lines)


def parse_result(data: Mapping[str, Any], active_slugs: list[str]) -> GeminiResult:
    slug = str(data.get("category", "")).strip()
    try:
        conf = float(data.get("confidence", 0.0))
    except ValueError, TypeError:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reason = str(data.get("reason", ""))[:300]
    if slug != NEEDS_REVIEW and slug not in active_slugs:
        return GeminiResult(NEEDS_REVIEW, conf, reason or "model returned unknown slug")
    return GeminiResult(slug, conf, reason)


async def classify(
    *,
    fields: Mapping[str, str | None],
    examples: Sequence[FewShotExample],
    active_slugs: list[str],
    api_key: str,
    model: str,
    name_tokens: Sequence[str] = (),
) -> GeminiResult:
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
