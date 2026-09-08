"""Bounded application context supplied to assistant model calls."""

import json
from collections.abc import Mapping, Sequence

from financial_dashboard.services.categorization.normalize import (
    redact_names,
    redact_pii,
)
from financial_dashboard.services.settings import get_redact_name_tokens

MAX_HISTORY_TURNS = 12
MAX_CONTEXT_CHARS = 18_000


def bounded_history(history: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    result = []
    name_tokens = get_redact_name_tokens()
    for turn in history[-MAX_HISTORY_TURNS:]:
        result.append(
            {
                "role": str(turn.get("role", ""))[:30],
                "text": redact_names(
                    redact_pii(str(turn.get("text", ""))), name_tokens
                )[:1500],
            }
        )
    return result


def serialize_tool_result(value: object) -> dict[str, object]:
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        return dict(asdict())
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {
            "items": [
                dict(item._asdict())
                if hasattr(item, "_asdict")
                else dict(item)
                if isinstance(item, Mapping)
                else str(item)[:1000]
                for item in value
            ]
        }
    return {"value": str(value)[:4000]}


def context_size(context: Mapping[str, object]) -> int:
    return len(json.dumps(context, default=str, ensure_ascii=False))
