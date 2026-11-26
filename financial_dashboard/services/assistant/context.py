"""Bounded application context supplied to assistant model calls."""

import json
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from pydantic import BaseModel

from financial_dashboard.services.assistant.transaction_reads import (
    AssistantTransaction,
)
from financial_dashboard.services.categorization.normalize import (
    redact_names,
    redact_pii,
)
from financial_dashboard.services.settings import get_redact_name_tokens

MAX_HISTORY_TURNS = 12
MAX_CONTEXT_CHARS = 18_000


class ToolResult(NamedTuple):
    """The result of one read, as the model sees it."""

    tool: str
    items: list[dict[str, object]]
    hint: str | None = None

    def as_json(self) -> dict[str, object]:
        """Return the JSON object that the prompt quotes for this result."""
        result: dict[str, object] = {"items": self.items}
        if self.hint:
            result["hint"] = self.hint
        return {"tool": self.tool, "result": result}


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


def tool_items(
    rows: Sequence[AssistantTransaction] | Sequence[BaseModel],
) -> list[dict[str, object]]:
    """Return the rows of a read as the JSON objects that the model sees.

    Args:
        rows: Transaction rows, or the accounts or receivables of a read.

    Returns:
        One object per row.
    """
    return [
        row._asdict()
        if isinstance(row, AssistantTransaction)
        else row.model_dump(mode="json")
        for row in rows
    ]


def context_size(context: Mapping[str, object]) -> int:
    return len(json.dumps(context, default=str, ensure_ascii=False))
