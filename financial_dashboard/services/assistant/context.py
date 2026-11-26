"""Bounded application context supplied to assistant model calls."""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from financial_dashboard.schemas.receivables import ReceivableRead
from financial_dashboard.services.assistant.transaction_reads import (
    AssistantAccount,
    AssistantTransaction,
)
from financial_dashboard.services.categorization.normalize import (
    redact_names,
    redact_pii,
)
from financial_dashboard.services.settings import get_redact_name_tokens

MAX_HISTORY_TURNS = 12
MAX_CONTEXT_CHARS = 18_000


class ToolResult(BaseModel):
    """The result of one read, as the model sees it.

    The prompt quotes it as JSON. The model never sees the database rows.

    Attributes:
        tool: The name of the read tool, such as ``list_transactions``.
        items: The rows that the read returns, already redacted.
        hint: What to do next when the read found nothing, or None.
    """

    model_config = ConfigDict(frozen=True)

    tool: str
    items: list[AssistantTransaction] | list[AssistantAccount] | list[ReceivableRead]
    hint: str | None = None


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
