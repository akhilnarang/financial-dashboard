"""Provider-neutral conversational transaction assistant."""

from financial_dashboard.services.assistant.contracts import (
    AssistantResponse,
    parse_response,
)

__all__ = ["AssistantResponse", "parse_response"]
