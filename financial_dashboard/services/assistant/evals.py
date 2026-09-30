"""Read-only JSONL projection of audited assistant turns for offline evals."""

import json
from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import AuditAction, AuditInteraction


async def iter_audit_eval_rows(
    session: AsyncSession,
) -> AsyncIterator[dict[str, object]]:
    """Yield stable, attachment-free evaluation records oldest first."""
    interactions = (
        (await session.execute(select(AuditInteraction).order_by(AuditInteraction.id)))
        .scalars()
        .all()
    )
    for interaction in interactions:
        actions = (
            (
                await session.execute(
                    select(AuditAction)
                    .where(AuditAction.interaction_id == interaction.id)
                    .order_by(AuditAction.id)
                )
            )
            .scalars()
            .all()
        )
        yield {
            "interaction_id": interaction.id,
            "conversation_id": interaction.conversation_id,
            "transaction_id": interaction.transaction_id,
            "trigger": interaction.trigger,
            "user_text": interaction.user_text,
            "model_input": _json_value(interaction.model_input_json),
            "model_output": _json_value(interaction.model_output_json),
            "model_explanation": interaction.model_explanation,
            "provider": interaction.provider,
            "model": interaction.model,
            "prompt_version": interaction.prompt_version,
            "output_mode": interaction.output_mode,
            "assistant_text": interaction.assistant_text,
            "outcome": interaction.outcome,
            "status": interaction.status,
            "error_code": interaction.error_code,
            "actions": [
                {
                    "id": action.id,
                    "type": action.action_type,
                    "target_type": action.target_type,
                    "target_id": action.target_id,
                    "arguments": _json_value(action.arguments_json),
                    "before": _json_value(action.before_json),
                    "after": _json_value(action.after_json),
                    "status": action.status,
                    "undo_status": action.undo_status,
                    "undone_by_interaction_id": action.undone_by_interaction_id,
                }
                for action in actions
            ],
        }


def _json_value(raw: str | None) -> object:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
