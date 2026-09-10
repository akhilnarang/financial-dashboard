"""Persistence primitives for assistant idempotency and audit trails.

These functions deliberately do not commit.  The caller owns the transaction so
that a successful financial mutation, its audit row, and its outbound response
are one durable unit.
"""

import datetime
import secrets
from typing import NamedTuple, cast

from sqlalchemy import and_, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import AuditInteraction, utc_now

PROCESSING_LEASE = datetime.timedelta(minutes=5)


class InteractionClaim(NamedTuple):
    interaction: AuditInteraction
    is_new: bool


class ProcessingClaim(NamedTuple):
    interaction: AuditInteraction | None
    worker_token: str


async def claim_interaction(
    session: AsyncSession,
    *,
    telegram_update_id: str | None,
    chat_id: int,
    message_id: int | None,
    reply_to_message_id: int | None,
    trigger: str,
    user_text: str | None,
    inbound_payload_json: str | None = None,
) -> InteractionClaim:
    """Insert an interaction exactly once and return ``(row, is_new)``.

    Telegram normally supplies an update id.  Deterministic internal callers
    may omit it; those callers cannot use this helper for idempotency.
    """
    if telegram_update_id is None:
        row = AuditInteraction(
            inbound_chat_id=chat_id,
            inbound_message_id=message_id,
            reply_to_message_id=reply_to_message_id,
            trigger=trigger,
            user_text=user_text,
            inbound_payload_json=inbound_payload_json,
        )
        session.add(row)
        await session.flush()
        return InteractionClaim(row, True)

    existing = await session.scalar(
        select(AuditInteraction).where(
            AuditInteraction.telegram_update_id == telegram_update_id
        )
    )
    if existing is not None:
        return InteractionClaim(existing, False)
    try:
        async with session.begin_nested():
            row = AuditInteraction(
                telegram_update_id=telegram_update_id,
                inbound_chat_id=chat_id,
                inbound_message_id=message_id,
                reply_to_message_id=reply_to_message_id,
                trigger=trigger,
                user_text=user_text,
                inbound_payload_json=inbound_payload_json,
                status="claimed",
                claimed_at=utc_now(),
            )
            session.add(row)
            await session.flush()
    except IntegrityError:
        existing = await session.scalar(
            select(AuditInteraction).where(
                AuditInteraction.telegram_update_id == telegram_update_id
            )
        )
        if existing is None:
            raise
        return InteractionClaim(existing, False)
    return InteractionClaim(row, True)


async def claim_processing(
    session: AsyncSession,
    interaction_id: int,
    *,
    lease: datetime.timedelta = PROCESSING_LEASE,
) -> ProcessingClaim:
    """CAS claim processing, reclaiming only an expired lease."""
    token = secrets.token_urlsafe(24)
    now = utc_now()
    until = now + lease
    result = cast(
        CursorResult,
        await session.execute(
            update(AuditInteraction)
            .where(
                AuditInteraction.id == interaction_id,
                or_(
                    AuditInteraction.status == "claimed",
                    and_(
                        AuditInteraction.status == "processing",
                        or_(
                            AuditInteraction.processing_lease_until.is_(None),
                            AuditInteraction.processing_lease_until < now,
                        ),
                    ),
                ),
            )
            .values(
                status="processing",
                worker_token=token,
                processing_lease_until=until,
                processing_at=now,
            )
            .execution_options(synchronize_session="fetch")
        ),
    )
    if result.rowcount != 1:
        return ProcessingClaim(None, token)
    return ProcessingClaim(await session.get(AuditInteraction, interaction_id), token)


async def renew_processing_lease(
    session: AsyncSession,
    interaction_id: int,
    worker_token: str,
    *,
    lease: datetime.timedelta = PROCESSING_LEASE,
) -> bool:
    """Extend a processing lease only while the same worker still owns it."""
    result = cast(
        CursorResult,
        await session.execute(
            update(AuditInteraction)
            .where(
                AuditInteraction.id == interaction_id,
                AuditInteraction.status == "processing",
                AuditInteraction.worker_token == worker_token,
            )
            .values(processing_lease_until=utc_now() + lease)
            .execution_options(synchronize_session="fetch")
        ),
    )
    return result.rowcount == 1


async def finalize_interaction(
    session: AsyncSession,
    interaction_id: int,
    worker_token: str,
    *,
    status: str,
    outcome: str,
    assistant_text: str | None = None,
    model_input_json: str | None = None,
    model_output_json: str | None = None,
    model_explanation: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    output_mode: str | None = None,
    error_code: str | None = None,
    error_detail: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: int | None = None,
) -> bool:
    """Fenced final write; a stolen lease cannot commit stale work."""
    result = cast(
        CursorResult,
        await session.execute(
            update(AuditInteraction)
            .where(
                AuditInteraction.id == interaction_id,
                AuditInteraction.status == "processing",
                AuditInteraction.worker_token == worker_token,
            )
            .values(
                status=status,
                outcome=outcome,
                assistant_text=assistant_text,
                model_input_json=model_input_json,
                model_output_json=model_output_json,
                model_explanation=model_explanation,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                output_mode=output_mode,
                error_code=error_code,
                error_detail=error_detail,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                completed_at=utc_now(),
                processing_lease_until=None,
                worker_token=None,
            )
        ),
    )
    return result.rowcount == 1


async def mark_authorization_changed(
    session: AsyncSession, interaction_id: int
) -> bool:
    """Apply the plan's pre/post-output authorization state transitions."""
    result = cast(
        CursorResult,
        await session.execute(
            update(AuditInteraction)
            .where(
                AuditInteraction.id == interaction_id,
                AuditInteraction.status.in_(["claimed", "processing"]),
            )
            .values(
                status="failed",
                outcome="error",
                error_code="authorization_changed",
                completed_at=utc_now(),
            )
        ),
    )
    if result.rowcount == 1:
        return True
    from financial_dashboard.services.assistant.delivery import (
        cancel_for_authorization_change,
    )

    result = cast(
        CursorResult,
        await session.execute(
            update(AuditInteraction)
            .where(
                AuditInteraction.id == interaction_id,
                AuditInteraction.status.in_(["ready_to_send", "delivery_partial"]),
            )
            .values(
                status="delivery_failed",
                error_code="authorization_changed",
                completed_at=utc_now(),
            )
        ),
    )
    if result.rowcount == 1:
        await cancel_for_authorization_change(session, interaction_id)
    return result.rowcount == 1
