"""Durable Telegram outbox primitives.

Sending is intentionally outside this module: callers claim a row, invoke the
Telegram API, then record the result.  Retrying therefore reuses the exact
stored payload and never re-runs assistant work.
"""

import datetime
import secrets
from typing import NamedTuple, cast

from sqlalchemy import and_, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    AuditInteraction,
    TelegramOutboundDelivery,
    utc_now,
)

DELIVERY_LEASE = datetime.timedelta(minutes=2)
MAX_DELIVERY_ATTEMPTS = 5


class DeliveryClaim(NamedTuple):
    delivery: TelegramOutboundDelivery | None
    worker_token: str


async def validate_delivery_proof(
    session: AsyncSession,
    delivery_id: int,
    *,
    recipient_chat_id: int | None = None,
    transaction_id: int | None = None,
    interaction_id: int | None = None,
    category_review_decision_id: int | None = None,
) -> bool:
    """Validate the exact outbound row before accepting Telegram proof."""
    delivery = await session.get(TelegramOutboundDelivery, delivery_id)
    if delivery is None:
        return False
    if (
        recipient_chat_id is not None
        and delivery.recipient_chat_id != recipient_chat_id
    ):
        return False
    if transaction_id is not None and delivery.transaction_id != transaction_id:
        return False
    if interaction_id is not None and delivery.interaction_id != interaction_id:
        return False
    if (
        category_review_decision_id is not None
        and delivery.category_review_decision_id != category_review_decision_id
    ):
        return False
    return delivery.status in {
        "pending",
        "delivering",
        "delivery_unknown",
        "abandoned",
        "delivered",
    }


def make_delivery(
    *,
    recipient_chat_id: int,
    text: str,
    ordinal: int,
    interaction_id: int | None = None,
    category_review_decision_id: int | None = None,
    transaction_id: int | None = None,
    parse_mode: str | None = None,
    reply_markup_json: str | None = None,
) -> TelegramOutboundDelivery:
    if (interaction_id is None) == (category_review_decision_id is None):
        raise ValueError("exactly one outbound owner is required")
    return TelegramOutboundDelivery(
        interaction_id=interaction_id,
        category_review_decision_id=category_review_decision_id,
        recipient_chat_id=recipient_chat_id,
        transaction_id=transaction_id,
        ordinal=ordinal,
        text=text,
        parse_mode=parse_mode,
        reply_markup_json=reply_markup_json,
        delivery_token=secrets.token_urlsafe(16),
        status="pending",
    )


async def claim_delivery(
    session: AsyncSession,
    delivery_id: int,
    *,
    lease: datetime.timedelta = DELIVERY_LEASE,
) -> DeliveryClaim:
    """CAS claim pending or stale unknown/delivering delivery work."""
    token = secrets.token_urlsafe(24)
    now = utc_now()
    result = cast(
        CursorResult,
        await session.execute(
            update(TelegramOutboundDelivery)
            .where(
                TelegramOutboundDelivery.id == delivery_id,
                TelegramOutboundDelivery.delivery_attempts < MAX_DELIVERY_ATTEMPTS,
                or_(
                    TelegramOutboundDelivery.status == "pending",
                    and_(
                        TelegramOutboundDelivery.status.in_(
                            ["delivery_unknown", "delivering"]
                        ),
                        or_(
                            TelegramOutboundDelivery.delivery_lease_until.is_(None),
                            TelegramOutboundDelivery.delivery_lease_until < now,
                        ),
                    ),
                ),
            )
            .values(
                status="delivering",
                worker_token=token,
                delivery_lease_until=now + lease,
                delivery_attempts=TelegramOutboundDelivery.delivery_attempts + 1,
                last_attempt_at=now,
            )
            .execution_options(synchronize_session="fetch")
        ),
    )
    if result.rowcount != 1:
        return DeliveryClaim(None, token)
    return DeliveryClaim(
        await session.get(TelegramOutboundDelivery, delivery_id), token
    )


async def mark_delivery_delivered(
    session: AsyncSession,
    delivery_id: int,
    worker_token: str,
) -> bool:
    result = cast(
        CursorResult,
        await session.execute(
            update(TelegramOutboundDelivery)
            .where(
                TelegramOutboundDelivery.id == delivery_id,
                TelegramOutboundDelivery.status == "delivering",
                TelegramOutboundDelivery.worker_token == worker_token,
            )
            .values(
                status="delivered",
                worker_token=None,
                delivery_lease_until=None,
                delivered_at=utc_now(),
            )
        ),
    )
    return result.rowcount == 1


async def settle_delivery_from_callback(
    session: AsyncSession,
    delivery_id: int,
) -> bool:
    """Settle a delivery when a callback proves Telegram rendered it."""
    await session.execute(
        update(TelegramOutboundDelivery)
        .where(
            TelegramOutboundDelivery.id == delivery_id,
            # A callback is proof that Telegram rendered the message even if
            # recovery already moved the row to ``abandoned``.  Keep the CAS
            # so a cancelled row (or a different delivery) cannot be revived.
            TelegramOutboundDelivery.status.in_(
                ["pending", "delivering", "delivery_unknown", "abandoned"]
            ),
        )
        .values(
            status="delivered",
            worker_token=None,
            delivery_lease_until=None,
            delivered_at=utc_now(),
        )
        .execution_options(synchronize_session="fetch")
    )
    delivery = await session.get(TelegramOutboundDelivery, delivery_id)
    settled = delivery is not None and delivery.status == "delivered"
    if settled and delivery.interaction_id is not None:
        await refresh_interaction_delivery_status(session, delivery.interaction_id)
    return settled


async def mark_delivery_unknown(
    session: AsyncSession,
    delivery_id: int,
    worker_token: str,
    error: str | None = None,
) -> bool:
    result = cast(
        CursorResult,
        await session.execute(
            update(TelegramOutboundDelivery)
            .where(
                TelegramOutboundDelivery.id == delivery_id,
                TelegramOutboundDelivery.status == "delivering",
                TelegramOutboundDelivery.worker_token == worker_token,
            )
            .values(
                status="delivery_unknown",
                worker_token=None,
                delivery_lease_until=None,
            )
        ),
    )
    return result.rowcount == 1


async def recover_assistant_work(session: AsyncSession) -> None:
    """Requeue stale processing/delivery leases during an existing cycle."""
    now = utc_now()
    await session.execute(
        update(AuditInteraction)
        .where(
            AuditInteraction.status == "processing",
            AuditInteraction.processing_lease_until < now,
        )
        .values(status="claimed", worker_token=None, processing_lease_until=None)
        .execution_options(synchronize_session="fetch")
    )
    await session.execute(
        update(TelegramOutboundDelivery)
        .where(
            TelegramOutboundDelivery.status == "delivering",
            TelegramOutboundDelivery.delivery_lease_until < now,
        )
        .values(status="delivery_unknown", worker_token=None, delivery_lease_until=None)
        .execution_options(synchronize_session="fetch")
    )
    await abandon_exhausted(session)


async def cancel_for_authorization_change(
    session: AsyncSession, interaction_id: int
) -> int:
    result = cast(
        CursorResult,
        await session.execute(
            update(TelegramOutboundDelivery)
            .where(
                TelegramOutboundDelivery.interaction_id == interaction_id,
                TelegramOutboundDelivery.status.in_(
                    ["pending", "delivering", "delivery_unknown"]
                ),
            )
            .values(status="cancelled", worker_token=None, delivery_lease_until=None)
        ),
    )
    return result.rowcount


async def abandon_exhausted(session: AsyncSession) -> int:
    exhausted_owner_ids = list(
        (
            await session.scalars(
                select(TelegramOutboundDelivery.interaction_id).where(
                    TelegramOutboundDelivery.status.in_(
                        ["pending", "delivery_unknown"]
                    ),
                    TelegramOutboundDelivery.delivery_attempts >= MAX_DELIVERY_ATTEMPTS,
                    TelegramOutboundDelivery.interaction_id.is_not(None),
                )
            )
        ).all()
    )
    result = cast(
        CursorResult,
        await session.execute(
            update(TelegramOutboundDelivery)
            .where(
                TelegramOutboundDelivery.status.in_(["pending", "delivery_unknown"]),
                TelegramOutboundDelivery.delivery_attempts >= MAX_DELIVERY_ATTEMPTS,
            )
            .values(status="abandoned")
        ),
    )
    if exhausted_owner_ids:
        await session.execute(
            update(AuditInteraction)
            .where(
                AuditInteraction.id.in_(exhausted_owner_ids),
                AuditInteraction.status.in_(["ready_to_send", "delivery_partial"]),
            )
            .values(status="delivery_failed", completed_at=utc_now())
        )
    return result.rowcount


async def refresh_interaction_delivery_status(
    session: AsyncSession, interaction_id: int
) -> str | None:
    rows = (
        await session.scalars(
            select(TelegramOutboundDelivery).where(
                TelegramOutboundDelivery.interaction_id == interaction_id
            )
        )
    ).all()
    if not rows:
        return None
    if all(row.status == "delivered" for row in rows):
        status = "delivered"
    elif any(row.status in {"abandoned", "cancelled"} for row in rows):
        status = "delivery_failed"
    elif any(row.status in {"delivering", "delivery_unknown"} for row in rows):
        status = "delivery_partial"
    else:
        status = "ready_to_send"
    await session.execute(
        update(AuditInteraction)
        .where(AuditInteraction.id == interaction_id)
        .values(
            status=status,
            completed_at=(
                utc_now() if status in {"delivered", "delivery_failed"} else None
            ),
        )
    )
    return status
