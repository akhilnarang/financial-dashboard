"""Shared lifecycle operations for active category-review decisions."""

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    CategoryReviewDecision,
    TelegramOutboundDelivery,
    utc_now,
)


async def supersede_active_decisions(
    session: AsyncSession,
    transaction_id: int,
    *,
    preserve_decision_id: int | None = None,
) -> None:
    """Expire active decisions and cancel every undelivered message they own."""
    active_ids = select(CategoryReviewDecision.id).where(
        CategoryReviewDecision.transaction_id == transaction_id,
        CategoryReviewDecision.status == "active",
    )
    source_interaction_ids_stmt = select(
        CategoryReviewDecision.source_interaction_id
    ).where(
        CategoryReviewDecision.transaction_id == transaction_id,
        CategoryReviewDecision.status == "active",
        CategoryReviewDecision.source_interaction_id.is_not(None),
    )
    if preserve_decision_id is not None:
        active_ids = active_ids.where(CategoryReviewDecision.id != preserve_decision_id)
        source_interaction_ids_stmt = source_interaction_ids_stmt.where(
            CategoryReviewDecision.id != preserve_decision_id
        )
    source_interaction_ids = [
        interaction_id
        for interaction_id in (await session.scalars(source_interaction_ids_stmt)).all()
        if interaction_id is not None
    ]
    await session.execute(
        update(TelegramOutboundDelivery)
        .where(
            (TelegramOutboundDelivery.category_review_decision_id.in_(active_ids))
            | (
                TelegramOutboundDelivery.interaction_id.in_(source_interaction_ids)
                & (TelegramOutboundDelivery.transaction_id == transaction_id)
            ),
            TelegramOutboundDelivery.status.in_(
                ["pending", "delivering", "delivery_unknown"]
            ),
        )
        .values(status="cancelled", worker_token=None, delivery_lease_until=None)
    )
    statement = update(CategoryReviewDecision).where(
        CategoryReviewDecision.transaction_id == transaction_id,
        CategoryReviewDecision.status == "active",
    )
    if preserve_decision_id is not None:
        statement = statement.where(CategoryReviewDecision.id != preserve_decision_id)
    await session.execute(
        statement.values(status="superseded", superseded_at=utc_now())
    )
    if source_interaction_ids:
        from financial_dashboard.services.assistant.delivery import (
            refresh_interaction_delivery_status,
        )

        for interaction_id in source_interaction_ids:
            await refresh_interaction_delivery_status(session, interaction_id)
