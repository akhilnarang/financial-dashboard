"""Durable category proposals and race-safe Telegram choice handling."""

import datetime
import json
from collections.abc import Iterable
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    AuditAction,
    CategoryReviewDecision,
    Account,
    Transaction,
    TelegramOutboundDelivery,
    as_utc,
    utc_now,
)
from financial_dashboard.services.categorization.llm import LlmCandidate, LlmResult
from financial_dashboard.services.categorization.manual import assign_category_no_commit
from financial_dashboard.services.categorization.manual import CategoryDirectionPolicy
from financial_dashboard.services.assistant.delivery import make_delivery
from financial_dashboard.services.categorization.hashing import (
    build_input_payload,
    compute_input_hash,
)
from financial_dashboard.services.categorization.decision_lifecycle import (
    supersede_active_decisions,
)

DECISION_TTL = datetime.timedelta(days=7)


async def ensure_decision_delivery(
    session: AsyncSession,
    decision: CategoryReviewDecision,
    *,
    recipient_chat_id: int,
    transaction_id: int,
    text: str,
    reply_markup_json: str | None,
    ordinal: int = 0,
    parse_mode: str | None = "HTML",
) -> TelegramOutboundDelivery:
    """Create the proactive decision-owned outbox row before Telegram send."""
    if decision.source_interaction_id is not None:
        raise ValueError("only background decisions may own proactive delivery")
    existing = await session.scalar(
        select(TelegramOutboundDelivery)
        .where(
            TelegramOutboundDelivery.category_review_decision_id == decision.id,
            TelegramOutboundDelivery.ordinal == ordinal,
        )
        .limit(1)
    )
    if existing is not None:
        return existing
    delivery = make_delivery(
        recipient_chat_id=recipient_chat_id,
        text=text,
        ordinal=ordinal,
        category_review_decision_id=decision.id,
        transaction_id=transaction_id,
        parse_mode=parse_mode,
        reply_markup_json=reply_markup_json,
    )
    session.add(delivery)
    await session.flush()
    return delivery


def candidates_from_result(result: LlmResult) -> list[dict[str, object]]:
    """Return at most three validated candidate records for persistence."""
    candidates: Iterable[LlmCandidate] = result.candidates
    if not result.candidates and result.slug != "needs_review":
        candidates = (LlmCandidate(result.slug, result.confidence),)
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.slug in seen or not candidate.slug:
            continue
        seen.add(candidate.slug)
        output.append({"category": candidate.slug, "confidence": candidate.confidence})
        if len(output) == 3:
            break
    return output


async def create_or_reuse_decision(
    session: AsyncSession,
    txn: Transaction,
    *,
    candidates: list[dict[str, object]],
    gate_reason: str | None,
    proposed_slug: str | None = None,
    confidence: float | None = None,
    threshold: float | None = None,
    source_interaction_id: int | None = None,
) -> CategoryReviewDecision:
    """Reuse a fresh proposal or supersede it and create a new one.

    This deliberately does not commit.  Categorization and its review decision
    therefore become visible together in the caller's transaction.
    """
    now = utc_now()
    encoded = json.dumps(candidates, separators=(",", ":"), sort_keys=True)
    existing = await session.scalar(
        select(CategoryReviewDecision)
        .where(
            CategoryReviewDecision.transaction_id == txn.id,
            CategoryReviewDecision.status == "active",
        )
        .order_by(CategoryReviewDecision.id.desc())
        .limit(1)
    )
    if (
        existing is not None
        and existing.category_input_hash == txn.category_input_hash
        and existing.candidates_json == encoded
        and (existing.expires_at is None or as_utc(existing.expires_at) > now)
    ):
        return existing
    if existing is not None:
        await supersede_active_decisions(session, txn.id)
    decision = CategoryReviewDecision(
        transaction_id=txn.id,
        source_interaction_id=source_interaction_id,
        category_input_hash=txn.category_input_hash or "",
        candidates_json=encoded,
        proposed_slug=proposed_slug,
        confidence=confidence,
        threshold=threshold,
        gate_reason=gate_reason,
        expires_at=now + DECISION_TTL,
    )
    session.add(decision)
    await session.flush()
    return decision


async def ensure_legacy_decision(
    session: AsyncSession, txn: Transaction
) -> CategoryReviewDecision:
    """Record a zero-candidate decision for rows predating this lifecycle."""
    existing = await session.scalar(
        select(CategoryReviewDecision)
        .where(
            CategoryReviewDecision.transaction_id == txn.id,
            CategoryReviewDecision.status == "active",
            CategoryReviewDecision.source_interaction_id.is_(None),
        )
        .order_by(CategoryReviewDecision.id.desc())
        .limit(1)
    )
    if existing is not None:
        return existing
    decision = CategoryReviewDecision(
        transaction_id=txn.id,
        category_input_hash=txn.category_input_hash or "legacy",
        candidates_json="[]",
        gate_reason=txn.review_reason,
    )
    session.add(decision)
    await session.flush()
    return decision


async def consume_decision(
    session: AsyncSession,
    decision_id: int,
    *,
    selected_slug: str,
    category_input_hash: str,
    interaction_id: int | None = None,
    delivery_id: int | None = None,
) -> AuditAction | None:
    """CAS-claim an active choice and apply category plus audit atomically."""
    now = utc_now()
    decision = await session.get(CategoryReviewDecision, decision_id)
    if decision is None or decision.status != "active":
        return None
    txn = await session.get(Transaction, decision.transaction_id)
    if txn is None:
        return None
    candidate_rows = json.loads(decision.candidates_json)
    candidate_slugs = {
        row.get("category", row.get("slug"))
        for row in candidate_rows
        if isinstance(row, dict)
    }
    if selected_slug not in candidate_slugs:
        return None
    if decision.source_interaction_id is None and txn.review_status not in {
        "pending",
        "notified",
    }:
        return None
    account_type = None
    if txn.account_id is not None:
        account = await session.get(Account, txn.account_id)
        account_type = account.type if account is not None else None
    fields = build_input_payload(txn, account_type)
    fresh_hash = compute_input_hash(fields)
    if fresh_hash != category_input_hash or fresh_hash != decision.category_input_hash:
        return None
    if delivery_id is not None:
        owner_clause = (
            TelegramOutboundDelivery.category_review_decision_id == decision_id
            if decision.source_interaction_id is None
            else TelegramOutboundDelivery.interaction_id
            == decision.source_interaction_id
        )
        delivered = await session.scalar(
            select(TelegramOutboundDelivery.id).where(
                TelegramOutboundDelivery.id == delivery_id,
                TelegramOutboundDelivery.transaction_id == txn.id,
                owner_clause,
            )
        )
        if delivered is None:
            return None

    class DecisionConflict(Exception):
        pass

    try:
        async with session.begin_nested():
            before = {"category": txn.category, "review_status": txn.review_status}
            ok, category = await assign_category_no_commit(
                session,
                txn.id,
                selected_slug,
                actor="assistant_button",
                direction_policy=CategoryDirectionPolicy.INFERRED_STRICT,
                preserve_decision_id=decision_id,
            )
            if not ok or category is None:
                raise DecisionConflict
            claimed = await session.execute(
                update(CategoryReviewDecision)
                .where(
                    CategoryReviewDecision.id == decision_id,
                    CategoryReviewDecision.status == "active",
                    CategoryReviewDecision.category_input_hash == category_input_hash,
                    (CategoryReviewDecision.expires_at.is_(None))
                    | (CategoryReviewDecision.expires_at > now),
                )
                .values(status="consumed", selected_slug=selected_slug, consumed_at=now)
                .execution_options(synchronize_session="fetch")
            )
            if cast(Any, claimed).rowcount != 1:
                raise DecisionConflict
            action = AuditAction(
                interaction_id=interaction_id,
                action_type="set_category",
                target_type="transaction",
                target_id=txn.id,
                arguments_json=json.dumps({"category": category}, sort_keys=True),
                before_json=json.dumps(before, sort_keys=True),
                after_json=json.dumps(
                    {"category": txn.category, "review_status": txn.review_status},
                    sort_keys=True,
                ),
                undo_status=None,
            )
            session.add(action)
            await session.flush()
    except DecisionConflict:
        return None
    return action
