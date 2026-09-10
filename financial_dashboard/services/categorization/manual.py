"""Single authoritative path for manual category assignment."""

from enum import StrEnum
from difflib import SequenceMatcher
import logging
from typing import NamedTuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    Account,
    Category,
    Transaction,
    utc_now,
)
from financial_dashboard.services.categorization.polarity import resolve_direction
from financial_dashboard.services.categorization.decision_lifecycle import (
    supersede_active_decisions,
)
from financial_dashboard.services.categorization.vocabulary import (
    canonicalize_slug,
    ensure_category,
    get_vocab_version,
    is_valid_slug,
    refresh_vocab_cache,
)


class CategoryDirectionPolicy(StrEnum):
    INFERRED_STRICT = "inferred_strict"
    EXPLICIT_MANUAL_OVERRIDE = "explicit_manual_override"


class AssistantCategoryResolution(NamedTuple):
    """Assistant-only category lookup result."""

    slug: str | None
    corrected: bool


class CategoryAssignment(NamedTuple):
    """Result of a manual category assignment attempt."""

    ok: bool
    slug: str | None


logger = logging.getLogger(__name__)


async def resolve_assistant_category_slug(
    session: AsyncSession, raw_category: str
) -> AssistantCategoryResolution:
    """Resolve an unambiguous assistant typo against active categories only.

    Returns ``(slug, was_corrected)``.  Exact active matches are returned as
    supplied; near matches require a clear 0.08 ratio margin.  Existing manual,
    API, and web callers must continue using exact assignment below.
    """
    slug = canonicalize_slug(raw_category)
    active = list(
        (
            await session.scalars(
                select(Category.slug).where(Category.active.is_(True))
            )
        ).all()
    )
    if slug in active:
        return AssistantCategoryResolution(slug, False)
    ranked = sorted(
        (
            (SequenceMatcher(None, slug, candidate).ratio(), candidate)
            for candidate in active
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.82:
        return AssistantCategoryResolution(None, False)
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08:
        return AssistantCategoryResolution(None, False)
    return AssistantCategoryResolution(ranked[0][1], True)


async def assign_category_manual(
    session: AsyncSession,
    txn_id: int,
    raw_category: str,
    *,
    actor: str = "user",
    create: bool = False,
) -> CategoryAssignment:
    """Set a transaction's category by hand (authoritative; sweeps never override).

    By default the slug must already exist in the controlled vocabulary — a typo
    like 'goceries' is rejected rather than silently minting a junk category
    (same guard as add_merchant_rule). Pass create=True to deliberately add a
    brand-new category.
    """
    result = await assign_category_no_commit(
        session, txn_id, raw_category, actor=actor, create=create
    )
    if result[0]:
        await session.commit()
        if create:
            try:
                await refresh_vocab_cache(session)
            except Exception:
                logger.exception(
                    "Category committed but vocabulary cache refresh failed; "
                    "the fetch cycle will retry"
                )
    return result


async def assign_category_no_commit(
    session: AsyncSession,
    txn_id: int,
    raw_category: str,
    *,
    actor: str = "user",
    create: bool = False,
    direction_policy: CategoryDirectionPolicy = CategoryDirectionPolicy.EXPLICIT_MANUAL_OVERRIDE,
    preserve_decision_id: int | None = None,
) -> CategoryAssignment:
    """Apply a manual category while leaving commit/rollback to the caller."""
    txn = await session.get(Transaction, txn_id)
    if not txn:
        return CategoryAssignment(False, None)

    cleaned = (raw_category or "").strip()
    if not cleaned:
        # Clearing a category.
        txn.category = None
        txn.category_method = "manual"
        txn.category_confidence = 1.0
        txn.categorized_at = utc_now()
        txn.review_status = "resolved"
        await supersede_active_decisions(
            session, txn.id, preserve_decision_id=preserve_decision_id
        )
        return CategoryAssignment(True, None)

    slug = canonicalize_slug(cleaned)
    if not is_valid_slug(slug):
        return CategoryAssignment(False, None)

    if not create:
        existing_category = await session.scalar(
            select(Category).where(Category.slug == slug)
        )
        if existing_category is None:
            return CategoryAssignment(False, None)
        if (
            direction_policy == CategoryDirectionPolicy.INFERRED_STRICT
            and not existing_category.active
        ):
            return CategoryAssignment(False, None)

    if direction_policy == CategoryDirectionPolicy.INFERRED_STRICT:
        account_type = None
        if txn.account_id is not None:
            account = await session.get(Account, txn.account_id)
            account_type = account.type if account is not None else None
        if resolve_direction(slug, txn.direction, account_type).slug != slug:
            return CategoryAssignment(False, None)

    await ensure_category(session, slug)
    txn.category = slug
    txn.category_method = "manual"
    txn.category_confidence = 1.0
    txn.category_model = f"manual:{actor}"
    version = await session.scalar(
        text("SELECT value FROM settings WHERE key = 'category_vocab_version'")
    )
    try:
        txn.category_vocab_version = (
            int(version) if version is not None else get_vocab_version()
        )
    except TypeError, ValueError:
        txn.category_vocab_version = get_vocab_version()
    txn.categorized_at = utc_now()
    txn.review_status = "resolved"
    await supersede_active_decisions(
        session, txn.id, preserve_decision_id=preserve_decision_id
    )
    return CategoryAssignment(True, slug)
