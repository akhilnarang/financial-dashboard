"""DB-backed merchant→category rules with in-memory cache.

Patterns are lowercased substrings matched against normalize_text(counterparty
+ ' ' + raw_description).  The cache is populated at startup via
load_merchant_rules() and read on every categorization call via
get_merchant_rules() — no DB round-trips in the hot path.
"""

import asyncio
import logging

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import async_session
from financial_dashboard.db.models import Category, MerchantRule
from financial_dashboard.services.categorization.normalize import normalize_text
from financial_dashboard.services.categorization.vocabulary import is_valid_slug

logger = logging.getLogger(__name__)

# Note: merchant rules are DB-only. There is no seed import here — the table is
# populated manually (scripts/merchant_rules.py seed/import) so the app runtime
# never depends on the untracked seed data.

# Module-level ordered cache: list of (pattern, category) in match priority
# order (lower priority number first, then longer pattern first for specificity).
_cache: list[tuple[str, str]] = []
# Hold across query and publish, so an older read cannot replace a newer one.
_reload_lock = asyncio.Lock()
# Rules match by substring. A shorter pattern matches almost every row.
MIN_PATTERN_LENGTH = 3


async def load_merchant_rules(_session: AsyncSession | None = None) -> None:
    """Populate the in-memory cache from active merchant_rules rows.

    Ordering: priority ASC, length(pattern) DESC, id ASC — so within a
    priority band, longer (more-specific) patterns are tried first.

    _session: inject an AsyncSession for testing (uses the global engine's
    async_session when None).
    """

    async def _query(s) -> None:
        stmt = (
            select(MerchantRule.pattern, MerchantRule.category)
            .where(MerchantRule.active.is_(True))
            .order_by(
                MerchantRule.priority.asc(),
                func.length(MerchantRule.pattern).desc(),
                MerchantRule.id.asc(),
            )
        )
        rows = (await s.execute(stmt)).all()
        _cache.clear()
        # Defensive: skip blank patterns — an empty substring matches everything.
        _cache.extend(
            (r.pattern, r.category) for r in rows if (r.pattern or "").strip()
        )
        logger.info("Loaded %d merchant rules", len(_cache))

    async with _reload_lock:
        if _session is not None:
            await _query(_session)
        else:
            async with async_session() as session:
                await _query(session)


def get_merchant_rules() -> tuple[tuple[str, str], ...]:
    """Return cached (pattern, category) tuples in priority order."""
    return tuple(_cache)


async def add_merchant_rule(
    session: AsyncSession, pattern: str, category: str, *, priority: int = 100
) -> str:
    """Insert or update a merchant rule and return its normalized pattern.

    Normalizes the pattern with normalize_text (lowercase, punctuation→space)
    so it matches the same text the matcher sees (e.g. "acme.pay@upi" →
    "acme pay upi"). Rejects a pattern shorter than MIN_PATTERN_LENGTH (it
    would match almost every row).
    Validates the category slug against the active vocabulary. ON CONFLICT
    updates category/active/priority.

    Does NOT refresh the module-level cache — caller calls load_merchant_rules()
    after committing if an up-to-date cache is needed.
    """
    pattern = normalize_text(pattern)
    if len(pattern) < MIN_PATTERN_LENGTH:
        raise ValueError(
            f"Merchant rule pattern must have at least {MIN_PATTERN_LENGTH} characters"
        )
    if not is_valid_slug(category):
        raise ValueError(f"Invalid category slug: {category!r}")
    # Category must exist in the controlled vocabulary — catches typos like
    # 'dinng' that would otherwise auto-write an out-of-vocab rule.
    exists = (
        await session.execute(
            select(Category.id).where(
                Category.slug == category, Category.active.is_(True)
            )
        )
    ).first()
    if not exists:
        raise ValueError(
            f"Unknown or inactive category {category!r} — create it first (add to the "
            "categories table) or fix the typo"
        )
    await session.execute(
        text(
            "INSERT INTO merchant_rules (pattern, category, active, priority, created_at) "
            "VALUES (:pattern, :category, 1, :priority, CURRENT_TIMESTAMP) "
            "ON CONFLICT(pattern) DO UPDATE SET "
            "category = :category, active = 1, priority = :priority"
        ),
        {"pattern": pattern, "category": category, "priority": priority},
    )
    return pattern


async def list_merchant_rules(session: AsyncSession) -> list[MerchantRule]:
    """Return all merchant rules ordered by priority ASC, length DESC, id ASC."""
    stmt = select(MerchantRule).order_by(
        MerchantRule.priority.asc(),
        func.length(MerchantRule.pattern).desc(),
        MerchantRule.id.asc(),
    )
    return list((await session.execute(stmt)).scalars().all())


async def save_merchant_rule(
    session: AsyncSession, pattern: str, category: str, *, priority: int = 100
) -> MerchantRule:
    """Insert or update a merchant rule and return the stored row.

    Args:
        session: Open async session. The caller commits.
        pattern: Raw pattern. The service normalizes it.
        category: Category slug from the active vocabulary.
        priority: Match priority. A lower number matches first.

    Returns:
        The stored merchant rule.

    Raises:
        ValueError: The pattern is too short or the category is not valid.
    """
    stored = await add_merchant_rule(session, pattern, category, priority=priority)
    rules = await session.scalars(
        select(MerchantRule)
        .where(MerchantRule.pattern == stored)
        .execution_options(populate_existing=True)
    )
    return rules.one()


async def deactivate_merchant_rule(session: AsyncSession, rule_id: int) -> bool:
    """Deactivate one merchant rule.

    The row stays as a tombstone. init_db reseeds a missing built-in rule, so a
    hard delete would bring the rule back at the next start.

    Args:
        session: Open async session. The caller commits.
        rule_id: Primary key of the rule.

    Returns:
        True when an active rule was deactivated, else False.
    """
    deactivated = await session.scalar(
        update(MerchantRule)
        .where(MerchantRule.id == rule_id, MerchantRule.active.is_(True))
        .values(active=False)
        .returning(MerchantRule.id)
    )
    return deactivated is not None
