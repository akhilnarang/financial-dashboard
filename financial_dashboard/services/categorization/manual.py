"""Single authoritative path for manual category assignment."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Category, Transaction, utc_now
from financial_dashboard.services.categorization.vocabulary import (
    canonicalize_slug,
    ensure_category,
    get_vocab_version,
    is_valid_slug,
)


async def assign_category_manual(
    session: AsyncSession,
    txn_id: int,
    raw_category: str,
    *,
    actor: str = "user",
    create: bool = False,
) -> tuple[bool, str | None]:
    """Set a transaction's category by hand (authoritative; sweeps never override).

    By default the slug must already exist in the controlled vocabulary — a typo
    like 'goceries' is rejected rather than silently minting a junk category
    (same guard as add_merchant_rule). Pass create=True to deliberately add a
    brand-new category.
    """
    txn = await session.get(Transaction, txn_id)
    if not txn:
        return False, None

    cleaned = (raw_category or "").strip()
    if not cleaned:
        # Clearing a category.
        txn.category = None
        txn.category_method = "manual"
        txn.category_confidence = 1.0
        txn.categorized_at = utc_now()
        txn.review_status = "resolved"
        await session.commit()
        return True, None

    slug = canonicalize_slug(cleaned)
    if not is_valid_slug(slug):
        return False, None

    if not create:
        exists = (
            await session.execute(select(Category.id).where(Category.slug == slug))
        ).first()
        if not exists:
            # Unknown slug and caller didn't opt in to creating one — reject so a
            # typo surfaces as a failed apply instead of a new one-off category.
            return False, None

    await ensure_category(session, slug)
    txn.category = slug
    txn.category_method = "manual"
    txn.category_confidence = 1.0
    txn.category_model = f"manual:{actor}"
    txn.category_vocab_version = get_vocab_version()
    txn.categorized_at = utc_now()
    txn.review_status = "resolved"
    await session.commit()
    return True, slug
