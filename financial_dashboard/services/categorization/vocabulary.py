"""Controlled-vocabulary access for transaction categories."""

import re

from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Category
from financial_dashboard.exceptions import (
    ConflictException,
    UnprocessableEntityException,
)
from financial_dashboard.schemas.transactions import CategoryListResponse, CategoryRead
from financial_dashboard.services.cashflow.buckets import (
    BUCKET_BY_SLUG,
    bucket_for_slug,
    label_for_slug,
)
from financial_dashboard.services.categorization.polarity import (
    USER_EXPENSE_SLUGS,
    USER_INCOME_SLUGS,
)
from financial_dashboard.services.categorization.slugs import (
    USER_CATEGORY_BUCKETS,
    UserCategoryBucket,
)
from financial_dashboard.services import settings as settings_mod
from financial_dashboard.services.settings import get_setting_int

SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

# Controlled vocabulary of category slugs. A human-facing display label per
# category will live alongside these once there's a categories UI to render it.
SEED_CATEGORIES: list[str] = [
    "salary",
    "interest",
    "refund",
    "cashback_rewards",
    "reimbursement",
    "other_income",
    "repayment",
    "expense",
    "investment",
    "investment_redemption",
    "self_transfer",
    "passthrough",
    "personal_loan",
    "family",
    "credit_card_payment",
    "bill_payment",
    "groceries",
    "dining",
    "cafe",
    "fuel",
    "car_maintenance",
    "transport",
    "shopping",
    "utilities",
    "subscriptions",
    "rent",
    "emi_loan",
    "insurance",
    "healthcare",
    "entertainment",
    "travel",
    "education",
    "personal_care",
    "fees_charges",
    "laundry",
    "professional_services",
    "tax",
    "tax_refund",
    "cash_withdrawal",
    "charity_gift",
    "gift",
    "misc",
    "unknown",
]


def canonicalize_slug(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def is_valid_slug(slug: str) -> bool:
    return bool(SLUG_RE.match(slug))


async def get_active_slugs(session: AsyncSession) -> list[str]:
    stmt = (
        select(Category.slug).where(Category.active.is_(True)).order_by(Category.slug)
    )
    return list((await session.execute(stmt)).scalars().all())


async def list_categories(session: AsyncSession) -> CategoryListResponse:
    """List every category with its label and bank-scope cashflow bucket.

    Args:
        session: Open async session.

    Returns:
        Every category, ordered by slug.
    """
    await load_user_buckets(session)
    categories = await session.scalars(select(Category).order_by(Category.slug))
    return CategoryListResponse(
        items=[
            CategoryRead(
                slug=category.slug,
                label=label_for_slug(category.slug),
                active=category.active,
                bucket=bucket_for_slug(category.slug, scope="bank"),
            )
            for category in categories
        ]
    )


async def load_user_buckets(session: AsyncSession) -> None:
    """Replace the cached user-category buckets with the stored ones.

    The cashflow report calls this before it reads, so a category created in
    another worker counts at once. A built-in slug keeps its code-map bucket.

    Args:
        session: Open async session.

    Returns:
        None. It replaces ``USER_EXPENSE_SLUGS`` and ``USER_INCOME_SLUGS``.
    """
    rows = (
        await session.execute(
            select(Category.slug, Category.bucket).where(
                Category.bucket.in_(USER_CATEGORY_BUCKETS),
                Category.slug.not_in(list(BUCKET_BY_SLUG)),
            )
        )
    ).all()
    USER_EXPENSE_SLUGS.clear()
    USER_EXPENSE_SLUGS.update(slug for slug, bucket in rows if bucket == "expense")
    USER_INCOME_SLUGS.clear()
    USER_INCOME_SLUGS.update(slug for slug, bucket in rows if bucket == "income")


async def create_category(
    session: AsyncSession, raw_slug: str, bucket: UserCategoryBucket
) -> Category:
    """Create a user category with a stored cashflow bucket. Do not commit.

    A slug that exists with no bucket, such as one a manual override made,
    gets the bucket.

    Args:
        session: Open async session.
        raw_slug: The category name. It is normalized to a slug.
        bucket: The cashflow bucket, ``expense`` or ``income``.

    Returns:
        The new or updated category row.

    Raises:
        UnprocessableEntityException: The name does not make a valid slug.
        ConflictException: The slug is a built-in or already has a bucket.
    """
    slug = canonicalize_slug(raw_slug)
    if not is_valid_slug(slug):
        raise UnprocessableEntityException(detail=f"'{raw_slug}' is not a valid slug")
    if slug in SEED_CATEGORIES:
        raise ConflictException(detail=f"'{slug}' is a built-in category")
    exists = ConflictException(detail=f"category '{slug}' already exists")
    if (
        category := await session.scalar(
            update(Category)
            .where(Category.slug == slug, Category.bucket.is_(None))
            .values(bucket=bucket, active=True)
            .returning(Category)
            .execution_options(populate_existing=True)
        )
    ) is None:
        if await session.scalar(select(Category.id).where(Category.slug == slug)):
            raise exists
        category = Category(slug=slug, active=True, bucket=bucket)
        try:
            async with session.begin_nested():
                session.add(category)
        except IntegrityError as exc:
            raise exists from exc
    await session.flush()
    await bump_vocab_version(session)
    return category


def get_vocab_version() -> int:
    return get_setting_int("category_vocab_version", 1)


async def bump_vocab_version(session: AsyncSession) -> int:
    """Persist a bumped version in the caller transaction.

    The database is authoritative.  In particular, do not update the process
    cache until the caller commits, otherwise a rollback leaks a version.
    """
    await session.execute(
        text(
            "INSERT INTO settings (key, value) VALUES "
            "('category_vocab_version', '2') "
            "ON CONFLICT(key) DO UPDATE SET value = "
            "CAST(settings.value AS INTEGER) + 1"
        ),
    )
    # Read the value written in this transaction.  The expression above is
    # evaluated by SQLite while holding the write lock, so concurrent category
    # creation cannot lose an increment.
    value = await session.scalar(
        text("SELECT value FROM settings WHERE key = 'category_vocab_version'")
    )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("category vocabulary version is not numeric") from exc


async def refresh_vocab_cache(session: AsyncSession) -> int:
    """Refresh the in-memory version and user buckets after a successful commit."""
    await load_user_buckets(session)
    value = await session.scalar(
        text("SELECT value FROM settings WHERE key = 'category_vocab_version'")
    )
    if value is not None:
        settings_mod._cache["category_vocab_version"] = str(value)
    return get_vocab_version()


async def ensure_category(session: AsyncSession, slug: str) -> bool:
    """Insert a category if absent. Returns True if newly created (and bumps the
    vocab version). Uses a SELECT existence check (reliable) rather than relying
    on rowcount after an upsert."""
    existing = (
        await session.execute(select(Category.id).where(Category.slug == slug))
    ).first()
    if existing:
        return False
    session.add(Category(slug=slug, active=True))
    await session.flush()
    await bump_vocab_version(session)
    return True
