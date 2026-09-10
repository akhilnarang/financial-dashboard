"""Controlled-vocabulary access for transaction categories."""

import re

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Category
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
    "family",
    "credit_card_payment",
    "bill_payment",
    "groceries",
    "dining",
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
    """Refresh the in-memory version after a successful commit."""
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
