"""Shared accounting policy for the Paisa projection and reconciliation.

This module owns the deterministic mapping from dashboard accounts/categories
to backend-valid ledger accounts.  It is pure over its arguments: callers own
database reads, rendering, publication, and DTO shaping.

Category contra accounts are selected by accounting semantics, never by the
transaction direction.  Operator mappings always win and are validated
strictly for the configured backend; generated defaults may be normalized by
the renderer strategy first (notably for Beancount's account grammar).
"""

from financial_dashboard.db.models import Account
from financial_dashboard.services.cashflow.scope import CARD_ACCOUNT_TYPES
from financial_dashboard.services.categorization.slugs import (
    CREDIT_CARD_ACCOUNT_TYPE,
    REPAYMENT_SLUG,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.renderers import (
    normalize_default_account,
    validate_account_name,
)
from financial_dashboard.services.paisa.renderers.base import (
    CARD_PAYMENT_CLEARING,
    INVESTMENT_UNALLOCATED_ACCOUNT,
    REPAYMENT_CLEARING_ACCOUNT,
    InvalidAccountName,
    LedgerAccount,
)

SELF_TRANSFER_SLUG = "self_transfer"
CREDIT_CARD_PAYMENT_SLUG = "credit_card_payment"

INCOME_CATEGORY_SLUGS = frozenset({"salary", "interest", "other_income"})
CONTRA_EXPENSE_CATEGORY_SLUGS = frozenset({"refund", "cashback_rewards"})
INVESTMENT_CATEGORY_SLUGS = frozenset({"investment", "investment_redemption"})

# Closed dashboard_kind taxonomy. Every emitted entry carries exactly one.
KIND_EXPENSE = "expense"
KIND_INCOME = "income"
KIND_CONTRA_EXPENSE = "contra_expense"
KIND_INVESTMENT = "investment"
KIND_REPAYMENT = "repayment"
KIND_SELF_TRANSFER = "self_transfer"
KIND_CARD_PAYMENT = "card_payment"
KIND_OPENING = "opening"
KIND_LOT = "investment_lot"
KIND_UNKNOWN = "unknown"


class ProjectionError(Exception):
    """Raised when projection policy cannot produce a valid ledger identity."""


def account_kind(account_type: str | None) -> str:
    """Map a dashboard account type to ``asset`` or ``liability``.

    Credit-card account types are liabilities; every other recognized account
    type (and the existing unknown-type fallback) is an asset.
    """
    if account_type == CREDIT_CARD_ACCOUNT_TYPE or account_type in CARD_ACCOUNT_TYPES:
        return "liability"
    return "asset"


def _title_segment(value: str | None) -> str:
    """Render a free-form value as a deterministic account-path segment."""
    text = (value or "").replace("_", " ")
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(part.capitalize() for part in cleaned.split())


def _default_account_name(account: Account, kind: str) -> str:
    bank = _title_segment(account.bank)
    label = _title_segment(account.label)
    if kind == "liability":
        return f"Liabilities:Card:{bank}:{label}"
    return f"Assets:Bank:{bank}:{label}"


def _finalize_name(raw: str, *, is_default: bool, backend: str, label: str) -> str:
    """Normalize a generated default or strictly validate an operator override."""
    name = normalize_default_account(raw, backend) if is_default else raw
    try:
        return validate_account_name(name, backend)
    except InvalidAccountName as exc:
        raise ProjectionError(
            f"{label}: invalid ledger name for {backend!r}: {exc}"
        ) from exc


def resolve_account(
    account: Account, mappings: dict[str, str], backend: str
) -> LedgerAccount:
    """Resolve one dashboard account to its configured ledger identity."""
    kind = account_kind(account.type)
    override = mappings.get(str(account.id))
    name = _finalize_name(
        override if override is not None else _default_account_name(account, kind),
        is_default=override is None,
        backend=backend,
        label=f"account {account.id}",
    )
    return LedgerAccount(account_id=account.id, name=name, kind=kind)


def category_kind(slug: str) -> str:
    """Classify a normalized category slug into the closed dashboard taxonomy."""
    if slug in INCOME_CATEGORY_SLUGS:
        return KIND_INCOME
    if slug in CONTRA_EXPENSE_CATEGORY_SLUGS:
        return KIND_CONTRA_EXPENSE
    if slug in INVESTMENT_CATEGORY_SLUGS:
        return KIND_INVESTMENT
    if slug == REPAYMENT_SLUG:
        return KIND_REPAYMENT
    if slug in ("", "unknown", "misc"):
        return KIND_UNKNOWN
    return KIND_EXPENSE


def contra_account(
    category: str | None,
    config: PaisaProjectionConfig,
    backend: str,
) -> str:
    """Resolve a category's semantic contra account for ``backend``.

    Direction is intentionally absent: credits on expense categories are
    reversals that net against Expenses, not income. Operator mappings take
    precedence for every category, including investment and repayment.
    """
    slug = (category or "").strip().lower() or "unknown"
    override = config.category_mappings.get(slug)
    if override is not None:
        return _finalize_name(
            override, is_default=False, backend=backend, label=f"category {slug!r}"
        )
    kind = category_kind(slug)
    title = _title_segment(slug)
    if kind == KIND_INVESTMENT:
        raw = INVESTMENT_UNALLOCATED_ACCOUNT
    elif kind == KIND_REPAYMENT:
        raw = REPAYMENT_CLEARING_ACCOUNT
    elif kind == KIND_INCOME:
        raw = f"Income:{title}"
    elif kind == KIND_CONTRA_EXPENSE:
        raw = f"Expenses:{title}"
    elif kind == KIND_UNKNOWN:
        raw = "Expenses:Unknown"
    else:
        raw = f"Expenses:{title}"
    return _finalize_name(
        raw, is_default=True, backend=backend, label=f"category {slug!r}"
    )


def card_clearing_account(config: PaisaProjectionConfig, backend: str) -> str:
    """Resolve the generic card-payment clearing liability for ``backend``."""
    override = config.category_mappings.get(CREDIT_CARD_PAYMENT_SLUG)
    raw = override if override is not None else CARD_PAYMENT_CLEARING
    return _finalize_name(
        raw,
        is_default=override is None,
        backend=backend,
        label=CREDIT_CARD_PAYMENT_SLUG,
    )


def normalize_policy_account(raw: str, *, backend: str, label: str) -> str:
    """Normalize and validate a generated policy-owned account name.

    Projection uses this for the investment-funding equity remap, which follows
    the same generated-default rule as the accounts resolved above.
    """
    return _finalize_name(raw, is_default=True, backend=backend, label=label)


__all__ = [
    "CONTRA_EXPENSE_CATEGORY_SLUGS",
    "CREDIT_CARD_PAYMENT_SLUG",
    "INCOME_CATEGORY_SLUGS",
    "INVESTMENT_CATEGORY_SLUGS",
    "KIND_CARD_PAYMENT",
    "KIND_CONTRA_EXPENSE",
    "KIND_EXPENSE",
    "KIND_INCOME",
    "KIND_INVESTMENT",
    "KIND_LOT",
    "KIND_OPENING",
    "KIND_REPAYMENT",
    "KIND_SELF_TRANSFER",
    "KIND_UNKNOWN",
    "ProjectionError",
    "SELF_TRANSFER_SLUG",
    "account_kind",
    "card_clearing_account",
    "category_kind",
    "contra_account",
    "normalize_policy_account",
    "resolve_account",
]
