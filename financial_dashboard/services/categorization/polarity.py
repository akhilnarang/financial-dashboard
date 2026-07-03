"""Direction/polarity guardrail for transaction categorization."""

from typing import NamedTuple

EXPENSE_SLUGS: frozenset = frozenset(
    {
        "expense",
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
        "cash_withdrawal",
        "charity_gift",
        "gift",
    }
)

INCOME_SLUGS: frozenset = frozenset(
    {
        "salary",
        "interest",
        "refund",
        "cashback_rewards",
        "other_income",
        "repayment",
        "investment_redemption",
    }
)

# neutral (valid either direction, passed through unchanged): self_transfer,
# credit_card_payment, investment, misc. NOTE: 'unknown' is NOT neutral — it is
# always replaced by the direction default below.
DEBIT_DEFAULT = "expense"
CREDIT_DEFAULT = "repayment"


class DirectionResult(NamedTuple):
    slug: str
    changed: bool


def resolve_direction(slug: str, direction: str) -> DirectionResult:
    """Return (resolved_slug, changed). Flip directionally-impossible categories
    and default 'unknown' by direction:
      debit  + (INCOME slug or 'unknown') -> DEBIT_DEFAULT ('expense')
      credit + (EXPENSE slug or 'unknown') -> CREDIT_DEFAULT ('repayment')
      otherwise unchanged.
    """
    if direction == "debit":
        if slug in INCOME_SLUGS or slug == "unknown":
            return DirectionResult(DEBIT_DEFAULT, True)
    elif direction == "credit":
        if slug in EXPENSE_SLUGS or slug == "unknown":
            return DirectionResult(CREDIT_DEFAULT, True)
    return DirectionResult(slug, False)
