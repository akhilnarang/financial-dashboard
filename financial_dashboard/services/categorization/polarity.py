"""Direction/polarity guardrail for transaction categorization."""

from typing import NamedTuple

from financial_dashboard.services.categorization.slugs import (
    CREDIT_CARD_ACCOUNT_TYPE,
    CREDIT_CARD_PAYMENT_SLUG,
    REPAYMENT_SLUG,
    UNKNOWN_SLUG,
)

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
CREDIT_DEFAULT = REPAYMENT_SLUG

# On a credit card there is no inbound money: the account holds what you OWE, so
# the only way it can be credited is a merchant handing money back or you paying
# the bill. 'repayment' (the bank-side credit default — somebody paying you back)
# is therefore impossible on a card and must never survive as the answer; the
# unexplained card credit defaults to a bill payment instead.
CC_CREDIT_DEFAULT = CREDIT_CARD_PAYMENT_SLUG
CC_IMPOSSIBLE_CREDIT_SLUGS: frozenset = frozenset(
    {REPAYMENT_SLUG, "salary", "other_income", "interest"}
)


class DirectionResult(NamedTuple):
    slug: str
    changed: bool


def resolve_direction(
    slug: str, direction: str, account_type: str | None = None
) -> DirectionResult:
    """Return (resolved_slug, changed). Flip directionally-impossible categories
    and default 'unknown' by direction:
      debit  + (INCOME slug or 'unknown') -> DEBIT_DEFAULT ('expense')
      credit + (EXPENSE slug or 'unknown') -> CREDIT_DEFAULT ('repayment')
      credit on a CREDIT CARD + (EXPENSE slug, 'unknown', or an inbound-money
        slug that a card cannot receive) -> CC_CREDIT_DEFAULT
      otherwise unchanged.
    """
    if direction == "debit":
        if slug in INCOME_SLUGS or slug == UNKNOWN_SLUG:
            return DirectionResult(DEBIT_DEFAULT, True)
    elif direction == "credit":
        if account_type == CREDIT_CARD_ACCOUNT_TYPE:
            if (
                slug in EXPENSE_SLUGS
                or slug in CC_IMPOSSIBLE_CREDIT_SLUGS
                or slug == UNKNOWN_SLUG
            ):
                return DirectionResult(CC_CREDIT_DEFAULT, True)
            return DirectionResult(slug, False)
        if slug in EXPENSE_SLUGS or slug == UNKNOWN_SLUG:
            return DirectionResult(CREDIT_DEFAULT, True)
    return DirectionResult(slug, False)
