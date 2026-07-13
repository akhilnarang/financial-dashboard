"""Slug -> report-bucket map for the cashflow report.

Derived from the categorization direction guard's slug sets with four re-homings,
so the map can never silently drift from the guard. This is a *reporting*
statement (income/investment/expense/transfers_in/internal), not a direction
guard: it answers "which line of the cashflow report does this slug belong on",
which is a different question from "can this slug occur on a debit".
"""

import logging

from financial_dashboard.services.categorization.polarity import (
    EXPENSE_SLUGS,
    INCOME_SLUGS,
)
from financial_dashboard.services.categorization.slugs import (
    REPAYMENT_SLUG,
    UNKNOWN_SLUG,
)

logger = logging.getLogger(__name__)

# The slug itself is shared, not re-spelled: this module's whole job is to stay
# in step with the categorization vocabulary, so it must not carry its own copy
# of a slug that vocabulary already names.
TRANSFERS_IN_SLUG = REPAYMENT_SLUG
CONTRA_EXPENSE_SLUGS = frozenset({"refund", "cashback_rewards"})
INTERNAL_SLUGS = frozenset({"self_transfer", "credit_card_payment"})

# Re-homings out of the income guard, so the report shows earnings rather than
# every credit: refund and cashback_rewards are money back on a purchase, so they
# net against spend (contra-expense); investment_redemption is a portfolio
# movement, not earnings; repayment is somebody handing money back and gets its
# own line so it never inflates income.
INCOME_BUCKET = frozenset(
    INCOME_SLUGS - CONTRA_EXPENSE_SLUGS - {"investment_redemption", TRANSFERS_IN_SLUG}
)
# cash_withdrawal is already an expense-guard slug; misc is neutral there. Both
# are unioned in explicitly so the expense line is complete regardless.
EXPENSE_BUCKET = frozenset(
    EXPENSE_SLUGS | CONTRA_EXPENSE_SLUGS | {"cash_withdrawal", "misc"}
)
INVESTMENT_BUCKET = frozenset({"investment", "investment_redemption"})

BUCKET_BY_SLUG: dict[str, str] = {
    **{s: "income" for s in INCOME_BUCKET},
    **{s: "expense" for s in EXPENSE_BUCKET},
    **{s: "investment" for s in INVESTMENT_BUCKET},
    **{s: "internal" for s in INTERNAL_SLUGS},
    TRANSFERS_IN_SLUG: "transfers_in",
}

LABEL_OVERRIDES = {
    "emi_loan": "EMI / Loan",
    "cash_withdrawal": "Cash Withdrawal",
    "cashback_rewards": "Cashback Rewards",
    "fees_charges": "Fees & Charges",
    "charity_gift": "Charity & Gifts",
}


def bucket_for_slug(slug: str | None) -> str:
    """Name the report bucket a category slug belongs on.

    Returns one of ``income``, ``expense``, ``investment``, ``transfers_in``,
    ``internal`` or ``uncategorized``. Every slug resolves to a bucket — there is
    no failure mode, because a row the report cannot place must still be shown:

    * ``None``, the empty string and the ``unknown`` sentinel are the *expected*
      absence of a category, so they short-circuit to ``uncategorized`` before
      the map is consulted and never look like drift.
    * A slug the map has never heard of (a manual override can mint one at
      runtime) also lands in ``uncategorized``, but is logged as a warning first:
      money must never vanish from the report just because a slug is new.
    """
    if not slug or slug == UNKNOWN_SLUG:
        return "uncategorized"
    bucket = BUCKET_BY_SLUG.get(slug)
    if bucket is None:
        logger.warning("cashflow: unmapped category slug %r -> uncategorized", slug)
        return "uncategorized"
    return bucket


def label_for_slug(slug: str | None) -> str:
    """Render a category slug as the label a report line shows.

    A known slug becomes its override, or its title-cased words. The two members
    of the uncategorized line get placeholders instead, because neither has a
    name of its own: ``None`` (a row carrying no category at all) reads as
    ``(uncategorized)``, and a slug no bucket maps reads as ``unmapped: <slug>``
    so an unrecognized slug is legible on the page rather than silently ordinary.
    """
    if slug is None:
        return "(uncategorized)"
    if slug in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[slug]
    if slug not in BUCKET_BY_SLUG and slug != UNKNOWN_SLUG:
        return f"unmapped: {slug}"
    return slug.replace("_", " ").title()
