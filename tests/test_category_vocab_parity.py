"""Exact-equality guard between the dashboard's seed category vocabulary and
the synthetic generator's mirror of it.

Two places name the same controlled vocabulary:

* ``financial_dashboard.services.categorization.vocabulary.SEED_CATEGORIES`` —
  the dashboard runtime's source of truth. ``init_db`` loads it into the
  ``categories`` table, the polarity guard's expense/income slug sets overlap
  it, and the cashflow bucket map is derived from it.
* ``scripts.synth.constants.SEED_CATEGORY_SLUGS`` — the synthetic seed
  generator's plain-literal mirror, kept pure so the generator never imports
  the dashboard runtime.

``scripts.synth.loader`` is the production module that asserts membership at
load time, and is owned elsewhere. This file owns the *vocabulary* parity: a
slug added to one and not the other is a parity bug regardless of which path
is loading data, and the only way to catch it without running either loader is
to import both constants in one test and compare. The order is not load-bearing
in either consumer (both write rows / emit categories without depending on it),
so the assertion is on the unordered set, with the per-side duplicate guards
that turn a quiet "two of the same slug" into a loud failure.
"""

from financial_dashboard.services.categorization.vocabulary import SEED_CATEGORIES
from scripts.synth.constants import SEED_CATEGORY_SLUGS


def test_dashboard_and_synth_seed_categories_match():
    """Both vocabularies hold the same slugs, with no duplicates, and both hold
    the slugs the polarity guard, cashflow buckets and reports depend on."""
    from financial_dashboard.services.categorization.slugs import (
        CREDIT_CARD_PAYMENT_SLUG,
        REPAYMENT_SLUG,
        UNKNOWN_SLUG,
    )
    from financial_dashboard.services.cashflow.buckets import CONTRA_EXPENSE_SLUGS

    dashboard = set(SEED_CATEGORIES)
    assert dashboard == set(SEED_CATEGORY_SLUGS)
    assert len(SEED_CATEGORIES) == len(SEED_CATEGORY_SLUGS) == len(dashboard)

    load_bearing = {
        "salary",
        "interest",
        "expense",
        "investment",
        "investment_redemption",
        "self_transfer",
        "misc",
        CREDIT_CARD_PAYMENT_SLUG,
        REPAYMENT_SLUG,
        UNKNOWN_SLUG,
    } | CONTRA_EXPENSE_SLUGS
    assert load_bearing <= dashboard, load_bearing - dashboard
