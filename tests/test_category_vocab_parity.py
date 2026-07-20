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


def test_dashboard_and_synth_seed_categories_are_exactly_equal_as_sets():
    """Every dashboard seed slug is a synth seed slug and vice versa, with no
    silent duplicates on either side.

    Set equality is what both consumers actually depend on: ``init_db`` writes
    ``categories`` rows from ``SEED_CATEGORIES`` under a UNIQUE constraint, and
    the synth generator emits ``SynthCategory`` rows from
    ``SEED_CATEGORY_SLUGS`` keyed by slug. Order is incidental in both paths.
    """
    dashboard = set(SEED_CATEGORIES)
    synth = set(SEED_CATEGORY_SLUGS)

    assert dashboard == synth
    # Either side silently growing a duplicate would still satisfy set equality
    # but would be a real bug (a UNIQUE constraint violation at write time, or
    # a synth row count that disagrees with the manifest). Pin the duplicate
    # count to zero on both sides so the equality stays meaningful.
    assert len(SEED_CATEGORIES) == len(dashboard), (
        "SEED_CATEGORIES contains duplicate slugs"
    )
    assert len(SEED_CATEGORY_SLUGS) == len(synth), (
        "SEED_CATEGORY_SLUGS contains duplicate slugs"
    )
    # And the two lengths match, which (with no duplicates) is the same
    # statement as the set equality above but reads as the count parity the
    # synth manifest's row count is meant to mirror.
    assert len(SEED_CATEGORIES) == len(SEED_CATEGORY_SLUGS)


def test_the_documented_reconciliation_slugs_are_in_both_vocabularies():
    """A few slugs are load-bearing across the dashboard — the polarity guard's
    named sentinels, the cashflow bucket map's contra-expense pair, and the
    report's transfers-in slug. Each one has to be in BOTH vocabularies, or the
    side that's missing it would emit an out-of-vocab slug the other cannot
    load."""
    from financial_dashboard.services.categorization.slugs import (
        CREDIT_CARD_PAYMENT_SLUG,
        REPAYMENT_SLUG,
        UNKNOWN_SLUG,
    )
    from financial_dashboard.services.cashflow.buckets import CONTRA_EXPENSE_SLUGS

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

    dashboard = set(SEED_CATEGORIES)
    synth = set(SEED_CATEGORY_SLUGS)
    for slug in load_bearing:
        assert slug in dashboard, f"missing from dashboard SEED_CATEGORIES: {slug}"
        assert slug in synth, f"missing from synth SEED_CATEGORY_SLUGS: {slug}"
