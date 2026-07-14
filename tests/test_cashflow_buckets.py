import logging

from financial_dashboard.services.cashflow.buckets import (
    BUCKET_BY_SLUG,
    INCOME_BUCKET,
    INTERNAL_SLUGS,
    TRANSFERS_IN_SLUG,
    bucket_for_slug,
    internal_slugs_for_scope,
    label_for_slug,
)
from financial_dashboard.services.categorization.polarity import (
    EXPENSE_SLUGS,
    INCOME_SLUGS,
)
from financial_dashboard.services.categorization.vocabulary import SEED_CATEGORIES


def test_income_bucket_is_earnings_only():
    assert INCOME_BUCKET == frozenset({"salary", "interest", "other_income"})


def test_repayment_is_transfers_in_not_income():
    assert TRANSFERS_IN_SLUG == "repayment"
    assert BUCKET_BY_SLUG["repayment"] == "transfers_in"
    assert "repayment" not in INCOME_BUCKET


def test_exhaustive_over_seed_vocabulary():
    # Every seed slug except the 'unknown' sentinel is mapped exactly once.
    seed = {s for s in SEED_CATEGORIES if s != "unknown"}
    assert set(BUCKET_BY_SLUG) == seed


def test_consistent_with_polarity_except_rehomings():
    # Expense-guard slugs map to expense; income-guard slugs map to income
    # EXCEPT the four documented re-homings.
    rehomed = {"refund", "cashback_rewards", "investment_redemption", "repayment"}
    for slug in EXPENSE_SLUGS:
        assert BUCKET_BY_SLUG[slug] == "expense", slug
    for slug in INCOME_SLUGS - rehomed:
        assert BUCKET_BY_SLUG[slug] == "income", slug
    assert BUCKET_BY_SLUG["refund"] == "expense"
    assert BUCKET_BY_SLUG["cashback_rewards"] == "expense"
    assert BUCKET_BY_SLUG["investment_redemption"] == "investment"


def test_internal_slugs_excluded():
    assert INTERNAL_SLUGS == frozenset({"self_transfer", "credit_card_payment"})
    assert BUCKET_BY_SLUG["self_transfer"] == "internal"
    assert BUCKET_BY_SLUG["credit_card_payment"] == "internal"


def test_card_payment_is_the_one_slug_whose_bucket_depends_on_scope():
    # Over the bank, paying the card bill is when the cash actually leaves.
    assert bucket_for_slug("credit_card_payment", scope="bank") == "expense"
    # Over every account it is internal churn — the swipes it settles are in
    # scope there, and counting both would charge the same rupee twice.
    assert bucket_for_slug("credit_card_payment") == "internal"
    assert bucket_for_slug("credit_card_payment", scope=None) == "internal"


def test_self_transfer_is_internal_under_every_scope():
    assert bucket_for_slug("self_transfer", scope="bank") == "internal"
    assert bucket_for_slug("self_transfer") == "internal"


def test_scope_changes_no_other_slug():
    for slug in BUCKET_BY_SLUG:
        if slug == "credit_card_payment":
            continue
        assert bucket_for_slug(slug, scope="bank") == bucket_for_slug(slug), slug


def test_internal_slugs_narrow_under_bank_scope():
    # What the footnote counts and what its drill-through lists are one set.
    assert internal_slugs_for_scope("bank") == frozenset({"self_transfer"})
    assert internal_slugs_for_scope() == INTERNAL_SLUGS
    assert internal_slugs_for_scope(None) == INTERNAL_SLUGS


def test_bucket_for_unknown_and_unmapped():
    assert bucket_for_slug(None) == "uncategorized"
    assert bucket_for_slug("unknown") == "uncategorized"
    assert bucket_for_slug("some_new_runtime_slug") == "uncategorized"


def test_label_helper():
    assert label_for_slug("cashback_rewards") == "Cashback Rewards"
    assert label_for_slug("emi_loan") == "EMI / Loan"  # override
    assert label_for_slug("groceries") == "Groceries"
    assert label_for_slug(None) == "(uncategorized)"  # NULL-category group
    assert label_for_slug("some_new_runtime_slug") == "unmapped: some_new_runtime_slug"


def test_unmapped_slug_is_logged(caplog):
    with caplog.at_level(logging.WARNING):
        bucket_for_slug("some_new_runtime_slug")
    assert any("unmapped" in r.message for r in caplog.records)


def test_expected_uncategorized_inputs_are_not_logged_as_drift(caplog):
    """The warning means "a slug appeared that the code map has never heard of" —
    a redeploy is owed. NULL and the 'unknown' sentinel are neither: they are the
    ordinary, expected inputs of the uncategorized line, arriving on every sweep.
    Warning on them would bury the one message that asks for an action under a
    steady stream of ones that do not."""
    with caplog.at_level(logging.WARNING):
        assert bucket_for_slug(None) == "uncategorized"
        assert bucket_for_slug("unknown") == "uncategorized"
        assert bucket_for_slug("") == "uncategorized"
    assert caplog.records == []
