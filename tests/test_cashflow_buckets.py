import logging

from financial_dashboard.services.cashflow.buckets import (
    BUCKET_BY_SLUG,
    INTERNAL_SLUGS,
    bucket_for_slug,
    internal_slugs_for_scope,
    label_for_slug,
)
from financial_dashboard.services.categorization.polarity import (
    EXPENSE_SLUGS,
    INCOME_SLUGS,
)
from financial_dashboard.services.categorization.vocabulary import SEED_CATEGORIES


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
    assert BUCKET_BY_SLUG["repayment"] == "transfers_in"


def test_only_card_payment_changes_bucket_with_scope():
    # Over the bank, the card bill is when the cash leaves. Over every account
    # it is internal: counting it and the swipes would charge a rupee twice.
    assert bucket_for_slug("credit_card_payment", scope="bank") == "expense"
    assert bucket_for_slug("credit_card_payment") == "internal"
    for slug in BUCKET_BY_SLUG:
        if slug != "credit_card_payment":
            assert bucket_for_slug(slug, scope="bank") == bucket_for_slug(slug), slug


def test_internal_slugs_narrow_under_bank_scope():
    # What the footnote counts and what its drill-through lists are one set.
    assert internal_slugs_for_scope("bank") == frozenset(
        {"self_transfer", "passthrough", "personal_loan"}
    )
    assert internal_slugs_for_scope() == INTERNAL_SLUGS
    assert INTERNAL_SLUGS == frozenset(
        {"self_transfer", "passthrough", "personal_loan", "credit_card_payment"}
    )


def test_bucket_for_unknown_and_unmapped():
    assert bucket_for_slug(None) == "uncategorized"
    assert bucket_for_slug("unknown") == "uncategorized"
    assert bucket_for_slug("some_new_runtime_slug") == "uncategorized"


def test_label_helper():
    assert label_for_slug("emi_loan") == "EMI / Loan"  # override
    assert label_for_slug("groceries") == "Groceries"
    assert label_for_slug(None) == "(uncategorized)"  # NULL-category group
    assert label_for_slug("some_new_runtime_slug") == "unmapped: some_new_runtime_slug"


def test_only_an_unmapped_slug_is_logged_as_drift(caplog):
    """NULL, blank and 'unknown' are expected inputs, so they do not warn."""
    with caplog.at_level(logging.WARNING):
        for slug in (None, "unknown", ""):
            bucket_for_slug(slug)
        assert caplog.records == []
        bucket_for_slug("some_new_runtime_slug")
    assert any("unmapped" in r.message for r in caplog.records)
