import pytest

from financial_dashboard.services.assistant.intent_policy import (
    derive_merchant_pattern,
    is_direct_affirmative,
    merchant_rule_is_explicit,
)


def test_merchant_rule_requires_durable_current_message_and_category():
    assert not merchant_rule_is_explicit(
        "set this to groceries", "set this to groceries", "groceries"
    )
    assert merchant_rule_is_explicit(
        "Always categorize this as groceries",
        "Always categorize this as groceries",
        "groceries",
    )


def test_negated_durable_cue_cannot_authorize_merchant_rule():
    text = "I don't always buy from Acme, but put this in dining"
    assert not merchant_rule_is_explicit(text, text, "dining")


def test_pending_confirmation_accepts_only_direct_affirmative():
    assert is_direct_affirmative("yes")
    assert is_direct_affirmative("go ahead")
    assert not is_direct_affirmative("yes, and change the note")


@pytest.mark.parametrize(
    "value",
    [
        "upi payment to",
        "abc 12345678",
    ],
)
def test_merchant_pattern_rejects_generic_or_reference_like_values(value):
    with pytest.raises(ValueError):
        derive_merchant_pattern(value)


def test_merchant_pattern_is_server_derived_and_normalized():
    assert derive_merchant_pattern(" Amazon Fresh ") == "amazon fresh"
    assert derive_merchant_pattern("upi payment to Amazon") == "upi payment to amazon"
