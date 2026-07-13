from financial_dashboard.services.categorization.normalize import (
    normalize_counterparty,
    normalize_text,
    redact_names,
    redact_pii,
)


def test_normalize_text_collapses_and_strips():
    assert normalize_text("  X9   ACME-STORE!! ") == "x9 acme store"
    assert normalize_text(None) == ""


def test_normalize_counterparty_alnum_only():
    assert normalize_counterparty("ALE X QUINN DOE") == "alexquinndoe"
    assert normalize_counterparty(None) == ""


def test_redact_pii_masks_long_digits_and_phones():
    assert "1234567890123456" not in redact_pii("paid via card 1234567890123456")
    assert "9876543210" not in redact_pii("upi to 9876543210@okbank")
    # short numbers (amounts) are preserved
    assert "10" in redact_pii("amount 10 at shop")


def test_redact_names_collapses_full_name_from_listed_token():
    # all listed -> single marker
    assert redact_names("ALEX QUINN DOE", ("alex", "doe")) == "[redacted-name]"
    # only one part listed; unlisted middle/edge parts still absorbed
    assert redact_names("Bob Quinn Doe", ("doe",)) == "[redacted-name]"
    assert redact_names("Mr ALEX QUINN DO", ("alex",)) == "[redacted-name]"


def test_redact_names_keeps_surrounding_text():
    out = redact_names("received from ALEX QUINN DOE.", ("alex", "doe"))
    assert out == "received from [redacted-name]."


def test_redact_names_noops_without_match_or_token():
    assert redact_names("WWW ACMESTORE", ("alex",)) == "WWW ACMESTORE"
    assert redact_names(None, ("alex",)) == ""
    # tokens under 3 chars are ignored (avoid over-matching)
    assert redact_names("AB CD", ("ab",)) == "AB CD"


def test_redact_names_handles_punctuation_and_concatenation():
    # unlisted middle part (quinn) flanked by listed parts is absorbed even
    # with no whitespace or with punctuation separators
    assert redact_names("ALEXQUINSHDOE", ("alex", "doe")) == "[redacted-name]"
    assert redact_names("UPI/ALEX/QUINN/DOE", ("alex", "doe")) == "UPI/[redacted-name]"
    # name embedded in a UPI handle is redacted; the @vpa suffix survives
    assert redact_names("username@vpa", ("username",)) == "[redacted-name]@vpa"
