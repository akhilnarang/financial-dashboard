from financial_dashboard.services.categorization.llm import (
    NEEDS_REVIEW,
    build_prompt,
    parse_result,
)
from financial_dashboard.services.categorization.normalize import redact_names


def test_prompt_lists_slugs_and_redacts():
    prompt = build_prompt(
        fields={
            "bank": "hdfc",
            "account_type": "credit_card",
            "email_type": "cc_transaction",
            "counterparty": "UPI to 9876543210",
            "raw_description": "card 1234567890123456 grocery",
            "direction": "debit",
            "channel": "upi",
            "amount": "500",
            "currency": "INR",
        },
        examples=[],
        active_slugs=["groceries", "dining"],
    )
    assert "groceries" in prompt and "dining" in prompt
    assert NEEDS_REVIEW in prompt
    assert "1234567890123456" not in prompt
    assert "9876543210" not in prompt
    assert "bank: hdfc" in prompt
    assert "account_type: credit_card" in prompt
    assert "email_type: cc_transaction" in prompt


def test_prompt_flags_dr_cr_only_for_banks_with_the_hint():
    def prompt_for(bank):
        return build_prompt(
            fields={
                "bank": bank,
                "counterparty": "someone@axl",
                "raw_description": "UPI/1234567890/DR/NAME/HDFC/someone@axl",
                "direction": "debit",
                "channel": "upi",
                "amount": "500",
                "currency": "INR",
            },
            examples=[],
            active_slugs=["groceries", "dining"],
        )

    prompt = prompt_for("indusind")
    assert "format note" in prompt
    assert "debit or credit" in prompt
    assert "healthcare" in prompt
    assert "format note" not in prompt_for("hdfc")


def test_parse_result_clamps_confidence():
    r = parse_result(
        {"category": "groceries", "confidence": 1.5, "reason": "x"}, ["groceries"]
    )
    assert r.slug == "groceries" and r.confidence == 1.0
    malformed = parse_result(
        {
            "category": "groceries",
            "confidence": "NaN",
            "candidates": [{"category": "groceries", "confidence": "NaN"}],
        },
        ["groceries"],
    )
    assert malformed.confidence == malformed.candidates[0].confidence == 0.0


def test_redact_names():
    # One listed part absorbs the unlisted middle/edge parts.
    assert redact_names("Bob Quinn Doe", ("doe",)) == "[redacted-name]"
    assert redact_names("Mr ALEX QUINN DO", ("alex",)) == "[redacted-name]"
    out = redact_names("received from ALEX QUINN DOE.", ("alex", "doe"))
    assert out == "received from [redacted-name]."
    # An unlisted middle part is absorbed with no whitespace or with
    # punctuation separators.
    assert redact_names("ALEXQUINSHDOE", ("alex", "doe")) == "[redacted-name]"
    assert redact_names("UPI/ALEX/QUINN/DOE", ("alex", "doe")) == "UPI/[redacted-name]"
    # A name in a UPI handle is redacted; the @vpa suffix survives.
    assert redact_names("username@vpa", ("username",)) == "[redacted-name]@vpa"
    assert redact_names("WWW ACMESTORE", ("alex",)) == "WWW ACMESTORE"
    assert redact_names(None, ("alex",)) == ""
    # Tokens under 3 chars are ignored.
    assert redact_names("AB CD", ("ab",)) == "AB CD"
