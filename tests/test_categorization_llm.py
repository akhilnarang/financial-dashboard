from financial_dashboard.services.categorization.llm import (
    NEEDS_REVIEW,
    LlmResult,
    build_prompt,
    parse_result,
)


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


def test_prompt_flags_dr_cr_as_a_direction_marker_for_banks_with_the_bug():
    for bank in ("indusind", "idfc", "sbi", "uboi"):
        prompt = build_prompt(
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
        assert "format note" in prompt
        assert "debit or credit" in prompt
        assert "healthcare" in prompt


def test_prompt_has_no_format_note_for_a_bank_without_a_hint():
    prompt = build_prompt(
        fields={
            "bank": "hdfc",
            "counterparty": "someone@axl",
            "raw_description": "some narration",
            "direction": "debit",
            "channel": "upi",
            "amount": "500",
            "currency": "INR",
        },
        examples=[],
        active_slugs=["groceries", "dining"],
    )
    assert "format note" not in prompt


def test_parse_result_clamps_and_defaults():
    r = parse_result(
        {"category": "groceries", "confidence": 1.5, "reason": "x"}, ["groceries"]
    )
    assert isinstance(r, LlmResult)
    assert r.slug == "groceries" and r.confidence == 1.0
    # unknown slug → needs_review
    bad = parse_result(
        {"category": "made_up", "confidence": 0.9, "reason": "y"}, ["groceries"]
    )
    assert bad.slug == NEEDS_REVIEW
    assert bad.reason == "invalid model category slug: made_up"
    malformed = parse_result(
        {
            "category": "groceries",
            "confidence": "NaN",
            "candidates": [{"category": "groceries", "confidence": "NaN"}],
        },
        ["groceries"],
    )
    assert malformed.confidence == malformed.candidates[0].confidence == 0.0
