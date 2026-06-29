from financial_dashboard.services.categorization.gemini import (
    NEEDS_REVIEW,
    GeminiResult,
    build_prompt,
    parse_result,
)


def test_prompt_lists_slugs_and_redacts():
    prompt = build_prompt(
        fields={
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


def test_parse_result_clamps_and_defaults():
    r = parse_result(
        {"category": "groceries", "confidence": 1.5, "reason": "x"}, ["groceries"]
    )
    assert isinstance(r, GeminiResult)
    assert r.slug == "groceries" and r.confidence == 1.0
    # unknown slug → needs_review
    bad = parse_result(
        {"category": "made_up", "confidence": 0.9, "reason": "y"}, ["groceries"]
    )
    assert bad.slug == NEEDS_REVIEW
