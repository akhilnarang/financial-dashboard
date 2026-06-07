"""Unit tests for core/masks.py."""

from financial_dashboard.core.masks import mask_digits, mask_last4


def test_mask_digits_strips_non_digits():
    assert mask_digits("XX4321") == "4321"
    assert mask_digits("4000 XXXX XXXX 4321") == "40004321"
    assert mask_digits("x0000") == "0000"


def test_mask_digits_empty_and_none():
    assert mask_digits(None) == ""
    assert mask_digits("") == ""
    assert mask_digits("XXXX") == ""


def test_mask_digits_strips_non_ascii_digits():
    # str.isdigit() accepts these; a card mask never contains them, so
    # mask_digits must drop them rather than treat them as significant.
    assert mask_digits("²³⁴⁵") == ""  # superscripts
    assert mask_digits("١٢٣٤") == ""  # Arabic-Indic
    assert mask_digits("１２３４") == ""  # fullwidth
    assert mask_digits("XX⁴12") == "12"  # mixed: only ASCII kept


def test_mask_last4_takes_last_four_digits():
    assert mask_last4("XX4321") == "4321"
    assert mask_last4("4000 XXXX XXXX 4321") == "4321"
    assert mask_last4("1234 XXXX XXXX 5678") == "5678"


def test_mask_last4_returns_none_below_four_digits_by_default():
    assert mask_last4("XX34") is None
    assert mask_last4("X9") is None
    assert mask_last4(None) is None
    assert mask_last4("XXXX") is None


def test_mask_last4_partial_returns_short_suffix():
    # partial=True keeps the short suffix a statement may show ("XX34").
    assert mask_last4("XX34", partial=True) == "34"
    assert mask_last4("X9", partial=True) == "9"
    # Still None when there are no digits at all.
    assert mask_last4("XXXX", partial=True) is None
    assert mask_last4(None, partial=True) is None
    # 4+ digits behave the same regardless of the flag.
    assert mask_last4("4000 XXXX XXXX 4321", partial=True) == "4321"
