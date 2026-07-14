"""Unit tests for the mask grammar.

The linker's correctness rests on normalize_mask: it decides which characters
are digits, which hide a digit, which are cosmetic, and which mean "this is not
a mask I understand". Getting that grammar wrong attributes money to the wrong
account, so each class is pinned directly here rather than only through the
linker. Every number below is fabricated.
"""

import pytest

from financial_dashboard.core.masks import (
    mask_matches,
    normalize_mask,
    trailing_visible_digits,
)


@pytest.mark.parametrize("wildcard", ["X", "x", "*", "#"])
def test_every_wildcard_normalizes_to_one_hidden_digit(wildcard):
    """All four hide exactly one digit, so they must be interchangeable and must
    not change the length — position is what makes a mask matchable."""
    assert normalize_mask(f"73{wildcard * 6}3942") == "73XXXXXX3942"
    assert mask_matches(normalize_mask(f"73{wildcard * 6}3942"), "730055113942")


@pytest.mark.parametrize(
    "mask,expected",
    [
        ("4000 XXXX XXXX 1234", "4000XXXXXXXX1234"),  # spaces
        ("4000-XXXX-XXXX-1234", "4000XXXXXXXX1234"),  # dashes
        ("4000\tXXXX 1234", "4000XXXX1234"),  # tab
        ("4000\nXXXX 1234", "4000XXXX1234"),  # newline
        (" XX1234 ", "XX1234"),  # leading/trailing
    ],
)
def test_separators_are_dropped_not_rejected(mask, expected):
    """Whitespace and the group dash are cosmetic: they occupy no digit position,
    so they vanish rather than invalidating the mask."""
    assert normalize_mask(mask) == expected


@pytest.mark.parametrize("mask", ["12A3456", "12?3456", "XX12.34", "XX12/34", "abc"])
def test_an_unreadable_mask_is_rejected_not_flattened(mask):
    """A character we do not understand means we cannot say which account this
    denotes. Dropping it would INVENT an answer: "12A3456" would become "123456"
    and match an account ending in that run. Rejecting fails to match instead.

    '?' is deliberately not a wildcard — no parser emits it, and admitting it on
    speculation would widen the grammar with no evidence.
    """
    assert normalize_mask(mask) == ""
    # A rejected mask can never match anything.
    assert mask_matches(normalize_mask(mask), "999999123456") is False


def test_a_bare_number_is_its_own_mask():
    assert normalize_mask("730055113942") == "730055113942"


def test_empty_and_none():
    assert normalize_mask(None) == ""
    assert normalize_mask("") == ""


@pytest.mark.parametrize(
    "normalized,expected",
    [
        ("XX1234", 4),
        ("73XXXXXX3942", 4),
        ("1234", 4),
        ("123XXXXXXXXX", 0),  # digits are all LEADING — identifies only a prefix
        ("XXXX", 0),
        ("", 0),
    ],
)
def test_trailing_visible_digits_counts_only_the_right_edge(normalized, expected):
    """What identifies an account is the digits at the END. Leading digits are a
    bank/branch prefix that accounts share, so they must not count toward the
    minimum-signal guard."""
    assert trailing_visible_digits(normalized) == expected
