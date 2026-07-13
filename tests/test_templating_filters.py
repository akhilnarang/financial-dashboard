"""Unit tests for the Jinja money filters."""

from decimal import Decimal

import pytest

from financial_dashboard.core.templating import format_inr_exact


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Below a lakh the two conventions agree, so these only pin the basics.
        (Decimal("0"), "₹0.00"),
        (Decimal("999"), "₹999.00"),
        (Decimal("1000"), "₹1,000.00"),
        (Decimal("99999.5"), "₹99,999.50"),
        # From a lakh up, the groups above the last three digits are pairs.
        (Decimal("100000"), "₹1,00,000.00"),
        (Decimal("1234567.89"), "₹12,34,567.89"),
        (Decimal("-1234567.89"), "-₹12,34,567.89"),
        (Decimal("123456789.01"), "₹12,34,56,789.01"),
        (Decimal("1234567890"), "₹1,23,45,67,890.00"),
        # A contra credit is negative and must not lose its sign.
        (Decimal("-500"), "-₹500.00"),
        # Aggregations can carry more than two places; they round, not truncate.
        (Decimal("100000.005"), "₹1,00,000.01"),
        (None, "₹0.00"),
    ],
)
def test_format_inr_exact(value, expected):
    assert format_inr_exact(value) == expected


def test_format_inr_exact_is_not_western_grouping():
    assert format_inr_exact(Decimal("1234567.89")) != "₹1,234,567.89"


def test_format_inr_exact_rejects_float():
    """A float is not merely imprecise here, it is silently *wrong*: 2.675 is really
    2.67499999999999982..., so the half that ROUND_HALF_UP is supposed to round up is
    already gone by the time the value arrives and the paisa renders a unit low. The
    same amount as a Decimal rounds the way the docstring promises, so the fix is to
    refuse the float rather than to round it differently."""
    with pytest.raises(TypeError, match="not float"):
        format_inr_exact(2.675)

    assert format_inr_exact(Decimal("2.675")) == "₹2.68"

    # Not a blanket ban on numbers: the types that hold money exactly still work.
    assert format_inr_exact(Decimal("1234567.89")) == "₹12,34,567.89"
    assert format_inr_exact(1500) == "₹1,500.00"
    assert format_inr_exact(None) == "₹0.00"
