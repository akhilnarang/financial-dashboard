"""Unit tests for the Jinja money filters."""

from decimal import Decimal

import pytest

from financial_dashboard.core.templating import format_inr_exact


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # From a lakh up, the groups above the last three digits are pairs.
        (Decimal("123456789.01"), "₹12,34,56,789.01"),
        # A contra credit is negative and must not lose its sign.
        (Decimal("-1234567.89"), "-₹12,34,567.89"),
        # Aggregations can carry more than two places; they round, not truncate.
        (Decimal("100000.005"), "₹1,00,000.01"),
    ],
)
def test_format_inr_exact(value, expected):
    assert format_inr_exact(value) == expected


def test_format_inr_exact_rejects_float():
    """A float 2.675 is really 2.67499..., so it would render a paisa low."""
    with pytest.raises(TypeError):
        format_inr_exact(2.675)
    assert format_inr_exact(Decimal("2.675")) == "₹2.68"
