import pytest

from financial_dashboard.services.categorization.polarity import resolve_direction


@pytest.mark.parametrize(
    "slug, direction, expected, changed",
    [
        ("interest", "debit", "expense", True),
        ("repayment", "credit", "repayment", False),
        ("self_transfer", "credit", "self_transfer", False),
    ],
)
def test_resolve_direction(slug, direction, expected, changed):
    assert resolve_direction(slug, direction) == (expected, changed)
