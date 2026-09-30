import datetime

import pytest

from financial_dashboard.core.dates import financial_year_start, quarter_start


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (datetime.date(2026, 3, 31), datetime.date(2026, 1, 1)),
        (datetime.date(2026, 4, 1), datetime.date(2026, 4, 1)),
        (datetime.date(2026, 12, 31), datetime.date(2026, 10, 1)),
    ],
)
def test_quarter_start_snaps_to_the_first_day_of_the_calendar_quarter(day, expected):
    assert quarter_start(day) == expected


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (datetime.date(2026, 4, 1), datetime.date(2026, 4, 1)),
        (datetime.date(2026, 3, 31), datetime.date(2025, 4, 1)),
    ],
)
def test_financial_year_start_rolls_back_for_january_to_march(day, expected):
    assert financial_year_start(day) == expected
