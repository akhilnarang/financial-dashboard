"""Shared validation and normalization for JSON API query parameters."""

import datetime
from typing import NamedTuple

from financial_dashboard.exceptions import UnprocessableEntityException


class DateTimeBounds(NamedTuple):
    """Inclusive wall-clock bounds for date-only query parameters."""

    start: datetime.datetime | None
    end: datetime.datetime | None


def validate_date_range(
    date_from: datetime.date | None,
    date_to: datetime.date | None,
) -> None:
    """Reject an inverted optional date range."""
    if date_from is not None and date_to is not None and date_from > date_to:
        raise UnprocessableEntityException(detail="date_from must not exceed date_to")


def inclusive_datetime_bounds(
    date_from: datetime.date | None,
    date_to: datetime.date | None,
) -> DateTimeBounds:
    """Expand optional dates to inclusive bounds for naive DateTime columns."""
    validate_date_range(date_from, date_to)

    start = (
        datetime.datetime.combine(date_from, datetime.time.min) if date_from else None
    )
    end = datetime.datetime.combine(date_to, datetime.time.max) if date_to else None
    return DateTimeBounds(start, end)
