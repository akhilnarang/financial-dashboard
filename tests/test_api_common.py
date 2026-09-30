import datetime

import pytest
from pydantic import BaseModel, ValidationError

from financial_dashboard.api.query import (
    inclusive_datetime_bounds,
    validate_date_range,
)
from financial_dashboard.exceptions import UnprocessableEntityException
from financial_dashboard.schemas.common import DatabaseId, DatabaseIdBatch


class DatabaseIdModel(BaseModel):
    """Test wrapper proving shared scalar and batch ID validation."""

    identifier: DatabaseId
    identifiers: DatabaseIdBatch


def test_database_id_types_require_positive_unique_ids():
    valid = DatabaseIdModel(identifier=2, identifiers=[2, 1])
    assert valid.identifier == 2
    assert valid.identifiers == [2, 1]

    with pytest.raises(ValidationError):
        DatabaseIdModel(identifier=0, identifiers=[1])
    with pytest.raises(ValidationError):
        DatabaseIdModel(identifier=1, identifiers=[1, 1])


def test_inclusive_datetime_bounds_expands_optional_dates():
    day = datetime.date(2030, 1, 2)

    bounds = inclusive_datetime_bounds(day, day)

    assert bounds.start == datetime.datetime(2030, 1, 2, 0, 0)
    assert bounds.end == datetime.datetime(2030, 1, 2, 23, 59, 59, 999999)


def test_validate_date_range_rejects_inverted_range():
    with pytest.raises(UnprocessableEntityException):
        validate_date_range(
            datetime.date(2030, 1, 3),
            datetime.date(2030, 1, 2),
        )
