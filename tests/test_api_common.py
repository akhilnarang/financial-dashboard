import datetime

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from financial_dashboard.api.query import (
    inclusive_datetime_bounds,
    validate_date_range,
)
from financial_dashboard.exceptions import (
    BadRequestException,
    ConflictException,
    FailedDependencyException,
    InternalServerException,
    NotFoundException,
    PayloadTooLargeException,
    UnprocessableEntityException,
)
from financial_dashboard.schemas.common import DatabaseId, DatabaseIdBatch


class DatabaseIdModel(BaseModel):
    """Test wrapper proving shared scalar and batch ID validation."""

    identifier: DatabaseId
    identifiers: DatabaseIdBatch


@pytest.mark.parametrize(
    ("exception_type", "status_code"),
    [
        (BadRequestException, 400),
        (NotFoundException, 404),
        (ConflictException, 409),
        (PayloadTooLargeException, 413),
        (UnprocessableEntityException, 422),
        (FailedDependencyException, 424),
        (InternalServerException, 500),
    ],
)
def test_named_api_exceptions_set_status_and_detail(exception_type, status_code):
    error = exception_type(detail="Synthetic error")

    assert isinstance(error, HTTPException)
    assert error.status_code == status_code
    assert error.detail == "Synthetic error"


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
