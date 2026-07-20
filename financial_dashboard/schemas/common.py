"""Reusable validation types shared by JSON API schemas and parameters."""

from typing import Annotated

from pydantic import AfterValidator, Field

DatabaseId = Annotated[int, Field(ge=1)]


def _require_unique_ids(values: list[DatabaseId]) -> list[DatabaseId]:
    """Reject ambiguous batch requests while preserving caller-supplied order."""
    if len(values) != len(set(values)):
        raise ValueError("ids must be unique")
    return values


DatabaseIdBatch = Annotated[
    list[DatabaseId],
    Field(min_length=1, max_length=100),
    AfterValidator(_require_unique_ids),
]
