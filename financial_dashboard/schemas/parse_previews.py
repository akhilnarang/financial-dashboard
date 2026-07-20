"""Shared schemas for side-effect-free parser match evidence."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MatchEvidencePreview(BaseModel):
    """Bounded explanation of the candidates and gates used by the matcher."""

    path: Literal["none", "reference", "fuzzy", "am_pm_alias"]
    candidate_ids: Annotated[list[int], Field(max_length=10)]
    observed_candidate_count: Annotated[int, Field(ge=0)]
    candidate_ids_truncated: bool
    gates: Annotated[list[str], Field(max_length=10)]
    reason: Annotated[str, Field(max_length=64)]
