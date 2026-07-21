"""Operational endpoints for configured email sources."""

from typing import Annotated

from fastapi import APIRouter, Path

from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.exceptions import NotFoundException
from financial_dashboard.schemas.common import DatabaseId
from financial_dashboard.schemas.sources import SourceTestResponse
from financial_dashboard.services.sources import (
    SourceNotFoundError,
    test_source_connectivity,
)

router = APIRouter()


@router.post("/sources/{source_id}/test")
async def test_source(
    source_id: Annotated[DatabaseId, Path()],
    session: AsyncSessionDep,
) -> SourceTestResponse:
    """Test one configured provider without returning stored credentials."""
    try:
        return await test_source_connectivity(session, source_id)
    except SourceNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
