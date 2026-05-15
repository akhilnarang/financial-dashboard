from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.sources import SourceTestResponse
from financial_dashboard.services.sources import (
    SourceNotFoundError,
    test_source_connectivity,
)

router = APIRouter()


@router.post("/sources/{source_id}/test", response_model=SourceTestResponse)
async def test_source(
    source_id: int,
    session: AsyncSession = Depends(get_session),
) -> SourceTestResponse:
    try:
        return await test_source_connectivity(session, source_id)
    except SourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
