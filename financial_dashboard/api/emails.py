"""Email JSON endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.emails import (
    DuplicateResolutionRequest,
    DuplicateResolutionResponse,
)
from financial_dashboard.services.duplicate_resolution import (
    DuplicateResolutionError,
    resolve_email_duplicate,
)

router = APIRouter()


@router.post("/emails/{email_id}/resolve-duplicate")
async def resolve_duplicate(
    email_id: int,
    payload: DuplicateResolutionRequest,
    session: AsyncSession = Depends(get_session),
) -> DuplicateResolutionResponse:
    try:
        return await resolve_email_duplicate(session, email_id, payload)
    except DuplicateResolutionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
