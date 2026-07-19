"""Email JSON endpoints."""

import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas import emails as email_schemas
from financial_dashboard.schemas.emails import (
    DuplicateResolutionRequest,
    DuplicateResolutionResponse,
)
from financial_dashboard.services.duplicate_resolution import (
    DuplicateResolutionError,
    resolve_email_duplicate,
)
from financial_dashboard.services.email_reads import (
    EmailRawReadError,
    get_email_detail,
    get_email_raw,
    get_emails_by_ids,
    list_emails,
)

router = APIRouter()
_DB_ID_MAX = 9_223_372_036_854_775_807


@router.get("/emails")
async def emails_list(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    email_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    source_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    rule_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    provider: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    status: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    email_kind: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    transaction_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    parser_type: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    direction: Annotated[str | None, Query(min_length=1, max_length=16)] = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    query: Annotated[str | None, Query(alias="q", min_length=1, max_length=128)] = None,
    session: AsyncSession = Depends(get_session),
) -> email_schemas.EmailListResponse:
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not exceed date_to")
    return await list_emails(
        session,
        limit=limit,
        offset=offset,
        email_id=email_id,
        source_id=source_id,
        rule_id=rule_id,
        provider=provider,
        status=status,
        bank=bank,
        email_kind=email_kind,
        transaction_id=transaction_id,
        parser_type=parser_type,
        direction=direction,
        date_from=datetime.datetime.combine(date_from, datetime.time.min)
        if date_from is not None
        else None,
        date_to=datetime.datetime.combine(date_to, datetime.time.max)
        if date_to is not None
        else None,
        query=query,
    )


@router.post("/emails/batch")
async def emails_batch(
    payload: email_schemas.EmailBatchRequest,
    session: AsyncSession = Depends(get_session),
) -> email_schemas.EmailBatchResponse:
    return await get_emails_by_ids(session, payload.ids)


@router.get("/emails/{email_id}/raw")
async def email_raw(
    email_id: Annotated[int, Path(ge=1, le=_DB_ID_MAX)],
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> email_schemas.EmailRawResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_email_raw(session, email_id)
    except EmailRawReadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/emails/{email_id}")
async def email_detail(
    email_id: Annotated[int, Path(ge=1, le=_DB_ID_MAX)],
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> email_schemas.EmailDetailResponse:
    response.headers["Cache-Control"] = "no-store"
    email = await get_email_detail(session, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")
    return email


@router.post("/emails/{email_id}/resolve-duplicate")
async def resolve_duplicate(
    email_id: Annotated[int, Path(ge=1, le=_DB_ID_MAX)],
    payload: DuplicateResolutionRequest,
    session: AsyncSession = Depends(get_session),
) -> DuplicateResolutionResponse:
    try:
        return await resolve_email_duplicate(session, email_id, payload)
    except DuplicateResolutionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
