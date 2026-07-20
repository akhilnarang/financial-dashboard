"""Read and duplicate-resolution endpoints for email source records."""

import datetime
from typing import Annotated

from fastapi import APIRouter, Path, Query, Response

from financial_dashboard.api.query import inclusive_datetime_bounds
from financial_dashboard.core.deps import SessionDep
from financial_dashboard.exceptions import ApiException, NotFoundException
from financial_dashboard.schemas import emails as email_schemas
from financial_dashboard.schemas.common import DatabaseId
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
from financial_dashboard.services.parse_previews import (
    EmailParsePreviewError,
    preview_email_parse,
)
from financial_dashboard.web.emails import reparse_email as reparse_email_service

router = APIRouter()


@router.get("/emails")
async def emails_list(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    email_id: Annotated[DatabaseId | None, Query()] = None,
    source_id: Annotated[DatabaseId | None, Query()] = None,
    rule_id: Annotated[DatabaseId | None, Query()] = None,
    provider: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    status: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    email_kind: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    transaction_id: Annotated[DatabaseId | None, Query()] = None,
    parser_type: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    direction: Annotated[str | None, Query(min_length=1, max_length=16)] = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    query: Annotated[str | None, Query(alias="q", min_length=1, max_length=128)] = None,
) -> email_schemas.EmailListResponse:
    """List a bounded page of email metadata without raw message bodies."""
    date_bounds = inclusive_datetime_bounds(date_from, date_to)

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
        date_from=date_bounds.start,
        date_to=date_bounds.end,
        query=query,
    )


@router.post("/emails/batch")
async def emails_batch(
    payload: email_schemas.EmailBatchRequest,
    session: SessionDep,
) -> email_schemas.EmailBatchResponse:
    """Return email summaries for an ordered, explicit set of IDs."""
    return await get_emails_by_ids(session, payload.ids)


@router.get("/emails/{email_id}/raw")
async def email_raw(
    email_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: SessionDep,
) -> email_schemas.EmailRawResponse:
    """Load one bounded raw email body without allowing response caching."""
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_email_raw(session, email_id)
    except EmailRawReadError as exc:
        raise ApiException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/emails/{email_id}")
async def email_detail(
    email_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: SessionDep,
) -> email_schemas.EmailDetailResponse:
    """Return one email's bounded metadata and linked provenance."""
    response.headers["Cache-Control"] = "no-store"
    if email := await get_email_detail(session, email_id):
        return email

    raise NotFoundException(detail="Email not found")


@router.post("/emails/{email_id}/parse-preview")
async def email_parse_preview(
    email_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: SessionDep,
) -> email_schemas.EmailParsePreviewResponse:
    """Refetch and parse one stored email without database or notification writes."""
    response.headers["Cache-Control"] = "no-store"
    try:
        if preview := await preview_email_parse(session, email_id):
            return preview
    except EmailParsePreviewError as exc:
        raise ApiException(status_code=exc.status_code, detail=str(exc)) from exc

    raise NotFoundException(detail="Email not found")


@router.post("/emails/{email_id}/reparse")
async def email_reparse(
    email_id: Annotated[DatabaseId, Path()],
    session: SessionDep,
    force_new: Annotated[bool, Query()] = False,
) -> email_schemas.ReparseEmailResponse:
    """Run the canonical email reparse behavior through the JSON API."""
    return await reparse_email_service(email_id, force_new, session)


@router.post("/emails/{email_id}/resolve-duplicate")
async def resolve_duplicate(
    email_id: Annotated[DatabaseId, Path()],
    payload: DuplicateResolutionRequest,
    session: SessionDep,
) -> DuplicateResolutionResponse:
    """Preview or apply explicit enrichment of a deferred duplicate email."""
    try:
        return await resolve_email_duplicate(session, email_id, payload)
    except DuplicateResolutionError as exc:
        raise ApiException(status_code=exc.status_code, detail=str(exc)) from exc
