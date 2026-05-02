"""SMS ingest endpoint."""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from bank_email_fetcher.core.deps import get_session
from bank_email_fetcher.schemas.sms import SmsIngestRequest
from bank_email_fetcher.services.sms import ingest_sms

router = APIRouter()


@router.post(
    "/sms",
    status_code=201,
    response_class=Response,
    responses={
        201: {"description": "SMS stored"},
        204: {"description": "Duplicate — existing row, no change"},
    },
)
async def post_sms(
    payload: SmsIngestRequest,
    session: AsyncSession = Depends(get_session),
) -> Response:
    _, stored = await ingest_sms(session, payload)
    return Response(status_code=201 if stored else 204)
