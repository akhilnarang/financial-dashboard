"""SMS ingest endpoint."""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from bank_email_fetcher.core.deps import get_session
from bank_email_fetcher.schemas.sms import SmsIngestRequest
from bank_email_fetcher.services.sms import ingest_sms

router = APIRouter()


@router.post("/sms")
async def post_sms(
    payload: SmsIngestRequest,
    session: AsyncSession = Depends(get_session),
) -> Response:
    _row, stored = await ingest_sms(session, payload)
    return Response(status_code=201 if stored else 204)
