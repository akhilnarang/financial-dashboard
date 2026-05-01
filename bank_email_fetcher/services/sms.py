"""SMS ingest service."""

from sqlalchemy.ext.asyncio import AsyncSession

from bank_email_fetcher.db import SmsMessage
from bank_email_fetcher.schemas.sms import SmsIngestRequest


async def ingest_sms(
    session: AsyncSession, payload: SmsIngestRequest
) -> tuple[SmsMessage, bool]:
    """Persist an SMS payload.

    Returns (row, stored). stored=True means a new row was inserted;
    stored=False means a duplicate was detected and the existing row is returned.
    """
    row = SmsMessage(
        bank=payload.bank,
        sender=payload.sender,
        body=payload.body,
        received_at=payload.received_at,
    )
    session.add(row)
    await session.commit()
    return row, True
