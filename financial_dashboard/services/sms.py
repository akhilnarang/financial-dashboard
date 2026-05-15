"""SMS ingest service."""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import SmsMessage
from financial_dashboard.schemas.sms import SmsIngestRequest


async def ingest_sms(
    session: AsyncSession, payload: SmsIngestRequest
) -> tuple[SmsMessage, bool]:
    """Persist an SMS payload.

    Returns (row, stored). stored=True means a new row was inserted;
    stored=False means a duplicate was detected (UNIQUE constraint on
    (sender, received_at, body)) and the existing row is returned. The
    existing row is NOT updated — see spec §4 "dedup is best-effort".
    """
    row = SmsMessage(
        bank=payload.bank,
        sender=payload.sender,
        body=payload.body,
        received_at=payload.received_at,
    )
    session.add(row)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await session.scalar(
            select(SmsMessage).where(
                SmsMessage.sender == payload.sender,
                SmsMessage.received_at == payload.received_at,
                SmsMessage.body == payload.body,
            )
        )
        if existing is None:
            # Should be unreachable: the UNIQUE constraint guarantees a
            # matching row exists when IntegrityError fires for this insert.
            raise RuntimeError(
                "ingest_sms: IntegrityError raised but no matching row found for dedup"
            )
        return existing, False
    return row, True
