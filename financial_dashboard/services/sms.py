"""SMS ingest service.

Two entry points:

- ``ingest_sms(session, payload)`` — public, commits internally.
- ``_ingest_sms_no_commit(session, payload)`` — used by ``POST /api/sms``,
  which wraps the raw insert + the parse/merge pipeline in one outer
  transaction so a parse crash can't leave a permanent ``pending`` row
  that the dedup constraint then blocks from ever being re-POSTed.
"""

from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import SmsMessage
from financial_dashboard.schemas.sms import SmsIngestRequest


class SmsIngestResult(NamedTuple):
    message: SmsMessage
    stored: bool


async def _ingest_sms_no_commit(
    session: AsyncSession, payload: SmsIngestRequest
) -> SmsIngestResult:
    """Insert (or detect duplicate) without committing the outer txn.

    Uses ``session.begin_nested()`` for the conflict-recovery so an
    IntegrityError doesn't poison the outer transaction the caller owns.
    """
    row = SmsMessage(
        bank=payload.bank,
        sender=payload.sender,
        body=payload.body,
        received_at=payload.received_at,
    )
    try:
        async with session.begin_nested():
            session.add(row)
            await session.flush()
    except IntegrityError:
        existing = await session.scalar(
            select(SmsMessage).where(
                SmsMessage.sender == payload.sender,
                SmsMessage.received_at == payload.received_at,
                SmsMessage.body == payload.body,
            )
        )
        if existing is None:
            raise RuntimeError(
                "_ingest_sms_no_commit: IntegrityError but no matching row"
            )
        return SmsIngestResult(existing, False)
    return SmsIngestResult(row, True)


async def ingest_sms(
    session: AsyncSession, payload: SmsIngestRequest
) -> SmsIngestResult:
    """Persist an SMS payload (commits its own transaction)."""
    async with session.begin():
        return await _ingest_sms_no_commit(session, payload)
