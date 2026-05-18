"""SMS ingest endpoint.

Synchronous pipeline: parse + merge + DB commit happen inside the
request; Telegram dispatch is ``await``ed after commit so a slow send
is visible to the forwarder but cannot leave the DB in a half-state.
"""

import logging

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.sms import SmsIngestRequest
from financial_dashboard.services.linker import build_link_context
from financial_dashboard.services.settings import (
    get_telegram_chat_id,
    should_notify_transactions,
)
from financial_dashboard.services.sms import _ingest_sms_no_commit
from financial_dashboard.services.sms_pipeline import process_sms_row
from financial_dashboard.services.telegram import (
    send_enrichment_notification,
    send_transaction_notification,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/sms",
    status_code=201,
    response_class=Response,
    responses={
        201: {"description": "SMS stored and processed"},
        204: {"description": "Duplicate — existing row, no change"},
    },
)
async def post_sms(
    payload: SmsIngestRequest,
    session: AsyncSession = Depends(get_session),
) -> Response:
    primary: dict | None = None
    enrichment: tuple[int, object, dict] | None = None
    outcome = None

    async with session.begin():
        sms_row, stored = await _ingest_sms_no_commit(session, payload)
        if not stored:
            return Response(status_code=204)
        link_ctx = await build_link_context(session)
        outcome = await process_sms_row(session, sms_row, link_ctx)
        primary = outcome.primary_notification
        enrichment = outcome.enrichment_notification

    # Telegram dispatch after commit — `await`ed inline so a slow send is
    # visible to the forwarder but cannot leave the DB in a half-state.
    if should_notify_transactions() and outcome is not None:
        chat_id = get_telegram_chat_id()
        if primary is not None:
            try:
                await send_transaction_notification(
                    outcome.transaction_id or 0, primary, chat_id, source="sms"
                )
            except Exception as exc:
                logger.warning("Telegram primary dispatch failed: %s", exc)
        if enrichment is not None:
            txn_id, diff, txn_info = enrichment
            try:
                await send_enrichment_notification(
                    txn_id, diff, chat_id, source="sms", txn_info=txn_info
                )
            except Exception as exc:
                logger.warning("Telegram enrichment dispatch failed: %s", exc)

    if outcome is not None and outcome.pending_payment_check is not None:
        from financial_dashboard.services.reminders import check_payment_received

        try:
            await check_payment_received(*outcome.pending_payment_check)
        except Exception as exc:
            logger.warning("Payment-received check failed for SMS-derived txn: %s", exc)

    if outcome is not None and outcome.pending_disambiguation is not None:
        from financial_dashboard.services.telegram import (
            send_disambiguation_prompt,
        )

        try:
            await send_disambiguation_prompt(
                outcome.pending_disambiguation, get_telegram_chat_id()
            )
        except Exception as exc:
            logger.warning("Disambiguation prompt failed: %s", exc)

    return Response(status_code=201)
