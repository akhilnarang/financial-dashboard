"""Read and ingest endpoints for SMS source records.

SMS ingestion parses and commits synchronously. Telegram dispatch runs after
commit so notification failure cannot leave database state half-written.
"""

import datetime
import logging
from typing import Annotated

from fastapi import APIRouter, Path, Query, Response

from financial_dashboard.api.query import inclusive_datetime_bounds
from financial_dashboard.core.deps import SessionDep
from financial_dashboard.exceptions import NotFoundException
from financial_dashboard.schemas import sms as sms_schemas
from financial_dashboard.schemas.common import DatabaseId
from financial_dashboard.schemas.sms import SmsIngestRequest
from financial_dashboard.services.linker import build_link_context
from financial_dashboard.services.parse_previews import preview_sms_parse
from financial_dashboard.services.settings import (
    get_telegram_chat_id,
    should_notify_transactions,
)
from financial_dashboard.services.sms import _ingest_sms_no_commit
from financial_dashboard.services.sms_pipeline import process_sms_row
from financial_dashboard.services.sms_reads import (
    get_sms_by_ids,
    get_sms_detail,
    list_sms,
)
from financial_dashboard.services.telegram import (
    send_enrichment_notification,
    send_transaction_notification,
)
from financial_dashboard.web.sms import reparse_sms as reparse_sms_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sms")
async def sms_list(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    sms_id: Annotated[DatabaseId | None, Query()] = None,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    status: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    transaction_id: Annotated[DatabaseId | None, Query()] = None,
    parser_type: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
) -> sms_schemas.SmsListResponse:
    """List a bounded page of SMS metadata without raw message bodies."""
    date_bounds = inclusive_datetime_bounds(date_from, date_to)

    return await list_sms(
        session,
        limit=limit,
        offset=offset,
        sms_id=sms_id,
        bank=bank,
        status=status,
        transaction_id=transaction_id,
        parser_type=parser_type,
        date_from=date_bounds.start,
        date_to=date_bounds.end,
    )


@router.post("/sms/batch")
async def sms_batch(
    payload: sms_schemas.SmsBatchRequest,
    session: SessionDep,
) -> sms_schemas.SmsBatchResponse:
    """Return SMS summaries for an ordered, explicit set of IDs."""
    return await get_sms_by_ids(session, payload.ids)


@router.get("/sms/{sms_id}")
async def sms_detail(
    sms_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: SessionDep,
) -> sms_schemas.SmsDetailResponse:
    """Return one SMS with a bounded raw body and linked transactions."""
    response.headers["Cache-Control"] = "no-store"
    if sms := await get_sms_detail(session, sms_id):
        return sms

    raise NotFoundException(detail="SMS not found")


@router.post("/sms/{sms_id}/parse-preview")
async def sms_parse_preview(
    sms_id: Annotated[DatabaseId, Path()],
    session: SessionDep,
) -> sms_schemas.SmsParsePreviewResponse:
    """Parse one stored SMS and project merge behavior without side effects."""
    if preview := await preview_sms_parse(session, sms_id):
        return preview

    raise NotFoundException(detail="SMS not found")


@router.post("/sms/{sms_id}/reparse")
async def sms_reparse(
    sms_id: Annotated[DatabaseId, Path()],
    session: SessionDep,
    force_new: Annotated[bool, Query()] = False,
) -> sms_schemas.ReparseSmsResponse:
    """Run the canonical SMS reparse behavior through the JSON API."""
    return await reparse_sms_service(sms_id, force_new, session)


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
    session: SessionDep,
) -> Response:
    """Store, parse, and reconcile one forwarded SMS before notifying."""
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
        from financial_dashboard.services.telegram import send_disambiguation_prompt

        try:
            await send_disambiguation_prompt(
                outcome.pending_disambiguation, get_telegram_chat_id()
            )
        except Exception as exc:
            logger.warning("Disambiguation prompt failed: %s", exc)

    return Response(status_code=201)
