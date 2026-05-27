"""Web routes for /sms (HTML pages + JSON reparse endpoints)."""

import logging
from datetime import date as _date, datetime as _datetime
from typing import TYPE_CHECKING, Annotated, Literal, cast
from urllib.parse import urlencode

if TYPE_CHECKING:
    from financial_dashboard.services.txn_merge import EnrichmentDiff

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import SmsMessage, Transaction, async_session
from financial_dashboard.services.linker import build_link_context
from financial_dashboard.services.settings import (
    get_telegram_chat_id,
    should_notify_transactions,
)
from financial_dashboard.services.sms_pipeline import process_sms_row
from financial_dashboard.services.telegram import (
    send_enrichment_notification,
    send_transaction_notification,
)

logger = logging.getLogger(__name__)

router = APIRouter()
templates = (
    get_templates()
)  # matches the pattern in web/emails.py and web/bank_statements.py


class ReparseSmsResponse(BaseModel):
    message: str
    new_status: Literal["parsed", "enriched", "error", "skipped"]
    txn_id: int | None = None
    diff: list[str] | None = None


@router.post("/sms/{sms_id}/reparse", response_model=ReparseSmsResponse)
async def reparse_sms(
    sms_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReparseSmsResponse:
    sms = await session.get(SmsMessage, sms_id)
    if sms is None:
        raise HTTPException(404, "SMS not found")

    # Mirror the email-side dance at web/emails.py:314-318: close the
    # implicit read txn opened by session.get() above so the explicit
    # session.begin() below doesn't collide with it.
    await session.rollback()

    async with session.begin():
        sms = await session.get(SmsMessage, sms_id)
        if sms is None:
            raise HTTPException(500, "SMS disappeared")
        link_ctx = await build_link_context(session)
        outcome = await process_sms_row(session, sms, link_ctx)

    # Telegram dispatch (best-effort, post-commit).
    if should_notify_transactions():
        chat_id = get_telegram_chat_id()
        if outcome.primary_notification is not None:
            try:
                await send_transaction_notification(
                    outcome.transaction_id or 0,
                    outcome.primary_notification,
                    chat_id,
                    source="sms",
                )
            except Exception as exc:
                logger.warning("Reparse Telegram primary dispatch failed: %s", exc)
        if outcome.enrichment_notification is not None:
            txn_id, diff, txn_info = outcome.enrichment_notification
            try:
                await send_enrichment_notification(
                    txn_id, diff, chat_id, source="sms", txn_info=txn_info
                )
            except Exception as exc:
                logger.warning("Reparse Telegram enrichment dispatch failed: %s", exc)

    # CC payment marking — same hooks as POST /api/sms. A reparse of a
    # CC-payment-received SMS should still mark the statement paid.
    if outcome.pending_payment_check is not None:
        from financial_dashboard.services.reminders import check_payment_received

        try:
            await check_payment_received(*outcome.pending_payment_check)
        except Exception as exc:
            logger.warning(
                "Reparse payment-received check failed for SMS-derived txn: %s",
                exc,
            )

    if outcome.pending_disambiguation is not None:
        from financial_dashboard.services.telegram import (
            send_disambiguation_prompt,
        )

        try:
            await send_disambiguation_prompt(
                outcome.pending_disambiguation, get_telegram_chat_id()
            )
        except Exception as exc:
            logger.warning("Reparse disambiguation prompt failed: %s", exc)

    if outcome.status == "error":
        raise HTTPException(status_code=422, detail=sms.parse_error or "Parse error")

    diff_list = None
    if outcome.enrichment_notification is not None:
        _, diff_obj, _ = outcome.enrichment_notification
        # ProcessSmsOutcome.enrichment_notification's second element is
        # typed `object` to avoid a circular import in the dataclass;
        # the runtime type is always EnrichmentDiff. Cast here so the
        # attribute access types cleanly.
        diff_list = cast("EnrichmentDiff", diff_obj).changed_fields

    return ReparseSmsResponse(
        message=f"SMS #{sms_id} → {outcome.status}",
        new_status=outcome.status,
        txn_id=outcome.transaction_id,
        diff=diff_list,
    )


class ReparseAllSmsResponse(BaseModel):
    processed: int
    enriched: int
    still_error: int
    skipped: int


@router.post("/sms/reparse-all-failed", response_model=ReparseAllSmsResponse)
async def reparse_all_failed_sms(
    session: AsyncSession = Depends(get_session),
) -> ReparseAllSmsResponse:
    # 1. Read row IDs only.
    ids = (
        (
            await session.execute(
                select(SmsMessage.id).where(SmsMessage.status.in_(("pending", "error")))
            )
        )
        .scalars()
        .all()
    )
    await session.rollback()

    # Build link_context once for the whole batch.
    async with async_session() as s:
        link_ctx = await build_link_context(s)

    # Lazy-imported here to avoid a top-level cycle.
    from financial_dashboard.services.reminders import check_payment_received
    from financial_dashboard.services.telegram import send_disambiguation_prompt

    counts = {"processed": 0, "enriched": 0, "still_error": 0, "skipped": 0}
    chat_id_for_dispatch = (
        get_telegram_chat_id() if should_notify_transactions() else None
    )
    for sms_id in ids:
        try:
            async with async_session() as s:
                async with s.begin():
                    row = await s.get(SmsMessage, sms_id)
                    if row is None:
                        continue
                    outcome = await process_sms_row(s, row, link_ctx)
                # Bucket outside the txn block — row attrs are still loaded.
            match outcome.status:
                case "parsed":
                    counts["processed"] += 1
                case "enriched":
                    counts["processed"] += 1
                    counts["enriched"] += 1
                case "error":
                    counts["still_error"] += 1
                case "skipped":
                    counts["skipped"] += 1

            # Telegram primary + enrichment notifications fire on reparse
            # too, matching the single-reparse path. Best-effort; one row's
            # dispatch failure does not abort the batch.
            if chat_id_for_dispatch is not None:
                if outcome.primary_notification is not None:
                    try:
                        await send_transaction_notification(
                            outcome.transaction_id or 0,
                            outcome.primary_notification,
                            chat_id_for_dispatch,
                            source="sms",
                        )
                    except Exception as exc:
                        logger.warning(
                            "Bulk reparse Telegram primary failed for SMS %d: %s",
                            sms_id,
                            exc,
                        )
                if outcome.enrichment_notification is not None:
                    enrich_txn_id, diff, enrich_info = outcome.enrichment_notification
                    try:
                        await send_enrichment_notification(
                            enrich_txn_id,
                            diff,
                            chat_id_for_dispatch,
                            source="sms",
                            txn_info=enrich_info,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Bulk reparse Telegram enrichment failed for SMS %d: %s",
                            sms_id,
                            exc,
                        )

            # CC payment marking + disambiguation also still fire on reparse.
            if outcome.pending_payment_check is not None:
                try:
                    await check_payment_received(*outcome.pending_payment_check)
                except Exception as exc:
                    logger.warning(
                        "Bulk reparse payment-check failed for SMS %d: %s",
                        sms_id,
                        exc,
                    )
            if (
                outcome.pending_disambiguation is not None
                and chat_id_for_dispatch is not None
            ):
                try:
                    await send_disambiguation_prompt(
                        outcome.pending_disambiguation, chat_id_for_dispatch
                    )
                except Exception as exc:
                    logger.warning(
                        "Bulk reparse disambiguation prompt failed for SMS %d: %s",
                        sms_id,
                        exc,
                    )
        except Exception:
            logger.exception("Bulk reparse: row %d crashed", sms_id)
            counts["still_error"] += 1

    return ReparseAllSmsResponse(**counts)


@router.get("/sms", response_class=HTMLResponse)
async def sms_list(
    request: FastAPIRequest,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    bank: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    date_from: Annotated[str | None, Query()] = None,
    date_to: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query()] = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(SmsMessage)
    if bank:
        stmt = stmt.where(SmsMessage.bank == bank)
    if status:
        stmt = stmt.where(SmsMessage.status == status)
    if date_from:
        try:
            stmt = stmt.where(SmsMessage.received_at >= _date.fromisoformat(date_from))
        except ValueError:
            pass
    if date_to:
        try:
            end_of_day = _datetime.combine(
                _date.fromisoformat(date_to), _datetime.max.time()
            )
            stmt = stmt.where(SmsMessage.received_at <= end_of_day)
        except ValueError:
            pass
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(SmsMessage.sender.ilike(like), SmsMessage.body.ilike(like))
        )

    total_count = (
        await session.execute(select(func.count()).select_from(stmt.subquery()))
    ).scalar() or 0
    failed_count = (
        await session.execute(
            select(func.count(SmsMessage.id)).where(
                SmsMessage.status.in_(("pending", "error"))
            )
        )
    ).scalar() or 0
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    offset = (page - 1) * page_size
    page_window = sorted(
        set(
            [1]
            + list(range(max(1, page - 2), min(total_pages, page + 2) + 1))
            + [total_pages]
        )
    )

    stmt = stmt.order_by(SmsMessage.id.desc()).offset(offset).limit(page_size)
    rows = (await session.execute(stmt)).scalars().all()

    bank_result = await session.execute(select(SmsMessage.bank).distinct())
    banks = sorted([row[0] for row in bank_result.all() if row[0]])

    filters = {
        "bank": bank,
        "status": status,
        "date_from": date_from,
        "date_to": date_to,
        "q": q,
    }
    qs_items: dict[str, str] = {k: v for k, v in filters.items() if v}
    if page_size != 50:
        qs_items["page_size"] = str(page_size)
    base_qs = urlencode(qs_items)

    return templates.TemplateResponse(
        request,
        "sms.html",
        {
            "active_page": "sms",
            "sms_rows": rows,
            "banks": banks,
            "filters": filters,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "page_window": page_window,
            "failed_count": failed_count,
            "base_qs": base_qs,
        },
    )


async def _load_sms(
    session: AsyncSession, sms_id: int
) -> tuple[SmsMessage, Transaction | None] | None:
    """Load an SmsMessage with its linked Transaction (if any), or None.

    Looks up the linked Transaction via sms.transaction_id (not the
    reverse FK on Transaction.sms_message_id — that points at whichever
    SMS arrived first, which may be a different row).
    """
    sms = await session.get(SmsMessage, sms_id)
    if sms is None:
        return None
    txn = (
        await session.get(Transaction, sms.transaction_id)
        if sms.transaction_id
        else None
    )
    return sms, txn


@router.get("/sms/{sms_id}/detail", response_class=HTMLResponse)
async def sms_detail(
    sms_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    loaded = await _load_sms(session, sms_id)
    if loaded is None:
        return HTMLResponse("<p>SMS not found.</p>", 404)
    sms, txn = loaded
    return templates.TemplateResponse(
        request,
        "partials/sms_detail.html",
        {"sms": sms, "txn": txn},
    )


@router.get("/sms/{sms_id}", response_class=HTMLResponse)
async def sms_page(
    sms_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    loaded = await _load_sms(session, sms_id)
    if loaded is None:
        return HTMLResponse("<p>SMS not found.</p>", 404)
    sms, txn = loaded
    return templates.TemplateResponse(
        request,
        "sms_page.html",
        {
            "active_page": "sms",
            "sms": sms,
            "txn": txn,
        },
    )
