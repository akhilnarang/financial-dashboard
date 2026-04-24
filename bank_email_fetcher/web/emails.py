# ty: ignore
"""Email HTML routes."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from bank_email_fetcher.core.crypto import decrypt_credentials
from bank_email_fetcher.core.deps import get_session
from bank_email_fetcher.core.templating import get_templates
from bank_email_fetcher.db import (
    Account,
    BankStatementUpload,
    Card,
    Email,
    EmailSource,
    FetchRule,
    StatementUpload,
    Transaction,
)
from bank_email_fetcher.integrations.email.body import (
    _extract_html_body,
    _extract_text_body,
    _save_failed_email,
    load_or_fetch_raw_email,
)
from bank_email_fetcher.integrations.email.imap_gmail import _fetch_gmail_single_sync
from bank_email_fetcher.integrations.email.jmap_fastmail import (
    _fetch_fastmail_single_sync,
)
from bank_email_fetcher.schemas.emails import (
    ReparseAllFailedResponse,
    ReparseEmailResponse,
)
from bank_email_fetcher.services.emails import parse_email_by_kind
from bank_email_fetcher.services.linker import build_link_context, link_transaction
from bank_email_fetcher.services.settings import (
    get_telegram_chat_id,
    should_notify_transactions,
)
from bank_email_fetcher.services.telegram import (
    build_account_label,
    send_transaction_notification,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()


@router.get("/emails", response_class=HTMLResponse)
async def email_list(
    request: FastAPIRequest,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    bank: Annotated[str | None, Query(description="Filter by bank (via rule)")] = None,
    provider: Annotated[
        str | None, Query(description="Filter by email provider")
    ] = None,
    status: Annotated[
        str | None, Query(description="Filter by processing status")
    ] = None,
    date_from: Annotated[
        str | None, Query(description="Received on/after (YYYY-MM-DD)")
    ] = None,
    date_to: Annotated[
        str | None, Query(description="Received on/before (YYYY-MM-DD)")
    ] = None,
    q: Annotated[
        str | None, Query(description="Case-insensitive search over sender and subject")
    ] = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Email)
    needs_rule_join = bool(bank)
    if needs_rule_join:
        stmt = stmt.join(FetchRule, Email.rule_id == FetchRule.id).where(
            FetchRule.bank == bank
        )
    if provider:
        stmt = stmt.where(Email.provider == provider)
    if status:
        stmt = stmt.where(Email.status == status)
    if date_from:
        try:
            stmt = stmt.where(Email.received_at >= date.fromisoformat(date_from))
        except ValueError:
            pass
    if date_to:
        try:
            end_of_day = datetime.combine(
                date.fromisoformat(date_to), datetime.max.time()
            )
            stmt = stmt.where(Email.received_at <= end_of_day)
        except ValueError:
            pass
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Email.sender.ilike(like), Email.subject.ilike(like)))

    total_count = (
        await session.execute(select(func.count()).select_from(stmt.subquery()))
    ).scalar() or 0
    failed_count = (
        await session.execute(
            select(func.count(Email.id)).where(Email.status == "failed")
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

    stmt = stmt.order_by(Email.id.desc()).offset(offset).limit(page_size)
    emails = (await session.execute(stmt)).scalars().all()

    bank_result = await session.execute(select(FetchRule.bank).distinct())
    banks = sorted([row[0] for row in bank_result.all() if row[0]])
    provider_result = await session.execute(select(Email.provider).distinct())
    providers = sorted([row[0] for row in provider_result.all() if row[0]])

    filters = {
        "bank": bank,
        "provider": provider,
        "status": status,
        "date_from": date_from,
        "date_to": date_to,
        "q": q,
    }
    qs_items = {k: v for k, v in filters.items() if v}
    if page_size != 50:
        qs_items["page_size"] = page_size
    base_qs = urlencode(qs_items)

    return templates.TemplateResponse(
        request,
        "emails.html",
        {
            "active_page": "emails",
            "emails": emails,
            "banks": banks,
            "providers": providers,
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


@router.get("/emails/{email_id}/detail", response_class=HTMLResponse)
async def email_detail(
    email_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Email, Transaction)
        .outerjoin(Transaction, Transaction.email_id == Email.id)
        .where(Email.id == email_id)
    )
    row = result.first()
    if not row:
        return HTMLResponse("<p>Email not found.</p>", 404)
    email_row, txn = row
    return templates.TemplateResponse(
        request,
        "partials/email_detail.html",
        {
            "email": email_row,
            "txn": txn,
        },
    )


@router.get("/emails/{email_id}/original", response_class=HTMLResponse)
async def view_original_email(
    email_id: int,
    session: AsyncSession = Depends(get_session),
):
    email_row = await session.get(Email, email_id)
    if not email_row or not email_row.source_id or not email_row.remote_id:
        return HTMLResponse("<p>Original email not available.</p>", 404)
    source = await session.get(EmailSource, email_row.source_id)
    if not source:
        return HTMLResponse("<p>Email source not found.</p>", 404)

    creds = decrypt_credentials(source.credentials)

    if source.provider == "gmail":
        raw = await asyncio.to_thread(
            _fetch_gmail_single_sync,
            creds["user"],
            creds["app_password"],
            email_row.remote_id,
        )
    elif source.provider == "fastmail":
        raw = await asyncio.to_thread(
            _fetch_fastmail_single_sync, creds["token"], email_row.remote_id
        )
    else:
        return HTMLResponse("<p>Unknown provider.</p>", 400)

    if not raw:
        return HTMLResponse("<p>Could not fetch original email from provider.</p>", 502)

    html_body = _extract_html_body(raw)
    if not html_body:
        # Fallback to plain text
        text_body = _extract_text_body(raw)
        import html

        html_body = (
            f"<pre>{html.escape(text_body)}</pre>"
            if text_body
            else "<p>No content.</p>"
        )

    # Return with restrictive headers
    return HTMLResponse(
        html_body,
        headers={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data: https:;",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
        },
    )


@router.post("/emails/{email_id}/reparse", response_model=ReparseEmailResponse)
async def reparse_email(
    email_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReparseEmailResponse:
    """Re-parse a failed email, loading raw bytes from the spool or re-fetching
    from the provider if the spool has aged out.

    Returns JSON so the caller can update the UI without a full-page redirect.
    """
    email_row = await session.get(Email, email_id)
    if not email_row:
        raise HTTPException(status_code=404, detail="Email not found")

    rule = (
        await session.get(FetchRule, email_row.rule_id) if email_row.rule_id else None
    )

    if not rule:
        raise HTTPException(
            status_code=400, detail="No fetch rule associated with this email"
        )

    raw_bytes, fetch_error = await load_or_fetch_raw_email(email_row)
    if raw_bytes is None:
        raise HTTPException(
            status_code=404, detail=fetch_error or "Unable to load raw email"
        )

    error, txn_data, password_hint, stmt_result = await parse_email_by_kind(
        bank=rule.bank,
        email_kind=getattr(rule, "email_kind", None),
        raw_bytes=raw_bytes,
        subject=email_row.subject or "",
        source_id=email_row.source_id,
        log_ref=f"reparse:{email_id}",
    )

    if not txn_data and not stmt_result:
        # Parsing still fails — update error message so it's fresh, but keep
        # status=failed. Re-save the raw bytes to the spool so the next retry
        # doesn't have to hit the provider again until the cleanup cron evicts them.
        _save_failed_email(email_row.provider, email_row.message_id, raw_bytes)
        em = await session.get(Email, email_id)
        if em:
            em.error = error
            await session.commit()
        raise HTTPException(
            status_code=422,
            detail=error or "Parsing failed (no transaction or statement found)",
        )

    # Success — update the email row and create transaction if needed

    # Close the implicit read transaction opened earlier (from session.get at
    # the top of this handler) so session.begin() below doesn't collide with it.
    await session.rollback()

    async with session.begin():
        em = await session.get(Email, email_id)
        if not em:
            raise HTTPException(status_code=500, detail="Email disappeared")

        em.status = "parsed"
        em.error = None

        if stmt_result and stmt_result.get("statement_upload_id"):
            su_id = stmt_result["statement_upload_id"]
            su = await session.get(StatementUpload, su_id)
            if su:
                su.email_id = em.id
            else:
                logger.warning(
                    "StatementUpload %s disappeared during reparse of email %d",
                    su_id,
                    email_id,
                )
        elif stmt_result and stmt_result.get("bank_statement_upload_id"):
            su_id = stmt_result["bank_statement_upload_id"]
            su = await session.get(BankStatementUpload, su_id)
            if su:
                su.email_id = em.id
            else:
                logger.warning(
                    "BankStatementUpload %s disappeared during reparse of email %d",
                    su_id,
                    email_id,
                )

        txn_id = None
        duplicate_error: str | None = None
        if txn_data:
            try:
                async with session.begin_nested():
                    txn_row = Transaction(email_id=em.id, **txn_data)
                    session.add(txn_row)
                    await session.flush()
                    _link_ctx = await build_link_context(session)
                    link_transaction(_link_ctx, txn_row)
                    await session.flush()
                    txn_id = txn_row.id
                    account_obj = (
                        await session.get(Account, txn_row.account_id)
                        if txn_row.account_id
                        else None
                    )
                    card_obj = (
                        await session.get(Card, txn_row.card_id)
                        if txn_row.card_id
                        else None
                    )
                    txn_data["account_label"] = build_account_label(
                        account_obj, card_obj
                    )
                    txn_data["channel"] = txn_row.channel
            except IntegrityError:
                em.status = "skipped"
                em.error = "Duplicate transaction skipped because an identical transaction row already exists"
                duplicate_error = em.error

    if duplicate_error:
        raise HTTPException(status_code=409, detail=duplicate_error)

    # Send Telegram notification for the new transaction
    if txn_id and txn_data and should_notify_transactions():
        try:
            await send_transaction_notification(
                txn_id, txn_data, get_telegram_chat_id()
            )
        except Exception as tg_err:
            logger.warning(
                "Telegram notification failed for reparsed txn #%s: %s", txn_id, tg_err
            )

    msg = "Email re-parsed successfully"
    if stmt_result:
        stmt_kind = "Bank" if stmt_result.get("bank_statement_upload_id") else "CC"
        msg = f"{stmt_kind} statement re-processed (matched={stmt_result.get('matched', 0)}, imported={stmt_result.get('imported', 0)})"
    logger.info("Reparse of email %d succeeded: %s", email_id, msg)
    return ReparseEmailResponse(message=msg, new_status="parsed", txn_id=txn_id)


@router.post("/emails/reparse-all-failed", response_model=ReparseAllFailedResponse)
async def reparse_all_failed(
    session: AsyncSession = Depends(get_session),
) -> ReparseAllFailedResponse:
    """Re-parse all emails with status='failed', loading raw bytes from the
    spool or re-fetching from the provider when the spool has aged out."""

    succeeded = 0
    skipped = 0
    still_failed = 0

    rows = (
        await session.execute(
            select(Email, FetchRule)
            .outerjoin(FetchRule, Email.rule_id == FetchRule.id)
            .where(Email.status == "failed")
        )
    ).all()

    # Close the implicit read transaction opened by the select above so the
    # per-email session.begin() below can start cleanly.
    await session.rollback()

    for email_row, rule in rows:
        if not rule:
            still_failed += 1
            continue

        raw_bytes, fetch_error = await load_or_fetch_raw_email(email_row)
        if raw_bytes is None:
            logger.info(
                "Skipping bulk reparse for email %d: %s",
                email_row.id,
                fetch_error,
            )
            still_failed += 1
            continue

        error, txn_data, _, stmt_result = await parse_email_by_kind(
            bank=rule.bank,
            email_kind=getattr(rule, "email_kind", None),
            raw_bytes=raw_bytes,
            subject=email_row.subject or "",
            source_id=email_row.source_id,
            log_ref=f"bulk-reparse:{email_row.id}",
        )

        if not txn_data and not stmt_result:
            # Re-save to spool so the next retry doesn't re-fetch from the
            # provider (cleanup cron will evict after FAILED_SPOOL_MAX_AGE_DAYS).
            _save_failed_email(email_row.provider, email_row.message_id, raw_bytes)
            still_failed += 1
            continue

        was_skipped = False
        async with session.begin():
            em = await session.get(Email, email_row.id)
            if not em:
                continue
            em.status = "parsed"
            em.error = None

            if stmt_result and stmt_result.get("statement_upload_id"):
                su_id = stmt_result["statement_upload_id"]
                su = await session.get(StatementUpload, su_id)
                if su:
                    su.email_id = em.id
                else:
                    logger.warning(
                        "StatementUpload %s disappeared during bulk reparse of email %d",
                        su_id,
                        email_row.id,
                    )

            if txn_data:
                try:
                    async with session.begin_nested():
                        txn_row = Transaction(email_id=em.id, **txn_data)
                        session.add(txn_row)
                        await session.flush()
                        _link_ctx = await build_link_context(session)
                        link_transaction(_link_ctx, txn_row)
                        await session.flush()
                except IntegrityError:
                    em.status = "skipped"
                    em.error = "Duplicate transaction skipped"
                    was_skipped = True

        if was_skipped:
            skipped += 1
        else:
            succeeded += 1

    return ReparseAllFailedResponse(
        succeeded=succeeded,
        skipped=skipped,
        failed=still_failed,
    )
