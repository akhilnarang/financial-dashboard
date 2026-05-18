"""Email HTML routes."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError, MultipleResultsFound
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.crypto import decrypt_credentials
from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    Card,
    Email,
    EmailSource,
    FetchRule,
    StatementUpload,
    Transaction,
)
from financial_dashboard.integrations.email.body import (
    _extract_html_body,
    _extract_text_body,
    _save_failed_email,
    load_or_fetch_raw_email,
)
from financial_dashboard.integrations.email.imap_gmail import _fetch_gmail_single_sync
from financial_dashboard.integrations.email.jmap_fastmail import (
    _fetch_fastmail_single_sync,
)
from financial_dashboard.schemas.emails import (
    ReparseAllFailedResponse,
    ReparseEmailResponse,
)
from financial_dashboard.services.cc_disambiguation import (
    resolve_cc_payment_account,
    should_auto_reconcile_statement,
)
from financial_dashboard.services.emails import parse_email_by_kind
from financial_dashboard.services.linker import build_link_context, link_transaction
from financial_dashboard.services.reminders import check_payment_received
from financial_dashboard.services.settings import (
    get_telegram_chat_id,
    should_notify_transactions,
)
from financial_dashboard.services.telegram import (
    build_account_label,
    send_disambiguation_prompt,
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
    qs_items: dict[str, str] = {k: v for k, v in filters.items() if v}
    if page_size != 50:
        qs_items["page_size"] = str(page_size)
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


async def _load_email(
    session: AsyncSession, email_id: int
) -> tuple[Email, Transaction | None] | None:
    """Load an Email with its linked Transaction (if any), or None."""
    result = await session.execute(
        select(Email, Transaction)
        .outerjoin(Transaction, Transaction.email_id == Email.id)
        .where(Email.id == email_id)
    )
    row = result.first()
    if not row:
        return None
    return tuple(row)  # type: ignore[return-value]


@router.get("/emails/{email_id}/detail", response_class=HTMLResponse)
async def email_detail(
    email_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    loaded = await _load_email(session, email_id)
    if loaded is None:
        return HTMLResponse("<p>Email not found.</p>", 404)
    email_row, txn = loaded
    return templates.TemplateResponse(
        request,
        "partials/email_detail.html",
        {"email": email_row, "txn": txn},
    )


@router.get("/emails/{email_id}", response_class=HTMLResponse)
async def email_page(
    email_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    loaded = await _load_email(session, email_id)
    if loaded is None:
        return HTMLResponse("<p>Email not found.</p>", 404)
    email_row, txn = loaded
    return templates.TemplateResponse(
        request,
        "email_page.html",
        {
            "active_page": "emails",
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
        pending_payment_check: tuple[int, int, Decimal] | None = None
        pending_disambiguation: dict | None = None
        if txn_data:
            try:
                async with session.begin_nested():
                    # Upsert: if a transaction is already attached to this
                    # email (the common case when reparse is invoked to
                    # pick up a downstream fix like the new CC
                    # disambiguation), update it in place rather than
                    # inserting a duplicate. Account/card FKs are cleared
                    # so the linker re-runs from scratch on the refreshed
                    # parser fields.
                    existing = (
                        await session.execute(
                            select(Transaction).where(Transaction.email_id == em.id)
                        )
                    ).scalar_one_or_none()
                    # `was_orphaned` distinguishes "fix a historical
                    # unlinked row" (account_id was None → the original
                    # parse never fired check_payment_received, so
                    # firing now is correct) from "re-apply an
                    # already-processed payment" (account_id was set →
                    # the statement was already credited, so firing
                    # again would double-count).
                    was_orphaned = existing is None or existing.account_id is None
                    if existing is not None:
                        for key, value in txn_data.items():
                            setattr(existing, key, value)
                        existing.account_id = None
                        existing.card_id = None
                        txn_row = existing
                    else:
                        txn_row = Transaction(email_id=em.id, **txn_data)
                        session.add(txn_row)
                    await session.flush()
                    _link_ctx = await build_link_context(session)
                    link_transaction(_link_ctx, txn_row)
                    await session.flush()
                    # Maskless CC bill-payment: try amount-match against
                    # open statements; else queue a Telegram prompt.
                    pending_disambiguation = await resolve_cc_payment_account(
                        session, txn_row
                    )
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
                    # Only auto-reconcile when the original parse did
                    # NOT already credit the statement. Two cases qualify:
                    #   - Fresh insert (no prior txn for this email).
                    #   - Upsert of a previously-orphaned txn (prior
                    #     account_id was None, so check_payment_received
                    #     never fired the first time round — this is the
                    #     "fix a historical orphan via reparse" workflow).
                    # If the prior row was already linked, the statement
                    # was already credited; re-firing would double-count
                    # payment_paid_amount on a PARTIALLY_PAID statement.
                    # The redundant `account_id is not None` is for ty —
                    # should_auto_reconcile_statement already guarantees
                    # it at runtime but ty can't narrow helper calls.
                    if (
                        was_orphaned
                        and should_auto_reconcile_statement(txn_row)
                        and txn_row.account_id is not None
                    ):
                        pending_payment_check = (
                            txn_row.id,
                            txn_row.account_id,
                            txn_row.amount,
                        )
            except IntegrityError:
                em.status = "skipped"
                em.error = "Duplicate transaction skipped because an identical transaction row already exists"
                duplicate_error = em.error
            except MultipleResultsFound:
                # The upsert lookup assumes 1:1 between Email and
                # Transaction. If a historical data quirk produced two
                # txns for one email, surface that loudly instead of
                # silently picking one — the operator needs to merge or
                # delete the duplicates first.
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Email {email_id} has more than one attached "
                        f"transaction; resolve the duplicates manually "
                        f"before reparsing."
                    ),
                )

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

    # Mirror the polling pipeline (services/emails.py:513-522): credit
    # transactions against an account may satisfy a pending statement payment.
    # Runs after the Telegram notification so the user sees the txn first.
    if pending_payment_check:
        try:
            await check_payment_received(*pending_payment_check)
        except Exception as exc:
            logger.warning(
                "Payment-received check failed for reparsed txn %s: %s",
                pending_payment_check[0],
                exc,
            )

    if pending_disambiguation is not None:
        try:
            await send_disambiguation_prompt(
                pending_disambiguation, get_telegram_chat_id()
            )
        except Exception as exc:
            logger.warning(
                "CC disambiguation prompt failed for reparsed txn %s: %s",
                pending_disambiguation.get("txn_id"),
                exc,
            )

    msg = "Email re-parsed successfully"
    if stmt_result:
        stmt_kind = "Bank" if stmt_result.get("bank_statement_upload_id") else "CC"
        if stmt_result.get("summary_only"):
            msg = f"{stmt_kind} statement summary created from email body"
        else:
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

    # Detach the loaded rows before closing the read transaction. Without
    # this, the rollback expires the ORM attributes; the next access (e.g.
    # email_row.provider in load_or_fetch_raw_email) would trigger an async
    # lazy-load with no greenlet attached and raise MissingGreenlet.
    # Detached instances keep their already-loaded simple-column values,
    # which is all we need until session.get(Email, ...) re-attaches inside
    # the per-iteration begin block.
    session.expunge_all()
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
        pending_payment_check: tuple[int, int, Decimal] | None = None
        pending_disambiguation: dict | None = None
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
                        # Upsert: failed emails reaching the bulk-reparse
                        # path rarely have an attached txn already, but
                        # mirror the single-email reparse logic so a
                        # mixed batch (e.g. some rows previously parsed
                        # and re-failed) doesn't insert duplicates.
                        existing = (
                            await session.execute(
                                select(Transaction).where(Transaction.email_id == em.id)
                            )
                        ).scalar_one_or_none()
                        was_orphaned = existing is None or existing.account_id is None
                        if existing is not None:
                            for key, value in txn_data.items():
                                setattr(existing, key, value)
                            existing.account_id = None
                            existing.card_id = None
                            txn_row = existing
                        else:
                            txn_row = Transaction(email_id=em.id, **txn_data)
                            session.add(txn_row)
                        await session.flush()
                        _link_ctx = await build_link_context(session)
                        link_transaction(_link_ctx, txn_row)
                        await session.flush()
                        # Maskless CC bill-payment: try amount-match
                        # against open statements; else queue a prompt.
                        pending_disambiguation = await resolve_cc_payment_account(
                            session, txn_row
                        )
                        # Only auto-reconcile when the original parse
                        # did not already credit the statement (either a
                        # fresh insert or an upsert of a previously-
                        # orphaned txn). See the per-email reparse path
                        # for the full rationale.
                        # The redundant account_id check is for ty.
                        if (
                            was_orphaned
                            and should_auto_reconcile_statement(txn_row)
                            and txn_row.account_id is not None
                        ):
                            pending_payment_check = (
                                txn_row.id,
                                txn_row.account_id,
                                txn_row.amount,
                            )
                except IntegrityError:
                    em.status = "skipped"
                    em.error = "Duplicate transaction skipped"
                    was_skipped = True
                except MultipleResultsFound:
                    # Skip this row but keep the batch going — the
                    # operator can deal with the duplicate manually.
                    em.status = "skipped"
                    em.error = (
                        "More than one attached transaction; resolve "
                        "manually before reparsing."
                    )
                    was_skipped = True

        if pending_payment_check:
            try:
                await check_payment_received(*pending_payment_check)
            except Exception as exc:
                logger.warning(
                    "Payment-received check failed for bulk-reparsed txn %s: %s",
                    pending_payment_check[0],
                    exc,
                )

        if pending_disambiguation is not None:
            try:
                await send_disambiguation_prompt(
                    pending_disambiguation, get_telegram_chat_id()
                )
            except Exception as exc:
                logger.warning(
                    "CC disambiguation prompt failed for bulk-reparsed txn %s: %s",
                    pending_disambiguation.get("txn_id"),
                    exc,
                )

        if was_skipped:
            skipped += 1
        else:
            succeeded += 1

    return ReparseAllFailedResponse(
        succeeded=succeeded,
        skipped=skipped,
        failed=still_failed,
    )
