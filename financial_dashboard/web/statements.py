"""Statement HTML routes."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from cc_parser.cli import write_transactions_csv
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    Request as FastAPIRequest,
    UploadFile,
)
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.config import get_fernet
from financial_dashboard.core.crypto import decrypt_credentials
from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    Email,
    EmailSource,
    FetchRule,
    PaymentStatus,
    StatementUpload,
    Transaction,
)
from financial_dashboard.integrations.email.body import (
    load_or_fetch_raw_email,
)
from financial_dashboard.integrations.email.imap_gmail import _fetch_gmail_single_sync
from financial_dashboard.integrations.email.jmap_fastmail import (
    _fetch_fastmail_single_sync,
)
from financial_dashboard.services.accounts import (
    retry_password_required_statements as accounts_retry_password_required_statements,
)
from financial_dashboard.services.reminders import init_payment_tracking
from financial_dashboard.services.statements.bank import process_bank_statement_email
from financial_dashboard.services.statements.cc import (
    _parse_pdf_bytes_sync,
    enrich_matched_transactions,
    extract_pdf_from_email,
    group_recon_by_person,
    import_missing_cc_txns,
    parse_cc_amount,
    parse_statement,
    process_statement_email,
    reconcile_statement,
    reconciliation_from_json as cc_reconciliation_from_json,
    reconciliation_to_json,
)
from financial_dashboard.services.statements.shared import (
    retry_bank_statement_upload,
    retry_cc_statement_upload,
)
from financial_dashboard.web.forms import (
    STATEMENTS_DIR,
    _safe_upload_filename,
    _unlink_statement_file,
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


@router.get("/statements", response_class=HTMLResponse)
async def statements_list(
    request: FastAPIRequest,
    type: Annotated[
        str | None, Query(description="Filter by statement type: cc or bank")
    ] = None,
    bank: Annotated[str | None, Query(description="Filter by bank name")] = None,
    account_id: Annotated[str | None, Query(description="Filter by account ID")] = None,
    status: Annotated[str | None, Query(description="Filter by upload status")] = None,
    date_from: Annotated[
        str | None, Query(description="Uploaded on/after (YYYY-MM-DD)")
    ] = None,
    date_to: Annotated[
        str | None, Query(description="Uploaded on/before (YYYY-MM-DD)")
    ] = None,
    session: AsyncSession = Depends(get_session),
):
    def _apply_common_filters(stmt, model):
        if bank:
            stmt = stmt.where(model.bank == bank)
        if account_id:
            try:
                stmt = stmt.where(model.account_id == int(account_id))
            except ValueError:
                pass
        if status:
            stmt = stmt.where(model.status == status)
        if date_from:
            try:
                stmt = stmt.where(model.created_at >= date.fromisoformat(date_from))
            except ValueError:
                pass
        if date_to:
            try:
                end_of_day = datetime.combine(
                    date.fromisoformat(date_to), datetime.max.time()
                )
                stmt = stmt.where(model.created_at <= end_of_day)
            except ValueError:
                pass
        return stmt

    if type == "bank":
        cc_uploads = []
    else:
        cc_stmt = _apply_common_filters(select(StatementUpload), StatementUpload)
        cc_stmt = cc_stmt.order_by(StatementUpload.created_at.desc())
        cc_uploads = (await session.execute(cc_stmt)).scalars().all()

    if type == "cc":
        bank_uploads = []
    else:
        bank_stmt = _apply_common_filters(
            select(BankStatementUpload), BankStatementUpload
        )
        bank_stmt = bank_stmt.order_by(BankStatementUpload.created_at.desc())
        bank_uploads = (await session.execute(bank_stmt)).scalars().all()

    cc_accounts = (
        (
            await session.execute(
                select(Account)
                .where(Account.type == "credit_card", Account.active.is_(True))
                .order_by(Account.bank, Account.label)
            )
        )
        .scalars()
        .all()
    )
    bank_accounts = (
        (
            await session.execute(
                select(Account)
                .where(Account.type == "bank_account", Account.active.is_(True))
                .order_by(Account.bank, Account.label)
            )
        )
        .scalars()
        .all()
    )

    cc_banks = list(
        (await session.execute(select(StatementUpload.bank).distinct())).all()
    )
    bank_banks = list(
        (await session.execute(select(BankStatementUpload.bank).distinct())).all()
    )
    banks = sorted({row[0] for row in cc_banks + bank_banks if row[0]})

    # Tag each upload with its type so templates can distinguish them
    for u in cc_uploads:
        u._statement_type = "cc"
    for u in bank_uploads:
        u._statement_type = "bank"
    uploads = sorted(
        [*cc_uploads, *bank_uploads],
        key=lambda u: u.created_at or datetime.min,
        reverse=True,
    )

    # Build JSON for cascading account dropdown (bank -> accounts)
    accounts_by_bank: dict[str, list] = {}
    for a in [*cc_accounts, *bank_accounts]:
        accounts_by_bank.setdefault(a.bank, []).append(
            {
                "id": a.id,
                "label": a.label,
                "type": a.type,
            }
        )

    # Map account id -> password hint, for inline display in the upload form
    account_hints = {
        a.id: a.statement_password_hint or "" for a in [*cc_accounts, *bank_accounts]
    }

    filters = {
        "type": type,
        "bank": bank,
        "account_id": account_id,
        "status": status,
        "date_from": date_from,
        "date_to": date_to,
    }

    return templates.TemplateResponse(
        request,
        "statements.html",
        {
            "active_page": "statements",
            "uploads": uploads,
            "cc_accounts": cc_accounts,
            "bank_accounts": bank_accounts,
            "banks": banks,
            "accounts_json": json.dumps(accounts_by_bank),
            "account_hints_json": json.dumps(account_hints),
            "filters": filters,
        },
    )


@router.post("/statements/upload")
async def statement_upload(
    request: FastAPIRequest,
    account_id: int = Form(...),
    password: str = Form(""),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    account = await session.get(Account, account_id)
    if not account or account.type != "credit_card":
        return RedirectResponse(url="/statements", status_code=303)

    # Save PDF to disk
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_upload_filename(file.filename)
    file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
    content = await file.read()
    file_path.write_bytes(content)

    # Parse the PDF
    try:
        parsed = await asyncio.to_thread(
            parse_statement, file_path, password or None, account.bank
        )
    except Exception as e:
        error_msg = str(e)
        is_encrypted = "encrypt" in error_msg.lower() or "password" in error_msg.lower()
        upload = StatementUpload(
            account_id=account_id,
            bank=account.bank,
            filename=safe_name,
            file_path=str(file_path),
            status="password_required" if is_encrypted else "parse_error",
            error=error_msg,
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id
        return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)

    # Reconcile against DB transactions for this account
    db_txns = (
        (
            await session.execute(
                select(Transaction).where(Transaction.account_id == account_id)
            )
        )
        .scalars()
        .all()
    )

    db_txns = list(db_txns)
    recon = reconcile_statement(parsed, db_txns, account_id)

    # Enrich matched DB transactions with statement narrations
    await enrich_matched_transactions(recon)

    # Create upload and auto-import missing transactions
    upload = StatementUpload(
        account_id=account_id,
        bank=parsed.bank,
        filename=safe_name,
        file_path=str(file_path),
        status="parsed",
        card_number=parsed.card_number,
        statement_name=parsed.name,
        due_date=parsed.due_date,
        total_amount_due=parsed.statement_total_amount_due,
        parsed_txn_count=len(recon["matched"]) + len(recon["missing"]),
        matched_count=len(recon["matched"]),
        missing_count=len(recon["missing"]),
        reconciliation_data=reconciliation_to_json(recon),
    )
    session.add(upload)
    await session.flush()

    imported = len(
        await import_missing_cc_txns(session, upload, parsed, account, recon)
    )

    upload.imported_count = imported
    upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
    upload.reconciliation_data = reconciliation_to_json(recon)
    if upload.missing_count == 0:
        upload.status = "imported"
    await session.commit()
    upload_id = upload.id

    await init_payment_tracking(upload_id)

    return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)


@router.get("/statements/{upload_id}", response_class=HTMLResponse)
async def statement_detail(
    upload_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    if not (upload := await session.get(StatementUpload, upload_id)):
        return HTMLResponse("<p>Statement not found.</p>", 404)

    recon = None
    person_groups = []
    card_summaries = []
    if upload.reconciliation_data:
        recon = cc_reconciliation_from_json(upload.reconciliation_data)
        person_groups = group_recon_by_person(recon)
        card_summaries = recon.get("card_summaries", [])

    return templates.TemplateResponse(
        request,
        "statement_reconcile.html",
        {
            "active_page": "statements",
            "upload": upload,
            "recon": recon,
            "person_groups": person_groups,
            "card_summaries": card_summaries,
            "error": request.query_params.get("error"),
        },
    )


@router.get("/statements/{upload_id}/csv")
async def statement_csv_download(
    upload_id: int,
    session: AsyncSession = Depends(get_session),
):
    upload = await session.get(StatementUpload, upload_id)
    if not upload:
        return HTMLResponse("<p>Statement not found.</p>", 404)

    def _csv_export_fail(msg: str):
        logger.warning("CSV export failed for statement %d: %s", upload_id, msg)
        return RedirectResponse(
            url=f"/statements/{upload_id}?{urlencode({'error': msg})}",
            status_code=303,
        )

    if upload.source_kind == "email_summary":
        return _csv_export_fail(
            "This statement was parsed from the email body — "
            "upload a PDF to export CSV."
        )

    if (
        not (file_path := upload.file_path)
        or not (pdf_path := Path(file_path)).exists()
    ):
        return _csv_export_fail(
            "PDF file missing; CSV export requires the saved statement PDF."
        )

    password = None
    if account := await session.get(Account, upload.account_id):
        bank = account.bank
        if encrypted_password := account.statement_password:
            try:
                password = get_fernet().decrypt(encrypted_password.encode()).decode()
            except Exception:
                pass
    else:
        bank = upload.bank

    try:
        parsed = await asyncio.to_thread(parse_statement, pdf_path, password, bank)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / f"statement-{upload_id}.csv"
            await asyncio.to_thread(write_transactions_csv, parsed, csv_path)
            csv_bytes = csv_path.read_bytes()
    except Exception as e:
        return _csv_export_fail(f"CSV export failed: {e}")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="statement-{upload_id}.csv"'
        },
    )


@router.post("/statements/{upload_id}/retry")
async def statement_retry(
    upload_id: int,
    password: str = Form(...),
    save_password: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    if not (upload := await session.get(StatementUpload, upload_id)):
        return RedirectResponse(url="/statements", status_code=303)
    account_id = upload.account_id

    # Only persist the password to the account if the retry succeeded — a
    # wrong password shouldn't overwrite a previously-good one. Saving the
    # password also unlocks any other password_required statements on the
    # same account (the coordinator skips this upload since its status is
    # no longer password_required after the retry above).
    if await retry_cc_statement_upload(upload_id, password) and save_password == "1":
        encrypted = get_fernet().encrypt(password.encode()).decode()
        if account := await session.get(Account, account_id):
            account.statement_password = encrypted
            await session.commit()
            logger.info("Saved statement password for account %s", account.label)
        await accounts_retry_password_required_statements(
            session,
            account_id,
            password,
            retry_cc_upload=retry_cc_statement_upload,
            retry_bank_upload=retry_bank_statement_upload,
        )

    return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)


@router.post("/statements/{upload_id}/delete")
async def statement_delete(
    upload_id: int,
    session: AsyncSession = Depends(get_session),
):
    upload = await session.get(StatementUpload, upload_id)
    if not upload:
        return RedirectResponse(url="/statements", status_code=303)
    await session.execute(
        update(Transaction)
        .where(Transaction.statement_upload_id == upload_id)
        .values(statement_upload_id=None)
    )
    _unlink_statement_file(upload.file_path)
    await session.delete(upload)
    await session.commit()

    return RedirectResponse(url="/statements", status_code=303)


@router.post("/statements/{upload_id}/payment")
async def statement_payment(
    upload_id: int,
    action: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    """Manually toggle a CC statement's payment status (mirrors the Telegram button).

    action=mark_paid   -> set PAID, stamp payment_paid_at, fill payment_paid_amount
                         from total_amount_due if parseable.
    action=mark_unpaid -> revert to UNPAID and replay reminders. Preserves any
                         real partial payment amount (from bank auto-detection)
                         so history isn't lost; only clears the manual full-pay
                         marker. No-op if there is no due date tracked.
    """

    upload = await session.get(StatementUpload, upload_id)
    if not upload:
        return RedirectResponse(url="/statements", status_code=303)

    if action == "mark_paid":
        if upload.payment_status != PaymentStatus.PAID:
            upload.payment_status = PaymentStatus.PAID
            upload.payment_paid_at = datetime.now(timezone.utc)
            if upload.total_amount_due:
                try:
                    upload.payment_paid_amount = parse_cc_amount(
                        upload.total_amount_due
                    )
                except ValueError, InvalidOperation:
                    pass
            await session.commit()
    elif action == "mark_unpaid":
        if upload.payment_status is not None:
            was_partial = upload.payment_status == PaymentStatus.PARTIALLY_PAID
            upload.payment_status = (
                PaymentStatus.PARTIALLY_PAID if was_partial else PaymentStatus.UNPAID
            )
            upload.payment_paid_at = None
            if not was_partial:
                upload.payment_paid_amount = Decimal(0)
            upload.payment_sent_offsets = "[]"
            upload.payment_last_reminded_at = None
            await session.commit()

    return RedirectResponse(url="/statements", status_code=303)


@router.post("/statements/{upload_id}/reprocess")
async def statement_reprocess(
    upload_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Re-parse the saved CC statement PDF and rebuild reconciliation data."""
    upload = await session.get(StatementUpload, upload_id)
    if not upload:
        return RedirectResponse(url="/statements", status_code=303)
    if upload.source_kind == "email_summary":
        return RedirectResponse(
            url=f"/statements/{upload_id}?"
            + urlencode(
                {
                    "error": "This statement was parsed from the email body — "
                    "upload a PDF to reconcile transactions."
                }
            ),
            status_code=303,
        )
    account_id = upload.account_id
    file_path = upload.file_path
    email_id = upload.email_id

    # Try to get password if needed
    password = None
    account = await session.get(Account, account_id)
    if account and account.statement_password:
        try:
            password = (
                get_fernet().decrypt(account.statement_password.encode()).decode()
            )
        except Exception:
            pass

    def _reprocess_fail(msg: str):
        logger.warning("Reprocess failed for statement %d: %s", upload_id, msg)
        return RedirectResponse(
            url=f"/statements/{upload_id}?{urlencode({'error': msg})}",
            status_code=303,
        )

    bank = account.bank if account else upload.bank
    pdf_path = Path(file_path) if file_path else None
    if pdf_path and pdf_path.exists():
        try:
            parsed = await asyncio.to_thread(parse_statement, pdf_path, password, bank)
        except Exception as e:
            return _reprocess_fail(f"Parse error: {e}")
    elif email_id:
        email_row = await session.get(Email, email_id)
        if not email_row or not email_row.source_id or not email_row.remote_id:
            return _reprocess_fail("Original email not available for re-fetch")
        source = await session.get(EmailSource, email_row.source_id)
        if not source:
            return _reprocess_fail("Email source not found")
        remote_id = email_row.remote_id
        provider = source.provider
        encrypted_creds = source.credentials

        try:
            creds = decrypt_credentials(encrypted_creds)
        except Exception:
            return _reprocess_fail("Email source credential decryption failed")

        if provider == "gmail":
            raw = await asyncio.to_thread(
                _fetch_gmail_single_sync,
                creds["user"],
                creds["app_password"],
                remote_id,
            )
        elif provider == "fastmail":
            raw = await asyncio.to_thread(
                _fetch_fastmail_single_sync, creds["token"], remote_id
            )
        else:
            raw = None

        if not raw:
            return _reprocess_fail("Could not fetch email from provider")

        pdfs = extract_pdf_from_email(raw)
        if not pdfs:
            return _reprocess_fail("No PDF attachment found in re-fetched email")

        try:
            parsed = await asyncio.to_thread(
                _parse_pdf_bytes_sync, pdfs[0][1], password, bank
            )
        except Exception as e:
            return _reprocess_fail(f"Parse error: {e}")
    else:
        return _reprocess_fail("PDF file missing and no linked email to re-fetch from")

    db_txns = (
        (
            await session.execute(
                select(Transaction).where(Transaction.account_id == account_id)
            )
        )
        .scalars()
        .all()
    )

    db_txns = list(db_txns)
    recon = reconcile_statement(parsed, db_txns, account_id)
    await enrich_matched_transactions(recon)

    upload = await session.get(StatementUpload, upload_id)
    if upload is None:
        return RedirectResponse(url="/statements", status_code=303)
    upload.bank = parsed.bank
    upload.card_number = parsed.card_number
    upload.statement_name = parsed.name
    if (
        upload.due_date != parsed.due_date
        or upload.total_amount_due != parsed.statement_total_amount_due
    ):
        upload.payment_status = None
        upload.payment_paid_amount = Decimal(0)
        upload.payment_paid_at = None
        upload.payment_sent_offsets = "[]"
        upload.payment_last_reminded_at = None

    upload.due_date = parsed.due_date
    upload.total_amount_due = parsed.statement_total_amount_due

    newly_imported = len(
        await import_missing_cc_txns(session, upload, parsed, account, recon)
    )

    upload.parsed_txn_count = len(recon["matched"]) + len(recon["missing"])
    upload.matched_count = len(recon["matched"])
    upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
    upload.imported_count = (upload.imported_count or 0) + newly_imported
    upload.reconciliation_data = reconciliation_to_json(recon)
    if upload.missing_count == 0:
        upload.status = "imported"
    elif newly_imported > 0:
        upload.status = "partial_import"
    await session.commit()

    await init_payment_tracking(upload_id)

    return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)


@router.post("/statements/reprocess-failed")
async def statements_reprocess_failed(
    session: AsyncSession = Depends(get_session),
):
    """Reprocess failed emails that have PDF attachments as statements (CC or bank)."""
    failed_emails = (
        await session.execute(
            select(Email, FetchRule)
            .join(FetchRule, Email.rule_id == FetchRule.id)
            .where(Email.status == "failed")
        )
    ).all()

    processed = 0
    for email_row, rule in failed_emails:
        raw_bytes, _fetch_error = await load_or_fetch_raw_email(email_row)
        if raw_bytes is None:
            continue

        pdfs = extract_pdf_from_email(raw_bytes)
        if not pdfs:
            continue

        # Try CC statement first, then bank account statement
        result = None
        try:
            result = await process_statement_email(
                rule.bank,
                raw_bytes,
                email_row.subject or "",
                source_id=email_row.source_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to reprocess email %s as CC statement: %s", email_row.id, e
            )

        if result is None:
            try:
                result = await process_bank_statement_email(
                    rule.bank,
                    raw_bytes,
                    email_row.subject or "",
                    source_id=email_row.source_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to reprocess email %s as bank statement: %s",
                    email_row.id,
                    e,
                )

        if result:
            em = await session.get(Email, email_row.id)
            if em:
                em.status = "parsed"
                em.error = None
            if result.get("statement_upload_id"):
                su = await session.get(StatementUpload, result["statement_upload_id"])
                if su:
                    su.email_id = email_row.id
            elif result.get("bank_statement_upload_id"):
                su = await session.get(
                    BankStatementUpload, result["bank_statement_upload_id"]
                )
                if su:
                    su.email_id = email_row.id
            await session.commit()
            processed += 1

    logger.info("Reprocessed %d failed emails as statements", processed)
    return RedirectResponse(url="/statements", status_code=303)
