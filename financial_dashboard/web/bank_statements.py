"""Bank statement HTML routes."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Request as FastAPIRequest,
    UploadFile,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.config import get_fernet
from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    Transaction,
)
from financial_dashboard.services.accounts import (
    retry_password_required_statements as accounts_retry_password_required_statements,
)
from financial_dashboard.services.linker import build_link_context, link_transaction
from financial_dashboard.services.snapshots import emit_bank_snapshot
from financial_dashboard.services.statements.bank import (
    _last4,
    _parse_amount,
    _parse_date,
    enrich_matched_transactions,
    parse_bank_statement,
    reconcile_bank_statement,
    reconciliation_from_json,
    reconciliation_to_json,
)
from financial_dashboard.services.statements.shared import (
    retry_bank_statement_upload,
    retry_cc_statement_upload,
)
from financial_dashboard.core.uploads import STATEMENTS_DIR, safe_upload_filename
from financial_dashboard.web.forms import _unlink_statement_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()


@router.post("/statements/upload-bank")
async def bank_statement_upload(
    request: FastAPIRequest,
    account_id: int = Form(...),
    password: str = Form(""),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    account = await session.get(Account, account_id)
    if not account or account.type != "bank_account":
        return RedirectResponse(url="/statements", status_code=303)

    # Save PDF to disk
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = safe_upload_filename(file.filename)
    file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
    content = await file.read()
    file_path.write_bytes(content)

    # Parse the PDF
    try:
        parsed = await asyncio.to_thread(
            parse_bank_statement, file_path, account.bank, password or None
        )
    except Exception as e:
        error_msg = str(e)
        is_encrypted = "encrypt" in error_msg.lower() or "password" in error_msg.lower()
        upload = BankStatementUpload(
            account_id=account_id,
            bank=account.bank,
            filename=safe_name,
            file_path=str(file_path),
            status="password_required" if is_encrypted else "parse_error",
            error=error_msg,
        )
        session.add(upload)
        await emit_bank_snapshot(session, upload)
        await session.commit()
        upload_id = upload.id
        return RedirectResponse(url=f"/statements/bank/{upload_id}", status_code=303)

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
    recon = reconcile_bank_statement(parsed, db_txns, account_id)
    await enrich_matched_transactions(recon)

    # Create upload and auto-import missing transactions
    upload = BankStatementUpload(
        account_id=account_id,
        bank=parsed.bank or account.bank,
        filename=safe_name,
        file_path=str(file_path),
        status="parsed",
        account_number=parsed.account_number,
        account_holder_name=parsed.account_holder_name,
        opening_balance=parsed.opening_balance,
        closing_balance=parsed.closing_balance,
        statement_period_start=parsed.statement_period_start,
        statement_period_end=parsed.statement_period_end,
        parsed_txn_count=len(recon["matched"]) + len(recon["missing"]),
        matched_count=len(recon["matched"]),
        missing_count=len(recon["missing"]),
        reconciliation_data=reconciliation_to_json(recon),
    )
    session.add(upload)
    await session.flush()

    link_ctx = await build_link_context(session)

    imported = 0
    for entry in recon["missing"]:
        try:
            amount = _parse_amount(entry["amount"])
            txn_date = _parse_date(entry["date"])
        except ValueError, KeyError:
            continue
        txn = Transaction(
            bank_statement_upload_id=upload.id,
            account_id=account_id,
            bank=parsed.bank or account.bank,
            email_type="bank_statement",
            direction=entry["direction"],
            amount=amount,
            currency="INR",
            transaction_date=txn_date,
            counterparty=entry.get("counterparty") or entry.get("narration"),
            account_mask=_last4(parsed.account_number),
            reference_number=entry.get("reference_number"),
            channel=entry.get("channel") or "bank_statement",
            raw_description=entry.get("narration"),
        )
        session.add(txn)
        await session.flush()
        link_transaction(link_ctx, txn)
        await session.flush()
        entry["imported"] = True
        entry["imported_txn_id"] = txn.id
        imported += 1

    upload.imported_count = imported
    upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
    upload.reconciliation_data = reconciliation_to_json(recon)
    if upload.missing_count == 0:
        upload.status = "imported"
    elif imported > 0:
        upload.status = "partial_import"
    await emit_bank_snapshot(session, upload)
    await session.commit()
    upload_id = upload.id

    return RedirectResponse(url=f"/statements/bank/{upload_id}", status_code=303)


@router.get("/statements/bank/{upload_id}", response_class=HTMLResponse)
async def bank_statement_detail(
    upload_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):

    upload = await session.get(BankStatementUpload, upload_id)
    if not upload:
        return HTMLResponse("<p>Bank statement not found.</p>", 404)

    recon = None
    if upload.reconciliation_data:
        recon = reconciliation_from_json(upload.reconciliation_data)

    return templates.TemplateResponse(
        request,
        "bank_statement_reconcile.html",
        {
            "active_page": "statements",
            "upload": upload,
            "recon": recon,
            "error": request.query_params.get("error"),
        },
    )


@router.post("/statements/bank/{upload_id}/retry")
async def bank_statement_retry(
    upload_id: int,
    password: str = Form(...),
    save_password: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    if not (upload := await session.get(BankStatementUpload, upload_id)):
        return RedirectResponse(url="/statements", status_code=303)
    account_id = upload.account_id

    # Only persist the password to the account if the retry succeeded — a
    # wrong password shouldn't overwrite a previously-good one. Saving the
    # password also unlocks any other password_required statements on the
    # same account (the coordinator skips this upload since its status is
    # no longer password_required after the retry above).
    if await retry_bank_statement_upload(upload_id, password) and save_password == "1":
        encrypted = get_fernet().encrypt(password.encode()).decode()
        if account := await session.get(Account, account_id):
            account.statement_password = encrypted
            await session.commit()
        await accounts_retry_password_required_statements(
            session,
            account_id,
            password,
            retry_cc_upload=retry_cc_statement_upload,
            retry_bank_upload=retry_bank_statement_upload,
        )

    return RedirectResponse(url=f"/statements/bank/{upload_id}", status_code=303)


@router.post("/statements/bank/{upload_id}/delete")
async def bank_statement_delete(
    upload_id: int,
    session: AsyncSession = Depends(get_session),
):
    upload = await session.get(BankStatementUpload, upload_id)
    if not upload:
        return RedirectResponse(url="/statements", status_code=303)
    await session.execute(
        update(Transaction)
        .where(Transaction.bank_statement_upload_id == upload_id)
        .values(bank_statement_upload_id=None)
    )
    _unlink_statement_file(upload.file_path)
    await session.delete(upload)
    await session.commit()

    return RedirectResponse(url="/statements", status_code=303)
