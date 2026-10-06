"""Bank statement HTML routes."""

import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Request as FastAPIRequest,
    UploadFile,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import update
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
from financial_dashboard.services.statements.bank import (
    reconciliation_from_json,
    upload_bank_statement,
)
from financial_dashboard.services.statements.shared import (
    retry_bank_statement_upload,
)
from financial_dashboard.web.forms import _unlink_statement_file
from financial_dashboard.web.transaction_display import (
    hydrate_reconciliation_transactions,
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

    result = await upload_bank_statement(
        session, account, file.filename, await file.read(), password or None
    )
    assert result.upload is not None
    return RedirectResponse(url=f"/statements/bank/{result.upload.id}", status_code=303)


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
        await hydrate_reconciliation_transactions(session, recon)

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
