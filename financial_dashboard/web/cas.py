"""CAS upload routes."""

from __future__ import annotations

import datetime
from urllib.parse import urlencode

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Request as FastAPIRequest,
    UploadFile,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.core.uploads import STATEMENTS_DIR, safe_upload_filename
from financial_dashboard.services.cas_ingestion import CasIngestError, ingest_cas_pdf

templates = get_templates()
router = APIRouter()

CAS_UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 10 MB; real CAS PDFs are <2 MB.


@router.get("/cas/upload", response_class=HTMLResponse)
async def cas_upload_form(request: FastAPIRequest):
    return templates.TemplateResponse(
        request,
        "cas/upload.html",
        {
            "active_page": "networth",
            "error": request.query_params.get("error"),
        },
    )


@router.post("/cas/upload")
async def cas_upload(
    password: str = Form(""),
    force_replace: bool = Form(False),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if file.size is not None and file.size > CAS_UPLOAD_MAX_BYTES:
        return RedirectResponse(
            url=f"/cas/upload?{urlencode({'error': 'PDF exceeds 10 MB limit.'})}",
            status_code=303,
        )
    payload = await file.read()
    if len(payload) > CAS_UPLOAD_MAX_BYTES:
        return RedirectResponse(
            url=f"/cas/upload?{urlencode({'error': 'PDF exceeds 10 MB limit.'})}",
            status_code=303,
        )

    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    safe_name = safe_upload_filename(file.filename)
    file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
    file_path.write_bytes(payload)

    try:
        await ingest_cas_pdf(
            session,
            file_path,
            password=password or None,
            force_replace=force_replace,
        )
    except CasIngestError as exc:
        await session.rollback()
        file_path.unlink(missing_ok=True)
        return RedirectResponse(
            url=f"/cas/upload?{urlencode({'error': str(exc)})}", status_code=303
        )
    except Exception as exc:
        await session.rollback()
        file_path.unlink(missing_ok=True)
        message = str(exc) or "Could not parse CAS PDF."
        return RedirectResponse(
            url=f"/cas/upload?{urlencode({'error': message})}", status_code=303
        )

    await session.commit()
    return RedirectResponse(url="/networth", status_code=303)
