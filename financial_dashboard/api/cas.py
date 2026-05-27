"""CAS JSON endpoints."""

from __future__ import annotations

import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.cas import CasUploadRead
from financial_dashboard.services.cas_ingestion import CasIngestError, ingest_cas_pdf
from financial_dashboard.core.uploads import STATEMENTS_DIR, safe_upload_filename
from financial_dashboard.web.cas import CAS_UPLOAD_MAX_BYTES

router = APIRouter()


@router.post("/cas/upload", response_model=CasUploadRead)
async def upload_cas(
    password: str = Form(""),
    force_replace: bool = Form(False),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if file.size is not None and file.size > CAS_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 10 MB limit.")
    payload = await file.read()
    if len(payload) > CAS_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 10 MB limit.")

    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    safe_name = safe_upload_filename(file.filename)
    file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
    file_path.write_bytes(payload)

    try:
        upload = await ingest_cas_pdf(
            session,
            file_path,
            password=password or None,
            force_replace=force_replace,
        )
    except (CasIngestError, ValueError) as exc:
        await session.rollback()
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await session.commit()
    await session.refresh(upload)
    return upload
