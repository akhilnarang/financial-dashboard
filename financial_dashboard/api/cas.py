"""CAS PDF ingestion endpoints."""

from fastapi import APIRouter, File, Form, UploadFile

from financial_dashboard.core.uploads import read_bounded_pdf
from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.core.uploads import safe_upload_filename, save_statement_pdf
from financial_dashboard.exceptions import (
    BadRequestException,
)
from financial_dashboard.schemas.cas import CasUploadRead
from financial_dashboard.services.cas_ingestion import CasIngestError, ingest_cas_pdf

router = APIRouter()


@router.post("/cas/upload")
async def upload_cas(
    session: AsyncSessionDep,
    password: str = Form(""),
    force_replace: bool = Form(False),
    file: UploadFile = File(...),
) -> CasUploadRead:
    """Parse and ingest one bounded CAS PDF upload."""
    payload = await read_bounded_pdf(file)

    safe_name = safe_upload_filename(file.filename)
    file_path = save_statement_pdf(payload, safe_name)

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
        raise BadRequestException(detail=str(exc)) from exc

    await session.commit()
    await session.refresh(upload)
    return CasUploadRead.model_validate(upload)
