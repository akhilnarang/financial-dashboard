"""JSON endpoints for credit-card and bank statements."""

import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, Path, Query, Response, UploadFile

from financial_dashboard.api.query import inclusive_datetime_bounds
from financial_dashboard.core.uploads import read_bounded_pdf
from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.exceptions import (
    ApiException,
    ConflictException,
    NotFoundException,
    StatementPreviewError,
    UnprocessableEntityException,
)
from financial_dashboard.schemas import statements as statement_schemas
from financial_dashboard.db import Account
from financial_dashboard.schemas.common import DatabaseId
from financial_dashboard.services.statement_previews import (
    preview_statement_parse,
    preview_statement_reconciliation,
    reconciliation_lists,
)
from financial_dashboard.services.statement_reads import (
    get_bank_statement_detail,
    get_bank_statements_by_ids,
    get_cc_statement_detail,
    get_cc_statements_by_ids,
    list_bank_statements,
    list_cc_statements,
)
from financial_dashboard.services.statements.bank import upload_bank_statement
from financial_dashboard.services.statements.shared import (
    retry_bank_statement_upload,
    retry_cc_statement_upload,
)

router = APIRouter()


@router.post("/statements/bank/upload")
async def bank_statement_upload(
    session: AsyncSessionDep,
    response: Response,
    account_id: Annotated[DatabaseId, Form()],
    file: Annotated[UploadFile, File()],
    password: Annotated[str, Form(max_length=128)] = "",
    dry_run: Annotated[bool, Form()] = True,
) -> statement_schemas.BankStatementUploadResponse:
    """Parse and reconcile one bank statement PDF against the account's rows.

    ``dry_run`` defaults to true. A dry run writes nothing. A real run saves the
    PDF, records the upload, enriches matched rows and imports missing rows.
    """
    response.headers["Cache-Control"] = "no-store"
    account = await session.get(Account, account_id)
    if account is None or account.type != "bank_account":
        raise NotFoundException(detail="Bank account not found")
    content = await read_bounded_pdf(file)

    result = await upload_bank_statement(
        session, account, file.filename, content, password or None, dry_run=dry_run
    )
    if result.error is not None and result.upload is None:
        raise UnprocessableEntityException(detail=f"Parse failed: {result.error}")

    parsed, upload = result.parsed, result.upload
    return statement_schemas.BankStatementUploadResponse(
        **reconciliation_lists(result.recon or {}).model_dump(),
        dry_run=dry_run,
        account_id=account.id,
        upload_id=upload.id if upload else None,
        status=upload.status if upload else "preview",
        error=upload.error if upload else None,
        statement_period_start=parsed.statement_period_start if parsed else None,
        statement_period_end=parsed.statement_period_end if parsed else None,
        opening_balance=parsed.opening_balance if parsed else None,
        closing_balance=parsed.closing_balance if parsed else None,
        imported_count=(upload.imported_count or 0) if upload else 0,
    )


@router.get("/statements/cc")
async def cc_statement_list(
    response: Response,
    session: AsyncSessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    statement_id: Annotated[DatabaseId | None, Query()] = None,
    account_id: Annotated[DatabaseId | None, Query()] = None,
    email_id: Annotated[DatabaseId | None, Query()] = None,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    status: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
) -> statement_schemas.CcStatementListResponse:
    """List bounded credit-card statement metadata and payment state."""
    response.headers["Cache-Control"] = "no-store"
    date_bounds = inclusive_datetime_bounds(date_from, date_to)

    return await list_cc_statements(
        session,
        limit=limit,
        offset=offset,
        statement_id=statement_id,
        account_id=account_id,
        email_id=email_id,
        bank=bank,
        status=status,
        date_from=date_bounds.start,
        date_to=date_bounds.end,
    )


@router.post("/statements/cc/batch")
async def cc_statement_batch(
    payload: statement_schemas.StatementBatchRequest,
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.CcStatementBatchResponse:
    """Return CC statement summaries for an ordered, explicit set of IDs."""
    response.headers["Cache-Control"] = "no-store"
    return await get_cc_statements_by_ids(session, payload.ids)


@router.get("/statements/cc/{statement_id}")
async def cc_statement_detail(
    statement_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.CcStatementDetailResponse:
    """Return one CC statement with bounded reconciliation evidence."""
    response.headers["Cache-Control"] = "no-store"
    if statement := await get_cc_statement_detail(session, statement_id):
        return statement

    raise NotFoundException(detail="CC statement not found")


@router.post("/statements/cc/{statement_id}/parse-preview")
async def cc_statement_parse_preview(
    statement_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.StatementParsePreviewResponse:
    """Reparse a stored CC statement PDF without imports or database writes."""
    response.headers["Cache-Control"] = "no-store"
    try:
        if preview := await preview_statement_parse(session, "cc", statement_id):
            return preview
    except StatementPreviewError as exc:
        raise ApiException(status_code=exc.status_code, detail=str(exc)) from exc

    raise NotFoundException(detail="CC statement not found")


@router.post("/statements/cc/{statement_id}/reconcile-preview")
async def cc_statement_reconcile_preview(
    statement_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.StatementReconciliationPreviewResponse:
    """Project CC statement matches and misses without enrichment or imports."""
    response.headers["Cache-Control"] = "no-store"
    try:
        if preview := await preview_statement_reconciliation(
            session, "cc", statement_id
        ):
            return preview
    except StatementPreviewError as exc:
        raise ApiException(status_code=exc.status_code, detail=str(exc)) from exc

    raise NotFoundException(detail="CC statement not found")


@router.post("/statements/cc/{statement_id}/reparse")
async def cc_statement_reparse(
    statement_id: Annotated[DatabaseId, Path()],
    payload: statement_schemas.StatementReparseRequest,
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.CcStatementDetailResponse:
    """Reparse and reconcile one stored CC statement through the canonical path.

    The supplied password is used only for this operation and is never persisted.
    A successful reparse may enrich matches, import missing transactions, update
    statement payment tracking, and emit the corresponding balance snapshot.
    """
    response.headers["Cache-Control"] = "no-store"
    current_statement = await get_cc_statement_detail(session, statement_id)
    if current_statement is None:
        raise NotFoundException(detail="CC statement not found")
    if current_statement.source_kind == "email_summary":
        raise ConflictException(
            detail="Email-summary statements have no PDF to reparse"
        )

    session.expunge_all()
    await session.rollback()
    succeeded = await retry_cc_statement_upload(
        statement_id, payload.password.get_secret_value()
    )
    if not succeeded:
        if await get_cc_statement_detail(session, statement_id) is None:
            raise NotFoundException(detail="CC statement not found")
        raise UnprocessableEntityException(detail="CC statement reparse failed")

    if statement := await get_cc_statement_detail(session, statement_id):
        return statement
    raise NotFoundException(detail="CC statement not found")


@router.get("/statements/bank")
async def bank_statement_list(
    response: Response,
    session: AsyncSessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    statement_id: Annotated[DatabaseId | None, Query()] = None,
    account_id: Annotated[DatabaseId | None, Query()] = None,
    email_id: Annotated[DatabaseId | None, Query()] = None,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    status: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
) -> statement_schemas.BankStatementListResponse:
    """List bounded bank-statement metadata and reconciliation counts."""
    response.headers["Cache-Control"] = "no-store"
    date_bounds = inclusive_datetime_bounds(date_from, date_to)

    return await list_bank_statements(
        session,
        limit=limit,
        offset=offset,
        statement_id=statement_id,
        account_id=account_id,
        email_id=email_id,
        bank=bank,
        status=status,
        date_from=date_bounds.start,
        date_to=date_bounds.end,
    )


@router.post("/statements/bank/batch")
async def bank_statement_batch(
    payload: statement_schemas.StatementBatchRequest,
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.BankStatementBatchResponse:
    """Return bank statement summaries for an ordered, explicit set of IDs."""
    response.headers["Cache-Control"] = "no-store"
    return await get_bank_statements_by_ids(session, payload.ids)


@router.post("/statements/bank/{statement_id}/parse-preview")
async def bank_statement_parse_preview(
    statement_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.StatementParsePreviewResponse:
    """Reparse a stored bank statement PDF without imports or database writes."""
    response.headers["Cache-Control"] = "no-store"
    try:
        if preview := await preview_statement_parse(session, "bank", statement_id):
            return preview
    except StatementPreviewError as exc:
        raise ApiException(status_code=exc.status_code, detail=str(exc)) from exc

    raise NotFoundException(detail="Bank statement not found")


@router.post("/statements/bank/{statement_id}/reconcile-preview")
async def bank_statement_reconcile_preview(
    statement_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.StatementReconciliationPreviewResponse:
    """Project bank statement matches and misses without enrichment or imports."""
    response.headers["Cache-Control"] = "no-store"
    try:
        if preview := await preview_statement_reconciliation(
            session, "bank", statement_id
        ):
            return preview
    except StatementPreviewError as exc:
        raise ApiException(status_code=exc.status_code, detail=str(exc)) from exc

    raise NotFoundException(detail="Bank statement not found")


@router.post("/statements/bank/{statement_id}/reparse")
async def bank_statement_reparse(
    statement_id: Annotated[DatabaseId, Path()],
    payload: statement_schemas.StatementReparseRequest,
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.BankStatementDetailResponse:
    """Reparse and reconcile one stored bank statement through the canonical path.

    The supplied password is used only for this operation and is never persisted.
    A successful reparse may enrich matches, import missing transactions, and
    emit the corresponding account balance snapshot.
    """
    response.headers["Cache-Control"] = "no-store"
    if await get_bank_statement_detail(session, statement_id) is None:
        raise NotFoundException(detail="Bank statement not found")

    session.expunge_all()
    await session.rollback()
    succeeded = await retry_bank_statement_upload(
        statement_id, payload.password.get_secret_value()
    )
    if not succeeded:
        if await get_bank_statement_detail(session, statement_id) is None:
            raise NotFoundException(detail="Bank statement not found")
        raise UnprocessableEntityException(detail="Bank statement reparse failed")

    if statement := await get_bank_statement_detail(session, statement_id):
        return statement
    raise NotFoundException(detail="Bank statement not found")


@router.get("/statements/bank/{statement_id}")
async def bank_statement_detail(
    statement_id: Annotated[DatabaseId, Path()],
    response: Response,
    session: AsyncSessionDep,
) -> statement_schemas.BankStatementDetailResponse:
    """Return one bank statement with bounded reconciliation evidence."""
    response.headers["Cache-Control"] = "no-store"
    if statement := await get_bank_statement_detail(session, statement_id):
        return statement

    raise NotFoundException(detail="Bank statement not found")
