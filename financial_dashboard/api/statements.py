import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas import statements as statement_schemas
from financial_dashboard.services.statement_reads import (
    get_bank_statement_detail,
    get_bank_statements_by_ids,
    get_cc_statement_detail,
    get_cc_statements_by_ids,
    list_bank_statements,
    list_cc_statements,
)

router = APIRouter()
_DB_ID_MAX = 9_223_372_036_854_775_807


def _date_bounds(
    date_from: datetime.date | None,
    date_to: datetime.date | None,
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not exceed date_to")
    return (
        datetime.datetime.combine(date_from, datetime.time.min)
        if date_from is not None
        else None,
        datetime.datetime.combine(date_to, datetime.time.max)
        if date_to is not None
        else None,
    )


@router.get("/statements/cc")
async def cc_statement_list(
    response: Response,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    statement_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    account_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    email_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    status: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    session: AsyncSession = Depends(get_session),
) -> statement_schemas.CcStatementListResponse:
    response.headers["Cache-Control"] = "no-store"
    start, end = _date_bounds(date_from, date_to)
    return await list_cc_statements(
        session,
        limit=limit,
        offset=offset,
        statement_id=statement_id,
        account_id=account_id,
        email_id=email_id,
        bank=bank,
        status=status,
        date_from=start,
        date_to=end,
    )


@router.post("/statements/cc/batch")
async def cc_statement_batch(
    payload: statement_schemas.StatementBatchRequest,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> statement_schemas.CcStatementBatchResponse:
    response.headers["Cache-Control"] = "no-store"
    return await get_cc_statements_by_ids(session, payload.ids)


@router.get("/statements/cc/{statement_id}")
async def cc_statement_detail(
    statement_id: Annotated[int, Path(ge=1, le=_DB_ID_MAX)],
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> statement_schemas.CcStatementDetailResponse:
    response.headers["Cache-Control"] = "no-store"
    statement = await get_cc_statement_detail(session, statement_id)
    if statement is None:
        raise HTTPException(status_code=404, detail="CC statement not found")
    return statement


@router.get("/statements/bank")
async def bank_statement_list(
    response: Response,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    statement_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    account_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    email_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    status: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    session: AsyncSession = Depends(get_session),
) -> statement_schemas.BankStatementListResponse:
    response.headers["Cache-Control"] = "no-store"
    start, end = _date_bounds(date_from, date_to)
    return await list_bank_statements(
        session,
        limit=limit,
        offset=offset,
        statement_id=statement_id,
        account_id=account_id,
        email_id=email_id,
        bank=bank,
        status=status,
        date_from=start,
        date_to=end,
    )


@router.post("/statements/bank/batch")
async def bank_statement_batch(
    payload: statement_schemas.StatementBatchRequest,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> statement_schemas.BankStatementBatchResponse:
    response.headers["Cache-Control"] = "no-store"
    return await get_bank_statements_by_ids(session, payload.ids)


@router.get("/statements/bank/{statement_id}")
async def bank_statement_detail(
    statement_id: Annotated[int, Path(ge=1, le=_DB_ID_MAX)],
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> statement_schemas.BankStatementDetailResponse:
    response.headers["Cache-Control"] = "no-store"
    statement = await get_bank_statement_detail(session, statement_id)
    if statement is None:
        raise HTTPException(status_code=404, detail="Bank statement not found")
    return statement
