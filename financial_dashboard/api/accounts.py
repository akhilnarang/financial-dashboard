from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas import accounts as account_schemas
from financial_dashboard.services.account_reads import (
    get_account_detail,
    list_accounts,
)

router = APIRouter()


@router.get("/accounts")
async def accounts_list(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    account_type: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    active: bool | None = None,
    session: AsyncSession = Depends(get_session),
) -> account_schemas.AccountListResponse:
    return await list_accounts(
        session,
        limit=limit,
        offset=offset,
        bank=bank,
        account_type=account_type,
        active=active,
    )


@router.get("/accounts/{account_id}")
async def account_detail(
    account_id: Annotated[int, Path(ge=1, le=9_223_372_036_854_775_807)],
    session: AsyncSession = Depends(get_session),
) -> account_schemas.AccountDetailResponse:
    account = await get_account_detail(session, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return account
