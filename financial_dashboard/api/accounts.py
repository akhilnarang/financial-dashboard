"""Read-only JSON endpoints for financial accounts."""

from typing import Annotated

from fastapi import APIRouter, Path, Query

from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.exceptions import NotFoundException
from financial_dashboard.schemas import accounts as account_schemas
from financial_dashboard.schemas.common import DatabaseId
from financial_dashboard.services.account_reads import (
    get_account_detail,
    list_accounts,
)

router = APIRouter()


@router.get("/accounts")
async def accounts_list(
    session: AsyncSessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    account_type: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    active: bool | None = None,
) -> account_schemas.AccountListResponse:
    """List a bounded page of accounts with redacted card identifiers."""
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
    account_id: Annotated[DatabaseId, Path()],
    session: AsyncSessionDep,
) -> account_schemas.AccountDetailResponse:
    """Return one account with balances, cards, and related-row counts."""
    if account := await get_account_detail(session, account_id):
        return account

    raise NotFoundException(detail="Account not found")
