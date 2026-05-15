from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.transactions import (
    TransactionCategoryResponse,
    TransactionCategoryUpdate,
    TransactionNoteResponse,
    TransactionNoteUpdate,
)
from financial_dashboard.services.transactions import (
    update_transaction_category,
    update_transaction_note,
)

router = APIRouter()


@router.post("/transactions/{txn_id}/note", response_model=TransactionNoteResponse)
async def update_note(
    txn_id: int,
    payload: TransactionNoteUpdate,
    session: AsyncSession = Depends(get_session),
) -> TransactionNoteResponse:
    ok, note = await update_transaction_note(session, txn_id, payload.note)
    if not ok:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return TransactionNoteResponse(ok=True, note=note)


@router.post(
    "/transactions/{txn_id}/category", response_model=TransactionCategoryResponse
)
async def update_category(
    txn_id: int,
    payload: TransactionCategoryUpdate,
    session: AsyncSession = Depends(get_session),
) -> TransactionCategoryResponse:
    ok, category = await update_transaction_category(session, txn_id, payload.category)
    if not ok:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return TransactionCategoryResponse(ok=True, category=category)
