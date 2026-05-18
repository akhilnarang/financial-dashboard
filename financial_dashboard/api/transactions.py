from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.transactions import (
    TransactionCategoryResponse,
    TransactionCategoryUpdate,
    TransactionNoteResponse,
    TransactionNoteUpdate,
    TransactionRelinkResponse,
    TransactionRelinkUpdate,
)
from financial_dashboard.services.transactions import (
    RelinkError,
    relink_transaction,
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


@router.post("/transactions/{txn_id}/relink", response_model=TransactionRelinkResponse)
async def relink(
    txn_id: int,
    payload: TransactionRelinkUpdate,
    session: AsyncSession = Depends(get_session),
) -> TransactionRelinkResponse:
    """Manually set or clear the account/card link on a Transaction.

    Body: ``{"account_id": <int|null>, "card_id": <int|null>}``. Either
    field may be null. When ``card_id`` is given without
    ``account_id``, the account is derived from the card's owning
    account. Returns the resolved labels so the UI can refresh
    without a second round-trip.

    Fires ``check_payment_received`` when the transaction was
    previously orphaned and the new state passes the auto-reconcile
    gate (CC bill-payment credits only). Subsequent re-relinks of an
    already-linked txn don't re-credit.
    """
    try:
        result = await relink_transaction(
            session,
            txn_id,
            account_id=payload.account_id,
            card_id=payload.card_id,
        )
    except RelinkError as exc:
        raise HTTPException(status_code=400, detail=exc.message)
    if result is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return TransactionRelinkResponse(
        ok=True,
        account_id=result.account_id,
        card_id=result.card_id,
        account_label=result.account_label,
        card_label=result.card_label,
        statement_marked_paid=result.statement_marked_paid,
    )
