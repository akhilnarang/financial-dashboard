import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas import transactions as transaction_schemas
from financial_dashboard.schemas.transactions import (
    TransactionCategoryResponse,
    TransactionCategoryUpdate,
    TransactionNoteResponse,
    TransactionNoteUpdate,
    TransactionRelinkResponse,
    TransactionRelinkUpdate,
)
from financial_dashboard.services.transaction_reads import (
    get_transaction_detail,
    get_transactions_by_ids,
    list_transactions,
)
from financial_dashboard.services.transactions import (
    RelinkError,
    relink_transaction,
    update_transaction_category,
    update_transaction_note,
)

router = APIRouter()
_DB_ID_MAX = 9_223_372_036_854_775_807


@router.get("/transactions")
async def transactions_list(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    transaction_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    account_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    card_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    email_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    sms_message_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    statement_upload_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    bank_statement_upload_id: Annotated[int | None, Query(ge=1, le=_DB_ID_MAX)] = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    direction: Annotated[str | None, Query(min_length=1, max_length=16)] = None,
    amount: Annotated[Decimal | None, Query(ge=0, le=9_999_999_999.99)] = None,
    bank: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    email_type: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    source: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    category: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    review_status: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    reference_number: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
    session: AsyncSession = Depends(get_session),
) -> transaction_schemas.TransactionListResponse:
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not exceed date_to")
    return await list_transactions(
        session,
        limit=limit,
        offset=offset,
        transaction_id=transaction_id,
        account_id=account_id,
        card_id=card_id,
        email_id=email_id,
        sms_message_id=sms_message_id,
        statement_upload_id=statement_upload_id,
        bank_statement_upload_id=bank_statement_upload_id,
        date_from=date_from,
        date_to=date_to,
        direction=direction,
        amount=amount,
        bank=bank,
        email_type=email_type,
        source=source,
        category=category,
        review_status=review_status,
        reference_number=reference_number,
    )


@router.post("/transactions/batch")
async def transactions_batch(
    payload: transaction_schemas.TransactionBatchRequest,
    session: AsyncSession = Depends(get_session),
) -> transaction_schemas.TransactionBatchResponse:
    return await get_transactions_by_ids(session, payload.ids)


@router.get("/transactions/{txn_id}")
async def transaction_detail(
    txn_id: Annotated[int, Path(ge=1, le=_DB_ID_MAX)],
    session: AsyncSession = Depends(get_session),
) -> transaction_schemas.TransactionDetailResponse:
    transaction = await get_transaction_detail(session, txn_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction


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
    try:
        ok, category = await update_transaction_category(
            session, txn_id, payload.category
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
