"""JSON endpoints for reading and editing transactions."""

import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Path, Query

from financial_dashboard.api.query import validate_date_range
from financial_dashboard.core.deps import SessionDep
from financial_dashboard.exceptions import BadRequestException, NotFoundException
from financial_dashboard.schemas import transactions as transaction_schemas
from financial_dashboard.schemas.common import DatabaseId
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


@router.get("/transactions")
async def transactions_list(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    transaction_id: Annotated[DatabaseId | None, Query()] = None,
    account_id: Annotated[DatabaseId | None, Query()] = None,
    card_id: Annotated[DatabaseId | None, Query()] = None,
    email_id: Annotated[DatabaseId | None, Query()] = None,
    sms_message_id: Annotated[DatabaseId | None, Query()] = None,
    statement_upload_id: Annotated[DatabaseId | None, Query()] = None,
    bank_statement_upload_id: Annotated[DatabaseId | None, Query()] = None,
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
) -> transaction_schemas.TransactionListResponse:
    """List a bounded page of transactions matching optional exact filters."""
    validate_date_range(date_from, date_to)

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
    session: SessionDep,
) -> transaction_schemas.TransactionBatchResponse:
    """Return transaction summaries for an ordered, explicit set of IDs."""
    return await get_transactions_by_ids(session, payload.ids)


@router.get("/transactions/{txn_id}")
async def transaction_detail(
    txn_id: Annotated[DatabaseId, Path()],
    session: SessionDep,
) -> transaction_schemas.TransactionDetailResponse:
    """Return one transaction with attribution and source provenance."""
    if transaction := await get_transaction_detail(session, txn_id):
        return transaction

    raise NotFoundException(detail="Transaction not found")


@router.post("/transactions/{txn_id}/note")
async def update_note(
    txn_id: Annotated[DatabaseId, Path()],
    payload: TransactionNoteUpdate,
    session: SessionDep,
) -> TransactionNoteResponse:
    """Replace the operator note on one transaction."""
    ok, note = await update_transaction_note(session, txn_id, payload.note)
    if ok:
        return TransactionNoteResponse(ok=True, note=note)

    raise NotFoundException(detail="Transaction not found")


@router.post("/transactions/{txn_id}/category")
async def update_category(
    txn_id: Annotated[DatabaseId, Path()],
    payload: TransactionCategoryUpdate,
    session: SessionDep,
) -> TransactionCategoryResponse:
    """Set or clear the manual category on one transaction."""
    try:
        ok, category = await update_transaction_category(
            session, txn_id, payload.category
        )
    except ValueError as exc:
        raise BadRequestException(detail=str(exc)) from exc

    if ok:
        return TransactionCategoryResponse(ok=True, category=category)

    raise NotFoundException(detail="Transaction not found")


@router.post("/transactions/{txn_id}/relink")
async def relink(
    txn_id: Annotated[DatabaseId, Path()],
    payload: TransactionRelinkUpdate,
    session: SessionDep,
) -> TransactionRelinkResponse:
    """Manually set or clear a transaction's account and card attribution.

    When only ``card_id`` is supplied, the service derives the account from the
    card. A newly linked CC payment may update statement payment tracking.
    """
    try:
        result = await relink_transaction(
            session,
            txn_id,
            account_id=payload.account_id,
            card_id=payload.card_id,
        )
    except RelinkError as exc:
        raise BadRequestException(detail=exc.message) from exc

    if result is None:
        raise NotFoundException(detail="Transaction not found")

    return TransactionRelinkResponse(
        ok=True,
        account_id=result.account_id,
        card_id=result.card_id,
        account_label=result.account_label,
        card_label=result.card_label,
        statement_marked_paid=result.statement_marked_paid,
    )
