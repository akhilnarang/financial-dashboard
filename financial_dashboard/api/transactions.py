"""JSON endpoints for reading and editing transactions."""

import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Path, Query
from fastapi.responses import FileResponse
from pydantic import Field

from financial_dashboard.api.query import validate_date_range
from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.exceptions import (
    BadRequestException,
    ConflictException,
    NotFoundException,
)
from financial_dashboard.schemas import transactions as transaction_schemas
from financial_dashboard.schemas.common import DatabaseId
from financial_dashboard.schemas.transactions import (
    TransactionCategoryResponse,
    TransactionCategoryUpdate,
    TransactionExcludeResponse,
    TransactionExcludeUpdate,
    TransactionMergeBatchRequest,
    TransactionMergeResponse,
    TransactionNoteResponse,
    TransactionNoteUpdate,
    TransactionRelinkResponse,
    TransactionRelinkUpdate,
)
from financial_dashboard.services.duplicate_merge import (
    MergeRefused,
    merge_duplicates,
)
from financial_dashboard.services.categorization.vocabulary import list_categories
from financial_dashboard.services.transaction_reads import (
    get_transaction_detail,
    get_transactions_by_ids,
    list_transactions,
)
from financial_dashboard.services.transactions import (
    RelinkError,
    categorize_transactions,
    relink_transaction,
    set_transaction_excluded,
    update_transaction_category,
    update_transaction_note,
)
from financial_dashboard.services.transaction_attachments import (
    AttachmentError,
    detect_attachment_type,
    resolve_attachment_path,
)

router = APIRouter()


@router.get("/transactions")
async def transactions_list(
    session: AsyncSessionDep,
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
    category: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=128)]] | None,
        Query(max_length=50),
    ] = None,
    q: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
    exclude_from_cashflow: bool | None = None,
    review_status: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    reference_number: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
) -> transaction_schemas.TransactionListResponse:
    """List a bounded page of transactions matching optional filters.

    ``category`` repeats to match any of several slugs. ``q`` is a
    case-insensitive literal substring search over the row's text fields.
    """
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
        categories=category,
        review_status=review_status,
        reference_number=reference_number,
        search=q,
        excluded=exclude_from_cashflow,
    )


@router.post("/transactions/batch")
async def transactions_batch(
    payload: transaction_schemas.TransactionBatchRequest,
    session: AsyncSessionDep,
) -> transaction_schemas.TransactionBatchResponse:
    """Return transaction summaries for an ordered, explicit set of IDs."""
    return await get_transactions_by_ids(session, payload.ids)


@router.post("/transactions/merge-batch")
async def transactions_merge_batch(
    payload: TransactionMergeBatchRequest,
    session: AsyncSessionDep,
) -> TransactionMergeResponse:
    """Merge duplicate rows into their keepers, all or nothing.

    A dry run writes nothing. A refused pair refuses the whole batch.
    """
    try:
        return await merge_duplicates(session, payload.pairs, dry_run=payload.dry_run)
    except MergeRefused as exc:
        raise ConflictException(detail={"refused": exc.refusals}) from exc


@router.post("/transactions/categorize")
async def transactions_categorize(
    payload: transaction_schemas.TransactionCategorizeRequest,
    session: AsyncSessionDep,
) -> transaction_schemas.TransactionCategorizeResponse:
    """Change category, note and cashflow exclusion on many rows at once.

    ``dry_run`` defaults to true. One invalid item rejects the whole request
    with a 400 and per-item errors, and nothing is written.
    """
    try:
        items = await categorize_transactions(
            session, payload.items, dry_run=payload.dry_run
        )
    except ValueError as exc:
        raise BadRequestException(detail={"errors": exc.args[0]}) from exc
    return transaction_schemas.TransactionCategorizeResponse(
        dry_run=payload.dry_run, items=items
    )


@router.get("/categories")
async def categories_list(
    session: AsyncSessionDep,
) -> transaction_schemas.CategoryListResponse:
    """List every category slug with its bank-scope cashflow bucket."""
    return await list_categories(session)


@router.get("/transactions/{txn_id}")
async def transaction_detail(
    txn_id: Annotated[DatabaseId, Path()],
    session: AsyncSessionDep,
) -> transaction_schemas.TransactionDetailResponse:
    """Return one transaction with attribution and source provenance."""
    if transaction := await get_transaction_detail(session, txn_id):
        return transaction

    raise NotFoundException(detail="Transaction not found")


@router.get("/transactions/{txn_id}/attachment")
async def transaction_attachment(
    txn_id: Annotated[DatabaseId, Path()],
    session: AsyncSessionDep,
) -> FileResponse:
    """Serve one stored receipt through the API's normal auth dependency."""
    from financial_dashboard.db import Transaction

    transaction = await session.get(Transaction, txn_id)
    if transaction is None or transaction.attachment_path is None:
        raise NotFoundException(detail="Transaction attachment not found")
    try:
        target = resolve_attachment_path(transaction.attachment_path)
        with target.open("rb") as stored:
            media_type, suffix = detect_attachment_type(stored.read(16))
    except (AttachmentError, OSError) as exc:
        raise NotFoundException(detail="Transaction attachment not found") from exc
    disposition = "inline" if media_type.startswith("image/") else "attachment"
    filename = f"transaction-{txn_id}-receipt{suffix}"
    return FileResponse(
        target,
        media_type=media_type,
        filename=filename,
        content_disposition_type=disposition,
    )


@router.post("/transactions/{txn_id}/note")
async def update_note(
    txn_id: Annotated[DatabaseId, Path()],
    payload: TransactionNoteUpdate,
    session: AsyncSessionDep,
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
    session: AsyncSessionDep,
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


@router.post("/transactions/{txn_id}/exclude")
async def update_exclude(
    txn_id: Annotated[DatabaseId, Path()],
    payload: TransactionExcludeUpdate,
    session: AsyncSessionDep,
) -> TransactionExcludeResponse:
    """Set whether one transaction is dropped from the cashflow report."""
    ok, state = await set_transaction_excluded(
        session, txn_id, payload.exclude_from_cashflow
    )
    if ok:
        return TransactionExcludeResponse(ok=True, exclude_from_cashflow=state)

    raise NotFoundException(detail="Transaction not found")


@router.post("/transactions/{txn_id}/relink")
async def relink(
    txn_id: Annotated[DatabaseId, Path()],
    payload: TransactionRelinkUpdate,
    session: AsyncSessionDep,
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
