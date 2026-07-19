import datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer, joinedload, noload

from financial_dashboard.core.masks import mask_last4
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    Email,
    SmsMessage,
    StatementUpload,
    Transaction,
)
from financial_dashboard.schemas import transactions as transaction_schemas
from financial_dashboard.services.cc_disambiguation import (
    should_auto_reconcile_statement,
)

_DETAIL_TEXT_LIMIT = 50_000


def _safe_mask(value: str | None) -> str | None:
    last_digits = mask_last4(value, partial=True)
    return f"XXXX{last_digits}" if last_digits else None


def _bounded_detail_text(value: str | None) -> tuple[str | None, bool]:
    if value is None or len(value) <= _DETAIL_TEXT_LIMIT:
        return value, False
    return value[:_DETAIL_TEXT_LIMIT], True


def _transaction_read(row: Transaction) -> transaction_schemas.TransactionRead:
    return transaction_schemas.TransactionRead(
        id=row.id,
        bank=row.bank,
        email_type=row.email_type,
        direction=row.direction,
        amount=row.amount,
        currency=row.currency,
        transaction_date=row.transaction_date,
        transaction_time=row.transaction_time,
        counterparty=row.counterparty,
        card_mask=_safe_mask(row.card_mask),
        account_mask=_safe_mask(row.account_mask),
        reference_number=row.reference_number,
        channel=row.channel,
        balance=row.balance,
        account_id=row.account_id,
        card_id=row.card_id,
        email_id=row.email_id,
        sms_message_id=row.sms_message_id,
        statement_upload_id=row.statement_upload_id,
        bank_statement_upload_id=row.bank_statement_upload_id,
        source=row.source,
        category=row.category,
        category_method=row.category_method,
        review_status=row.review_status,
        created_at=row.created_at,
        enriched_at=row.enriched_at,
    )


def _filters(
    *,
    transaction_id: int | None,
    account_id: int | None,
    card_id: int | None,
    email_id: int | None,
    sms_message_id: int | None,
    statement_upload_id: int | None,
    bank_statement_upload_id: int | None,
    date_from: datetime.date | None,
    date_to: datetime.date | None,
    direction: str | None,
    amount: Decimal | None,
    bank: str | None,
    email_type: str | None,
    source: str | None,
    category: str | None,
    review_status: str | None,
    reference_number: str | None,
) -> list:
    clauses = []
    equality_filters = (
        (Transaction.id, transaction_id),
        (Transaction.account_id, account_id),
        (Transaction.card_id, card_id),
        (Transaction.email_id, email_id),
        (Transaction.sms_message_id, sms_message_id),
        (Transaction.statement_upload_id, statement_upload_id),
        (Transaction.bank_statement_upload_id, bank_statement_upload_id),
        (Transaction.amount, amount),
    )
    clauses.extend(
        column == value for column, value in equality_filters if value is not None
    )
    if date_from is not None:
        clauses.append(Transaction.transaction_date >= date_from)
    if date_to is not None:
        clauses.append(Transaction.transaction_date <= date_to)
    text_filters = (
        (Transaction.direction, direction),
        (Transaction.email_type, email_type),
        (Transaction.source, source),
        (Transaction.category, category),
        (Transaction.review_status, review_status),
        (Transaction.reference_number, reference_number),
    )
    clauses.extend(
        column == value.strip() for column, value in text_filters if value is not None
    )
    if bank is not None:
        clauses.append(func.lower(Transaction.bank) == bank.strip().lower())
    return clauses


async def list_transactions(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
    transaction_id: int | None,
    account_id: int | None,
    card_id: int | None,
    email_id: int | None,
    sms_message_id: int | None,
    statement_upload_id: int | None,
    bank_statement_upload_id: int | None,
    date_from: datetime.date | None,
    date_to: datetime.date | None,
    direction: str | None,
    amount: Decimal | None,
    bank: str | None,
    email_type: str | None,
    source: str | None,
    category: str | None,
    review_status: str | None,
    reference_number: str | None,
) -> transaction_schemas.TransactionListResponse:
    clauses = _filters(
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
    with session.no_autoflush:
        total_count = await session.scalar(
            select(func.count(Transaction.id))
            .where(*clauses)
            .execution_options(autoflush=False)
        )
        rows = (
            (
                await session.execute(
                    select(Transaction)
                    .options(
                        noload(Transaction.account),
                        noload(Transaction.card),
                        defer(Transaction.raw_description),
                        defer(Transaction.note),
                        defer(Transaction.review_reason),
                    )
                    .where(*clauses)
                    .order_by(Transaction.id.desc())
                    .offset(offset)
                    .limit(limit)
                    .execution_options(autoflush=False)
                )
            )
            .unique()
            .scalars()
            .all()
        )

    items = [_transaction_read(row) for row in rows]
    return transaction_schemas.TransactionListResponse(
        items=items,
        returned_count=len(items),
        total_count=total_count or 0,
        limit=limit,
        offset=offset,
    )


async def _source_links(
    session: AsyncSession,
    row: Transaction,
) -> tuple[
    transaction_schemas.TransactionSourceLink | None,
    transaction_schemas.TransactionSourceLink | None,
    transaction_schemas.TransactionStatementLink | None,
]:
    email_link: transaction_schemas.TransactionSourceLink | None = None
    if row.email_id is not None:
        email = (
            await session.execute(
                select(Email.id, Email.status, Email.received_at).where(
                    Email.id == row.email_id
                )
            )
        ).one_or_none()
        if email is not None:
            email_link = transaction_schemas.TransactionSourceLink(
                id=email.id,
                status=email.status,
                timestamp=email.received_at,
            )

    sms_link: transaction_schemas.TransactionSourceLink | None = None
    if row.sms_message_id is not None:
        sms = (
            await session.execute(
                select(SmsMessage.id, SmsMessage.status, SmsMessage.received_at).where(
                    SmsMessage.id == row.sms_message_id
                )
            )
        ).one_or_none()
        if sms is not None:
            sms_link = transaction_schemas.TransactionSourceLink(
                id=sms.id,
                status=sms.status,
                timestamp=sms.received_at,
            )

    statement_link: transaction_schemas.TransactionStatementLink | None = None
    if row.statement_upload_id is not None:
        statement = (
            await session.execute(
                select(
                    StatementUpload.id,
                    StatementUpload.status,
                    StatementUpload.account_id,
                ).where(StatementUpload.id == row.statement_upload_id)
            )
        ).one_or_none()
        if statement is not None:
            statement_link = transaction_schemas.TransactionStatementLink(
                id=statement.id,
                kind="cc",
                status=statement.status,
                account_id=statement.account_id,
            )
    elif row.bank_statement_upload_id is not None:
        statement = (
            await session.execute(
                select(
                    BankStatementUpload.id,
                    BankStatementUpload.status,
                    BankStatementUpload.account_id,
                ).where(BankStatementUpload.id == row.bank_statement_upload_id)
            )
        ).one_or_none()
        if statement is not None:
            statement_link = transaction_schemas.TransactionStatementLink(
                id=statement.id,
                kind="bank",
                status=statement.status,
                account_id=statement.account_id,
            )

    return email_link, sms_link, statement_link


async def get_transaction_detail(
    session: AsyncSession,
    transaction_id: int,
) -> transaction_schemas.TransactionDetailResponse | None:
    with session.no_autoflush:
        row = await session.scalar(
            select(Transaction)
            .options(
                joinedload(Transaction.account).noload(Account.cards),
                joinedload(Transaction.card),
            )
            .where(Transaction.id == transaction_id)
            .execution_options(autoflush=False)
        )
        if row is None:
            return None
        email, sms, statement = await _source_links(session, row)

    raw_description, raw_description_truncated = _bounded_detail_text(
        row.raw_description
    )
    note, note_truncated = _bounded_detail_text(row.note)
    summary = _transaction_read(row)
    account = row.account
    card = row.card
    return transaction_schemas.TransactionDetailResponse(
        **summary.model_dump(),
        raw_description=raw_description,
        raw_description_truncated=raw_description_truncated,
        note=note,
        note_truncated=note_truncated,
        category_confidence=row.category_confidence,
        category_model=row.category_model,
        category_input_hash=row.category_input_hash,
        category_vocab_version=row.category_vocab_version,
        categorized_at=row.categorized_at,
        review_reason=row.review_reason,
        last_notified_at=row.last_notified_at,
        notify_attempts=row.notify_attempts,
        notified_channel=row.notified_channel,
        account=transaction_schemas.TransactionAccountLink(
            id=account.id,
            bank=account.bank,
            label=account.label,
            type=account.type,
        )
        if account is not None
        else None,
        card=transaction_schemas.TransactionCardLink(
            id=card.id,
            label=card.label,
            card_mask=_safe_mask(card.card_mask),
            is_primary=bool(card.is_primary),
        )
        if card is not None
        else None,
        email=email,
        sms=sms,
        statement=statement,
        may_affect_cc_payment_state=bool(
            account is not None
            and account.type == "credit_card"
            and should_auto_reconcile_statement(row)
        ),
    )


async def get_transactions_by_ids(
    session: AsyncSession,
    ids: list[int],
) -> transaction_schemas.TransactionBatchResponse:
    with session.no_autoflush:
        rows = (
            (
                await session.execute(
                    select(Transaction)
                    .options(
                        noload(Transaction.account),
                        noload(Transaction.card),
                        defer(Transaction.raw_description),
                        defer(Transaction.note),
                        defer(Transaction.review_reason),
                    )
                    .where(Transaction.id.in_(ids))
                    .execution_options(autoflush=False)
                )
            )
            .unique()
            .scalars()
            .all()
        )
    by_id = {row.id: row for row in rows}
    return transaction_schemas.TransactionBatchResponse(
        items=[_transaction_read(by_id[row_id]) for row_id in ids if row_id in by_id],
        missing_ids=[row_id for row_id in ids if row_id not in by_id],
    )
