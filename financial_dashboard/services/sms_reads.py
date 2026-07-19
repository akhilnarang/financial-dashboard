import datetime

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import SmsMessage, Transaction
from financial_dashboard.schemas import sms as sms_schemas

_METADATA_LIMIT = 1_000
_LIST_ERROR_LIMIT = 1_000
_DETAIL_TEXT_LIMIT = 100_000
_ATTACHMENT_LIMIT = 100


def _summary_columns(*, error_limit: int = _LIST_ERROR_LIMIT):
    return (
        SmsMessage.id,
        func.substr(SmsMessage.bank, 1, _METADATA_LIMIT).label("bank"),
        case((func.length(SmsMessage.bank) > _METADATA_LIMIT, True), else_=False).label(
            "bank_truncated"
        ),
        func.substr(SmsMessage.sender, 1, _METADATA_LIMIT).label("sender"),
        case(
            (func.length(SmsMessage.sender) > _METADATA_LIMIT, True), else_=False
        ).label("sender_truncated"),
        SmsMessage.received_at,
        SmsMessage.created_at,
        SmsMessage.status,
        func.substr(SmsMessage.parse_error, 1, error_limit).label("parse_error"),
        case(
            (func.length(SmsMessage.parse_error) > error_limit, True),
            else_=False,
        ).label("parse_error_truncated"),
        SmsMessage.parsed_at,
        Transaction.id.label("transaction_id"),
        Transaction.email_type.label("transaction_email_type"),
        Transaction.direction.label("transaction_direction"),
        Transaction.source.label("transaction_source"),
    )


def _summary(row) -> sms_schemas.SmsRead:
    transaction = (
        sms_schemas.SmsTransactionLink(
            id=row.transaction_id,
            email_type=row.transaction_email_type,
            direction=row.transaction_direction,
            source=row.transaction_source,
        )
        if row.transaction_id is not None
        else None
    )
    return sms_schemas.SmsRead(
        id=row.id,
        bank=row.bank,
        bank_truncated=bool(row.bank_truncated),
        sender=row.sender,
        sender_truncated=bool(row.sender_truncated),
        received_at=row.received_at,
        created_at=row.created_at,
        status=row.status,
        parse_error=row.parse_error,
        parse_error_truncated=bool(row.parse_error_truncated),
        parsed_at=row.parsed_at,
        transaction=transaction,
    )


def _filters(
    *,
    sms_id: int | None,
    bank: str | None,
    status: str | None,
    transaction_id: int | None,
    parser_type: str | None,
    date_from: datetime.datetime | None,
    date_to: datetime.datetime | None,
) -> list:
    clauses = []
    if sms_id is not None:
        clauses.append(SmsMessage.id == sms_id)
    if bank is not None:
        clauses.append(func.lower(SmsMessage.bank) == bank.strip().lower())
    if status is not None:
        clauses.append(SmsMessage.status == status.strip())
    if transaction_id is not None:
        clauses.append(SmsMessage.transaction_id == transaction_id)
    if parser_type is not None:
        clauses.append(Transaction.email_type == parser_type.strip())
    if date_from is not None:
        clauses.append(SmsMessage.received_at >= date_from)
    if date_to is not None:
        clauses.append(SmsMessage.received_at <= date_to)
    return clauses


async def list_sms(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
    sms_id: int | None,
    bank: str | None,
    status: str | None,
    transaction_id: int | None,
    parser_type: str | None,
    date_from: datetime.datetime | None,
    date_to: datetime.datetime | None,
) -> sms_schemas.SmsListResponse:
    clauses = _filters(
        sms_id=sms_id,
        bank=bank,
        status=status,
        transaction_id=transaction_id,
        parser_type=parser_type,
        date_from=date_from,
        date_to=date_to,
    )
    base_join = SmsMessage.__table__.outerjoin(
        Transaction.__table__, Transaction.id == SmsMessage.transaction_id
    )
    with session.no_autoflush:
        total_count = await session.scalar(
            select(func.count(SmsMessage.id))
            .select_from(base_join)
            .where(*clauses)
            .execution_options(autoflush=False)
        )
        rows = (
            await session.execute(
                select(*_summary_columns())
                .select_from(base_join)
                .where(*clauses)
                .order_by(SmsMessage.id.desc())
                .offset(offset)
                .limit(limit)
                .execution_options(autoflush=False)
            )
        ).all()
    items = [_summary(row) for row in rows]
    return sms_schemas.SmsListResponse(
        items=items,
        returned_count=len(items),
        total_count=total_count or 0,
        limit=limit,
        offset=offset,
    )


async def get_sms_detail(
    session: AsyncSession,
    sms_id: int,
) -> sms_schemas.SmsDetailResponse | None:
    base_join = SmsMessage.__table__.outerjoin(
        Transaction.__table__, Transaction.id == SmsMessage.transaction_id
    )
    with session.no_autoflush:
        sms = (
            await session.execute(
                select(
                    *_summary_columns(error_limit=_DETAIL_TEXT_LIMIT),
                    func.substr(SmsMessage.body, 1, _DETAIL_TEXT_LIMIT).label("body"),
                    case(
                        (func.length(SmsMessage.body) > _DETAIL_TEXT_LIMIT, True),
                        else_=False,
                    ).label("body_truncated"),
                )
                .select_from(base_join)
                .where(SmsMessage.id == sms_id)
                .execution_options(autoflush=False)
            )
        ).one_or_none()
        if sms is None:
            return None
        attached_ids = (
            await session.scalars(
                select(Transaction.id)
                .where(Transaction.sms_message_id == sms_id)
                .order_by(Transaction.id)
                .limit(_ATTACHMENT_LIMIT + 1)
                .execution_options(autoflush=False)
            )
        ).all()

    summary = _summary(sms)
    return sms_schemas.SmsDetailResponse(
        **summary.model_dump(),
        body=sms.body,
        body_truncated=bool(sms.body_truncated),
        attached_transaction_ids=attached_ids[:_ATTACHMENT_LIMIT],
        attached_transactions_truncated=len(attached_ids) > _ATTACHMENT_LIMIT,
    )


async def get_sms_by_ids(
    session: AsyncSession,
    ids: list[int],
) -> sms_schemas.SmsBatchResponse:
    base_join = SmsMessage.__table__.outerjoin(
        Transaction.__table__, Transaction.id == SmsMessage.transaction_id
    )
    with session.no_autoflush:
        rows = (
            await session.execute(
                select(*_summary_columns())
                .select_from(base_join)
                .where(SmsMessage.id.in_(ids))
                .execution_options(autoflush=False)
            )
        ).all()
    by_id = {row.id: _summary(row) for row in rows}
    return sms_schemas.SmsBatchResponse(
        items=[by_id[row_id] for row_id in ids if row_id in by_id],
        missing_ids=[row_id for row_id in ids if row_id not in by_id],
    )
