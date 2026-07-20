"""Bounded email source and raw-body queries for the JSON API."""

import datetime
import logging

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import (
    BankStatementUpload,
    CasUpload,
    Email,
    FetchRule,
    StatementUpload,
    Transaction,
)
from financial_dashboard.integrations.email.body import (
    _extract_html_body,
    _extract_text_body,
    load_or_fetch_raw_email,
)
from financial_dashboard.schemas import emails as email_schemas
from financial_dashboard.services.read_helpers import bound_text, order_batch

_LIST_ERROR_LIMIT = 1_000
_HEADER_LIMIT = 1_000
_DETAIL_ID_LIMIT = 10_000
_DETAIL_ERROR_LIMIT = 100_000
_RAW_BODY_LIMIT = 100_000
_RAW_BYTES_LIMIT = 10_000_000
_LINK_LIMIT = 10

logger = logging.getLogger(__name__)


class EmailRawReadError(Exception):
    """A sanitized raw-email failure and its intended HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _raw_load_reason(error: str | None) -> str:
    """Classify a loader error without retaining paths or credential details."""
    normalized = (error or "").lower()
    if "no source/remote id" in normalized:
        return "missing-source-metadata"
    if "source" in normalized and "not found" in normalized:
        return "source-not-found"
    if "credential decryption" in normalized:
        return "credential-decryption-failed"
    if "unknown provider" in normalized:
        return "unsupported-provider"
    if "provider returned no data" in normalized:
        return "provider-returned-no-data"
    return "loader-failed"


def _email_columns():
    """Build the bounded SQL projection shared by email summary reads."""
    return (
        Email.id,
        Email.provider,
        Email.source_id,
        func.substr(Email.sender, 1, _HEADER_LIMIT).label("sender"),
        case((func.length(Email.sender) > _HEADER_LIMIT, True), else_=False).label(
            "sender_truncated"
        ),
        func.substr(Email.subject, 1, _HEADER_LIMIT).label("subject"),
        case((func.length(Email.subject) > _HEADER_LIMIT, True), else_=False).label(
            "subject_truncated"
        ),
        Email.received_at,
        Email.fetched_at,
        Email.status,
        func.substr(Email.error, 1, _LIST_ERROR_LIMIT).label("error"),
        case((func.length(Email.error) > _LIST_ERROR_LIMIT, True), else_=False).label(
            "error_truncated"
        ),
        FetchRule.id.label("rule_id"),
        FetchRule.bank.label("rule_bank"),
        FetchRule.email_kind.label("rule_email_kind"),
    )


async def _load_links(
    session: AsyncSession,
    email_ids: list[int],
) -> tuple[
    dict[int, list[email_schemas.EmailTransactionLink]],
    set[int],
    dict[int, list[email_schemas.EmailStatementLink]],
    set[int],
]:
    """Load capped transaction and statement links for several emails."""
    if not email_ids:
        return {}, set(), {}, set()

    txn_ranked = (
        select(
            Transaction.email_id.label("email_id"),
            Transaction.id.label("id"),
            Transaction.email_type.label("email_type"),
            Transaction.direction.label("direction"),
            Transaction.source.label("source"),
            func.row_number()
            .over(partition_by=Transaction.email_id, order_by=Transaction.id)
            .label("position"),
        )
        .where(Transaction.email_id.in_(email_ids))
        .subquery()
    )
    txn_rows = (
        await session.execute(
            select(txn_ranked)
            .where(txn_ranked.c.position <= _LINK_LIMIT + 1)
            .order_by(txn_ranked.c.email_id, txn_ranked.c.id)
            .execution_options(autoflush=False)
        )
    ).all()
    transactions: dict[int, list[email_schemas.EmailTransactionLink]] = {}
    transactions_truncated: set[int] = set()
    for row in txn_rows:
        links = transactions.setdefault(row.email_id, [])
        if len(links) == _LINK_LIMIT:
            transactions_truncated.add(row.email_id)
            continue
        links.append(
            email_schemas.EmailTransactionLink(
                id=row.id,
                email_type=row.email_type,
                direction=row.direction,
                source=row.source,
            )
        )

    cc_ranked = (
        select(
            StatementUpload.email_id.label("email_id"),
            StatementUpload.id.label("id"),
            StatementUpload.status.label("status"),
            func.row_number()
            .over(partition_by=StatementUpload.email_id, order_by=StatementUpload.id)
            .label("position"),
        )
        .where(StatementUpload.email_id.in_(email_ids))
        .subquery()
    )
    bank_ranked = (
        select(
            BankStatementUpload.email_id.label("email_id"),
            BankStatementUpload.id.label("id"),
            BankStatementUpload.status.label("status"),
            func.row_number()
            .over(
                partition_by=BankStatementUpload.email_id,
                order_by=BankStatementUpload.id,
            )
            .label("position"),
        )
        .where(BankStatementUpload.email_id.in_(email_ids))
        .subquery()
    )
    cc_rows = (
        await session.execute(
            select(cc_ranked)
            .where(cc_ranked.c.position <= _LINK_LIMIT + 1)
            .execution_options(autoflush=False)
        )
    ).all()
    cas_ranked = (
        select(
            CasUpload.email_id.label("email_id"),
            CasUpload.id.label("id"),
            case((CasUpload.portfolio_ok.is_(True), "parsed"), else_="invalid").label(
                "status"
            ),
            func.row_number()
            .over(partition_by=CasUpload.email_id, order_by=CasUpload.id)
            .label("position"),
        )
        .where(CasUpload.email_id.in_(email_ids))
        .subquery()
    )
    bank_rows = (
        await session.execute(
            select(bank_ranked)
            .where(bank_ranked.c.position <= _LINK_LIMIT + 1)
            .execution_options(autoflush=False)
        )
    ).all()
    cas_rows = (
        await session.execute(
            select(cas_ranked)
            .where(cas_ranked.c.position <= _LINK_LIMIT + 1)
            .execution_options(autoflush=False)
        )
    ).all()
    statement_candidates: dict[int, list[email_schemas.EmailStatementLink]] = {}
    for kind, rows in (("cc", cc_rows), ("bank", bank_rows), ("cas", cas_rows)):
        for row in rows:
            statement_candidates.setdefault(row.email_id, []).append(
                email_schemas.EmailStatementLink(
                    id=row.id,
                    kind=kind,
                    status=row.status,
                )
            )

    statements: dict[int, list[email_schemas.EmailStatementLink]] = {}
    statements_truncated: set[int] = set()
    for email_id, candidates in statement_candidates.items():
        kind_order = {"cc": 0, "bank": 1, "cas": 2}
        candidates.sort(key=lambda item: (kind_order[item.kind], item.id))
        statements[email_id] = candidates[:_LINK_LIMIT]
        if len(candidates) > _LINK_LIMIT:
            statements_truncated.add(email_id)

    return transactions, transactions_truncated, statements, statements_truncated


def _summary(
    row,
    *,
    transactions: dict[int, list[email_schemas.EmailTransactionLink]],
    transactions_truncated: set[int],
    statements: dict[int, list[email_schemas.EmailStatementLink]],
    statements_truncated: set[int],
    detail_error: tuple[str | None, bool] | None = None,
) -> email_schemas.EmailRead:
    """Map one projected email row and its capped links to a schema."""
    error, error_truncated = (
        detail_error
        if detail_error is not None
        else (row.error, bool(row.error_truncated))
    )
    return email_schemas.EmailRead(
        id=row.id,
        provider=row.provider,
        source_id=row.source_id,
        sender=row.sender,
        sender_truncated=bool(row.sender_truncated),
        subject=row.subject,
        subject_truncated=bool(row.subject_truncated),
        received_at=row.received_at,
        fetched_at=row.fetched_at,
        status=row.status,
        error=error,
        error_truncated=error_truncated,
        rule=email_schemas.EmailRuleSummary(
            id=row.rule_id,
            bank=row.rule_bank,
            email_kind=row.rule_email_kind,
        )
        if row.rule_id is not None
        else None,
        transactions=transactions.get(row.id, []),
        transactions_truncated=row.id in transactions_truncated,
        statements=statements.get(row.id, []),
        statements_truncated=row.id in statements_truncated,
    )


def _filters(
    *,
    email_id: int | None,
    source_id: int | None,
    rule_id: int | None,
    provider: str | None,
    status: str | None,
    bank: str | None,
    email_kind: str | None,
    transaction_id: int | None,
    parser_type: str | None,
    direction: str | None,
    date_from: datetime.datetime | None,
    date_to: datetime.datetime | None,
    query: str | None,
) -> list:
    """Build email clauses, correlating all transaction filters to one row."""
    clauses = []
    if email_id is not None:
        clauses.append(Email.id == email_id)
    if source_id is not None:
        clauses.append(Email.source_id == source_id)
    if rule_id is not None:
        clauses.append(Email.rule_id == rule_id)
    if provider is not None:
        clauses.append(Email.provider == provider.strip())
    if status is not None:
        clauses.append(Email.status == status.strip())
    if bank is not None:
        clauses.append(func.lower(FetchRule.bank) == bank.strip().lower())
    if email_kind is not None:
        clauses.append(FetchRule.email_kind == email_kind.strip())
    transaction_clauses = [Transaction.email_id == Email.id]
    if transaction_id is not None:
        transaction_clauses.append(Transaction.id == transaction_id)
    if parser_type is not None:
        transaction_clauses.append(Transaction.email_type == parser_type.strip())
    if direction is not None:
        transaction_clauses.append(Transaction.direction == direction.strip())
    if len(transaction_clauses) > 1:
        clauses.append(select(Transaction.id).where(*transaction_clauses).exists())
    if date_from is not None:
        clauses.append(Email.received_at >= date_from)
    if date_to is not None:
        clauses.append(Email.received_at <= date_to)
    if query is not None:
        pattern = f"%{query.strip()}%"
        clauses.append(or_(Email.sender.ilike(pattern), Email.subject.ilike(pattern)))
    return clauses


async def list_emails(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
    email_id: int | None,
    source_id: int | None,
    rule_id: int | None,
    provider: str | None,
    status: str | None,
    bank: str | None,
    email_kind: str | None,
    transaction_id: int | None,
    parser_type: str | None,
    direction: str | None,
    date_from: datetime.datetime | None,
    date_to: datetime.datetime | None,
    query: str | None,
) -> email_schemas.EmailListResponse:
    """Return one stable email metadata page without raw bodies or provider IDs."""
    clauses = _filters(
        email_id=email_id,
        source_id=source_id,
        rule_id=rule_id,
        provider=provider,
        status=status,
        bank=bank,
        email_kind=email_kind,
        transaction_id=transaction_id,
        parser_type=parser_type,
        direction=direction,
        date_from=date_from,
        date_to=date_to,
        query=query,
    )
    base_join = Email.__table__.outerjoin(
        FetchRule.__table__, FetchRule.id == Email.rule_id
    )
    with session.no_autoflush:
        total_count = await session.scalar(
            select(func.count(Email.id))
            .select_from(base_join)
            .where(*clauses)
            .execution_options(autoflush=False)
        )
        rows = (
            await session.execute(
                select(*_email_columns())
                .select_from(base_join)
                .where(*clauses)
                .order_by(Email.id.desc())
                .offset(offset)
                .limit(limit)
                .execution_options(autoflush=False)
            )
        ).all()
        links = await _load_links(session, [row.id for row in rows])
    items = [
        _summary(
            row,
            transactions=links[0],
            transactions_truncated=links[1],
            statements=links[2],
            statements_truncated=links[3],
        )
        for row in rows
    ]
    return email_schemas.EmailListResponse(
        items=items,
        returned_count=len(items),
        total_count=total_count or 0,
        limit=limit,
        offset=offset,
    )


async def get_email_detail(
    session: AsyncSession,
    email_id: int,
) -> email_schemas.EmailDetailResponse | None:
    """Return one email's bounded metadata and source provenance."""
    base_join = Email.__table__.outerjoin(
        FetchRule.__table__, FetchRule.id == Email.rule_id
    )
    with session.no_autoflush:
        row = (
            await session.execute(
                select(*_email_columns(), Email.message_id, Email.remote_id)
                .select_from(base_join)
                .where(Email.id == email_id)
                .execution_options(autoflush=False)
            )
        ).one_or_none()
        if row is None:
            return None
        full_error = await session.scalar(
            select(Email.error)
            .where(Email.id == email_id)
            .execution_options(autoflush=False)
        )
        links = await _load_links(session, [email_id])

    error = bound_text(full_error, _DETAIL_ERROR_LIMIT)
    summary = _summary(
        row,
        transactions=links[0],
        transactions_truncated=links[1],
        statements=links[2],
        statements_truncated=links[3],
        detail_error=error,
    )
    message_id, message_id_truncated = bound_text(row.message_id, _DETAIL_ID_LIMIT)
    remote_id, remote_id_truncated = bound_text(row.remote_id, _DETAIL_ID_LIMIT)
    assert message_id is not None
    return email_schemas.EmailDetailResponse(
        **summary.model_dump(),
        message_id=message_id,
        message_id_truncated=message_id_truncated,
        remote_id=remote_id,
        remote_id_truncated=remote_id_truncated,
    )


async def get_emails_by_ids(
    session: AsyncSession,
    ids: list[int],
) -> email_schemas.EmailBatchResponse:
    """Return summaries in requested ID order and report missing IDs."""
    base_join = Email.__table__.outerjoin(
        FetchRule.__table__, FetchRule.id == Email.rule_id
    )
    with session.no_autoflush:
        rows = (
            await session.execute(
                select(*_email_columns())
                .select_from(base_join)
                .where(Email.id.in_(ids))
                .execution_options(autoflush=False)
            )
        ).all()
        links = await _load_links(session, [row.id for row in rows])
    by_id = {
        row.id: _summary(
            row,
            transactions=links[0],
            transactions_truncated=links[1],
            statements=links[2],
            statements_truncated=links[3],
        )
        for row in rows
    }
    ordered = order_batch(ids, by_id)
    return email_schemas.EmailBatchResponse(
        items=ordered.items,
        missing_ids=ordered.missing_ids,
    )


async def get_email_raw(
    session: AsyncSession,
    email_id: int,
) -> email_schemas.EmailRawResponse:
    """Load and extract one bounded raw email body without mutating its row."""
    with session.no_autoflush:
        email = await session.get(Email, email_id)
    if email is None:
        await session.rollback()
        raise EmailRawReadError(404, "Email not found")

    # Provider fallback opens its own short-lived session. Detach the simple
    # source row and release this request's connection before any provider I/O
    # so concurrent raw reads cannot deadlock a small connection pool.
    session.expunge(email)
    await session.rollback()

    try:
        raw_email_result = await load_or_fetch_raw_email(email)
    except Exception:
        logger.warning(
            "Raw email loader raised for email %d (loader-exception)", email_id
        )
        raise EmailRawReadError(424, "Raw email source is unavailable") from None
    if raw_email_result.raw_bytes is None:
        reason = _raw_load_reason(raw_email_result.error)
        logger.warning("Raw email unavailable for email %d (%s)", email_id, reason)
        raise EmailRawReadError(424, "Raw email source is unavailable")
    raw_bytes = raw_email_result.raw_bytes
    if len(raw_bytes) > _RAW_BYTES_LIMIT:
        raise EmailRawReadError(413, "Raw email source exceeds the read limit")

    try:
        body = _extract_text_body(raw_bytes)
    except LookupError:
        body = None
    content_type = "text/plain"
    if body is None:
        try:
            body = _extract_html_body(raw_bytes)
        except LookupError:
            body = None
        content_type = "text/html"
    if body is None:
        raise EmailRawReadError(422, "Raw email has no readable body")
    bounded_body, truncated = bound_text(body, _RAW_BODY_LIMIT)
    assert bounded_body is not None
    return email_schemas.EmailRawResponse(
        email_id=email_id,
        content_type=content_type,
        body=bounded_body,
        body_truncated=truncated,
        raw_byte_size=len(raw_bytes),
    )
