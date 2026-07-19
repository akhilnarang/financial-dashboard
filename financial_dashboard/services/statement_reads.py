"""Bounded CC and bank statement queries for the JSON API."""

import datetime
import json
from typing import NamedTuple

from sqlalchemy import case, func, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.masks import display_mask
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    StatementUpload,
    Transaction,
)
from financial_dashboard.schemas import statements as statement_schemas
from financial_dashboard.services.read_helpers import order_batch

_METADATA_LIMIT = 1_000
_VALUE_LIMIT = 256
_LIST_ERROR_LIMIT = 1_000
_DETAIL_ERROR_LIMIT = 100_000
_RECONCILIATION_LIMIT = 1_000_000
_TRANSACTION_ID_LIMIT = 100


class _StatementPage(NamedTuple):
    """Projected rows and total count for one statement page."""

    rows: list[Row]
    total_count: int


def _text(column, label: str, limit: int = _VALUE_LIMIT):
    """Build a bounded text projection with a stable result label."""
    return func.substr(column, 1, limit).label(label)


def _truncated(column, label: str, limit: int):
    """Build the truncation flag paired with a bounded text projection."""
    return case((func.length(column) > limit, True), else_=False).label(label)


def _account(row) -> statement_schemas.StatementAccountLink | None:
    """Map optional projected account columns to a provenance link."""
    if row.linked_account_id is None:
        return None
    return statement_schemas.StatementAccountLink(
        id=row.linked_account_id,
        bank=row.account_bank,
        label=row.account_label,
        type=row.account_type,
    )


def _common_columns(model, *, error_limit: int):
    """Build columns shared by CC and bank statement summaries."""
    return (
        model.id,
        model.account_id,
        model.email_id,
        _text(model.bank, "bank", _METADATA_LIMIT),
        _text(model.filename, "filename", _METADATA_LIMIT),
        _truncated(model.filename, "filename_truncated", _METADATA_LIMIT),
        _text(model.status, "status", 64),
        model.parsed_txn_count,
        model.matched_count,
        model.missing_count,
        model.imported_count,
        _text(model.error, "error", error_limit),
        _truncated(model.error, "error_truncated", error_limit),
        model.created_at,
        Account.id.label("linked_account_id"),
        _text(Account.bank, "account_bank", _METADATA_LIMIT),
        _text(Account.label, "account_label", _METADATA_LIMIT),
        _text(Account.type, "account_type", 64),
    )


def _cc_columns(*, error_limit: int):
    """Build the bounded CC statement projection."""
    return (
        *_common_columns(StatementUpload, error_limit=error_limit),
        _text(StatementUpload.source_kind, "source_kind", 64),
        func.substr(StatementUpload.card_number, -4, 4).label("card_number"),
        _text(StatementUpload.statement_name, "statement_name", _METADATA_LIMIT),
        _truncated(
            StatementUpload.statement_name,
            "statement_name_truncated",
            _METADATA_LIMIT,
        ),
        _text(StatementUpload.due_date, "due_date"),
        _text(StatementUpload.total_amount_due, "total_amount_due"),
        _text(StatementUpload.minimum_amount_due, "minimum_amount_due"),
        _text(StatementUpload.payment_status, "payment_status", 64),
        StatementUpload.payment_paid_at,
        StatementUpload.payment_paid_amount,
        StatementUpload.payment_last_reminded_at,
    )


def _bank_columns(*, error_limit: int):
    """Build the bounded bank statement projection."""
    return (
        *_common_columns(BankStatementUpload, error_limit=error_limit),
        func.substr(BankStatementUpload.account_number, -4, 4).label("account_number"),
        _text(
            BankStatementUpload.account_holder_name,
            "account_holder_name",
            _METADATA_LIMIT,
        ),
        _truncated(
            BankStatementUpload.account_holder_name,
            "account_holder_name_truncated",
            _METADATA_LIMIT,
        ),
        _text(BankStatementUpload.opening_balance, "opening_balance"),
        _text(BankStatementUpload.closing_balance, "closing_balance"),
        _text(BankStatementUpload.statement_period_start, "statement_period_start"),
        _text(BankStatementUpload.statement_period_end, "statement_period_end"),
    )


def _cc_read(row) -> statement_schemas.CcStatementRead:
    """Map one projected CC statement to its summary schema."""
    return statement_schemas.CcStatementRead(
        id=row.id,
        account_id=row.account_id,
        email_id=row.email_id,
        bank=row.bank,
        filename=row.filename,
        filename_truncated=bool(row.filename_truncated),
        status=row.status,
        parsed_transaction_count=row.parsed_txn_count,
        matched_count=row.matched_count,
        missing_count=row.missing_count,
        imported_count=row.imported_count,
        error=row.error,
        error_truncated=bool(row.error_truncated),
        created_at=row.created_at,
        account=_account(row),
        source_kind=row.source_kind,
        card_mask=display_mask(row.card_number),
        statement_name=row.statement_name,
        statement_name_truncated=bool(row.statement_name_truncated),
        due_date=row.due_date,
        total_amount_due=row.total_amount_due,
        minimum_amount_due=row.minimum_amount_due,
        payment_status=row.payment_status,
        payment_paid_at=row.payment_paid_at,
        payment_paid_amount=row.payment_paid_amount,
        payment_last_reminded_at=row.payment_last_reminded_at,
    )


def _bank_read(row) -> statement_schemas.BankStatementRead:
    """Map one projected bank statement to its summary schema."""
    return statement_schemas.BankStatementRead(
        id=row.id,
        account_id=row.account_id,
        email_id=row.email_id,
        bank=row.bank,
        filename=row.filename,
        filename_truncated=bool(row.filename_truncated),
        status=row.status,
        parsed_transaction_count=row.parsed_txn_count,
        matched_count=row.matched_count,
        missing_count=row.missing_count,
        imported_count=row.imported_count,
        error=row.error,
        error_truncated=bool(row.error_truncated),
        created_at=row.created_at,
        account=_account(row),
        account_mask=display_mask(row.account_number),
        account_holder_name=row.account_holder_name,
        account_holder_name_truncated=bool(row.account_holder_name_truncated),
        opening_balance=row.opening_balance,
        closing_balance=row.closing_balance,
        statement_period_start=row.statement_period_start,
        statement_period_end=row.statement_period_end,
    )


def _filters(
    model,
    *,
    statement_id: int | None,
    account_id: int | None,
    email_id: int | None,
    bank: str | None,
    status: str | None,
    date_from: datetime.datetime | None,
    date_to: datetime.datetime | None,
) -> list:
    """Build exact-match clauses shared by both statement kinds."""
    clauses = []
    if statement_id is not None:
        clauses.append(model.id == statement_id)
    if account_id is not None:
        clauses.append(model.account_id == account_id)
    if email_id is not None:
        clauses.append(model.email_id == email_id)
    if bank is not None:
        clauses.append(func.lower(model.bank) == bank.strip().lower())
    if status is not None:
        clauses.append(model.status == status.strip())
    if date_from is not None:
        clauses.append(model.created_at >= date_from)
    if date_to is not None:
        clauses.append(model.created_at <= date_to)
    return clauses


def _account_join(model):
    """Join a statement model to its optional account projection."""
    return model.__table__.outerjoin(Account.__table__, Account.id == model.account_id)


async def _statement_page(
    session: AsyncSession,
    model,
    columns,
    clauses: list,
    *,
    limit: int,
    offset: int,
) -> _StatementPage:
    """Execute the shared count and stable page queries for a statement model."""
    with session.no_autoflush:
        total_count = await session.scalar(
            select(func.count(model.id))
            .where(*clauses)
            .execution_options(autoflush=False)
        )
        rows = (
            await session.execute(
                select(*columns)
                .select_from(_account_join(model))
                .where(*clauses)
                .order_by(model.id.desc())
                .offset(offset)
                .limit(limit)
                .execution_options(autoflush=False)
            )
        ).all()
    return _StatementPage(list(rows), total_count or 0)


async def _statement_batch_rows(
    session: AsyncSession,
    model,
    columns,
    ids: list[int],
) -> list[Row]:
    """Execute the shared explicit-ID query for a statement model."""
    with session.no_autoflush:
        return list(
            (
                await session.execute(
                    select(*columns)
                    .select_from(_account_join(model))
                    .where(model.id.in_(ids))
                    .execution_options(autoflush=False)
                )
            ).all()
        )


async def list_cc_statements(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
    statement_id: int | None,
    account_id: int | None,
    email_id: int | None,
    bank: str | None,
    status: str | None,
    date_from: datetime.datetime | None,
    date_to: datetime.datetime | None,
) -> statement_schemas.CcStatementListResponse:
    """Return one stable page of CC statement summaries."""
    clauses = _filters(
        StatementUpload,
        statement_id=statement_id,
        account_id=account_id,
        email_id=email_id,
        bank=bank,
        status=status,
        date_from=date_from,
        date_to=date_to,
    )
    page = await _statement_page(
        session,
        StatementUpload,
        _cc_columns(error_limit=_LIST_ERROR_LIMIT),
        clauses,
        limit=limit,
        offset=offset,
    )
    items = [_cc_read(row) for row in page.rows]
    return statement_schemas.CcStatementListResponse(
        items=items,
        returned_count=len(items),
        total_count=page.total_count,
        limit=limit,
        offset=offset,
    )


async def list_bank_statements(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
    statement_id: int | None,
    account_id: int | None,
    email_id: int | None,
    bank: str | None,
    status: str | None,
    date_from: datetime.datetime | None,
    date_to: datetime.datetime | None,
) -> statement_schemas.BankStatementListResponse:
    """Return one stable page of bank statement summaries."""
    clauses = _filters(
        BankStatementUpload,
        statement_id=statement_id,
        account_id=account_id,
        email_id=email_id,
        bank=bank,
        status=status,
        date_from=date_from,
        date_to=date_to,
    )
    page = await _statement_page(
        session,
        BankStatementUpload,
        _bank_columns(error_limit=_LIST_ERROR_LIMIT),
        clauses,
        limit=limit,
        offset=offset,
    )
    items = [_bank_read(row) for row in page.rows]
    return statement_schemas.BankStatementListResponse(
        items=items,
        returned_count=len(items),
        total_count=page.total_count,
        limit=limit,
        offset=offset,
    )


def _valid_id(value) -> int | None:
    """Accept positive integer IDs from untrusted reconciliation JSON."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _reconciliation_summary(
    data: str | None,
    data_length: int | None,
    imported_ids: list[int],
) -> statement_schemas.StatementReconciliationSummary:
    """Extract bounded transaction IDs and counts from stored reconciliation JSON."""
    imported_truncated = len(imported_ids) > _TRANSACTION_ID_LIMIT
    if data_length is None:
        status = "absent"
        matched_ids: list[int] = []
        matched_truncated = False
        ambiguous_count = None
        import_error_count = None
    elif data_length > _RECONCILIATION_LIMIT:
        status = "too_large"
        matched_ids = []
        matched_truncated = False
        ambiguous_count = None
        import_error_count = None
    else:
        try:
            parsed = json.loads(data or "")
            if not isinstance(parsed, dict):
                raise ValueError
            if "matched" not in parsed or "missing" not in parsed:
                raise ValueError
            matched = parsed["matched"]
            missing = parsed["missing"]
            if not isinstance(matched, list) or not isinstance(missing, list):
                raise ValueError
            all_entries = [*matched, *missing]
            if not all(isinstance(entry, dict) for entry in all_entries):
                raise ValueError
            matched_candidates = [
                identifier
                for entry in matched
                if (identifier := _valid_id(entry.get("db_txn_id"))) is not None
            ]
            matched_ids = list(dict.fromkeys(matched_candidates))
            matched_truncated = len(matched_ids) > _TRANSACTION_ID_LIMIT
            ambiguous_count = sum(bool(entry.get("ambiguous")) for entry in all_entries)
            import_error_count = sum(
                bool(entry.get("import_error")) for entry in all_entries
            )
            status = "parsed"
        except TypeError, ValueError, json.JSONDecodeError:
            status = "malformed"
            matched_ids = []
            matched_truncated = False
            ambiguous_count = None
            import_error_count = None

    return statement_schemas.StatementReconciliationSummary(
        status=status,
        matched_transaction_ids=matched_ids[:_TRANSACTION_ID_LIMIT],
        matched_transaction_ids_truncated=matched_truncated,
        imported_transaction_ids=imported_ids[:_TRANSACTION_ID_LIMIT],
        imported_transaction_ids_truncated=imported_truncated,
        ambiguous_entry_count=ambiguous_count,
        import_error_entry_count=import_error_count,
    )


async def _statement_detail_row(
    session: AsyncSession,
    model,
    columns,
    statement_id: int,
) -> Row | None:
    """Load one summary plus reconciliation JSON only when it is bounded."""
    return (
        await session.execute(
            select(
                *columns,
                func.length(model.reconciliation_data).label("reconciliation_length"),
                case(
                    (
                        func.length(model.reconciliation_data) <= _RECONCILIATION_LIMIT,
                        model.reconciliation_data,
                    )
                ).label("reconciliation_data"),
            )
            .select_from(_account_join(model))
            .where(model.id == statement_id)
            .execution_options(autoflush=False)
        )
    ).one_or_none()


async def _imported_ids(
    session: AsyncSession,
    link_column,
    statement_id: int,
) -> list[int]:
    """Load transaction IDs canonically linked to one statement upload."""
    return list(
        (
            await session.scalars(
                select(Transaction.id)
                .where(link_column == statement_id)
                .order_by(Transaction.id)
                .limit(_TRANSACTION_ID_LIMIT + 1)
                .execution_options(autoflush=False)
            )
        ).all()
    )


async def get_cc_statement_detail(
    session: AsyncSession,
    statement_id: int,
) -> statement_schemas.CcStatementDetailResponse | None:
    """Return one CC statement with bounded reconciliation evidence."""
    with session.no_autoflush:
        row = await _statement_detail_row(
            session,
            StatementUpload,
            _cc_columns(error_limit=_DETAIL_ERROR_LIMIT),
            statement_id,
        )
        if row is None:
            return None
        imported_ids = await _imported_ids(
            session,
            Transaction.statement_upload_id,
            statement_id,
        )
    summary = _cc_read(row)
    return statement_schemas.CcStatementDetailResponse(
        **summary.model_dump(),
        reconciliation=_reconciliation_summary(
            row.reconciliation_data,
            row.reconciliation_length,
            imported_ids,
        ),
    )


async def get_bank_statement_detail(
    session: AsyncSession,
    statement_id: int,
) -> statement_schemas.BankStatementDetailResponse | None:
    """Return one bank statement with bounded reconciliation evidence."""
    with session.no_autoflush:
        row = await _statement_detail_row(
            session,
            BankStatementUpload,
            _bank_columns(error_limit=_DETAIL_ERROR_LIMIT),
            statement_id,
        )
        if row is None:
            return None
        imported_ids = await _imported_ids(
            session,
            Transaction.bank_statement_upload_id,
            statement_id,
        )
    summary = _bank_read(row)
    return statement_schemas.BankStatementDetailResponse(
        **summary.model_dump(),
        reconciliation=_reconciliation_summary(
            row.reconciliation_data,
            row.reconciliation_length,
            imported_ids,
        ),
    )


async def get_cc_statements_by_ids(
    session: AsyncSession,
    ids: list[int],
) -> statement_schemas.CcStatementBatchResponse:
    """Return CC summaries in requested ID order and report missing IDs."""
    rows = await _statement_batch_rows(
        session,
        StatementUpload,
        _cc_columns(error_limit=_LIST_ERROR_LIMIT),
        ids,
    )
    ordered = order_batch(ids, {row.id: _cc_read(row) for row in rows})
    return statement_schemas.CcStatementBatchResponse(
        items=ordered.items,
        missing_ids=ordered.missing_ids,
    )


async def get_bank_statements_by_ids(
    session: AsyncSession,
    ids: list[int],
) -> statement_schemas.BankStatementBatchResponse:
    """Return bank summaries in requested ID order and report missing IDs."""
    rows = await _statement_batch_rows(
        session,
        BankStatementUpload,
        _bank_columns(error_limit=_LIST_ERROR_LIMIT),
        ids,
    )
    ordered = order_batch(ids, {row.id: _bank_read(row) for row in rows})
    return statement_schemas.BankStatementBatchResponse(
        items=ordered.items,
        missing_ids=ordered.missing_ids,
    )
