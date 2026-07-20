"""Side-effect-free parsing and reconciliation for stored statement uploads."""

import asyncio
import datetime
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.config import get_fernet
from financial_dashboard.core.dates import parse_date
from financial_dashboard.core.masks import display_mask
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    StatementUpload,
    Transaction,
)
from financial_dashboard.schemas import statements as statement_schemas
from financial_dashboard.services.statements.bank import (
    parse_bank_statement,
    reconcile_bank_statement,
)
from financial_dashboard.services.statements.cc import (
    load_account_card_masks,
    parse_statement,
    reconcile_statement,
)
from financial_dashboard.services.statements.dates import (
    bank_stmt_date_range,
    cc_stmt_date_range,
)
from financial_dashboard.services.statements.shared import (
    STMT_RECONCILE_DATE_BUFFER_DAYS,
)

_ROW_LIMIT = 100
_FILE_SIZE_LIMIT = 25_000_000
_TEXT_LIMIT = 1_000


class StatementPreviewError(Exception):
    """A sanitized statement-preview failure and its intended HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class _LoadedStatement:
    kind: Literal["cc", "bank"]
    statement_id: int
    account_id: int
    bank: str
    path: Path
    password: str | None


def _bounded(value: object | None, limit: int = _TEXT_LIMIT) -> str | None:
    """Convert optional parser text to a bounded response value."""
    return str(value)[:limit] if value is not None else None


async def _load_statement(
    session: AsyncSession,
    kind: Literal["cc", "bank"],
    statement_id: int,
) -> _LoadedStatement | None:
    """Capture parser inputs without exposing paths or retaining a DB transaction."""
    with session.no_autoflush:
        if kind == "cc":
            upload = await session.get(StatementUpload, statement_id)
            if upload is None:
                return None
            if upload.source_kind == "email_summary":
                raise StatementPreviewError(409, "Email-summary statement has no PDF")
        else:
            upload = await session.get(BankStatementUpload, statement_id)
            if upload is None:
                return None
        account = await session.get(Account, upload.account_id)

    if not upload.file_path:
        raise StatementPreviewError(404, "Statement PDF is unavailable")
    path = Path(upload.file_path)
    if not path.is_file():
        raise StatementPreviewError(404, "Statement PDF is unavailable")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise StatementPreviewError(404, "Statement PDF is unavailable") from exc
    if size > _FILE_SIZE_LIMIT:
        raise StatementPreviewError(413, "Statement PDF exceeds preview limit")

    password = None
    if account is not None and account.statement_password:
        try:
            password = (
                get_fernet().decrypt(account.statement_password.encode()).decode()
            )
        except Exception:
            password = None
    loaded = _LoadedStatement(
        kind=kind,
        statement_id=statement_id,
        account_id=upload.account_id,
        bank=account.bank if account is not None else upload.bank,
        path=path,
        password=password,
    )
    session.expunge_all()
    await session.rollback()
    return loaded


async def _parse(loaded: _LoadedStatement):
    """Run the appropriate synchronous PDF parser in a worker thread."""
    try:
        if loaded.kind == "cc":
            return await asyncio.to_thread(
                parse_statement,
                loaded.path,
                loaded.password,
                loaded.bank,
            )
        return await asyncio.to_thread(
            parse_bank_statement,
            loaded.path,
            loaded.bank,
            loaded.password,
        )
    except Exception as exc:
        raise StatementPreviewError(422, "Statement parse failed") from exc


def _row(
    kind: Literal["cc", "bank"],
    index: int,
    section: Literal["transactions", "payments_refunds"],
    transaction,
) -> statement_schemas.StatementParsedRow:
    """Map one parser row to bounded operational output."""
    if kind == "cc":
        card_mask = display_mask(transaction.card_number)
        person = _bounded(transaction.person, 256)
        reference_number = channel = balance = None
    else:
        card_mask = person = None
        reference_number = _bounded(transaction.reference_number, 256)
        channel = _bounded(transaction.channel, 64)
        balance = _bounded(transaction.balance, 64)

    return statement_schemas.StatementParsedRow(
        index=index,
        section=section,
        date=_bounded(transaction.date, 32) or "",
        amount=_bounded(transaction.amount, 64) or "",
        direction=transaction.transaction_type,
        narration=_bounded(transaction.narration),
        card_mask=card_mask,
        person=person,
        reference_number=reference_number,
        channel=channel,
        balance=balance,
    )


def _parse_response(
    loaded: _LoadedStatement,
    parsed,
) -> statement_schemas.StatementParsePreviewResponse:
    """Build a capped parser preview shared by CC and bank statements."""
    parser_rows: list[tuple[Literal["transactions", "payments_refunds"], object]] = []
    if loaded.kind == "cc":
        for transaction in parsed.transactions or []:
            parser_rows.append(("transactions", transaction))
        for transaction in parsed.payments_refunds or []:
            parser_rows.append(("payments_refunds", transaction))
        account_mask = None
        card_mask = display_mask(parsed.card_number)
        period_start = period_end = None
        opening_balance = closing_balance = None
    else:
        parser_rows.extend(("transactions", row) for row in parsed.transactions or [])
        account_mask = display_mask(parsed.account_number)
        card_mask = None
        period_start = _bounded(parsed.statement_period_start, 32)
        period_end = _bounded(parsed.statement_period_end, 32)
        opening_balance = _bounded(parsed.opening_balance, 64)
        closing_balance = _bounded(parsed.closing_balance, 64)

    rows = [
        _row(loaded.kind, index, section, transaction)
        for index, (section, transaction) in enumerate(parser_rows[:_ROW_LIMIT])
    ]
    return statement_schemas.StatementParsePreviewResponse(
        statement_id=loaded.statement_id,
        kind=loaded.kind,
        account_id=loaded.account_id,
        parser_bank=_bounded(parsed.bank, 64) or "",
        card_mask=card_mask,
        account_mask=account_mask,
        statement_period_start=period_start,
        statement_period_end=period_end,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        parsed_row_count=len(parser_rows),
        rows=rows,
        rows_truncated=len(parser_rows) > _ROW_LIMIT,
    )


async def preview_statement_parse(
    session: AsyncSession,
    kind: Literal["cc", "bank"],
    statement_id: int,
) -> statement_schemas.StatementParsePreviewResponse | None:
    """Parse one stored PDF without mutating its upload or transactions."""
    loaded = await _load_statement(session, kind, statement_id)
    if loaded is None:
        return None
    return _parse_response(loaded, await _parse(loaded))


def _reconciliation_entry(
    entry: dict,
) -> statement_schemas.StatementReconciliationEntry:
    """Map one reconciler classification without leaking unbounded narration."""
    return statement_schemas.StatementReconciliationEntry(
        statement_row_index=entry.get("stmt_idx"),
        date=_bounded(entry.get("date"), 32),
        amount=_bounded(entry.get("amount"), 64),
        direction=_bounded(entry.get("direction"), 16),
        narration=_bounded(entry.get("narration")),
        reference_number=_bounded(entry.get("reference_number"), 256),
        matched_transaction_id=entry.get("db_txn_id"),
        ambiguous=bool(entry.get("ambiguous")),
        candidate_transaction_ids=entry.get("candidate_transaction_ids", []),
        candidate_count=entry.get("candidate_count", 0),
        candidate_ids_truncated=bool(entry.get("candidate_ids_truncated")),
        decision_reason=_bounded(entry.get("decision_reason"), 64) or "unavailable",
        gates=[str(gate)[:64] for gate in entry.get("gates", [])[:10]],
    )


def _statement_candidate_index(
    entries: list[dict],
) -> tuple[set[tuple[str, Decimal, datetime.date]], set[str]]:
    """Normalize statement identities once for conservative extra detection."""
    identities: set[tuple[str, Decimal, datetime.date]] = set()
    uncertain_directions: set[str] = set()
    for entry in entries:
        direction = entry.get("direction")
        if not isinstance(direction, str):
            continue
        try:
            amount = Decimal(str(entry.get("amount")).replace(",", ""))
            txn_date = parse_date(str(entry.get("date")), dayfirst=True)
        except InvalidOperation, TypeError, ValueError:
            uncertain_directions.add(direction)
            continue
        if txn_date is None:
            uncertain_directions.add(direction)
            continue
        identities.add((direction, amount, txn_date))
    return identities, uncertain_directions


def _could_be_statement_candidate(
    transaction: Transaction,
    identities: set[tuple[str, Decimal, datetime.date]],
    uncertain_directions: set[str],
) -> bool:
    """Conservatively keep possible statement counterparts out of ``extra``."""
    if (
        transaction.transaction_date is None
        or transaction.direction in uncertain_directions
    ):
        return True
    return any(
        (
            transaction.direction,
            transaction.amount,
            transaction.transaction_date + datetime.timedelta(days=offset),
        )
        in identities
        for offset in (-1, 0, 1)
    )


async def preview_statement_reconciliation(
    session: AsyncSession,
    kind: Literal["cc", "bank"],
    statement_id: int,
) -> statement_schemas.StatementReconciliationPreviewResponse | None:
    """Parse and reconcile a stored PDF without imports, enrichment, or writes."""
    loaded = await _load_statement(session, kind, statement_id)
    if loaded is None:
        return None
    parsed = await _parse(loaded)
    date_range = (
        cc_stmt_date_range(parsed) if kind == "cc" else bank_stmt_date_range(parsed)
    )
    if date_range is None:
        raise StatementPreviewError(422, "Statement has no parseable date range")
    lo, hi = date_range
    if lo > hi:
        raise StatementPreviewError(422, "Statement date range is invalid")
    statement = select(Transaction).where(
        Transaction.account_id == loaded.account_id,
        Transaction.transaction_date.between(
            lo - datetime.timedelta(days=STMT_RECONCILE_DATE_BUFFER_DAYS),
            hi + datetime.timedelta(days=STMT_RECONCILE_DATE_BUFFER_DAYS),
        ),
    )
    with session.no_autoflush:
        db_transactions = list(
            (await session.execute(statement.execution_options(autoflush=False)))
            .scalars()
            .all()
        )
        try:
            if kind == "cc":
                reconciliation = reconcile_statement(
                    parsed,
                    db_transactions,
                    loaded.account_id,
                    await load_account_card_masks(session, loaded.account_id),
                )
            else:
                reconciliation = reconcile_bank_statement(
                    parsed, db_transactions, loaded.account_id
                )
        except Exception as exc:
            raise StatementPreviewError(422, "Statement reconciliation failed") from exc

    matched_all = reconciliation.get("matched", [])
    missing_all = reconciliation.get("missing", [])
    ambiguous_all = [entry for entry in missing_all if entry.get("ambiguous")]
    unambiguous_missing = [entry for entry in missing_all if not entry.get("ambiguous")]
    matched_ids = {
        entry["db_txn_id"]
        for entry in matched_all
        if entry.get("db_txn_id") is not None
    }
    all_statement_entries = [*matched_all, *missing_all]
    candidate_identities, uncertain_directions = _statement_candidate_index(
        all_statement_entries
    )
    extra_ids = sorted(
        transaction.id
        for transaction in db_transactions
        if transaction.id not in matched_ids
        and transaction.transaction_date is not None
        and lo <= transaction.transaction_date <= hi
        and not _could_be_statement_candidate(
            transaction, candidate_identities, uncertain_directions
        )
    )
    return statement_schemas.StatementReconciliationPreviewResponse(
        statement_id=statement_id,
        kind=kind,
        account_id=loaded.account_id,
        date_from=lo,
        date_to=hi,
        matched_count=len(matched_all),
        missing_count=len(unambiguous_missing),
        ambiguous_count=len(ambiguous_all),
        extra_count=len(extra_ids),
        matched=[_reconciliation_entry(row) for row in matched_all[:_ROW_LIMIT]],
        matched_truncated=len(matched_all) > _ROW_LIMIT,
        missing=[
            _reconciliation_entry(row) for row in unambiguous_missing[:_ROW_LIMIT]
        ],
        missing_truncated=len(unambiguous_missing) > _ROW_LIMIT,
        ambiguous=[_reconciliation_entry(row) for row in ambiguous_all[:_ROW_LIMIT]],
        ambiguous_truncated=len(ambiguous_all) > _ROW_LIMIT,
        extra_transaction_ids=extra_ids[:_ROW_LIMIT],
        extra_transaction_ids_truncated=len(extra_ids) > _ROW_LIMIT,
    )
