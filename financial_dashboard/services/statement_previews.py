"""Side-effect-free parsing and reconciliation for stored statement uploads."""

import asyncio
import datetime
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import status

from financial_dashboard.config import get_fernet
from financial_dashboard.core.dates import parse_date
from financial_dashboard.core.masks import display_mask
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    StatementUpload,
    Transaction,
)
from financial_dashboard.exceptions import StatementPreviewError
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

logger = logging.getLogger(__name__)

_ROW_LIMIT = 100
_FILE_SIZE_LIMIT = 25_000_000
_TEXT_LIMIT = 1_000
_REFERENCE_LIMIT = 5_000
_REFERENCE_CHUNK_SIZE = 500


@dataclass(frozen=True, slots=True)
class _LoadedStatement:
    """Detached inputs required to parse one stored statement.

    Attributes:
        kind: Statement pipeline selected by the API route.
        statement_id: Database ID of the statement upload.
        account_id: Account whose transactions may be reconciled.
        bank: Parser bank identifier stored on the upload.
        path: Verified local PDF path; never exposed in preview responses.
        password: Decrypted statement password when available.
    """

    kind: Literal["cc", "bank"]
    statement_id: int
    account_id: int
    bank: str
    path: Path
    password: str | None


class _StatementCandidateIndex(NamedTuple):
    """Normalized identities and uncertainty found in statement rows.

    Attributes:
        identities: Parsed direction, amount, and date identities.
        uncertain_directions: Directions containing an unparseable identity.
        uncertain_all_directions: Whether a row lacked even a usable direction.
    """

    identities: set[tuple[str, Decimal, datetime.date]]
    uncertain_directions: set[str]
    uncertain_all_directions: bool


def _bounded(value: object | None, limit: int = _TEXT_LIMIT) -> str | None:
    """Convert an optional parser value to bounded response text.

    Args:
        value: Parser or reconciler value to stringify, or ``None``.
        limit: Maximum number of characters returned.

    Returns:
        The stringified value truncated to ``limit``, or ``None`` when absent.
    """
    return str(value)[:limit] if value is not None else None


async def _load_statement(
    session: AsyncSession,
    kind: Literal["cc", "bank"],
    statement_id: int,
) -> _LoadedStatement | None:
    """Load and detach the inputs required for statement parsing.

    Args:
        session: Request-scoped asynchronous database session.
        kind: Credit-card or bank-statement pipeline to load.
        statement_id: Database ID of the corresponding statement upload.

    Returns:
        Verified, detached parser inputs, or ``None`` when the upload does not
        exist. Local paths and decrypted passwords remain internal.

    Raises:
        StatementPreviewError: If the upload has no PDF, the file is unavailable
            or too large, or the statement is an email-only summary.
    """
    with session.no_autoflush:
        if kind == "cc":
            upload = await session.get(StatementUpload, statement_id)
            if upload is None:
                return None
            if upload.source_kind == "email_summary":
                raise StatementPreviewError(
                    status.HTTP_409_CONFLICT, "Email-summary statement has no PDF"
                )
        else:
            upload = await session.get(BankStatementUpload, statement_id)
            if upload is None:
                return None
        account = await session.get(Account, upload.account_id)

    if not upload.file_path:
        raise StatementPreviewError(
            status.HTTP_404_NOT_FOUND, "Statement PDF is unavailable"
        )
    path = Path(upload.file_path)
    if not path.is_file():
        raise StatementPreviewError(
            status.HTTP_404_NOT_FOUND, "Statement PDF is unavailable"
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise StatementPreviewError(
            status.HTTP_404_NOT_FOUND, "Statement PDF is unavailable"
        ) from exc
    if size > _FILE_SIZE_LIMIT:
        raise StatementPreviewError(
            status.HTTP_413_CONTENT_TOO_LARGE, "Statement PDF exceeds preview limit"
        )

    password = None
    if account is not None and account.statement_password:
        try:
            password = (
                get_fernet().decrypt(account.statement_password.encode()).decode()
            )
        except Exception:
            logger.warning(
                "Could not decrypt password for %s statement %d",
                kind,
                statement_id,
            )
            password = None
    loaded = _LoadedStatement(
        kind=kind,
        statement_id=statement_id,
        account_id=upload.account_id,
        bank=upload.bank,
        path=path,
        password=password,
    )
    session.expunge_all()
    await session.rollback()
    return loaded


async def _parse(loaded: _LoadedStatement) -> Any:
    """Run the selected synchronous statement parser outside the event loop.

    Args:
        loaded: Detached and verified parser inputs.

    Returns:
        The parsed credit-card or bank-statement model produced by the selected
        parser package.

    Raises:
        StatementPreviewError: If the parser rejects or cannot read the PDF.
    """
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
        raise StatementPreviewError(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Statement parse failed"
        ) from exc


def _row(
    kind: Literal["cc", "bank"],
    index: int,
    section: Literal["transactions", "payments_refunds"],
    transaction: Any,
) -> statement_schemas.StatementParsedRow:
    """Map one parser transaction row to bounded operational output.

    Args:
        kind: Parser model family represented by ``transaction``.
        index: Zero-based row position in the flattened preview.
        section: Source collection containing the row.
        transaction: Credit-card or bank parser transaction model.

    Returns:
        Bounded, display-safe row data with only fields supported by the selected
        parser family populated.
    """
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
    parsed: Any,
) -> statement_schemas.StatementParsePreviewResponse:
    """Build a capped parse response for either statement pipeline.

    Args:
        loaded: Detached upload metadata associated with the parsed document.
        parsed: Credit-card or bank-statement parser result.

    Returns:
        Statement metadata and at most ``_ROW_LIMIT`` bounded parser rows,
        including the total row count and truncation indicator.
    """
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
    """Parse one stored statement PDF without database mutations.

    Args:
        session: Request-scoped asynchronous database session.
        kind: Credit-card or bank-statement pipeline to preview.
        statement_id: Database ID of the statement upload.

    Returns:
        Bounded parser output, or ``None`` when the upload does not exist.

    Raises:
        StatementPreviewError: If the PDF cannot be safely loaded or parsed.
    """
    loaded = await _load_statement(session, kind, statement_id)
    if loaded is None:
        return None
    return _parse_response(loaded, await _parse(loaded))


def _reconciliation_entry(
    entry: dict[str, Any],
) -> statement_schemas.StatementReconciliationEntry:
    """Map one reconciler classification to bounded API evidence.

    Args:
        entry: Internal matched, missing, or ambiguous reconciliation record.

    Returns:
        Public reconciliation evidence with narration, references, reasons, and
        gate names bounded to their response limits.
    """
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
    entries: list[dict[str, Any]],
) -> _StatementCandidateIndex:
    """Index normalized statement identities for conservative extra detection.

    Args:
        entries: Every matched and unmatched statement reconciliation record.

    Returns:
        A ``_StatementCandidateIndex`` containing parsed identities and the
        directions for which incomplete evidence requires conservative handling.
    """
    identities: set[tuple[str, Decimal, datetime.date]] = set()
    uncertain_directions: set[str] = set()
    uncertain_all_directions = False
    for entry in entries:
        direction = entry.get("direction")
        if not isinstance(direction, str):
            uncertain_all_directions = True
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
    return _StatementCandidateIndex(
        identities=identities,
        uncertain_directions=uncertain_directions,
        uncertain_all_directions=uncertain_all_directions,
    )


def _could_be_statement_candidate(
    transaction: Transaction,
    identities: set[tuple[str, Decimal, datetime.date]],
    uncertain_directions: set[str],
    uncertain_all_directions: bool,
) -> bool:
    """Decide whether a database row could correspond to a statement row.

    Args:
        transaction: Existing account transaction not selected as a match.
        identities: Parsed statement direction, amount, and date identities.
        uncertain_directions: Directions with at least one unparseable identity.
        uncertain_all_directions: Whether any statement row lacked a usable
            direction.

    Returns:
        ``True`` when missing or fuzzy evidence means the transaction must not be
        labeled extra; otherwise, whether its ±1-day identity appears in the
        statement index.
    """
    if (
        uncertain_all_directions
        or transaction.transaction_date is None
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
    """Preview reconciliation of one stored statement without side effects.

    Args:
        session: Request-scoped asynchronous database session. Candidate queries
            run with autoflush disabled.
        kind: Credit-card or bank-statement reconciliation pipeline.
        statement_id: Database ID of the statement upload.

    Returns:
        Bounded matched, missing, ambiguous, and extra classifications with
        candidate counts, decision reasons, gates, and truncation indicators; or
        ``None`` when the upload does not exist.

    Raises:
        StatementPreviewError: If the PDF cannot be loaded or parsed, its date
            range is unusable, reference closure exceeds the limit, or the
            reconciler fails.
    """
    loaded = await _load_statement(session, kind, statement_id)
    if loaded is None:
        return None
    parsed = await _parse(loaded)
    date_range = (
        cc_stmt_date_range(parsed) if kind == "cc" else bank_stmt_date_range(parsed)
    )
    if date_range is None:
        raise StatementPreviewError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Statement has no parseable date range",
        )
    lo, hi = date_range
    if lo > hi:
        raise StatementPreviewError(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Statement date range is invalid"
        )
    # Candidate closure equivalent to email ingest without loading all history:
    # fuzzy matching can only reach the buffered date range, while bank rows can
    # additionally match a misdated/NULL-dated DB row by exact reference.
    statement = select(Transaction).where(
        Transaction.account_id == loaded.account_id,
        Transaction.transaction_date.between(
            lo - datetime.timedelta(days=STMT_RECONCILE_DATE_BUFFER_DAYS),
            hi + datetime.timedelta(days=STMT_RECONCILE_DATE_BUFFER_DAYS),
        ),
    )
    references = (
        sorted(
            {
                row.reference_number
                for row in parsed.transactions or []
                if row.reference_number
            }
        )
        if kind == "bank"
        else []
    )
    if len(references) > _REFERENCE_LIMIT:
        raise StatementPreviewError(
            status.HTTP_413_CONTENT_TOO_LARGE, "Statement has too many references"
        )

    with session.no_autoflush:
        candidate_by_id = {
            row.id: row
            for row in (
                await session.execute(statement.execution_options(autoflush=False))
            )
            .scalars()
            .all()
        }
        for start in range(0, len(references), _REFERENCE_CHUNK_SIZE):
            reference_rows = (
                (
                    await session.execute(
                        select(Transaction)
                        .where(
                            Transaction.account_id == loaded.account_id,
                            Transaction.reference_number.in_(
                                references[start : start + _REFERENCE_CHUNK_SIZE]
                            ),
                        )
                        .execution_options(autoflush=False)
                    )
                )
                .scalars()
                .all()
            )
            candidate_by_id.update((row.id, row) for row in reference_rows)
        db_transactions = list(candidate_by_id.values())
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
            raise StatementPreviewError(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "Statement reconciliation failed"
            ) from exc

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
    candidate_index = _statement_candidate_index(all_statement_entries)
    extra_ids = sorted(
        transaction.id
        for transaction in db_transactions
        if transaction.id not in matched_ids
        and transaction.transaction_date is not None
        and lo <= transaction.transaction_date <= hi
        and not _could_be_statement_candidate(
            transaction,
            candidate_index.identities,
            candidate_index.uncertain_directions,
            candidate_index.uncertain_all_directions,
        )
    )
    return statement_schemas.StatementReconciliationPreviewResponse(
        statement_id=statement_id,
        kind=kind,
        account_id=loaded.account_id,
        candidate_scope="date_buffer_plus_statement_references",
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
