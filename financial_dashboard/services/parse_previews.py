"""Side-effect-free parser previews for stored source records."""

import datetime
from typing import Literal

from bank_sms_parser import parse_sms
from bank_sms_parser.exceptions import ParseError, UnsupportedSmsTypeError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.masks import display_mask
from financial_dashboard.db import Email, EmailKind, FetchRule, SmsMessage, Transaction
from financial_dashboard.integrations.email.body import load_or_fetch_raw_email
from financial_dashboard.schemas import emails as email_schemas
from financial_dashboard.schemas import sms as sms_schemas
from financial_dashboard.services.emails import _process_email_full
from financial_dashboard.services.sms_pipeline import (
    DECLINED_DIRECTION,
    NOTIFY_ONLY_ROLES,
    parsed_sms_to_txn_data,
)
from financial_dashboard.services.txn_merge import (
    DUP_DEFER_PREFIX,
    compute_applied_enrichment_diff,
    find_match,
)

_IDENTITY_FIELDS = ("bank", "direction", "amount", "currency")
_ERROR_LIMIT = 1_000
_LINK_LIMIT = 10
MergePreviewAction = Literal[
    "none",
    "notify_only",
    "declined",
    "match",
    "insert",
    "defer",
]


def _error_text(error: Exception) -> str:
    """Bound parser diagnostics before returning them through the API."""
    return str(error)[:_ERROR_LIMIT]


def _bounded(value: object | None, limit: int) -> str | None:
    """Convert an optional parser value to bounded response text."""
    return str(value)[:limit] if value is not None else None


def _transaction_preview(txn_data: dict) -> sms_schemas.SmsParsedTransactionPreview:
    """Map normalized parser data to a bounded, redacted response schema."""
    return sms_schemas.SmsParsedTransactionPreview(
        bank=_bounded(txn_data["bank"], 64) or "",
        email_type=_bounded(txn_data["email_type"], 128) or "",
        direction=_bounded(txn_data["direction"], 16) or "",
        amount=txn_data["amount"],
        currency=_bounded(txn_data.get("currency"), 8),
        transaction_date=txn_data.get("transaction_date"),
        transaction_time=txn_data.get("transaction_time"),
        counterparty=_bounded(txn_data.get("counterparty"), 1_000),
        card_mask=display_mask(txn_data.get("card_mask")),
        account_mask=display_mask(txn_data.get("account_mask")),
        reference_number=_bounded(txn_data.get("reference_number"), 256),
        channel=_bounded(txn_data.get("channel"), 64),
        balance=txn_data.get("balance"),
    )


def _identity_conflicts(
    transaction: Transaction,
    txn_data: dict,
) -> list[str]:
    """Return immutable identity fields changed by the current parser."""
    conflicts = []
    for field in _IDENTITY_FIELDS:
        stored = getattr(transaction, field)
        parsed = txn_data.get(field)
        if field == "currency":
            stored = str(stored or "INR").upper()
            parsed = str(parsed or "INR").upper()
        if stored != parsed:
            conflicts.append(field)
    return conflicts


def _empty_merge(
    action: MergePreviewAction = "none",
) -> sms_schemas.SmsMergePreview:
    """Build a merge projection with no target or field changes."""
    return sms_schemas.SmsMergePreview(
        action=action,
        target_transaction_id=None,
        match_kind=None,
        changed_fields=[],
        identity_conflicts=[],
    )


async def preview_sms_parse(
    session: AsyncSession,
    sms_id: int,
) -> sms_schemas.SmsParsePreviewResponse | None:
    """Parse one stored SMS and project merge behavior without writes or flushes."""
    with session.no_autoflush:
        sms = await session.get(SmsMessage, sms_id)
        if sms is None:
            return None
        linked = (
            await session.get(Transaction, sms.transaction_id)
            if sms.transaction_id is not None
            else None
        )

    received_at = sms.received_at
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=datetime.UTC)

    try:
        parsed = parse_sms(
            sms.bank,
            sms.body,
            sender=sms.sender,
            received_at=received_at,
        )
    except (ParseError, UnsupportedSmsTypeError) as error:
        skipped = (
            isinstance(error, UnsupportedSmsTypeError) or "_stub" in str(error).lower()
        )
        return sms_schemas.SmsParsePreviewResponse(
            sms_id=sms.id,
            current_status=sms.status,
            current_transaction_id=sms.transaction_id,
            parser=sms_schemas.SmsParserPreview(
                disposition="skipped" if skipped else "error",
                email_type=None,
                ledger_role=None,
                error=_error_text(error),
                transaction=None,
            ),
            merge=_empty_merge(),
        )

    txn_data = parsed_sms_to_txn_data(parsed, sms)
    parser = sms_schemas.SmsParserPreview(
        disposition="transaction" if txn_data is not None else "non_transaction",
        email_type=_bounded(parsed.email_type, 128),
        ledger_role=_bounded(parsed.ledger_role, 32),
        error=None,
        transaction=_transaction_preview(txn_data) if txn_data is not None else None,
    )
    if txn_data is None:
        merge = _empty_merge()
    elif parsed.ledger_role in NOTIFY_ONLY_ROLES and txn_data["direction"] == "credit":
        merge = _empty_merge("notify_only")
    elif txn_data["direction"] == DECLINED_DIRECTION:
        merge = _empty_merge("declined")
    else:
        conflicts = _identity_conflicts(linked, txn_data) if linked is not None else []
        with session.no_autoflush:
            decision = await find_match(session, txn_data, "sms")
        diff = (
            compute_applied_enrichment_diff(
                decision.transaction,
                txn_data,
                "sms",
                decision.kind,
            )
            if decision.transaction is not None
            else None
        )
        merge = sms_schemas.SmsMergePreview(
            action=decision.action,
            target_transaction_id=(
                decision.transaction.id if decision.transaction is not None else None
            ),
            match_kind=decision.kind,
            changed_fields=diff.changed_fields if diff is not None else [],
            identity_conflicts=conflicts,
        )

    return sms_schemas.SmsParsePreviewResponse(
        sms_id=sms.id,
        current_status=sms.status,
        current_transaction_id=sms.transaction_id,
        parser=parser,
        merge=merge,
    )


def _email_transaction_preview(
    txn_data: dict,
) -> email_schemas.EmailParsedTransactionPreview:
    """Map email parser output to bounded values without raw descriptions."""
    return email_schemas.EmailParsedTransactionPreview(
        bank=_bounded(txn_data["bank"], 64) or "",
        email_type=_bounded(txn_data["email_type"], 128) or "",
        direction=_bounded(txn_data["direction"], 16) or "",
        amount=txn_data["amount"],
        currency=_bounded(txn_data.get("currency"), 8),
        transaction_date=txn_data.get("transaction_date"),
        transaction_time=txn_data.get("transaction_time"),
        counterparty=_bounded(txn_data.get("counterparty"), 1_000),
        card_mask=display_mask(txn_data.get("card_mask")),
        account_mask=display_mask(txn_data.get("account_mask")),
        reference_number=_bounded(txn_data.get("reference_number"), 256),
        channel=_bounded(txn_data.get("channel"), 64),
        balance=txn_data.get("balance"),
    )


def _statement_summary_preview(parsed) -> email_schemas.EmailStatementSummaryPreview:
    """Map a body statement summary without returning parser debug text."""
    statement = parsed.statement
    return email_schemas.EmailStatementSummaryPreview(
        total_amount_due=(
            statement.total_amount_due.amount
            if statement.total_amount_due is not None
            else None
        ),
        minimum_amount_due=(
            statement.minimum_amount_due.amount
            if statement.minimum_amount_due is not None
            else None
        ),
        due_date=statement.due_date,
        card_mask=display_mask(statement.card_mask),
        statement_period_start=statement.statement_period_start,
        statement_period_end=statement.statement_period_end,
    )


def _email_refresh_fields(existing: Transaction, txn_data: dict) -> list[str]:
    """Project direct parser-field changes; attribution reruns separately."""
    changed = [
        field
        for field, value in txn_data.items()
        if value is not None and getattr(existing, field) != value
    ]
    return changed


def _email_merge(
    action: Literal[
        "none",
        "refresh_linked",
        "match",
        "insert",
        "defer",
        "conflict",
        "multiple_linked",
    ],
    *,
    target_id: int | None = None,
    match_kind: str | None = None,
    changed_fields: list[str] | None = None,
    identity_conflicts: list[str] | None = None,
    linked_attribution_refresh: bool = False,
) -> email_schemas.EmailMergePreview:
    """Build one bounded email reparse merge projection."""
    return email_schemas.EmailMergePreview(
        action=action,
        target_transaction_id=target_id,
        match_kind=match_kind,
        changed_fields=changed_fields or [],
        identity_conflicts=identity_conflicts or [],
        linked_attribution_refresh=linked_attribution_refresh,
    )


class EmailParsePreviewError(Exception):
    """A sanitized email-preview failure and its intended HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


async def preview_email_parse(
    session: AsyncSession,
    email_id: int,
) -> email_schemas.EmailParsePreviewResponse | None:
    """Refetch and parse one email, projecting the UI reparse path without writes."""
    with session.no_autoflush:
        email_row = await session.get(Email, email_id)
        if email_row is None:
            return None
        rule = (
            await session.get(FetchRule, email_row.rule_id)
            if email_row.rule_id is not None
            else None
        )
    if rule is None:
        raise EmailParsePreviewError(409, "Email has no associated fetch rule")

    bank = rule.bank
    email_kind = rule.email_kind
    routing: Literal["transaction", "statement", "cas"] = (
        "cas"
        if email_kind == EmailKind.CAS_STATEMENT
        else "statement"
        if email_kind
        in {EmailKind.CC_STATEMENT, EmailKind.BANK_STATEMENT, EmailKind.STATEMENT}
        else "transaction"
    )
    current_status = email_row.status
    current_error = email_row.error
    session.expunge_all()
    await session.rollback()

    raw_email_result = await load_or_fetch_raw_email(email_row)
    if raw_email_result.raw_bytes is None or raw_email_result.provenance is None:
        raise EmailParsePreviewError(404, "Raw email is unavailable")
    raw_bytes = raw_email_result.raw_bytes

    if routing == "cas":
        error = txn_data = password_hint = parsed = None
    else:
        error, txn_data, password_hint, parsed = _process_email_full(bank, raw_bytes)
    with session.no_autoflush:
        linked = list(
            (
                await session.execute(
                    select(Transaction)
                    .where(Transaction.email_id == email_id)
                    .order_by(Transaction.id)
                    .limit(_LINK_LIMIT + 1)
                    .execution_options(autoflush=False)
                )
            )
            .scalars()
            .all()
        )
    linked_ids = [row.id for row in linked[:_LINK_LIMIT]]

    if routing == "cas":
        parser = email_schemas.EmailParserPreview(
            disposition="routed_elsewhere",
            email_type=None,
            error=None,
            password_hint_present=False,
            transaction=None,
            statement=None,
        )
        merge = _email_merge("none", match_kind="cas_rule")
    elif parsed is None:
        parser = email_schemas.EmailParserPreview(
            disposition="error",
            email_type=None,
            error=_bounded(
                error or raw_email_result.error or "Parser failed", _ERROR_LIMIT
            ),
            password_hint_present=False,
            transaction=None,
            statement=None,
        )
        merge = _email_merge("none")
    else:
        statement = (
            _statement_summary_preview(parsed) if parsed.statement is not None else None
        )
        parser = email_schemas.EmailParserPreview(
            disposition=(
                "transaction"
                if txn_data is not None
                else "statement_summary"
                if statement is not None
                else "non_transaction"
            ),
            email_type=_bounded(parsed.email_type, 128),
            error=_bounded(error, _ERROR_LIMIT),
            password_hint_present=password_hint is not None,
            transaction=(
                _email_transaction_preview(txn_data) if txn_data is not None else None
            ),
            statement=statement,
        )
        if txn_data is None:
            merge = _email_merge("none")
        elif routing != "transaction":
            merge = _email_merge("none", match_kind="statement_rule")
        elif len(linked) > 1:
            merge = _email_merge("multiple_linked")
        elif linked:
            existing = linked[0]
            merge = _email_merge(
                "refresh_linked",
                target_id=existing.id,
                match_kind="linked_source",
                changed_fields=_email_refresh_fields(existing, txn_data),
                identity_conflicts=_identity_conflicts(existing, txn_data),
                linked_attribution_refresh=True,
            )
        elif (
            current_status == "skipped"
            and current_error
            and current_error.startswith(DUP_DEFER_PREFIX)
        ):
            merge = _email_merge("defer", match_kind="existing_dup_defer")
        else:
            with session.no_autoflush:
                decision = await find_match(session, txn_data, "email")
            if (
                decision.action == "match"
                and decision.transaction is not None
                and decision.transaction.email_id is None
            ):
                diff = compute_applied_enrichment_diff(
                    decision.transaction, txn_data, "email", decision.kind
                )
                merge = _email_merge(
                    "match",
                    target_id=decision.transaction.id,
                    match_kind=decision.kind,
                    changed_fields=diff.changed_fields,
                )
            elif (
                decision.action == "match"
                and decision.transaction is not None
                and decision.transaction.email_id is not None
                and (txn_data.get("reference_number") or "").strip()
                and decision.transaction.reference_number
                == (txn_data.get("reference_number") or "").strip()
            ):
                merge = _email_merge(
                    "conflict",
                    target_id=decision.transaction.id,
                    match_kind="claimed_reference",
                )
            elif decision.action == "defer" and decision.kind == "ref_amount_mismatch":
                merge = _email_merge("defer", match_kind=decision.kind)
            else:
                merge = _email_merge("insert")

    return email_schemas.EmailParsePreviewResponse(
        email_id=email_id,
        current_status=current_status,
        current_transaction_ids=linked_ids,
        raw_provenance=raw_email_result.provenance,
        routing=routing,
        parser=parser,
        merge=merge,
    )
