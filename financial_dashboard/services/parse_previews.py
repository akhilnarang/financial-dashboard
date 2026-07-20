"""Side-effect-free parser previews for stored source records."""

import datetime
from typing import Literal

from bank_sms_parser import parse_sms
from bank_sms_parser.exceptions import ParseError, UnsupportedSmsTypeError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.masks import display_mask
from financial_dashboard.db import SmsMessage, Transaction
from financial_dashboard.schemas import sms as sms_schemas
from financial_dashboard.services.sms_pipeline import (
    DECLINED_DIRECTION,
    NOTIFY_ONLY_ROLES,
    parsed_sms_to_txn_data,
)
from financial_dashboard.services.txn_merge import (
    compute_applied_enrichment_diff,
    find_match,
)

_IDENTITY_FIELDS = ("bank", "direction", "amount", "currency")
_ERROR_LIMIT = 1_000
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
