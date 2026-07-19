"""Explicit, preview-gated resolution of deferred email duplicates."""

import datetime
import hashlib
import hmac
import json
import logging
from decimal import Decimal
from typing import Literal, cast

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Email, FetchRule, Transaction
from financial_dashboard.db.enums import EmailKind
from financial_dashboard.integrations.email.body import load_or_fetch_raw_email
from financial_dashboard.schemas.emails import (
    DuplicateEnrichmentDiff,
    DuplicateResolutionRequest,
    DuplicateResolutionResponse,
    EnrichmentFieldChange,
    EnrichmentValue,
    TransactionEnrichmentState,
)
from financial_dashboard.services.duplicate_resolution_types import (
    EligibleResolutionRows,
    ResolutionEvaluation,
)
from financial_dashboard.services.email_attachments import (
    TransactionSlotConflict,
    lock_email_for_attachment,
)
from financial_dashboard.services.emails import _process_email_full
from financial_dashboard.services.linker import build_link_context, link_transaction
from financial_dashboard.services.txn_merge import (
    DUP_DEFER_PREFIX,
    EnrichmentDiff,
    apply_transaction_enrichment,
    compute_enrichment_diff,
    qualifies_as_explicit_match,
)

logger = logging.getLogger(__name__)

_STATE_FIELDS = (
    "transaction_date",
    "transaction_time",
    "counterparty",
    "card_mask",
    "account_mask",
    "reference_number",
    "channel",
    "balance",
    "raw_description",
)
_TOKEN_TARGET_FIELDS = (
    "id",
    "email_id",
    "sms_message_id",
    "account_id",
    "card_id",
    "bank",
    "email_type",
    "direction",
    "amount",
    "currency",
    *_STATE_FIELDS,
    "source",
    "enriched_at",
    "category",
    "category_method",
    "category_confidence",
    "category_model",
    "category_input_hash",
    "category_vocab_version",
    "categorized_at",
    "review_status",
    "review_reason",
)


class DuplicateResolutionError(Exception):
    def __init__(self, status_code: Literal[404, 409, 422], message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _state(row: Transaction) -> TransactionEnrichmentState:
    return TransactionEnrichmentState(
        **{field: getattr(row, field) for field in _STATE_FIELDS},
        email_id=row.email_id,
        source=row.source,
    )


def _projected_state(
    row: Transaction, diff: EnrichmentDiff, email_id: int
) -> TransactionEnrichmentState:
    values = {field: getattr(row, field) for field in _STATE_FIELDS}
    values.update(diff.filled)
    values.update({field: new for field, (_old, new) in diff.overwritten.items()})
    source = row.source
    if source != "email" and source is not None:
        source = "sms+email"
    return TransactionEnrichmentState(
        **values,
        email_id=email_id if row.email_id is None else row.email_id,
        source=source,
    )


def _response_diff(diff: EnrichmentDiff) -> DuplicateEnrichmentDiff:
    changes = {
        field: EnrichmentFieldChange(before=None, after=cast(EnrichmentValue, value))
        for field, value in diff.filled.items()
    }
    changes.update(
        {
            field: EnrichmentFieldChange(
                before=cast(EnrichmentValue, old),
                after=cast(EnrichmentValue, new),
            )
            for field, (old, new) in diff.overwritten.items()
        }
    )
    return DuplicateEnrichmentDiff(
        changed_fields=diff.changed_fields,
        filled=list(diff.filled),
        overwritten=list(diff.overwritten),
        changes=changes,
    )


def _jsonable(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _preview_token(
    email_row: Email,
    rule: FetchRule,
    target: Transaction,
    txn_data: dict,
    diff: EnrichmentDiff,
    raw_bytes: bytes,
) -> str:
    payload = {
        "version": 1,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "email": {
            "id": email_row.id,
            "status": email_row.status,
            "error": email_row.error,
            "rule_id": email_row.rule_id,
            "source_id": email_row.source_id,
            "remote_id": email_row.remote_id,
            "message_id": email_row.message_id,
        },
        "rule": {
            "id": rule.id,
            "bank": rule.bank,
            "email_kind": rule.email_kind,
        },
        "target": {field: getattr(target, field) for field in _TOKEN_TARGET_FIELDS},
        "parsed": txn_data,
        "diff": {
            "filled": diff.filled,
            "overwritten": diff.overwritten,
        },
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return f"v1.{hashlib.sha256(encoded).hexdigest()}"


async def _load_eligible_rows(
    session: AsyncSession,
    email_id: int,
    transaction_id: int,
    *,
    lock: bool,
    locked_email: Email | None = None,
) -> EligibleResolutionRows:
    """Load and validate DB-only resolution preconditions.

    This is used both as a cheap preflight before provider I/O and again in the
    preview/apply transaction. The preflight only avoids unnecessary I/O; the
    locked apply invocation remains the concurrency authorization.
    """
    email_stmt = select(Email).where(Email.id == email_id)
    target_stmt = select(Transaction).where(Transaction.id == transaction_id)
    if lock:
        email_stmt = email_stmt.with_for_update()
        target_stmt = target_stmt.with_for_update()

    email_row = locked_email
    if email_row is None:
        email_row = (await session.scalars(email_stmt)).one_or_none()
    if email_row is None:
        raise DuplicateResolutionError(404, "Email not found")
    target = (await session.scalars(target_stmt)).one_or_none()
    if target is None:
        raise DuplicateResolutionError(404, "Transaction not found")
    if (
        email_row.status != "skipped"
        or not email_row.error
        or not email_row.error.startswith(DUP_DEFER_PREFIX)
    ):
        raise DuplicateResolutionError(
            409, "Email is not a deferred possible duplicate"
        )

    attached_stmt = (
        select(Transaction.id).where(Transaction.email_id == email_id).limit(1)
    )
    if lock:
        attached_stmt = attached_stmt.with_for_update()
    attached = await session.scalar(attached_stmt)
    if attached is not None:
        raise DuplicateResolutionError(409, "Email already has an attached transaction")
    if target.email_id is not None:
        raise DuplicateResolutionError(
            409, "Selected transaction already has an email attached"
        )

    if email_row.rule_id is None:
        raise DuplicateResolutionError(404, "Email fetch rule not found")
    rule_stmt = select(FetchRule).where(FetchRule.id == email_row.rule_id)
    if lock:
        rule_stmt = rule_stmt.with_for_update()
    rule = (await session.scalars(rule_stmt)).one_or_none()
    if rule is None:
        raise DuplicateResolutionError(404, "Email fetch rule not found")
    if rule.email_kind not in (None, EmailKind.TRANSACTION):
        raise DuplicateResolutionError(
            422, "Current parser routing does not produce a transaction"
        )
    return EligibleResolutionRows(email_row, target, rule)


async def _load_rows_and_evaluate(
    session: AsyncSession,
    email_id: int,
    transaction_id: int,
    raw_bytes: bytes,
    *,
    lock: bool,
    locked_email: Email | None = None,
) -> ResolutionEvaluation:
    email_row, target, rule = await _load_eligible_rows(
        session,
        email_id,
        transaction_id,
        lock=lock,
        locked_email=locked_email,
    )

    parsed = _process_email_full(rule.bank, raw_bytes)
    if parsed.txn_data is None:
        raise DuplicateResolutionError(
            422, "Current parser did not produce a transaction"
        )
    txn_data = parsed.txn_data

    if not await qualifies_as_explicit_match(session, target, txn_data):
        raise DuplicateResolutionError(
            409, "Selected transaction is not a compatible duplicate candidate"
        )

    diff = compute_enrichment_diff(target, txn_data, "email")
    before = _state(target)
    after = _projected_state(target, diff, email_id)
    token = _preview_token(email_row, rule, target, txn_data, diff, raw_bytes)
    return ResolutionEvaluation(
        email=email_row,
        target=target,
        txn_data=txn_data,
        diff=diff,
        token=token,
        before=before,
        after=after,
    )


async def resolve_email_duplicate(
    session: AsyncSession,
    email_id: int,
    request: DuplicateResolutionRequest,
) -> DuplicateResolutionResponse:
    """Preview or atomically apply an explicit email-to-transaction merge."""
    # Reject stale or structurally ineligible requests before a missing spool can
    # trigger live provider I/O. Detach the fully loaded Email before rollback so
    # the loader cannot cause an async lazy refresh while using its scalar fields.
    try:
        eligible = await _load_eligible_rows(
            session, email_id, request.transaction_id, lock=False
        )
        email_row = eligible.email
        session.expunge(email_row)
    finally:
        await session.rollback()

    loaded = await load_or_fetch_raw_email(email_row)
    if loaded.raw_bytes is None:
        logger.warning(
            "Raw email load failed during duplicate resolution for email %d: %s",
            email_id,
            loaded.error,
        )
        raise DuplicateResolutionError(404, "Raw email source is unavailable")
    raw_bytes = loaded.raw_bytes

    if not request.apply:
        async with session.begin():
            evaluation = await _load_rows_and_evaluate(
                session,
                email_id,
                request.transaction_id,
                raw_bytes,
                lock=False,
            )
            return DuplicateResolutionResponse(
                mode="preview",
                email_id=email_id,
                transaction_id=evaluation.target.id,
                email_status="skipped",
                preview_token=evaluation.token,
                before=evaluation.before,
                after=evaluation.after,
                diff=_response_diff(evaluation.diff),
            )

    try:
        async with session.begin():
            # This must be the first DB operation in the transaction. Existing
            # email reparse writers take the same lock before checking/inserting,
            # so both paths serialize even though transactions.email_id is not
            # unique.
            locked_email = await lock_email_for_attachment(session, email_id)
            if locked_email is None:
                raise DuplicateResolutionError(404, "Email not found")
            evaluation = await _load_rows_and_evaluate(
                session,
                email_id,
                request.transaction_id,
                raw_bytes,
                lock=True,
                locked_email=locked_email,
            )
            assert request.preview_token is not None
            if not hmac.compare_digest(request.preview_token, evaluation.token):
                raise DuplicateResolutionError(
                    409, "Preview is stale; request a new preview"
                )

            applied_diff = await apply_transaction_enrichment(
                session,
                evaluation.target,
                evaluation.txn_data,
                "email",
                email_id=email_id,
            )
            # This should be identical to the tokened projection; keep the check
            # close to the mutation as a defense against future rule drift.
            if applied_diff != evaluation.diff:
                raise DuplicateResolutionError(
                    409, "Preview is stale; request a new preview"
                )
            mask_changed = any(
                field in applied_diff.filled or field in applied_diff.overwritten
                for field in ("card_mask", "account_mask")
            )
            if evaluation.target.account_id is None and mask_changed:
                link_context = await build_link_context(session)
                link_transaction(link_context, evaluation.target)
                await session.flush()

            # Explicit resolution intentionally only attaches/enriches. It does
            # not replay payment checks or transaction notifications.
            evaluation.email.status = "parsed"
            evaluation.email.error = None
            await session.flush()
            after = _state(evaluation.target)
            return DuplicateResolutionResponse(
                mode="applied",
                email_id=email_id,
                transaction_id=evaluation.target.id,
                email_status="parsed",
                preview_token=evaluation.token,
                before=evaluation.before,
                after=after,
                diff=_response_diff(applied_diff),
            )
    except TransactionSlotConflict as exc:
        raise DuplicateResolutionError(
            409, "Selected transaction already has an email attached"
        ) from exc
    except IntegrityError as exc:
        raise DuplicateResolutionError(
            409, "Duplicate resolution conflicts with current transaction state"
        ) from exc
