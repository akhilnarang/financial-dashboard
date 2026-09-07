"""Stable content hash over the classifier's input fields.

Written on every categorization path but not yet read: the requeue query
(select_needs_work_stmt) keys off category_method/vocab_version, not this hash.
TODO: compare the stored hash against a freshly computed one to requeue
non-manual rows whose inputs changed (e.g. an SMS row later enriched by an email
merge, or a reparse). Gate any such requeue so it doesn't stampede the LLM.
"""

import hashlib
import json
from typing import TypedDict

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.normalize import (
    normalize_counterparty,
    normalize_text,
)


class InputPayload(TypedDict):
    raw_description: str
    counterparty: str
    direction: str | None
    amount: str
    currency: str
    channel: str
    bank: str
    email_type: str
    account_type: str


def build_input_payload(txn: Transaction, account_type: str | None) -> InputPayload:
    return {
        "raw_description": normalize_text(txn.raw_description),
        "counterparty": normalize_counterparty(txn.counterparty),
        "direction": txn.direction,
        "amount": str(txn.amount),
        "currency": txn.currency or "INR",
        "channel": txn.channel or "",
        "bank": txn.bank,
        "email_type": txn.email_type,
        "account_type": account_type or "",
    }


def compute_input_hash(payload: InputPayload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_confirmation_payload(
    txn: Transaction, account_type: str | None
) -> dict[str, object]:
    """Build the state fingerprint used by pending confirmations.

    Confirmation state includes both the mutable fields a pending action may
    change and the classifier input hash. This lets a later affirmative reply
    fail closed when enrichment or another edit changed the transaction.
    """
    return {
        "category": txn.category,
        "note": txn.note,
        "exclude_from_cashflow": txn.exclude_from_cashflow,
        "category_input_hash": txn.category_input_hash,
        "input_state": build_input_payload(txn, account_type),
    }


def compute_confirmation_hash(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_transaction_confirmation_hash(
    txn: Transaction, account_type: str | None
) -> str:
    return compute_confirmation_hash(build_confirmation_payload(txn, account_type))
