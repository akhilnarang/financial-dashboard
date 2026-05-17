"""Disambiguate which credit-card account a maskless payment-received
transaction belongs to.

When a CC bill-payment email/SMS arrives with no card_mask, the linker
can resolve account_id via its bank-only fallback if and only if the
bank has exactly one credit-card account on file. With multiple CCs
under the same bank, the linker refuses to guess and leaves the row
unlinked.

Public API:

- ``is_cc_payment_received_email(email_type)`` — predicate.

- ``resolve_cc_payment_account(session, txn_row)`` — the single entry
  point used by every CC bill-payment ingestion path (handle_polled_email,
  /emails/{id}/reparse, bulk reparse, SMS pipeline). Encapsulates:
    1. Gate on direction == "credit" and CC-payment email_type.
    2. Look up CC candidate accounts for the bank — ONE query.
    3. 0 candidates: silent no-op.
    4. 1 candidate: auto-resolve account_id on the row, flush.
    5. >1 candidates: try amount vs open statement total_amount_due
       (find_cc_account_by_total_due). On a unique hit, auto-resolve.
       Otherwise, return the Telegram prompt payload for the caller to
       dispatch post-commit.

- ``find_cc_account_by_total_due(session, bank, amount, *, candidate_ids=...)``
  remains exposed for tests and for callers that need the lower-level
  amount-match step independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Account, StatementUpload, Transaction

logger = logging.getLogger(__name__)


# Suffixes for "credit card bill payment received" alerts. Covers every
# bank-email-parser and bank-sms-parser shape whose downstream event is
# a credit hitting one of the user's CCs as a result of a bill payment.
# Adding a suffix here also opts the matching email_type into the
# `check_payment_received` reconciliation pass on the email/reparse
# pipelines (services/emails.py, web/emails.py).
#
# Refund/reversal shapes (e.g. `_cc_reversal`, `_cc_refund_alert`) are
# deliberately NOT included: those are credits but not bill payments,
# and treating them as such would silently mark open statements as
# partially paid against a merchant refund.
#
# This list exists because bank-email-parser and bank-sms-parser do not
# yet agree on a single canonical name for the bill-payment shape.
# A future refactor in those parsers could collapse this down to a
# single suffix (see #TODO-canonical-cc-payment-type).
_CC_PAYMENT_RECEIVED_SUFFIXES = (
    "_cc_payment_alert",
    "_cc_upi_payment_alert",
    "_cc_credit_alert",
    "_cc_payment",
    "_cc_bill_paid",
    "_cc_payment_received_alert",
    "_cc_bill_paid_alert",
    # `sbi_payment_ack` doesn't follow any `_cc_*` convention because its
    # source email is BillDesk's payment-acknowledgement template, not an
    # SBI Card email. Matched here as a literal full-string. If a future
    # parser introduces a non-CC `*_payment_ack` shape, switch this to an
    # exact-string check instead of broadening the suffix list.
    "sbi_payment_ack",
)


def is_cc_payment_received_email(email_type: str | None) -> bool:
    """True if the parsed alert represents a credit-card bill-payment
    credit hitting one of the user's CCs (regardless of bank or whether
    the source was an email body or an SMS)."""
    if not email_type:
        return False
    return any(email_type.endswith(s) for s in _CC_PAYMENT_RECEIVED_SUFFIXES)


def should_auto_reconcile_statement(txn_row: Transaction) -> bool:
    """True if a freshly-created Transaction should fire
    ``check_payment_received`` against an open statement.

    Centralizes the gate so the email poll, single reparse, and bulk
    reparse paths can't drift apart. Callers should additionally ensure
    they're only invoking this for *newly created* rows — a second
    source enriching an existing row must NOT re-fire the check, since
    ``payment_paid_amount`` is cumulative and would double-count.
    """
    return (
        txn_row.direction == "credit"
        and txn_row.account_id is not None
        and is_cc_payment_received_email(txn_row.email_type)
    )


@dataclass(frozen=True)
class _CandidateAccount:
    id: int
    label: str


async def _load_cc_candidates(
    session: AsyncSession, bank: str
) -> list[_CandidateAccount]:
    rows = (
        await session.execute(
            select(Account.id, Account.label).where(
                Account.bank == bank,
                Account.type == "credit_card",
            )
        )
    ).all()
    return [_CandidateAccount(id=r[0], label=r[1]) for r in rows]


async def find_cc_account_by_total_due(
    session: AsyncSession,
    bank: str,
    amount: Decimal,
    *,
    candidate_ids: list[int] | None = None,
) -> int | None:
    """Resolve a maskless CC payment to a single account by matching
    ``amount`` against an open statement's ``total_amount_due``.

    Returns the account_id when exactly one candidate CC account in
    ``bank`` has an active statement whose total matches ``amount``
    exactly. Returns ``None`` on zero or multiple hits (caller falls
    back to the Telegram disambiguation prompt).

    ``candidate_ids`` may be passed when the caller has already loaded
    the CC candidate set (avoids a redundant DB query).
    """
    from financial_dashboard.services.reminders import (
        ACTIVE_STATUSES,
        latest_per_account,
    )
    from financial_dashboard.services.statements.cc import parse_cc_amount

    if candidate_ids is None:
        candidate_ids = [c.id for c in await _load_cc_candidates(session, bank)]
    if len(candidate_ids) < 2:
        # 0: nothing to disambiguate. 1: linker's bank-only fallback
        # already handles this case.
        return None

    uploads = (
        (
            await session.execute(
                select(StatementUpload).where(
                    StatementUpload.account_id.in_(candidate_ids),
                    StatementUpload.payment_status.in_(ACTIVE_STATUSES),
                    StatementUpload.due_date.isnot(None),
                    StatementUpload.total_amount_due.isnot(None),
                )
            )
        )
        .scalars()
        .all()
    )
    # Mirror check_payment_received: only the most recent cycle per
    # account is eligible — older unpaid balances roll into the new
    # statement.
    latest = latest_per_account(list(uploads))

    target = Decimal(str(amount))
    matches: list[int] = []
    for upload in latest:
        try:
            due = parse_cc_amount(upload.total_amount_due)
        except (ValueError, InvalidOperation):
            # An unparseable total_amount_due in a row that passed the
            # NOT NULL filter is a data-quality issue, not a routine
            # condition — log it so a silent skip is debuggable.
            logger.warning(
                "Skipping statement %s during amount-based disambiguation: "
                "total_amount_due=%r is not parseable.",
                upload.id, upload.total_amount_due,
            )
            continue
        if due == target:
            matches.append(upload.account_id)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "Amount-based CC disambiguation ambiguous for bank=%r amount=%s: "
            "matched %d statements %r — refusing to guess.",
            bank, target, len(matches), matches,
        )
    return None


def _build_payload_from_candidates(
    candidates: list[_CandidateAccount],
    *,
    txn_id: int,
    bank: str,
    amount: Decimal,
) -> dict:
    return {
        "txn_id": txn_id,
        "candidate_account_ids": [c.id for c in candidates],
        "candidate_labels": {c.id: c.label for c in candidates},
        "amount": amount,
        "bank": bank,
    }


async def resolve_cc_payment_account(
    session: AsyncSession, txn_row: Transaction
) -> dict | None:
    """Top-level resolver for maskless CC bill-payment credits.

    Mutates ``txn_row.account_id`` in place when an auto-resolve succeeds
    (single CC for bank, or unique amount-match against open statement
    totals) and flushes the change so subsequent reads in the same
    transaction see the new value.

    Returns:
      - ``None`` when there's nothing to do (txn isn't a maskless CC
        bill-payment credit, no candidates, or an auto-resolve
        succeeded).
      - The Telegram disambiguation payload (dict) when the caller
        should dispatch ``send_disambiguation_prompt`` after commit.

    Callers must invoke this AFTER the linker has run (so account_id is
    None only when the linker couldn't resolve), and BEFORE building any
    notification payload that needs to read txn_row.account_id.
    """
    if (
        txn_row.account_id is not None
        or txn_row.direction != "credit"
        or not is_cc_payment_received_email(txn_row.email_type)
    ):
        return None

    # Schema marks Transaction.amount NOT NULL, but a parser bug could
    # in principle deliver a None — guard so a downstream Decimal(str(None))
    # doesn't blow up. Defense in depth, not a routine condition.
    if txn_row.amount is None:
        return None

    candidates = await _load_cc_candidates(session, txn_row.bank)
    if not candidates:
        return None

    if len(candidates) == 1:
        txn_row.account_id = candidates[0].id
        await session.flush()
        return None

    amount_match = await find_cc_account_by_total_due(
        session,
        txn_row.bank,
        txn_row.amount,
        candidate_ids=[c.id for c in candidates],
    )
    if amount_match is not None:
        txn_row.account_id = amount_match
        await session.flush()
        return None

    return _build_payload_from_candidates(
        candidates, txn_id=txn_row.id, bank=txn_row.bank, amount=txn_row.amount
    )
