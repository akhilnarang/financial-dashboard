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
    5. >1 candidates, cascading resolution:
       a. Exact match: amount == total_amount_due on latest statement.
       b. Outstanding match: amount == total_amount_due - payment_paid_amount.
       c. Sole outstanding: only one card has remaining balance > 0.
       d. All fail → Telegram disambiguation prompt.
"""

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
    # Slice sends the manual-repayment SMS under the shape
    # `slice_cc_repayment_received_alert`. The "re" prefix breaks the
    # `_cc_payment_received_alert` suffix match, so the shape needs its
    # own entry. Without it, a manual slice bill payment is silently
    # excluded from `_qualifying_payment_credit_sum` and the statement
    # stays UNPAID even after the SMS lands.
    "_cc_repayment_received_alert",
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


async def _latest_statements(
    session: AsyncSession, candidate_ids: list[int]
) -> list[StatementUpload]:
    """Get the latest statement (by due_date) per candidate account.
    Does not filter on payment_status."""
    from financial_dashboard.services.reminders import latest_per_account

    uploads = (
        (
            await session.execute(
                select(StatementUpload).where(
                    StatementUpload.account_id.in_(candidate_ids),
                    StatementUpload.due_date.isnot(None),
                    StatementUpload.total_amount_due.isnot(None),
                )
            )
        )
        .scalars()
        .all()
    )
    return latest_per_account(list(uploads))


def _parse_outstanding(upload: StatementUpload) -> Decimal | None:
    """Calculate remaining balance: total_amount_due minus payment_paid_amount.
    Return None if the total is not parseable."""
    from financial_dashboard.services.statements.cc import parse_cc_amount

    if upload.total_amount_due is None:
        return None
    try:
        due = parse_cc_amount(upload.total_amount_due)
    except ValueError, InvalidOperation:
        logger.warning(
            "Skipping statement %s: total_amount_due=%r is not parseable.",
            upload.id,
            upload.total_amount_due,
        )
        return None
    paid = upload.payment_paid_amount or Decimal(0)
    return due - paid


async def _find_cc_account_by_total_due(
    session: AsyncSession,
    bank: str,
    amount: Decimal,
    candidate_ids: list[int],
) -> int | None:
    """Resolve a maskless CC payment to a single account by matching
    ``amount`` against an active statement's ``total_amount_due``.

    Returns the account_id when exactly one candidate CC account in
    ``bank`` has an active (unpaid/partially-paid/late) statement whose
    total matches ``amount`` exactly. Returns ``None`` on zero or
    multiple hits (caller falls back to outstanding-based fallbacks).
    """
    from financial_dashboard.services.reminders import (
        ACTIVE_STATUSES,
        latest_per_account,
    )
    from financial_dashboard.services.statements.cc import parse_cc_amount

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
    # Only the most recent cycle per account is eligible: older unpaid
    # balances roll into the new statement.
    latest = latest_per_account(list(uploads))

    target = Decimal(str(amount))
    matches: list[int] = []
    for upload in latest:
        if upload.total_amount_due is None:
            continue
        try:
            due = parse_cc_amount(upload.total_amount_due)
        except ValueError, InvalidOperation:
            logger.warning(
                "Skipping statement %s during amount-based disambiguation: "
                "total_amount_due=%r is not parseable.",
                upload.id,
                upload.total_amount_due,
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
            bank,
            target,
            len(matches),
            matches,
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
    if txn_row.amount is None or txn_row.amount <= 0:
        return None

    candidates = await _load_cc_candidates(session, txn_row.bank)
    if not candidates:
        return None

    if len(candidates) == 1:
        txn_row.account_id = candidates[0].id
        await session.flush()
        return None

    candidate_ids = [c.id for c in candidates]
    amount_match = await _find_cc_account_by_total_due(
        session, txn_row.bank, txn_row.amount, candidate_ids
    )
    if amount_match is not None:
        txn_row.account_id = amount_match
        await session.flush()
        return None

    target = Decimal(str(txn_row.amount))
    latest = await _latest_statements(session, candidate_ids)

    # All candidates must have a latest statement. If any card has none,
    # outstanding-based tiers cannot operate safely.
    covered_ids = {u.account_id for u in latest}
    if covered_ids != set(candidate_ids):
        return _build_payload_from_candidates(
            candidates, txn_id=txn_row.id, bank=txn_row.bank, amount=txn_row.amount
        )

    outstanding_matches: list[int] = []
    accounts_with_outstanding: list[int] = []
    has_untracked = False
    for upload in latest:
        if upload.payment_status is None:
            has_untracked = True
            continue
        remaining = _parse_outstanding(upload)
        if remaining is None:
            has_untracked = True
            continue
        if remaining > 0:
            accounts_with_outstanding.append(upload.account_id)
        if remaining == target:
            outstanding_matches.append(upload.account_id)

    if not has_untracked and len(outstanding_matches) == 1:
        logger.info(
            "CC disambiguation: outstanding-match for bank=%r amount=%s → account %d",
            txn_row.bank,
            target,
            outstanding_matches[0],
        )
        txn_row.account_id = outstanding_matches[0]
        await session.flush()
        return None

    sole_outstanding = None
    if not has_untracked and len(accounts_with_outstanding) == 1:
        sole_upload = next(
            upload
            for upload in latest
            if upload.account_id == accounts_with_outstanding[0]
        )
        sole_outstanding = _parse_outstanding(sole_upload)

    if sole_outstanding is not None and target <= sole_outstanding:
        logger.info(
            "CC disambiguation: sole-outstanding for bank=%r amount=%s → account %d",
            txn_row.bank,
            target,
            accounts_with_outstanding[0],
        )
        txn_row.account_id = accounts_with_outstanding[0]
        await session.flush()
        return None

    return _build_payload_from_candidates(
        candidates, txn_id=txn_row.id, bank=txn_row.bank, amount=txn_row.amount
    )
