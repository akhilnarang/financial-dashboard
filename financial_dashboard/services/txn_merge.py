"""Shared transaction merge logic.

`merge_transaction(...)` is called by both the email pipeline
(handle_polled_email) and the SMS pipeline (process_sms_row). It decides
whether an incoming txn_data dict creates a new Transaction or enriches
an existing one. Caller owns the transaction boundary — this module
does NOT commit and does NOT fire Telegram.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Transaction

Channel = Literal["sms", "email"]
MergeOutcome = Literal["created", "enriched"]


@dataclass(frozen=True)
class EnrichmentDiff:
    """What changed when a second source enriched an existing Transaction."""

    filled: dict[str, object] = field(default_factory=dict)
    """Fields that were NULL before and now have a value. {field_name: new_value}"""
    overwritten: dict[str, tuple[object, object]] = field(default_factory=dict)
    """Fields that had a value and were overwritten. {field_name: (old, new)}"""

    @property
    def changed_fields(self) -> list[str]:
        return list(self.filled.keys()) + list(self.overwritten.keys())


# Fields considered for enrichment. Match key fields (bank, direction,
# amount, currency) are deliberately excluded — by construction they
# match. email_type is also excluded: it's a routing tag, not a property
# of the underlying event, and downstream filters that branch on it
# shouldn't see it flip mid-row.
_ENRICHMENT_FIELDS = (
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


def compute_enrichment_diff(existing, incoming: dict, channel: Channel) -> EnrichmentDiff:
    """Pure function. Does NOT mutate.

    Conflict rule:
    - existing is None, incoming is not None → fill (always).
    - existing is not None, incoming is None → keep (do nothing).
    - existing == incoming → keep (no diff).
    - existing != incoming AND channel == "email" → overwrite.
    - existing != incoming AND channel == "sms" → keep (SMS does NOT overwrite email).
    """
    filled: dict[str, object] = {}
    overwritten: dict[str, tuple[object, object]] = {}
    for f in _ENRICHMENT_FIELDS:
        if f not in incoming:
            continue
        new_val = incoming[f]
        if new_val is None:
            continue
        old_val = getattr(existing, f, None)
        if old_val is None:
            filled[f] = new_val
        elif old_val == new_val:
            continue
        elif channel == "email":
            overwritten[f] = (old_val, new_val)
        # channel == "sms" with both non-null and unequal: keep existing.
    return EnrichmentDiff(filled=filled, overwritten=overwritten)


_FUZZY_MATCH_WINDOW_MINUTES = 10


def _normalize_counterparty(s: str | None) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _counterparty_match(a: str | None, b: str | None) -> bool:
    na, nb = _normalize_counterparty(a), _normalize_counterparty(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


async def find_match(
    session: AsyncSession, txn_data: dict
) -> Transaction | None:
    """Return an existing Transaction representing the same logical
    event, or None.

    Strategy:
    1. (bank, reference_number, direction) when ref is non-empty on
       both sides — the strongest signal we have.
    2. Fuzzy fallback on (bank, direction, amount, currency) within a
       ±10-minute window in IST-local wall time, with merchant substring
       as a tiebreaker. Date-only windows (one side has no time)
       additionally require counterparty agreement — prevents wrongly
       merging two same-day same-amount card swipes.
    """
    ref = (txn_data.get("reference_number") or "").strip()
    if ref:
        result = await session.execute(
            select(Transaction)
            .where(
                Transaction.bank == txn_data["bank"],
                Transaction.direction == txn_data["direction"],
                Transaction.reference_number == ref,
            )
            .limit(2)
        )
        rows = result.scalars().all()
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            return None
        # 0 rows: fall through to fuzzy.

    # Fuzzy fallback.
    txn_date = txn_data.get("transaction_date")
    if txn_date is None:
        # No date → can't fuzzy-match.
        return None

    incoming_currency = txn_data.get("currency") or "INR"
    incoming_time = txn_data.get("transaction_time")

    # Date window: if we have a time, narrow to ±10 min; otherwise whole day.
    if incoming_time is not None:
        anchor = datetime.combine(txn_date, incoming_time)
        lower = anchor - timedelta(minutes=_FUZZY_MATCH_WINDOW_MINUTES)
        upper = anchor + timedelta(minutes=_FUZZY_MATCH_WINDOW_MINUTES)
        date_lower, date_upper = lower.date(), upper.date()
    else:
        date_lower = date_upper = txn_date

    from sqlalchemy import func as sa_func
    result = await session.execute(
        select(Transaction).where(
            Transaction.bank == txn_data["bank"],
            Transaction.direction == txn_data["direction"],
            Transaction.amount == txn_data["amount"],
            sa_func.coalesce(Transaction.currency, "INR") == incoming_currency,
            Transaction.transaction_date.is_not(None),
            Transaction.transaction_date >= date_lower,
            Transaction.transaction_date <= date_upper,
        )
    )
    candidates = list(result.scalars().all())

    if incoming_time is not None:
        # Filter candidates to those within the time window too.
        def in_window(c):
            if c.transaction_time is None:
                # Other side lacks time — date-only safety applies below.
                return True
            c_dt = datetime.combine(c.transaction_date, c.transaction_time)
            return lower <= c_dt <= upper
        candidates = [c for c in candidates if in_window(c)]

    if not candidates:
        return None

    # Date-only safety: if either side lacks transaction_time, a singleton
    # candidate is NOT auto-accepted — require counterparty agreement.
    # Two ₹500 card swipes on the same card on the same day would otherwise
    # silently merge.
    incoming_cp = txn_data.get("counterparty")
    if incoming_time is None or any(c.transaction_time is None for c in candidates):
        filtered = [
            c for c in candidates if _counterparty_match(c.counterparty, incoming_cp)
        ]
        if len(filtered) == 1:
            return filtered[0]
        return None

    if len(candidates) == 1:
        return candidates[0]

    # >1 candidates in time window: apply counterparty tiebreaker.
    filtered = [
        c for c in candidates if _counterparty_match(c.counterparty, incoming_cp)
    ]
    if len(filtered) == 1:
        return filtered[0]
    return None


def _is_duplicate_transaction_error(exc: IntegrityError) -> bool:
    """Check whether an IntegrityError came from uq_transactions_ref."""
    message = str(exc.orig)
    return "uq_transactions_ref" in message or (
        "UNIQUE constraint failed:" in message
        and "transactions." in message
        and "reference_number" in message
    )


async def merge_transaction(
    session: AsyncSession,
    channel: Channel,
    txn_data: dict,
    *,
    sms_message_id: int | None = None,
    email_id: int | None = None,
) -> tuple[MergeOutcome, Transaction, EnrichmentDiff]:
    """Match against existing rows; insert or enrich accordingly.

    Caller owns the transaction boundary. This function does NOT commit
    and does NOT fire Telegram. Returns (outcome, row, diff); when
    outcome=="created" the diff is empty.
    """
    match = await find_match(session, txn_data)
    if match is None:
        try:
            async with session.begin_nested():
                row = Transaction(
                    **txn_data,
                    source=channel,
                    notified_channel=channel,
                    sms_message_id=sms_message_id,
                    email_id=email_id,
                )
                session.add(row)
                await session.flush()
        except IntegrityError as exc:
            if not _is_duplicate_transaction_error(exc):
                raise
            match = await find_match(session, txn_data)
            if match is None:
                # Constraint hit we didn't model; surface it.
                raise
        else:
            return "created", row, EnrichmentDiff()

    # Enrichment path.
    diff = compute_enrichment_diff(match, txn_data, channel)
    for key, value in diff.filled.items():
        setattr(match, key, value)
    for key, (_old, new) in diff.overwritten.items():
        setattr(match, key, new)

    if match.source != channel and match.source is not None:
        match.source = "sms+email"
    if sms_message_id is not None and match.sms_message_id is None:
        match.sms_message_id = sms_message_id
    if email_id is not None and match.email_id is None:
        match.email_id = email_id
    match.enriched_at = _dt.datetime.now(_dt.UTC)

    await session.flush()
    return "enriched", match, diff
