"""Shared transaction merge logic.

`merge_transaction(...)` is called by both the email pipeline
(handle_polled_email) and the SMS pipeline (process_sms_row). It decides
whether an incoming txn_data dict creates a new Transaction or enriches
an existing one. Caller owns the transaction boundary — this module
does NOT commit and does NOT fire Telegram.
"""

import datetime as _dt
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, NamedTuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Transaction
from financial_dashboard.services.parser_quirks import (
    AMBIGUOUS_12H_TIME_EMAIL_TYPES,
)

Channel = Literal["sms", "email"]
MergeOutcome = Literal["created", "enriched"]
MatchKind = Literal["standard", "am_pm_alias"]


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


class MergeTransactionResult(NamedTuple):
    outcome: MergeOutcome
    transaction: Transaction
    diff: EnrichmentDiff


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


_MASK_FIELDS = ("card_mask", "account_mask")


def _normalize_mask(s: str | None) -> str:
    """Reduce a card/account mask to its significant digits so cosmetic
    format differences between sources — "XX0000" vs "0000", "x0000" vs
    "0000" — compare equal (same card/account, different masking style)."""
    if not s:
        return ""
    return "".join(ch for ch in s if ch.isdigit())


def compute_enrichment_diff(
    existing, incoming: dict, channel: Channel
) -> EnrichmentDiff:
    """Pure function. Does NOT mutate.

    Conflict rule:
    - existing is None, incoming is not None → fill (always).
    - existing is not None, incoming is None → keep (do nothing).
    - existing == incoming → keep (no diff).
    - card_mask/account_mask that normalize to the same digits ("XX0000"
      vs "0000") → keep (no diff); the format differs, not the card.
    - existing != incoming AND channel == "email" → overwrite, EXCEPT
      transaction_time only overwrites when the incoming time is strictly
      earlier than the existing one. A later timestamp from the second
      source reflects notification/parsing delay, not a more accurate
      clock — the bank's underlying event happened at the earliest
      observation. (The am_pm_alias path in merge_transaction handles the
      12h-off correction separately by force-overwriting.)
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
        elif f in _MASK_FIELDS and _normalize_mask(old_val) == _normalize_mask(new_val):
            # Same card/account, only the mask format differs (e.g.
            # "XX0000" vs "0000"). Not a real enrichment — keep existing,
            # do not overwrite or notify.
            continue
        elif channel == "email":
            if f == "transaction_time" and not _incoming_time_is_earlier(
                existing, incoming, new_val, old_val
            ):
                continue
            overwritten[f] = (old_val, new_val)
        # channel == "sms" with both non-null and unequal: keep existing.
    return EnrichmentDiff(filled=filled, overwritten=overwritten)


def _incoming_time_is_earlier(existing, incoming: dict, new_time, old_time) -> bool:
    """True iff the incoming (date, transaction_time) point is strictly
    earlier than the existing one. When both sides have transaction_date,
    compare full datetimes so a cross-midnight pair (23:59 vs 00:01 the
    next day) is handled correctly; otherwise compare time-of-day alone.
    """
    existing_date = getattr(existing, "transaction_date", None)
    incoming_date = incoming.get("transaction_date") or existing_date
    if existing_date is not None and incoming_date is not None:
        return datetime.combine(incoming_date, new_time) < datetime.combine(
            existing_date, old_time
        )
    return new_time < old_time


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
) -> tuple[Transaction, MatchKind] | None:
    """Return an existing Transaction representing the same logical
    event, paired with the match kind, or None.

    Strategy:
    1. (bank, reference_number, direction) when ref is non-empty on
       both sides — the strongest signal we have. Returns ("standard").
    2. Fuzzy fallback on (bank, direction, amount, currency) within a
       ±10-minute window in IST-local wall time, with merchant substring
       as a tiebreaker. Date-only windows (one side has no time)
       additionally require counterparty agreement — prevents wrongly
       merging two same-day same-amount card swipes. Returns ("standard").
    3. AM/PM alias retry — when the fuzzy pass found nothing, retry with
       the incoming time shifted -12h, restricted to candidates whose
       ``email_type`` is in ``AMBIGUOUS_12H_TIME_EMAIL_TYPES`` (i.e. the
       bank's parser is known to drop AM/PM markers and may have stored
       an AM time for a real PM transaction). Counterparty must agree.
       Returns ("am_pm_alias"); the caller overwrites the candidate's
       transaction_time on enrichment.
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
            return rows[0], "standard"
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

    incoming_cp = txn_data.get("counterparty")

    if candidates:
        # Date-only safety: if either side lacks transaction_time, a
        # singleton candidate is NOT auto-accepted — require counterparty
        # agreement. Two ₹500 card swipes on the same card on the same
        # day would otherwise silently merge.
        if incoming_time is None or any(c.transaction_time is None for c in candidates):
            filtered = [
                c
                for c in candidates
                if _counterparty_match(c.counterparty, incoming_cp)
            ]
            if len(filtered) == 1:
                return filtered[0], "standard"
            return None

        if len(candidates) == 1:
            return candidates[0], "standard"

        # >1 candidates in time window: apply counterparty tiebreaker.
        filtered = [
            c for c in candidates if _counterparty_match(c.counterparty, incoming_cp)
        ]
        if len(filtered) == 1:
            return filtered[0], "standard"
        return None

    # Standard fuzzy returned nothing. Try the AM/PM alias retry.
    if incoming_time is None:
        return None
    aliased = await _find_am_pm_alias_match(
        session,
        txn_data,
        txn_date=txn_date,
        incoming_time=incoming_time,
        incoming_currency=incoming_currency,
        incoming_cp=incoming_cp,
    )
    if aliased is not None:
        return aliased, "am_pm_alias"
    return None


async def _find_am_pm_alias_match(
    session: AsyncSession,
    txn_data: dict,
    *,
    txn_date,
    incoming_time,
    incoming_currency: str,
    incoming_cp: str | None,
) -> Transaction | None:
    """Retry the fuzzy match with the incoming time shifted by ±12h,
    restricted to candidates of known-AM/PM-ambiguous email_types.

    Why: ICICI CC transaction-alert emails parsed before the email-side
    AM/PM disambiguator shipped have a transaction_time that may be 12h
    off, but only in shape-specific directions:

      - PM-stored-as-AM (the hour<12 case in `_disambiguate_am_pm`):
        real 22:33 stored as 10:33. The CANDIDATE'S hour is in 1–11.
        The INCOMING is the corrected PM time, hour ≥ 12. Recover by
        shifting incoming by -12h and matching candidates with
        ``transaction_time.hour < 12``.
      - midnight-stored-as-noon (the hour==12 case): body says
        "12:55:20" — both midnight (00:55) and noon (12:55) are valid
        12-hour readings, and the pre-fix parser defaulted to 12:55.
        The CANDIDATE'S hour is exactly 12. The INCOMING is the
        corrected 00:xx time, hour < 12. Recover by shifting incoming
        by +12h and matching candidates with
        ``transaction_time.hour == 12``.

    These are the ONLY two miscoding shapes the disambiguator
    produced. A broad ±12h search risks false-merging unrelated
    same-amount same-merchant transactions that happen to differ by
    exactly 12h (e.g., a 03:00 AM SMS over a correctly-stored 15:00
    PM email). The shape-aware filter rules those out.

    Counterparty must be present on BOTH sides AND match — primary
    protection against false-merging two genuine same-amount same-card
    same-day purchases that happen to be exactly 12h apart at the same
    merchant.

    Returns the single matching candidate across both alias windows,
    or None on zero / multiple matches (including the counterparty-
    mismatch refuse case).
    """
    if not incoming_cp:
        return None

    # Decide which alias directions are worth searching. Hour ranges
    # map directly to the two storage bugs.
    search_minus = incoming_time.hour >= 12  # PM incoming → AM candidate
    search_plus = incoming_time.hour < 12  # AM incoming → noon candidate
    if not search_minus and not search_plus:
        return None  # defensive; the hour predicates are exhaustive

    anchor = datetime.combine(txn_date, incoming_time)

    def _window(offset_hours: int) -> tuple[datetime, datetime]:
        center = anchor + timedelta(hours=offset_hours)
        lo = center - timedelta(minutes=_FUZZY_MATCH_WINDOW_MINUTES)
        hi = center + timedelta(minutes=_FUZZY_MATCH_WINDOW_MINUTES)
        return lo, hi

    from sqlalchemy import and_, or_
    from sqlalchemy import func as sa_func

    date_clauses = []
    minus_lo = minus_hi = plus_lo = plus_hi = None
    if search_minus:
        minus_lo, minus_hi = _window(-12)
        date_clauses.append(
            and_(
                Transaction.transaction_date >= minus_lo.date(),
                Transaction.transaction_date <= minus_hi.date(),
            )
        )
    if search_plus:
        plus_lo, plus_hi = _window(+12)
        date_clauses.append(
            and_(
                Transaction.transaction_date >= plus_lo.date(),
                Transaction.transaction_date <= plus_hi.date(),
            )
        )

    # One query covering the active date range(s). Per-row window
    # membership AND the storage-shape gate (hour<12 vs hour==12) are
    # enforced in Python so the SQL stays simple.
    result = await session.execute(
        select(Transaction).where(
            Transaction.bank == txn_data["bank"],
            Transaction.direction == txn_data["direction"],
            Transaction.amount == txn_data["amount"],
            sa_func.coalesce(Transaction.currency, "INR") == incoming_currency,
            Transaction.email_type.in_(AMBIGUOUS_12H_TIME_EMAIL_TYPES),
            Transaction.transaction_time.is_not(None),
            Transaction.transaction_date.is_not(None),
            or_(*date_clauses),
        )
    )
    rows = list(result.scalars().all())

    def matches_minus(c) -> bool:
        # PM-stored-as-AM: candidate's hour must be in 1..11.
        if not search_minus or minus_lo is None or minus_hi is None:
            return False
        if not (1 <= c.transaction_time.hour < 12):
            return False
        c_dt = datetime.combine(c.transaction_date, c.transaction_time)
        return minus_lo <= c_dt <= minus_hi

    def matches_plus(c) -> bool:
        # midnight-stored-as-noon: candidate's hour must be exactly 12.
        if not search_plus or plus_lo is None or plus_hi is None:
            return False
        if c.transaction_time.hour != 12:
            return False
        c_dt = datetime.combine(c.transaction_date, c.transaction_time)
        return plus_lo <= c_dt <= plus_hi

    candidates = [c for c in rows if matches_minus(c) or matches_plus(c)]
    # Counterparty prerequisite: BOTH sides must have a counterparty
    # AND they must agree. Already gated above that incoming_cp is
    # truthy; here filter candidates whose counterparty matches.
    candidates = [
        c for c in candidates if _counterparty_match(c.counterparty, incoming_cp)
    ]
    if len(candidates) == 1:
        return candidates[0]
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
) -> MergeTransactionResult:
    """Match against existing rows; insert or enrich accordingly.

    Caller owns the transaction boundary. This function does NOT commit
    and does NOT fire Telegram. Returns a ``MergeTransactionResult``
    (NamedTuple of outcome, transaction, diff); when outcome=="created"
    the diff is empty.
    """
    match_result = await find_match(session, txn_data)
    if match_result is None:
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
            match_result = await find_match(session, txn_data)
            if match_result is None:
                # Constraint hit we didn't model; surface it.
                raise
        else:
            return MergeTransactionResult("created", row, EnrichmentDiff())

    match, match_kind = match_result

    # Enrichment path.
    diff = compute_enrichment_diff(match, txn_data, channel)
    for key, value in diff.filled.items():
        setattr(match, key, value)
    for key, (_old, new) in diff.overwritten.items():
        setattr(match, key, new)

    # AM/PM alias match: by construction the candidate's stored
    # transaction_time is known-wrong by 12h (that's why the alias pass
    # had to fire), and the incoming row's time is the corrected source
    # of truth. Force-overwrite it, bypassing the usual channel rule
    # that says "SMS does NOT overwrite email" — the email's time here
    # is a pre-fix artifact, not real evidence.
    if match_kind == "am_pm_alias":
        new_time = txn_data.get("transaction_time")
        if new_time is not None and match.transaction_time != new_time:
            old_time = match.transaction_time
            match.transaction_time = new_time
            diff.overwritten["transaction_time"] = (old_time, new_time)

    if match.source != channel and match.source is not None:
        match.source = "sms+email"
    if sms_message_id is not None and match.sms_message_id is None:
        match.sms_message_id = sms_message_id
    if email_id is not None and match.email_id is None:
        match.email_id = email_id
    match.enriched_at = _dt.datetime.now(_dt.UTC)

    await session.flush()
    return MergeTransactionResult("enriched", match, diff)
