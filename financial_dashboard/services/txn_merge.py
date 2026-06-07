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
from decimal import Decimal
from typing import Literal, NamedTuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.masks import mask_digits, mask_last4
from financial_dashboard.db import Transaction
from financial_dashboard.services.parser_quirks import (
    AMBIGUOUS_12H_TIME_EMAIL_TYPES,
    CARD_PAYMENT_LINK_BY_MASK_EMAIL_TYPES,
)

Channel = Literal["sms", "email"]
MergeOutcome = Literal["created", "enriched", "deferred"]
MatchKind = Literal["standard", "am_pm_alias"]

# find_match's three terminal outcomes:
#   match  → enrich the carried Transaction
#   insert → no candidate is the same event; create a new row
#   defer  → ambiguous and not safely decidable from balance; skip the row
#            for manual resolution rather than risk a wrong merge/split.
MatchAction = Literal["match", "insert", "defer"]


class MatchDecision(NamedTuple):
    action: MatchAction
    # Only set when action == "match".
    transaction: Transaction | None = None
    kind: MatchKind | None = None


# Sentinel prefix on the channel-specific error note for a deferred row, so
# the manual-review queue can tell duplicate-defers apart from the other
# `skipped` shapes (unsupported, _stub, non-transaction, ref-race). Asserted
# in tests.
DUP_DEFER_PREFIX = "[dup-defer]"
DUP_DEFER_NOTE = (
    f"{DUP_DEFER_PREFIX} possible duplicate (no balance to confirm) — "
    "click Parse if this is a real separate transaction"
)


def _quantize_balance(value) -> Decimal | None:
    """Normalize a balance to 2dp for equality comparison. The incoming
    balance is built as ``Decimal(str(float))`` while the candidate's is a
    DB ``Numeric(12,2)`` round-trip; an unnormalized ``==`` can spuriously
    fail and wrongly DEFER a real same-event pair."""
    if value is None:
        return None
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return value.quantize(Decimal("0.01"))


def _slot_open(candidate: Transaction, channel: Channel) -> bool:
    """True when ``candidate`` has no source row yet on the incoming
    channel — its SMS slot (incoming SMS) or email slot (incoming email)
    is unfilled."""
    if channel == "sms":
        return candidate.sms_message_id is None
    return candidate.email_id is None


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
    # None only when outcome == "deferred" (no row created).
    transaction: Transaction | None
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


def _is_information_downgrade(field: str, old_val: object, new_val: object) -> bool:
    # Only the three string-valued fields below can degrade; every other
    # field short-circuits to False without touching old_val/new_val, so the
    # `object` annotation is honest about what callers actually pass.
    if field == "counterparty":
        old_norm = _normalize_counterparty(old_val)
        new_norm = _normalize_counterparty(new_val)
        return bool(new_norm) and new_norm in old_norm and new_norm != old_norm
    if field in _MASK_FIELDS:
        old_digits = mask_digits(old_val)
        new_digits = mask_digits(new_val)
        return len(new_digits) < len(old_digits) and old_digits.endswith(new_digits)
    if field == "raw_description":
        # Substring check is case-sensitive here (no .lower()), unlike the
        # counterparty path: a pure case flip is treated as a real change,
        # not a downgrade, since we can't tell which casing is more correct.
        old_stripped = old_val.strip()
        new_stripped = new_val.strip()
        return (
            bool(new_stripped)
            and new_stripped in old_stripped
            and new_stripped != old_stripped
        )
    return False


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
        elif f in _MASK_FIELDS and mask_digits(old_val) == mask_digits(new_val):
            # Same card/account, only the mask format differs (e.g.
            # "XX0000" vs "0000"). Not a real enrichment — keep existing,
            # do not overwrite or notify.
            continue
        elif channel == "email":
            if f == "transaction_time" and not _incoming_time_is_earlier(
                existing, incoming, new_val, old_val
            ):
                continue
            if _is_information_downgrade(f, old_val, new_val):
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


def _card_payment_mask_match(existing: Transaction, txn_data: dict) -> bool:
    """True iff both sides are a card payment alert (see
    CARD_PAYMENT_LINK_BY_MASK_EMAIL_TYPES) on the same card by last-4."""
    if existing.email_type not in CARD_PAYMENT_LINK_BY_MASK_EMAIL_TYPES:
        return False
    if txn_data.get("email_type") not in CARD_PAYMENT_LINK_BY_MASK_EMAIL_TYPES:
        return False
    existing_last4 = mask_last4(existing.card_mask)
    incoming_last4 = mask_last4(txn_data.get("card_mask"))
    return existing_last4 is not None and existing_last4 == incoming_last4


async def _gather_fuzzy_candidates(
    session: AsyncSession, txn_data: dict
) -> list[Transaction]:
    """Return the time-window candidate *list* for the fuzzy path, with the
    existing counterparty / date-only / card-mask gates applied as candidate
    *filters*. These gates no longer make a terminal accept/reject — they
    only shape the list the balance/slot decision in :func:`find_match` then
    runs on. An empty list means "no plausible same-event candidate", not
    "definitely a new event".
    """
    txn_date = txn_data.get("transaction_date")
    if txn_date is None:
        return []  # No date → can't fuzzy-match.

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
                # Other side lacks time — date-only gate applies below.
                return True
            c_dt = datetime.combine(c.transaction_date, c.transaction_time)
            return lower <= c_dt <= upper

        candidates = [c for c in candidates if in_window(c)]

    if not candidates:
        return []

    incoming_cp = txn_data.get("counterparty")

    # Date-only gate: if either side lacks transaction_time, two ₹500 swipes
    # on the same card on the same day are indistinguishable on time alone —
    # require counterparty agreement, or (for the no-counterparty card-payment
    # pair shape) card last-4 agreement. Applied as a filter, not an early
    # return, so the balance/slot decision still runs on the survivors.
    if incoming_time is None or any(c.transaction_time is None for c in candidates):
        by_cp = [
            c for c in candidates if _counterparty_match(c.counterparty, incoming_cp)
        ]
        if by_cp:
            return by_cp
        return [c for c in candidates if _card_payment_mask_match(c, txn_data)]

    # Timed window with >1 candidate: counterparty narrows it when it can.
    # A single timed candidate passes through as-is (no counterparty info
    # needed); the balance/slot decision is the real discriminator now.
    if len(candidates) > 1:
        by_cp = [
            c for c in candidates if _counterparty_match(c.counterparty, incoming_cp)
        ]
        if by_cp:
            return by_cp
    return candidates


def _decide(
    candidates: list[Transaction], txn_data: dict, channel: Channel
) -> MatchDecision:
    """The unified balance/slot decision, run on the gathered candidate
    list. Returns match / insert / defer.

    1. Authoritative split — drop candidates with a different *known*
       balance; if that empties the set, the incoming is a new event.
    2. Positive confirmation — a single equal known balance is the same
       event (a legit pair or a bank's duplicate notification).
    3. Incoming has no balance — merge only the clean balance-less 1+1
       pair; defer any balance-less multiplicity.
    """
    if not candidates:
        return MatchDecision("insert")

    incoming_balance = _quantize_balance(txn_data.get("balance"))

    # Step 1 — authoritative split: drop candidates with a DIFFERENT *known*
    # balance. A balance that differs can never be the same event. If this
    # empties the set, the incoming is a genuinely new distinct charge.
    if incoming_balance is not None:
        candidates = [
            c
            for c in candidates
            if c.balance is None or _quantize_balance(c.balance) == incoming_balance
        ]
        if not candidates:
            return MatchDecision("insert")  # ← THE FIX

        # Step 2 — positive balance confirmation. Equal known balance proves
        # same event (a legit cross-channel pair OR a bank's duplicate
        # notification), regardless of slot state.
        eq = [
            c
            for c in candidates
            if c.balance is not None
            and _quantize_balance(c.balance) == incoming_balance
        ]
        if len(eq) == 1:
            return MatchDecision("match", eq[0], "standard")
        # eq>1 (two stored rows same balance) or survivors all balance-None
        # (incoming has a balance, candidate doesn't — presence mismatch):
        # ambiguous, don't guess.
        return MatchDecision("defer")

    # Step 3 — incoming has no balance.
    balanceless = [c for c in candidates if c.balance is None]
    open_bl = [c for c in balanceless if _slot_open(c, channel)]
    if len(candidates) == 1 and len(balanceless) == 1 and len(open_bl) == 1:
        # Clean balance-less 1+1 (e.g. HDFC CC) → merge as today.
        return MatchDecision("match", candidates[0], "standard")
    return MatchDecision("defer")


async def find_match(
    session: AsyncSession, txn_data: dict, channel: Channel = "sms"
) -> MatchDecision:
    """Decide whether an incoming ``txn_data`` matches an existing row,
    is a new transaction, or is too ambiguous to decide.

    Returns a :class:`MatchDecision`:
    - ``"match"`` — enrich the carried Transaction (with its match kind).
    - ``"insert"`` — no candidate is the same event; create a new row.
    - ``"defer"`` — ambiguous; skip for manual resolution rather than risk a
      wrong merge (silently loses a txn) or wrong split.

    Strategy:
    1. (bank, reference_number, direction) when ref is non-empty — the
       strongest signal. A unique hit is a match; >1 is ambiguous (defer).
    2. Fuzzy fallback on (bank, direction, amount, currency) within a
       ±10-minute window, narrowed by counterparty / date-only / card-mask
       gates, then resolved by the balance/slot decision: a different known
       balance splits (insert), an equal known balance confirms (match),
       and balance-less multiplicity defers.
    3. AM/PM alias retry — when the fuzzy pass found no candidates, retry
       with the incoming time shifted ±12h for known-ambiguous email_types,
       then apply the same balance filter (differing balances → insert).
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
            return MatchDecision("match", rows[0], "standard")
        if len(rows) > 1:
            return MatchDecision("defer")
        # 0 rows: fall through to fuzzy.

    candidates = await _gather_fuzzy_candidates(session, txn_data)
    if candidates:
        return _decide(candidates, txn_data, channel)

    # No fuzzy candidates. Try the AM/PM alias retry.
    txn_date = txn_data.get("transaction_date")
    incoming_time = txn_data.get("transaction_time")
    if txn_date is None or incoming_time is None:
        return MatchDecision("insert")
    aliased = await _find_am_pm_alias_match(
        session,
        txn_data,
        txn_date=txn_date,
        incoming_time=incoming_time,
        incoming_currency=txn_data.get("currency") or "INR",
        incoming_cp=txn_data.get("counterparty"),
    )
    if aliased is None:
        return MatchDecision("insert")
    # Apply the balance filter to the alias candidate too: if both balances
    # are present and differ, it's a distinct event → insert. Otherwise
    # MATCH — pre-AM/PM-fix rows may have balance=None and must still merge
    # (treat candidate.balance is None as merge, NOT presence-mismatch defer).
    incoming_balance = _quantize_balance(txn_data.get("balance"))
    cand_balance = _quantize_balance(aliased.balance)
    if (
        incoming_balance is not None
        and cand_balance is not None
        and incoming_balance != cand_balance
    ):
        return MatchDecision("insert")
    return MatchDecision("match", aliased, "am_pm_alias")


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
    force_new: bool = False,
) -> MergeTransactionResult:
    """Match against existing rows; insert, enrich, or defer accordingly.

    Caller owns the transaction boundary. This function does NOT commit
    and does NOT fire Telegram. Returns a ``MergeTransactionResult``
    (NamedTuple of outcome, transaction, diff):
    - ``"created"`` — a new row was inserted; diff is empty.
    - ``"enriched"`` — an existing row was matched and updated.
    - ``"deferred"`` — find_match was too ambiguous to decide; NO row is
      created and ``transaction`` is ``None``. The caller marks the source
      row ``skipped`` with the ``[dup-defer]`` note.

    ``force_new=True`` bypasses ``find_match`` and inserts a new row
    directly — used by the manual "Parse" reparse of a deferred row, which
    would otherwise DEFER again forever. It is idempotent on a source row
    already linked to a transaction: if the incoming ``sms_message_id`` /
    ``email_id`` already owns a row, that row is returned untouched.
    """
    if force_new:
        existing = await _existing_for_source(session, sms_message_id, email_id)
        if existing is not None:
            return MergeTransactionResult("enriched", existing, EnrichmentDiff())
        return await _insert_new(session, channel, txn_data, sms_message_id, email_id)

    decision = await find_match(session, txn_data, channel)
    if decision.action == "insert":
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
            decision = await find_match(session, txn_data, channel)
            # A ref-duplicate that re-resolves to insert or defer is still a
            # hard duplicate we can't enrich — re-raise the original error.
            if decision.action != "match":
                raise
        else:
            return MergeTransactionResult("created", row, EnrichmentDiff())

    if decision.action == "defer":
        return MergeTransactionResult("deferred", None, EnrichmentDiff())

    assert decision.transaction is not None  # action == "match"
    match, match_kind = decision.transaction, decision.kind

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
    # Gate the timestamp churn on a real change: a no-op duplicate (equal
    # balance, slot already filled) must not rewrite the row.
    if diff.changed_fields:
        match.enriched_at = _dt.datetime.now(_dt.UTC)

    await session.flush()
    return MergeTransactionResult("enriched", match, diff)


async def _existing_for_source(
    session: AsyncSession, sms_message_id: int | None, email_id: int | None
) -> Transaction | None:
    """The transaction already linked to this source row, if any — the
    idempotency key for a force-create reparse (so a double Parse of one SMS
    or email can't create two rows). Channel-specific: SMS guards on
    ``sms_message_id``, email on ``email_id``."""
    if sms_message_id is not None:
        result = await session.execute(
            select(Transaction)
            .where(Transaction.sms_message_id == sms_message_id)
            .limit(1)
        )
        row = result.scalars().first()
        if row is not None:
            return row
    if email_id is not None:
        result = await session.execute(
            select(Transaction).where(Transaction.email_id == email_id).limit(1)
        )
        return result.scalars().first()
    return None


async def _insert_new(
    session: AsyncSession,
    channel: Channel,
    txn_data: dict,
    sms_message_id: int | None,
    email_id: int | None,
) -> MergeTransactionResult:
    row = Transaction(
        **txn_data,
        source=channel,
        notified_channel=channel,
        sms_message_id=sms_message_id,
        email_id=email_id,
    )
    session.add(row)
    await session.flush()
    return MergeTransactionResult("created", row, EnrichmentDiff())
