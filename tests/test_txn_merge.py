"""Unit tests for services/txn_merge.py."""

from datetime import date, time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Base, Transaction
from financial_dashboard.services.txn_merge import (
    EnrichmentDiff,
    compute_enrichment_diff,
    find_match,
    merge_transaction,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


def test_enrichment_diff_changed_fields_empty_by_default():
    diff = EnrichmentDiff()
    assert diff.changed_fields == []


def test_enrichment_diff_changed_fields_combines_filled_and_overwritten():
    diff = EnrichmentDiff(
        filled={"counterparty": "Phone Pe", "channel": "upi"},
        overwritten={"reference_number": ("OLD", "NEW")},
    )
    assert set(diff.changed_fields) == {"counterparty", "channel", "reference_number"}


def _make_txn(**fields):
    """Build a mock Transaction-like row with the given attributes; all others None."""
    defaults = {
        "transaction_date": None,
        "transaction_time": None,
        "counterparty": None,
        "card_mask": None,
        "account_mask": None,
        "reference_number": None,
        "channel": None,
        "balance": None,
        "raw_description": None,
    }
    defaults.update(fields)
    txn = MagicMock()
    for k, v in defaults.items():
        setattr(txn, k, v)
    return txn


def test_compute_diff_fills_null_field():
    existing = _make_txn(counterparty=None)
    incoming = {"counterparty": "Phone Pe", "channel": None}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.filled == {"counterparty": "Phone Pe"}
    assert diff.overwritten == {}


def test_compute_diff_email_overwrites_existing_value():
    existing = _make_txn(counterparty="PZCREDIT0000000")
    incoming = {"counterparty": "Phone Pe Private Limited"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.filled == {}
    assert diff.overwritten == {
        "counterparty": ("PZCREDIT0000000", "Phone Pe Private Limited")
    }


def test_compute_diff_sms_does_not_overwrite_existing_value():
    existing = _make_txn(counterparty="Phone Pe Private Limited")
    incoming = {"counterparty": "PZCREDIT0000000"}
    diff = compute_enrichment_diff(existing, incoming, "sms")
    assert diff.filled == {}
    assert diff.overwritten == {}


def test_compute_diff_silent_when_values_match():
    existing = _make_txn(counterparty="Phone Pe", channel="upi")
    incoming = {"counterparty": "Phone Pe", "channel": "upi"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.changed_fields == []


def test_compute_diff_incoming_null_never_overwrites_existing():
    existing = _make_txn(counterparty="Phone Pe")
    incoming = {"counterparty": None}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.filled == {}
    assert diff.overwritten == {}


def test_compute_diff_email_does_not_overwrite_transaction_time_with_later_value():
    """Real-world repro: HDFC #7101 -₹10 RFHOSPITAL — the SMS-derived
    09:30:26 was getting bumped to email-derived 09:30:27. A second-source
    timestamp that's LATER than the existing one is notification/parse
    delay, not new evidence about when the transaction happened, so keep
    the earlier value."""
    existing = _make_txn(
        transaction_date=date(2026, 5, 19),
        transaction_time=time(9, 30, 26),
    )
    incoming = {
        "transaction_date": date(2026, 5, 19),
        "transaction_time": time(9, 30, 27),
    }
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.filled == {}
    assert diff.overwritten == {}


def test_compute_diff_email_overwrites_transaction_time_with_earlier_value():
    """The mirror case: when the second source's time is strictly earlier,
    it is treated as a more accurate observation and replaces the
    existing value."""
    existing = _make_txn(
        transaction_date=date(2026, 5, 19),
        transaction_time=time(9, 30, 27),
    )
    incoming = {
        "transaction_date": date(2026, 5, 19),
        "transaction_time": time(9, 30, 26),
    }
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.filled == {}
    assert diff.overwritten == {"transaction_time": (time(9, 30, 27), time(9, 30, 26))}


def test_compute_diff_email_transaction_time_handles_midnight_crossing():
    """When the date differs across midnight, the (date, time) datetime is
    what's compared — a 23:59 existing should not be overwritten by a
    00:01 incoming on the following date (later by 2 minutes)."""
    existing = _make_txn(
        transaction_date=date(2026, 5, 19),
        transaction_time=time(23, 59, 0),
    )
    incoming = {
        "transaction_date": date(2026, 5, 20),
        "transaction_time": time(0, 1, 0),
    }
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert "transaction_time" not in diff.overwritten


def test_compute_diff_ignores_unparticipating_keys():
    existing = _make_txn(counterparty=None)
    # email_type, bank, direction, amount, currency should never enter the diff
    incoming = {
        "counterparty": "Phone Pe",
        "email_type": "irrelevant",
        "bank": "irrelevant",
        "direction": "credit",
        "amount": Decimal("100"),
        "currency": "INR",
    }
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.filled == {"counterparty": "Phone Pe"}
    assert "email_type" not in diff.changed_fields
    assert "bank" not in diff.changed_fields


# ---------------------------------------------------------------------------
# Async tests for find_match
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_find_match_by_reference_number_hits(session: AsyncSession):
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        reference_number="IMPS:000000000001",
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "debit",
            "amount": Decimal("500"),
            "reference_number": "IMPS:000000000001",
        },
    )
    assert match is not None
    row, kind = match
    assert row.id == existing.id
    assert kind == "standard"


@pytest.mark.anyio
async def test_find_match_by_reference_number_direction_distinguishes(
    session: AsyncSession,
):
    debit = Transaction(
        bank="hdfc",
        email_type="t1",
        direction="debit",
        amount=Decimal("500"),
        reference_number="IMPS:1",
    )
    credit = Transaction(
        bank="hdfc",
        email_type="t2",
        direction="credit",
        amount=Decimal("500"),
        reference_number="IMPS:2",  # different ref because of partial unique index
    )
    session.add_all([debit, credit])
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "credit",
            "amount": Decimal("500"),
            "reference_number": "IMPS:2",
        },
    )
    assert match is not None
    assert match[0].id == credit.id
    assert match[1] == "standard"


@pytest.mark.anyio
async def test_find_match_empty_reference_treated_as_null(
    session: AsyncSession,
):
    # No txn exists with reference_number=""; find_match should not match.
    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "debit",
            "amount": Decimal("500"),
            "reference_number": "",
        },
    )
    assert match is None


@pytest.mark.anyio
async def test_find_match_fuzzy_window_hits_within_10min(
    session: AsyncSession,
):
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 23, 0),
    )
    session.add(existing)
    await session.flush()

    # 5 minutes later — within window
    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 2),
            "transaction_time": time(14, 28, 0),
        },
    )
    assert match is not None
    assert match[0].id == existing.id
    assert match[1] == "standard"


@pytest.mark.anyio
async def test_find_match_fuzzy_window_misses_outside_10min(
    session: AsyncSession,
):
    existing = Transaction(
        bank="hdfc",
        email_type="t",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 23, 0),
    )
    session.add(existing)
    await session.flush()

    # 15 minutes later — outside window
    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 2),
            "transaction_time": time(14, 38, 0),
        },
    )
    assert match is None


@pytest.mark.anyio
async def test_find_match_fuzzy_date_only_requires_counterparty_agreement(
    session: AsyncSession,
):
    # When the window degrades to whole-day (no time on either side), a
    # singleton candidate is NOT auto-accepted — counterparty must agree.
    existing = Transaction(
        bank="axis",
        email_type="t",
        direction="credit",
        amount=Decimal("15000"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=None,
        counterparty=None,
    )
    session.add(existing)
    await session.flush()

    # No counterparty on either side → must NOT match.
    match = await find_match(
        session,
        {
            "bank": "axis",
            "direction": "credit",
            "amount": Decimal("15000"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 2),
            "transaction_time": None,
            "counterparty": None,
        },
    )
    assert match is None


@pytest.mark.anyio
async def test_find_match_fuzzy_counterparty_tiebreaker(
    session: AsyncSession,
):
    # Two candidates in window; one matches counterparty substring.
    t1 = Transaction(
        bank="hdfc",
        email_type="t",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 23),
        counterparty="Zomato Online Order",
    )
    t2 = Transaction(
        bank="hdfc",
        email_type="t",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 27),
        counterparty="Swiggy Instamart",
    )
    session.add_all([t1, t2])
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 2),
            "transaction_time": time(14, 25),
            "counterparty": "ZOMATO",
        },
    )
    assert match is not None
    assert match[0].id == t1.id
    assert match[1] == "standard"


@pytest.mark.anyio
async def test_find_match_fuzzy_currency_must_match(
    session: AsyncSession,
):
    existing = Transaction(
        bank="onecard",
        email_type="t",
        direction="debit",
        amount=Decimal("100"),
        currency="USD",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(10, 0),
    )
    session.add(existing)
    await session.flush()

    # Same amount, INR — must not match.
    match = await find_match(
        session,
        {
            "bank": "onecard",
            "direction": "debit",
            "amount": Decimal("100"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 2),
            "transaction_time": time(10, 5),
        },
    )
    assert match is None


@pytest.mark.anyio
async def test_merge_transaction_create_path(session: AsyncSession):
    txn_data = {
        "bank": "hdfc",
        "email_type": "hdfc_dc_transaction_alert",
        "direction": "debit",
        "amount": Decimal("500"),
        "currency": "INR",
        "transaction_date": date(2026, 5, 2),
        "transaction_time": time(14, 23),
        "counterparty": "Zomato",
        "card_mask": "x1234",
        "account_mask": None,
        "reference_number": None,
        "channel": None,
        "balance": None,
        "raw_description": None,
    }
    outcome, row, diff = await merge_transaction(
        session, "sms", txn_data, sms_message_id=None
    )
    assert outcome == "created"
    assert row.id is not None
    assert row.source == "sms"
    assert row.notified_channel == "sms"
    assert row.bank == "hdfc"
    assert diff.changed_fields == []


@pytest.mark.anyio
async def test_merge_transaction_create_path_sets_email_id(
    session: AsyncSession,
):
    txn_data = {
        "bank": "hdfc",
        "email_type": "hdfc_dc_transaction_alert",
        "direction": "debit",
        "amount": Decimal("100"),
        "currency": "INR",
        "transaction_date": date(2026, 5, 2),
    }
    outcome, row, _ = await merge_transaction(session, "email", txn_data, email_id=42)
    assert outcome == "created"
    assert row.email_id == 42
    assert row.source == "email"


@pytest.mark.anyio
async def test_merge_transaction_enrich_fills_null(session: AsyncSession):
    sms_row = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 23),
        reference_number="IMPS:1234",
        source="sms",
        notified_channel="sms",
        counterparty=None,
    )
    session.add(sms_row)
    await session.flush()

    outcome, row, diff = await merge_transaction(
        session,
        "email",
        {
            "bank": "hdfc",
            "email_type": "hdfc_dc_transaction_alert",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "transaction_date": date(2026, 5, 2),
            "transaction_time": time(14, 23),
            "reference_number": "IMPS:1234",
            "counterparty": "Phone Pe Private Limited",
            "channel": "upi",
        },
        email_id=99,
    )
    assert outcome == "enriched"
    assert row.id == sms_row.id
    assert row.counterparty == "Phone Pe Private Limited"
    assert row.channel == "upi"
    assert row.source == "sms+email"
    assert row.notified_channel == "sms"  # unchanged
    assert row.email_id == 99  # filled at enrich time
    assert row.enriched_at is not None
    assert "counterparty" in diff.filled
    assert "channel" in diff.filled


@pytest.mark.anyio
async def test_merge_transaction_email_overrides_sms_value(session: AsyncSession):
    sms_row = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 23),
        reference_number="IMPS:5678",
        source="sms",
        notified_channel="sms",
        counterparty="PZCREDIT0000000",
    )
    session.add(sms_row)
    await session.flush()

    outcome, row, diff = await merge_transaction(
        session,
        "email",
        {
            "bank": "hdfc",
            "email_type": "hdfc_dc_transaction_alert",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": "IMPS:5678",
            "counterparty": "Phone Pe Private Limited",
        },
        email_id=100,
    )
    assert outcome == "enriched"
    assert row.counterparty == "Phone Pe Private Limited"
    assert diff.overwritten["counterparty"] == (
        "PZCREDIT0000000",
        "Phone Pe Private Limited",
    )


@pytest.mark.anyio
async def test_merge_transaction_sms_does_not_override_email_value(
    session: AsyncSession,
):
    email_row = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 23),
        reference_number="IMPS:7777",
        source="email",
        notified_channel="email",
        counterparty="Phone Pe Private Limited",
    )
    session.add(email_row)
    await session.flush()

    outcome, row, diff = await merge_transaction(
        session,
        "sms",
        {
            "bank": "hdfc",
            "email_type": "hdfc_dc_transaction_alert",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": "IMPS:7777",
            "counterparty": "PZCREDIT0000000",
        },
        sms_message_id=5,
    )
    assert outcome == "enriched"
    assert row.counterparty == "Phone Pe Private Limited"  # SMS does NOT overwrite
    assert row.sms_message_id == 5  # but FK fills in
    assert row.source == "sms+email"
    assert diff.changed_fields == []  # silent enrichment


@pytest.mark.anyio
async def test_merge_transaction_email_type_is_immutable(
    session: AsyncSession,
):
    sms_row = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("100"),
        currency="INR",
        reference_number="IMPS:X",
        source="sms",
        notified_channel="sms",
    )
    session.add(sms_row)
    await session.flush()

    _, row, _ = await merge_transaction(
        session,
        "email",
        {
            "bank": "hdfc",
            "email_type": "some_other_classification",  # would-be different
            "direction": "debit",
            "amount": Decimal("100"),
            "currency": "INR",
            "reference_number": "IMPS:X",
        },
    )
    # email_type stays at first-arrival's classification.
    assert row.email_type == "hdfc_dc_transaction_alert"


# ---------------------------------------------------------------------------
# AM/PM alias-pass match (services/txn_merge.find_match step 3)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_am_pm_alias_match_recovers_pm_stored_as_am(session: AsyncSession):
    """hour<12 case: a pre-fix ICICI CC email row stored
    transaction_time=10:33:11 (interpreted as 24h AM, but the real txn
    was at 22:33 PM). The matching SMS arrives later with the correct
    received_at-derived time 22:33:30. The alias pass at incoming-12h
    finds it, gated by counterparty + AM/PM-ambiguous email_type."""
    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("320000"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 33, 11),  # wrong: should be 22:33:11
        counterparty="INDIAN INSTITUTE OF MA",
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "icici",
            "direction": "debit",
            "amount": Decimal("320000"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(22, 33, 30),  # SMS-derived correct time
            "counterparty": "INDIAN INSTITUT",
        },
    )
    assert match is not None
    row, kind = match
    assert row.id == existing.id
    assert kind == "am_pm_alias"


@pytest.mark.anyio
async def test_am_pm_alias_match_recovers_midnight_stored_as_noon(
    session: AsyncSession,
):
    """hour==12 case: a pre-fix ICICI CC email for a real 00:55 IST
    (midnight) transaction stored transaction_time=12:55:20 — because
    the body said '12:55:20' which on a 12-hour clock means either
    00:55 or 12:55 and the pre-fix parser stored 24-hour 12:55 by
    default. Reality is 00:55 (12:55 AM). The matching SMS arrives
    later with received_at-derived time 00:55:30. The alias pass at
    incoming+12h must find this candidate (the mirror of the
    PM-stored-as-AM case)."""
    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(12, 55, 20),  # wrong: should be 00:55:20
        counterparty="LATE NIGHT KITCHEN",
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "icici",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(0, 55, 30),  # SMS-derived correct time
            "counterparty": "LATE NIGHT KITCHEN",
        },
    )
    assert match is not None
    row, kind = match
    assert row.id == existing.id
    assert kind == "am_pm_alias"


@pytest.mark.anyio
async def test_am_pm_alias_match_does_not_fire_for_safe_email_types(
    session: AsyncSession,
):
    """Same shape, but the candidate's email_type is NOT in the
    AM/PM-ambiguous set. Alias pass must skip it — those types are
    24-hour and any 12h-offset row is a genuinely different event."""
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",  # 24-hour, not ambiguous
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 33, 11),
        counterparty="ZOMATO",
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(22, 33, 11),
            "counterparty": "ZOMATO",
        },
    )
    assert match is None


@pytest.mark.anyio
async def test_am_pm_alias_match_requires_counterparty_agreement(
    session: AsyncSession,
):
    """Two ICICI CC purchases of the same amount on the same card on
    the same day, exactly 12h apart, at DIFFERENT merchants. The
    counterparty-prerequisite guard must refuse the alias merge."""
    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 30, 0),
        counterparty="STARBUCKS",
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "icici",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(22, 30, 0),  # 12h offset
            "counterparty": "DOMINOS PIZZA",  # different merchant
        },
    )
    assert match is None


@pytest.mark.anyio
async def test_am_pm_alias_refuses_when_either_side_lacks_counterparty(
    session: AsyncSession,
):
    """Counterparty is the alias pass's primary safety. If either side
    lacks one, refuse the merge — better to land a duplicate than
    silently glue together two same-amount events that might be
    distinct."""
    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 30, 0),
        counterparty="STARBUCKS",
    )
    session.add(existing)
    await session.flush()

    # Incoming with no counterparty — must NOT match.
    match = await find_match(
        session,
        {
            "bank": "icici",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(22, 30, 0),
            "counterparty": None,
        },
    )
    assert match is None


@pytest.mark.anyio
async def test_am_pm_alias_returns_none_when_multiple_alias_candidates(
    session: AsyncSession,
):
    """Two pre-fix ICICI rows survive the alias-window + counterparty
    filter (same merchant, same amount, both stored ~12h off). The
    pass must refuse rather than guess."""
    t1 = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 30, 0),
        counterparty="STARBUCKS",
    )
    t2 = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 32, 0),  # also within the aliased window
        counterparty="STARBUCKS",
    )
    session.add_all([t1, t2])
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "icici",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(22, 30, 30),  # 12h offset from t1
            "counterparty": "STARBUCKS",
        },
    )
    assert match is None


@pytest.mark.anyio
async def test_am_pm_alias_plus12h_only_targets_noon_stored_candidates(
    session: AsyncSession,
):
    """The +12h alias direction exists only to recover the
    midnight-stored-as-noon case (real 00:xx stored as 12:xx). It must
    NOT match a correctly-stored PM email when an unrelated morning
    SMS for the same amount/merchant happens to differ by exactly 12h.

    Scenario: a real PM ICICI debit at 15:00:00 was correctly parsed by
    the post-fix email pipeline (so its stored transaction_time is the
    truth, 15:00:00). A separate, unrelated SMS at 03:00:00 with the
    same amount and merchant must NOT trigger the +12h alias and
    overwrite the PM row's correct time."""
    real_pm = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(15, 0, 0),  # correctly stored PM
        counterparty="STARBUCKS",
    )
    session.add(real_pm)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "icici",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(3, 0, 0),  # unrelated AM SMS, 12h offset
            "counterparty": "STARBUCKS",
        },
    )
    assert match is None, (
        "alias +12h must not match candidates whose stored hour != 12 — "
        "those are correctly-stored PM rows, not midnight-as-noon bugs"
    )


@pytest.mark.anyio
async def test_am_pm_alias_does_not_run_when_standard_pass_succeeds(
    session: AsyncSession,
):
    """Standard pass must take precedence. If a candidate is in the
    standard ±10-min window, the alias pass should not fire at all."""
    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("100"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(22, 30, 0),
        counterparty="ZEPTO",
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "icici",
            "direction": "debit",
            "amount": Decimal("100"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(22, 33, 0),  # within ±10 min
            "counterparty": "ZEPTO",
        },
    )
    assert match is not None
    row, kind = match
    assert row.id == existing.id
    assert kind == "standard"  # not am_pm_alias


@pytest.mark.anyio
async def test_merge_transaction_alias_match_overwrites_transaction_time(
    session: AsyncSession,
):
    """End-to-end through merge_transaction: alias-pass match must
    rewrite the candidate's transaction_time to the incoming value.
    This is how the pre-fix email row self-heals when the SMS arrives."""
    from financial_dashboard.services.txn_merge import merge_transaction

    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("320000"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 33, 11),  # wrong AM
        counterparty="INDIAN INSTITUTE OF MA",
        source="email",
    )
    session.add(existing)
    await session.flush()

    outcome, row, diff = await merge_transaction(
        session,
        "sms",
        {
            "bank": "icici",
            "email_type": "icici_cc_payment_received_alert",  # SMS shape
            "direction": "debit",
            "amount": Decimal("320000"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(22, 33, 30),
            "counterparty": "INDIAN INSTITUT",
        },
        sms_message_id=99,
    )
    assert outcome == "enriched"
    assert row.id == existing.id
    # Critical: the pre-fix email's time was rewritten to the SMS time,
    # bypassing the channel rule that normally blocks SMS overwrites.
    assert row.transaction_time == time(22, 33, 30)
    assert "transaction_time" in diff.overwritten
    assert diff.overwritten["transaction_time"] == (time(10, 33, 11), time(22, 33, 30))
