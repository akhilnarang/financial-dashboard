"""Unit tests for services/txn_merge.py."""

from datetime import date, time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Base, Transaction
import financial_dashboard.services.txn_merge as txn_merge_module
from financial_dashboard.services.txn_merge import (
    MatchDecision,
    compute_enrichment_diff,
    find_match,
    is_duplicate_transaction_error,
    merge_transaction,
    sync_counterparty_source,
)


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
        "counterparty_source": "bank",
    }
    defaults.update(fields)
    txn = MagicMock()
    for k, v in defaults.items():
        setattr(txn, k, v)
    return txn


def test_compute_diff_email_keeps_richer_or_equivalent_values():
    """An email overwrites a differing value, but not a value that holds less.

    A shorter counterparty, mask, or description is a downgrade. A mask in a
    different format for the same digits is the same card.
    """
    existing = _make_txn(
        counterparty="alicetest0000@upi",
        card_mask="XX0000",
        account_mask="99XXXXXX0000",
        raw_description="Sent via UPI to merchant for order #1234",
    )
    incoming = {
        "counterparty": "test0000@upi",
        "card_mask": "0000",
        "account_mask": "XX0000",
        "raw_description": "Sent via UPI to merchant",
    }
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.changed_fields == []


def test_compute_diff_email_never_moves_the_event_later():
    """A later date or time from a second source is notification delay.

    Compare the full (date, time): 00:01 on the next day is later than 23:59,
    although the time of day alone is earlier.
    """
    existing = _make_txn(
        transaction_date=date(2026, 5, 19),
        transaction_time=time(23, 59, 0),
    )
    incoming = {
        "transaction_date": date(2026, 5, 20),
        "transaction_time": time(0, 1, 0),
    }
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.changed_fields == []

    # An earlier date is the real event date. It overwrites a later one.
    existing = _make_txn(transaction_date=date(2026, 6, 9))
    incoming = {"transaction_date": date(2026, 5, 3)}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert list(diff.overwritten) == ["transaction_date"]


# ---------------------------------------------------------------------------
# Async tests for find_match
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_find_match_distinct_reference_cross_channel_does_not_split(
    session: AsyncSession,
):
    """A differing reference must not split a cross-channel pair. An SMS and an
    email for one event can carry different reference formats. The email row
    still has an open SMS slot. So the incoming SMS enriches it. It must not
    insert a duplicate."""
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_cc_payment_received_alert",
        direction="credit",
        amount=Decimal("50000"),
        currency="INR",
        transaction_date=date(2026, 8, 20),
        transaction_time=time(10, 27),
        reference_number="EMAIL-REF-A",
        card_mask="XXXX0000",
        source="email",
        email_id=5,  # The email slot is full. The SMS slot is open, so enrichable.
    )
    session.add(existing)
    await session.flush()

    decision = await find_match(
        session,
        {
            "bank": "hdfc",
            "email_type": "hdfc_cc_payment_received_alert",
            "direction": "credit",
            "amount": Decimal("50000"),
            "currency": "INR",
            "transaction_date": date(2026, 8, 20),
            "transaction_time": time(10, 27),
            "reference_number": "SMS-REF-B",
            "card_mask": "XXXX0000",
        },
        "sms",
    )
    assert decision.action == "match"
    assert decision.transaction.id == existing.id


@pytest.mark.anyio
async def test_find_match_shortened_reference_same_template_does_not_split(
    session: AsyncSession,
):
    """A shortened form of the same reference is not a difference. A provider
    can truncate its own reference across two messages. So the split must not
    fire, and it must not make a new row."""
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_cc_payment_received_alert",
        direction="credit",
        amount=Decimal("50000"),
        currency="INR",
        transaction_date=date(2026, 8, 20),
        transaction_time=time(10, 27),
        reference_number="231RI76LV8KJ1B8",
        card_mask="XXXX0000",
        source="sms",
        sms_message_id=1,
    )
    session.add(existing)
    await session.flush()

    decision = await find_match(
        session,
        {
            "bank": "hdfc",
            "email_type": "hdfc_cc_payment_received_alert",
            "direction": "credit",
            "amount": Decimal("50000"),
            "currency": "INR",
            "transaction_date": date(2026, 8, 20),
            "transaction_time": time(10, 27),
            "reference_number": "231RI76LV8KJ1B",  # suffix-truncated form
            "card_mask": "XXXX0000",
        },
        "sms",
    )
    assert decision.action != "insert"


@pytest.mark.anyio
async def test_find_match_reparse_changed_reference_keeps_own_row(
    session: AsyncSession,
):
    """A reparse can change a message's reference. The split must not drop the
    row the message already owns and insert a duplicate. Passing the message's
    own source_id keeps that row and lets the slot logic defer instead."""
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_cc_payment_received_alert",
        direction="credit",
        amount=Decimal("50000"),
        currency="INR",
        transaction_date=date(2026, 8, 20),
        transaction_time=time(10, 27),
        reference_number="OLD-REF-A",
        card_mask="XXXX0000",
        source="sms",
        sms_message_id=7,
    )
    session.add(existing)
    await session.flush()

    incoming = {
        "bank": "hdfc",
        "email_type": "hdfc_cc_payment_received_alert",
        "direction": "credit",
        "amount": Decimal("50000"),
        "currency": "INR",
        "transaction_date": date(2026, 8, 20),
        "transaction_time": time(10, 27),
        "reference_number": "NEW-REF-B",
        "card_mask": "XXXX0000",
    }
    # Without the guard, the split drops the message's own row and inserts.
    assert (await find_match(session, incoming, "sms")).action == "insert"
    # With the guard, sms_message_id 7 is the message's own row, so no insert.
    assert (await find_match(session, incoming, "sms", source_id=7)).action != "insert"


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

    incoming = {
        "bank": "hdfc",
        "direction": "debit",
        "amount": Decimal("500"),
        "currency": "INR",
        "reference_number": None,
        "transaction_date": date(2026, 5, 2),
        "transaction_time": time(14, 28, 0),
    }
    match = await find_match(session, incoming)
    assert match.action == "match"
    assert match.transaction.id == existing.id
    assert match.kind == "standard"

    outside = {**incoming, "transaction_time": time(14, 38, 0)}
    assert (await find_match(session, outside)).action == "insert"


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
    assert match.action == "insert"


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
    assert match.action == "match"
    assert match.transaction.id == t1.id
    assert match.kind == "standard"


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
    assert match.action == "insert"


@pytest.mark.anyio
async def test_merge_transaction_same_ref_diff_balance_defers_cleanly(
    session: AsyncSession,
):
    """End-to-end guard for the exact-ref balance split: a same-ref,
    same-amount, DIFFERENT-balance incoming must defer through
    merge_transaction WITHOUT crashing. The (bank, ref, direction) unique
    index forbids a second same-ref row, so an 'insert' decision would hit an
    IntegrityError and re-raise — defer is the only safe outcome."""
    existing = Transaction(
        bank="kotak",
        email_type="kotak_digital_transaction",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        reference_number="Transaction Successful",
        balance=Decimal("9000.00"),
        source="email",
    )
    session.add(existing)
    await session.flush()
    incoming = {
        "bank": "kotak",
        "email_type": "kotak_digital_transaction",
        "direction": "debit",
        "amount": Decimal("500"),
        "currency": "INR",
        "reference_number": "Transaction Successful",
        "balance": None,
    }

    # One unknown balance is not proof of a different event.
    match = await find_match(session, incoming)
    assert match.action == "match"
    assert match.transaction.id == existing.id

    # Two equal known balances are the same event.
    match = await find_match(session, {**incoming, "balance": Decimal("9000.00")})
    assert match.action == "match"
    assert match.transaction.id == existing.id

    outcome, row, diff = await merge_transaction(
        session,
        "sms",
        {**incoming, "balance": Decimal("8500.00")},
        sms_message_id=None,
    )
    assert outcome == "deferred"
    assert row is None
    # The pre-existing row is untouched; no second row was inserted.
    all_rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(all_rows) == 1
    assert all_rows[0].balance == Decimal("9000.00")


@pytest.mark.anyio
async def test_merge_transaction_email_enriches_sms_row(session: AsyncSession):
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
        counterparty="PZCREDIT0000000",
    )
    session.add(sms_row)
    await session.flush()

    outcome, row, diff = await merge_transaction(
        session,
        "email",
        {
            "bank": "hdfc",
            "email_type": "some_other_classification",
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
    # The first classification stays.
    assert row.email_type == "hdfc_dc_transaction_alert"
    assert diff.filled == {"channel": "upi"}
    assert diff.overwritten == {
        "counterparty": ("PZCREDIT0000000", "Phone Pe Private Limited")
    }


@pytest.mark.anyio
async def test_merge_transaction_stores_and_replaces_a_user_alias(
    session: AsyncSession,
) -> None:
    """A label fills an empty name. A bank name then replaces it.

    The stored source must follow the stored name, or the next write reads a
    claim that no longer holds.
    """
    row = Transaction(
        bank="hdfc",
        email_type="hdfc_account_neft_debit_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 23),
        reference_number="NEFT:1234",
        source="sms",
        counterparty=None,
    )
    session.add(row)
    await session.flush()

    base = {
        "bank": "hdfc",
        "email_type": "hdfc_account_neft_debit_alert",
        "direction": "debit",
        "amount": Decimal("500"),
        "currency": "INR",
        "transaction_date": date(2026, 5, 2),
        "transaction_time": time(14, 23),
        "reference_number": "NEFT:1234",
    }

    _, filled, _ = await merge_transaction(
        session,
        "email",
        {**base, "counterparty": "My Payee", "counterparty_source": "user_alias"},
        email_id=101,
    )
    assert filled.counterparty == "My Payee"
    assert filled.counterparty_source == "user_alias"

    _, replaced, _ = await merge_transaction(
        session,
        "sms",
        {**base, "counterparty": "SAMPLE BENEFICIARY"},
        sms_message_id=102,
    )
    assert replaced.counterparty == "SAMPLE BENEFICIARY"
    assert replaced.counterparty_source == "bank"

    # A label must not replace a name a bank stated.
    _, kept, _ = await merge_transaction(
        session,
        "email",
        {**base, "counterparty": "My Payee", "counterparty_source": "user_alias"},
        email_id=101,
    )
    assert kept.counterparty == "SAMPLE BENEFICIARY"
    assert kept.counterparty_source == "bank"


def test_a_bank_that_states_the_stored_label_clears_the_label_claim() -> None:
    """A bank can send the same text the user saved as a label.

    The name does not change, so no field changes. The text is still a name
    the bank states. If the column keeps saying "user_alias", the next label
    replaces a name the bank confirmed.
    """
    txn = _make_txn(counterparty="My Savings Payee", counterparty_source="user_alias")
    sync_counterparty_source(txn, {"counterparty": "My Savings Payee"})
    assert txn.counterparty_source == "bank"


@pytest.mark.anyio
async def test_lost_destination_slot_race_defers_without_enrichment(
    tmp_path, monkeypatch
):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'lost-slot-race.sqlite'}"
    )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with maker() as automatic_session:
            stale_target = Transaction(
                bank="hdfc",
                email_type="hdfc_dc_transaction_alert",
                direction="debit",
                amount=Decimal("500"),
                currency="INR",
                transaction_date=date(2026, 5, 2),
                transaction_time=time(14, 23),
                counterparty=None,
                source="sms",
            )
            automatic_session.add(stale_target)
            await automatic_session.commit()
            target_id = stale_target.id
            assert stale_target.email_id is None

            # Simulate a different email winning after automatic matching loaded
            # this target but before its canonical enrichment writer runs.
            async with maker() as winner_session:
                winner = await winner_session.get(Transaction, stale_target.id)
                assert winner is not None
                winner.email_id = 101
                await winner_session.commit()

            monkeypatch.setattr(
                txn_merge_module,
                "find_match",
                AsyncMock(
                    return_value=MatchDecision("match", stale_target, "standard")
                ),
            )
            outcome, row, diff = await merge_transaction(
                automatic_session,
                "email",
                {
                    "bank": "hdfc",
                    "email_type": "hdfc_dc_transaction_alert",
                    "direction": "debit",
                    "amount": Decimal("500"),
                    "currency": "INR",
                    "transaction_date": date(2026, 5, 2),
                    "transaction_time": time(14, 23),
                    "counterparty": "Losing email enrichment",
                },
                email_id=202,
            )

            assert outcome == "deferred"
            assert row is None
            assert diff.changed_fields == []
            assert stale_target.counterparty is None
            await automatic_session.rollback()

        async with maker() as check_session:
            stored = await check_session.get(Transaction, target_id)
            assert stored is not None
            assert stored.email_id == 101
            assert stored.counterparty is None
            assert stored.source == "sms"
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# AM/PM alias-pass match (services/txn_merge.find_match step 3)
# ---------------------------------------------------------------------------


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
    assert match.action == "match"
    assert match.transaction.id == existing.id
    assert match.kind == "am_pm_alias"


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
    assert match.action == "insert"


@pytest.mark.anyio
async def test_am_pm_alias_match_requires_counterparty_agreement(
    session: AsyncSession,
):
    """Two ICICI CC purchases of the same amount on the same card on
    the same day, exactly 12h apart, at DIFFERENT merchants. The
    counterparty-prerequisite guard must refuse the alias merge. It also
    refuses when the incoming side has no counterparty."""
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

    incoming = {
        "bank": "icici",
        "direction": "debit",
        "amount": Decimal("500"),
        "currency": "INR",
        "reference_number": None,
        "transaction_date": date(2026, 5, 16),
        "transaction_time": time(22, 30, 0),  # 12h offset
        "counterparty": "DOMINOS PIZZA",  # different merchant
    }
    assert (await find_match(session, incoming)).action == "insert"
    no_counterparty = {**incoming, "counterparty": None}
    assert (await find_match(session, no_counterparty)).action == "insert"


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
    assert match.action == "insert"


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
    assert match.action == "insert", (
        "alias +12h must not match candidates whose stored hour != 12 — "
        "those are correctly-stored PM rows, not midnight-as-noon bugs"
    )


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


# ICICI CC payment-received pair: SMS (time, no counterparty) + email
# (counterparty, no time), no reference on either side. Linked by card
# last-4 — see CARD_PAYMENT_LINK_BY_MASK_EMAIL_TYPES.


def _icici_payment_sms(**overrides):
    data = {
        "bank": "icici",
        "email_type": "icici_cc_payment_received_alert",
        "direction": "credit",
        "amount": Decimal("50000.00"),
        "currency": "INR",
        "transaction_date": date(2026, 6, 6),
        "transaction_time": time(17, 58, 10),
        "counterparty": "",
        "card_mask": "XX4321",
        "reference_number": None,
        # The parser declares this; the pipelines copy it into txn_data.
        "identifies_by": "card_mask",
    }
    data.update(overrides)
    return data


def _icici_payment_email(**overrides):
    data = {
        "bank": "icici",
        "email_type": "icici_cc_payment_alert",
        "direction": "credit",
        "amount": Decimal("50000.00"),
        "currency": "INR",
        "transaction_date": date(2026, 6, 6),
        "transaction_time": None,
        "counterparty": "Payment received",
        "card_mask": "4000 XXXX XXXX 4321",
        "reference_number": None,
        # The parser declares this; the pipelines copy it into txn_data.
        "identifies_by": "card_mask",
    }
    data.update(overrides)
    return data


@pytest.mark.anyio
async def test_icici_payment_sms_then_email_merges_by_card_last4(
    session: AsyncSession,
):
    _, sms_row, _ = await merge_transaction(
        session, "sms", _icici_payment_sms(), sms_message_id=369
    )
    outcome, row, diff = await merge_transaction(
        session, "email", _icici_payment_email(), email_id=3310
    )
    assert outcome == "enriched"
    assert row.id == sms_row.id
    assert row.source == "sms+email"
    assert row.email_id == 3310
    # SMS counterparty was "" (empty, not null), so the email value
    # overwrites rather than fills — either way the row ends up correct.
    assert row.counterparty == "Payment received"
    assert "counterparty" in diff.changed_fields
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1


@pytest.mark.anyio
async def test_card_mask_fallback_excludes_spend_alerts(session: AsyncSession):
    # Two distinct same-day same-amount swipes on the same card: spend
    # alerts are NOT in the link-by-mask set, so they must stay split.
    await merge_transaction(
        session,
        "email",
        {
            "bank": "icici",
            "email_type": "icici_cc_transaction_alert",
            "direction": "debit",
            "amount": Decimal("500.00"),
            "currency": "INR",
            "transaction_date": date(2026, 6, 6),
            "transaction_time": None,
            "counterparty": "Zomato",
            "card_mask": "XX4321",
            "reference_number": None,
        },
    )
    outcome, _row, _ = await merge_transaction(
        session,
        "email",
        {
            "bank": "icici",
            "email_type": "icici_cc_transaction_alert",
            "direction": "debit",
            "amount": Decimal("500.00"),
            "currency": "INR",
            "transaction_date": date(2026, 6, 6),
            "transaction_time": None,
            "counterparty": "Swiggy",
            "card_mask": "4000 XXXX XXXX 4321",
            "reference_number": None,
        },
    )
    assert outcome == "created"
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 2


@pytest.mark.anyio
async def test_card_mask_fallback_refuses_two_same_card_payments(
    session: AsyncSession,
):
    # Two genuine same-card same-amount payments on one day are ambiguous;
    # an arriving email must not merge into either. These payment alerts
    # carry no balance, so the email hits balance-less multiplicity →
    # DEFER (skip for manual resolution) rather than guess a merge.
    await merge_transaction(
        session, "sms", _icici_payment_sms(transaction_time=time(10, 0, 0))
    )
    await merge_transaction(
        session, "sms", _icici_payment_sms(transaction_time=time(17, 58, 10))
    )
    outcome, _row, _ = await merge_transaction(session, "email", _icici_payment_email())
    assert outcome == "deferred"
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 2


@pytest.mark.anyio
async def test_card_mask_fallback_refuses_different_card(session: AsyncSession):
    # Same email_type pair but different cards (last-4 differs) → no merge.
    await merge_transaction(
        session, "sms", _icici_payment_sms(card_mask="XX9999"), sms_message_id=1
    )
    outcome, _row, _ = await merge_transaction(
        session, "email", _icici_payment_email(), email_id=2
    )
    assert outcome == "created"
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Balance-based de-merge. Balance is the event identity:
# a different *known* balance means a distinct event and must split.
# ---------------------------------------------------------------------------


def _quantize(value):
    return None if value is None else Decimal(str(value)).quantize(Decimal("0.01"))


def _icici_spend_sms(
    *,
    amount="5000",
    balance,
    transaction_time,
    counterparty="TESTMERCHANT",
    card_mask="XX1234",
    transaction_date=date(2026, 6, 7),
):
    """An ICICI CC spend alert as it reaches merge_transaction from the SMS
    pipeline: no reference_number, carries an available-limit balance."""
    return {
        "bank": "icici",
        "email_type": "icici_cc_transaction_alert",
        "direction": "debit",
        "amount": Decimal(amount),
        "currency": "INR",
        "transaction_date": transaction_date,
        "transaction_time": transaction_time,
        "counterparty": counterparty,
        "card_mask": card_mask,
        "account_mask": None,
        "reference_number": None,
        "channel": "card",
        "balance": Decimal(balance) if balance is not None else None,
        "raw_description": None,
    }


@pytest.mark.anyio
async def test_worked_trace_all_four_notifications_pair_by_balance(
    session: AsyncSession,
):
    """Two distinct ₹5,000 charges, each reported by SMS + email. All four
    notifications converge to exactly two rows, each paired by balance into
    an sms+email row."""
    # SMS1 (charge A), SMS1 again, SMS2 (charge B), Email1 (A), Email2 (B).
    await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=389,
    )
    # The bank re-sends SMS1. An equal balance proves the same event, even
    # with the SMS slot full. It is a no-op enrich.
    outcome, resent, diff = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=395,
    )
    assert outcome == "enriched"
    assert diff.changed_fields == []
    assert resent.enriched_at is None
    await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="95000.00", transaction_time=time(21, 36, 55)),
        sms_message_id=390,
    )
    await merge_transaction(
        session,
        "email",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 12)),
        email_id=3320,
    )
    await merge_transaction(
        session,
        "email",
        _icici_spend_sms(balance="95000.00", transaction_time=time(21, 36, 40)),
        email_id=3321,
    )
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 2
    assert all(r.source == "sms+email" for r in rows)
    assert {_quantize(r.balance) for r in rows} == {
        Decimal("100000.00"),
        Decimal("95000.00"),
    }


@pytest.mark.anyio
async def test_am_pm_alias_with_differing_balance_inserts(session: AsyncSession):
    """An alias-window candidate that would normally merge, but the two
    balances differ → a distinct event → INSERT, not merge."""
    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 33, 11),  # PM-stored-as-AM shape
        counterparty="STARBUCKS",
        balance=Decimal("1000.00"),
    )
    session.add(existing)
    await session.flush()

    incoming = {
        "bank": "icici",
        "direction": "debit",
        "amount": Decimal("500"),
        "currency": "INR",
        "reference_number": None,
        "transaction_date": date(2026, 5, 16),
        "transaction_time": time(22, 33, 30),
        "counterparty": "STARBUCKS",
        "balance": Decimal("500.00"),  # differs from candidate
    }
    assert (await find_match(session, incoming)).action == "insert"

    # A row from before the AM/PM fix may have no balance. It must still match.
    existing.balance = None
    await session.flush()
    match = await find_match(session, incoming)
    assert match.action == "match"
    assert match.kind == "am_pm_alias"


@pytest.mark.anyio
async def test_force_new_bypasses_find_match(session: AsyncSession):
    """force_new inserts a new row even when a same-balance candidate exists
    — the manual Parse of a deferred row."""
    await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=389,
    )
    outcome, txn, _ = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=395,
        force_new=True,
    )
    assert outcome == "created"
    assert txn is not None
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 2


@pytest.mark.anyio
async def test_force_new_idempotent_on_already_linked_source(session: AsyncSession):
    """A double Parse of the same SMS must not create two rows: force_new is
    idempotent on a source row already linked to a transaction."""
    o1, r1, _ = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=389,
        force_new=True,
    )
    o2, r2, _ = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=389,
        force_new=True,
    )
    assert o1 == "created"
    assert r2.id == r1.id
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# No-date fuzzy insert: an incoming row with no date and no ref cannot fuzzy
# match and must insert.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_find_match_no_date_no_ref_inserts(session: AsyncSession):
    """An incoming txn with neither reference_number nor transaction_date can
    neither exact-ref-match nor fuzzy-match (the fuzzy window is anchored on
    the date). It must insert — never silently merge into an unrelated row."""
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 2),
        transaction_time=time(14, 23),
        counterparty="Zomato",
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
            "transaction_date": None,  # no date at all
            "transaction_time": None,
            "counterparty": "Zomato",
        },
    )
    assert match.action == "insert"


# ---------------------------------------------------------------------------
# merge insert IntegrityError re-resolve: a concurrent insert (race) lands the
# ref row between find_match and the insert; the savepoint IntegrityError must
# re-resolve and enrich instead of crashing.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_merge_insert_integrity_error_re_resolves_to_enrich(
    session: AsyncSession, monkeypatch
):
    """Repro of the race ``merge_transaction``'s savepoint catch defends
    against: ``find_match`` first returns ``insert`` (nothing found), but a
    concurrent commit adds the ref row before the insert flushes, so the
    insert hits ``uq_transactions_ref``. The handler must re-run find_match,
    see the now-present row, and enrich it — never leak the IntegrityError.

    The race is simulated deterministically by wrapping find_match so its
    first call inserts the conflicting row (mimicking the concurrent commit)
    before returning insert, then returns the match on the re-resolve call.
    """
    import financial_dashboard.services.txn_merge as txn_merge_mod

    real_find_match = txn_merge_mod.find_match
    call_count = {"n": 0}

    ref = "IMPS:RACE-001"

    async def _racing_find_match(sess, txn_data, channel="sms", **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate the concurrent commit landing between find_match and
            # the insert: insert the row this very call "didn't see".
            sess.add(
                Transaction(
                    bank="hdfc",
                    email_type="hdfc_dc_transaction_alert",
                    direction="debit",
                    amount=Decimal("500"),
                    currency="INR",
                    reference_number=ref,
                    source="email",
                    counterparty="Zomato",
                )
            )
            await sess.flush()
            return txn_merge_mod.MatchDecision("insert")
        # Re-resolve call after the IntegrityError: now it matches.
        return await real_find_match(sess, txn_data, channel)

    monkeypatch.setattr(txn_merge_mod, "find_match", _racing_find_match)

    outcome, row, _ = await merge_transaction(
        session,
        "sms",
        {
            "bank": "hdfc",
            "email_type": "hdfc_dc_transaction_alert",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": ref,
            "counterparty": "Zomato",
        },
        sms_message_id=77,
    )
    assert outcome == "enriched"
    assert row is not None
    assert row.reference_number == ref
    # Exactly one row exists (the race-injected one), now enriched with SMS.
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1
    assert rows[0].sms_message_id == 77


@pytest.mark.anyio
async def test_merge_insert_integrity_error_reraises_when_still_unresolvable(
    session: AsyncSession, monkeypatch
):
    """When the IntegrityError re-resolves to anything other than a clean
    match (still 'insert' or 'defer'), the original error must be re-raised
    rather than silently swallowed — there's no safe recovery."""
    import financial_dashboard.services.txn_merge as txn_merge_mod

    # Pre-seed the conflicting row. find_match is patched to ignore it and
    # always return 'insert', so merge's insert hits uq_transactions_ref and
    # the re-resolve (still 'insert') must re-raise.
    session.add(
        Transaction(
            bank="hdfc",
            email_type="t",
            direction="debit",
            amount=Decimal("500"),
            reference_number="IMPS:RERAISE",
            source="email",
        )
    )
    await session.flush()

    async def _always_insert(sess, txn_data, channel="sms", **_kwargs):
        return txn_merge_mod.MatchDecision("insert")

    monkeypatch.setattr(txn_merge_mod, "find_match", _always_insert)

    from sqlalchemy.exc import IntegrityError as SAIntegrityError

    with pytest.raises(SAIntegrityError):
        await merge_transaction(
            session,
            "sms",
            {
                "bank": "hdfc",
                "email_type": "t",
                "direction": "debit",
                "amount": Decimal("500"),
                "reference_number": "IMPS:RERAISE",
            },
            sms_message_id=1,
        )


# ---------------------------------------------------------------------------
# Canonical IntegrityError classifier (shared by txn_merge + emails).
# ---------------------------------------------------------------------------


def _exc_with_orig(message: str):
    from sqlalchemy.exc import IntegrityError

    return IntegrityError(statement=None, params=None, orig=Exception(message))


def test_is_duplicate_transaction_error_classifies_ref_index():
    for message in (
        "UNIQUE constraint failed: uq_transactions_ref",
        "UNIQUE constraint failed: uq_transaction_dedup",
        "UNIQUE constraint failed: transactions.bank, "
        "transactions.reference_number, transactions.direction",
    ):
        assert is_duplicate_transaction_error(_exc_with_orig(message))
    assert not is_duplicate_transaction_error(
        _exc_with_orig("UNIQUE constraint failed: emails.message_id")
    )
    assert not is_duplicate_transaction_error(
        _exc_with_orig("NOT NULL constraint failed: transactions.amount")
    )


@pytest.mark.anyio
async def test_find_match_deferral_metadata_keeps_only_compatible_candidates(
    session: AsyncSession,
):
    compatible = Transaction(
        bank="samplebank",
        email_type="sample_debit",
        direction="debit",
        amount=Decimal("246.80"),
        currency="INR",
        transaction_date=date(2026, 8, 12),
        transaction_time=time(10, 15),
        counterparty="Synthetic Shop",
        balance=None,
        source="email",
    )
    incompatible = Transaction(
        bank="samplebank",
        email_type="sample_debit",
        direction="debit",
        amount=Decimal("246.80"),
        currency="INR",
        transaction_date=date(2026, 8, 12),
        transaction_time=time(10, 16),
        counterparty="Synthetic Shop",
        balance=Decimal("6000.00"),
        source="email",
    )
    session.add_all([compatible, incompatible])
    await session.flush()

    decision = await find_match(
        session,
        {
            "bank": "samplebank",
            "direction": "debit",
            "amount": Decimal("246.80"),
            "currency": "INR",
            "transaction_date": date(2026, 8, 12),
            "transaction_time": time(10, 15),
            "counterparty": "Synthetic Shop",
            "balance": Decimal("5753.20"),
        },
    )

    assert decision.action == "defer"
    assert decision.deferral_reason == "balance_ambiguous"
    assert decision.resolution_candidate_ids == (compatible.id,)


@pytest.mark.anyio
async def test_merge_transaction_threads_reference_mismatch_metadata(
    session: AsyncSession,
):
    existing = Transaction(
        bank="samplebank",
        email_type="sample_debit",
        direction="debit",
        amount=Decimal("111.00"),
        reference_number="SYNTH-REF-42",
    )
    session.add(existing)
    await session.flush()

    result = await merge_transaction(
        session,
        "sms",
        {
            "bank": "samplebank",
            "email_type": "sample_debit",
            "direction": "debit",
            "amount": Decimal("222.00"),
            "reference_number": "SYNTH-REF-42",
        },
    )
    outcome, transaction, diff = result

    assert outcome == "deferred"
    assert transaction is None
    assert diff.changed_fields == []
    assert result.deferral_reason == "reference_amount_mismatch"
    assert result.resolution_candidate_ids == ()
