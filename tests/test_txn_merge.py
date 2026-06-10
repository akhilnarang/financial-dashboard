"""Unit tests for services/txn_merge.py."""

from datetime import date, time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
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


def test_compute_diff_card_mask_format_difference_is_not_enrichment():
    """A card_mask that differs only in masking style ("XX0000" vs "0000")
    is the same card — not a real enrichment. Must not overwrite or notify."""
    existing = _make_txn(card_mask="XX0000")
    incoming = {"card_mask": "0000"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.changed_fields == []


def test_compute_diff_account_mask_format_difference_is_not_enrichment():
    existing = _make_txn(account_mask="XX000")
    incoming = {"account_mask": "000"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.changed_fields == []


def test_compute_diff_genuinely_different_card_mask_still_overwrites():
    """Guard the normalization: different last-4 digits is a real change."""
    existing = _make_txn(card_mask="XX0000")
    incoming = {"card_mask": "9999"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.overwritten == {"card_mask": ("XX0000", "9999")}


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
    assert match.action == "match"
    assert match.transaction.id == existing.id
    assert match.kind == "standard"


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
    assert match.action == "match"
    assert match.transaction.id == credit.id
    assert match.kind == "standard"


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
    assert match.action == "insert"


def test_compute_diff_email_does_not_overwrite_transaction_date_with_later_value():
    """Repro: a Kotak digital-transaction email carries no transaction date,
    so the pipeline fills it from the email's received_at. Re-handling that
    email_type on a later day (or a second event sharing a bad
    reference_number) then arrives with transaction_date = today and clobbers
    the real, earlier date. A LATER incoming date is a notification/parse-delay
    artifact, not new evidence about when the event happened — keep the earlier
    value, mirroring transaction_time."""
    existing = _make_txn(transaction_date=date(2026, 5, 3))
    incoming = {"transaction_date": date(2026, 6, 9)}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.filled == {}
    assert diff.overwritten == {}


def test_compute_diff_email_overwrites_transaction_date_with_earlier_value():
    """Mirror case: a strictly earlier incoming date is a more accurate
    observation of when the event happened and replaces the existing one."""
    existing = _make_txn(transaction_date=date(2026, 6, 9))
    incoming = {"transaction_date": date(2026, 5, 3)}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.overwritten == {
        "transaction_date": (date(2026, 6, 9), date(2026, 5, 3))
    }


@pytest.mark.anyio
async def test_find_match_by_reference_with_amount_mismatch_defers(
    session: AsyncSession,
):
    """Repro: two distinct kotak_digital_transaction debits of different
    amounts both extracted the same boilerplate string as reference_number.
    The exact-ref match path keyed on (bank, ref, direction) and merged the
    second event into the first row without ever comparing amounts — silently
    destroying one transaction. A ref hit whose amount disagrees is not a
    confident same-event signal; defer for manual resolution instead of
    collapsing."""
    existing = Transaction(
        bank="kotak",
        email_type="kotak_digital_transaction",
        direction="debit",
        amount=Decimal("5555"),
        reference_number="If you are unable to view the below e-mailer.",
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "kotak",
            "direction": "debit",
            "amount": Decimal("7777"),
            "reference_number": "If you are unable to view the below e-mailer.",
        },
    )
    assert match.action == "defer"


@pytest.mark.anyio
async def test_find_match_by_reference_with_matching_amount_still_hits(
    session: AsyncSession,
):
    """The amount guard must not break the normal case: a ref hit whose
    amount agrees is still a confident match."""
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        reference_number="IMPS:000000000042",
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "debit",
            "amount": Decimal("500"),
            "reference_number": "IMPS:000000000042",
        },
    )
    assert match.action == "match"
    assert match.transaction.id == existing.id


@pytest.mark.anyio
async def test_find_match_by_reference_equal_amount_different_balance_defers(
    session: AsyncSession,
):
    """Two distinct same-amount debits that share a recycled/garbage
    reference_number but have DIFFERENT known available balances are not the
    same event — so they must NOT silently collapse into one row. The exact-ref
    path applies the same balance guard as the fuzzy path, but DEFERS rather
    than inserts: the `uq_transactions_ref` partial unique index forbids a
    second row with the same (bank, ref, direction), so a new insert can't
    physically land. We can neither merge (different balance) nor insert
    (unique ref) — park for manual resolution."""
    existing = Transaction(
        bank="kotak",
        email_type="kotak_digital_transaction",
        direction="debit",
        amount=Decimal("500"),
        reference_number="Transaction Successful",
        balance=Decimal("9000.00"),
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "kotak",
            "direction": "debit",
            "amount": Decimal("500"),
            "reference_number": "Transaction Successful",
            "balance": Decimal("8500.00"),
        },
    )
    assert match.action == "defer"
    assert match.kind == "ref_amount_mismatch"


@pytest.mark.anyio
async def test_find_match_by_reference_equal_amount_equal_balance_matches(
    session: AsyncSession,
):
    """A genuine same-ref pair with equal known balances is still a confident
    match — the balance guard must not break the legitimate cross-channel
    merge case."""
    existing = Transaction(
        bank="kotak",
        email_type="kotak_digital_transaction",
        direction="debit",
        amount=Decimal("500"),
        reference_number="UTR000000111222",
        balance=Decimal("8500.00"),
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "kotak",
            "direction": "debit",
            "amount": Decimal("500"),
            "reference_number": "UTR000000111222",
            "balance": Decimal("8500.00"),
        },
    )
    assert match.action == "match"
    assert match.transaction.id == existing.id


@pytest.mark.anyio
async def test_find_match_by_reference_equal_amount_one_balance_unknown_matches(
    session: AsyncSession,
):
    """When only one side has a known balance, the exact-ref hit still
    matches — the guard only splits when BOTH balances are known and
    differ (consistent with the fuzzy path keeping balance-None candidates)."""
    existing = Transaction(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        reference_number="IMPS:000000000077",
        balance=None,
    )
    session.add(existing)
    await session.flush()

    match = await find_match(
        session,
        {
            "bank": "hdfc",
            "direction": "debit",
            "amount": Decimal("500"),
            "reference_number": "IMPS:000000000077",
            "balance": Decimal("1234.00"),
        },
    )
    assert match.action == "match"
    assert match.transaction.id == existing.id


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
    assert match.action == "match"
    assert match.transaction.id == existing.id
    assert match.kind == "standard"


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
    assert match.action == "insert"


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

    outcome, row, diff = await merge_transaction(
        session,
        "sms",
        {
            "bank": "kotak",
            "email_type": "kotak_digital_transaction",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": "Transaction Successful",
            "balance": Decimal("8500.00"),
        },
        sms_message_id=None,
    )
    assert outcome == "deferred"
    assert row is None
    # The pre-existing row is untouched; no second row was inserted.
    all_rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(all_rows) == 1
    assert all_rows[0].balance == Decimal("9000.00")


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
    assert match.action == "match"
    assert match.transaction.id == existing.id
    assert match.kind == "am_pm_alias"


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
    assert match.action == "insert"


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
    assert match.action == "insert"


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
    assert match.action == "match"
    assert match.transaction.id == existing.id
    assert match.kind == "standard"  # not am_pm_alias


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


def test_email_enrichment_counterparty_no_downgrade():
    existing = _make_txn(counterparty="alicetest0000@upi")
    incoming = {"counterparty": "test0000@upi"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert "counterparty" not in diff.overwritten


def test_email_enrichment_counterparty_upgrade_allowed():
    existing = _make_txn(counterparty="test0000@upi")
    incoming = {"counterparty": "alicetest0000@upi"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.overwritten["counterparty"] == (
        "test0000@upi",
        "alicetest0000@upi",
    )


def test_email_enrichment_mask_upgrade_allowed():
    existing = _make_txn(account_mask="XX0000")
    incoming = {"account_mask": "99XXXXXX0000"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.overwritten["account_mask"] == ("XX0000", "99XXXXXX0000")


def test_email_enrichment_mask_no_downgrade():
    existing = _make_txn(account_mask="99XXXXXX0000")
    incoming = {"account_mask": "XX0000"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert "account_mask" not in diff.overwritten


def test_email_enrichment_unrelated_counterparty_still_overwrites():
    existing = _make_txn(counterparty="PZCREDIT123")
    incoming = {"counterparty": "Phone Pe Private Limited"}
    diff = compute_enrichment_diff(existing, incoming, "email")
    assert diff.overwritten["counterparty"] == (
        "PZCREDIT123",
        "Phone Pe Private Limited",
    )


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
async def test_icici_payment_email_then_sms_merges_by_card_last4(
    session: AsyncSession,
):
    # Reverse arrival: email row has no time, so the SMS still lands in
    # the date-only branch (candidate has NULL transaction_time).
    _, email_row, _ = await merge_transaction(
        session, "email", _icici_payment_email(), email_id=3310
    )
    outcome, row, diff = await merge_transaction(
        session, "sms", _icici_payment_sms(), sms_message_id=369
    )
    assert outcome == "enriched"
    assert row.id == email_row.id
    assert row.source == "sms+email"
    assert row.sms_message_id == 369
    assert "transaction_time" in diff.filled
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
async def test_two_distinct_charges_different_balance_split(session: AsyncSession):
    """The de-merge bug: two distinct ₹5,000 charges ~30s apart, SMS-only,
    identical on every old match field but with different available limits,
    must produce TWO rows — not silently collapse into one."""
    o1, _r1, _ = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=389,
    )
    o2, _r2, _ = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="95000.00", transaction_time=time(21, 36, 55)),
        sms_message_id=390,
    )
    assert o1 == "created"
    assert o2 == "created"
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 2


@pytest.mark.anyio
async def test_clean_cross_channel_pair_equal_balance_merges(session: AsyncSession):
    """One real charge, SMS then email, identical balance → one row.
    Balance equality is positive same-event confirmation."""
    o1, r1, _ = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=389,
    )
    o2, r2, _ = await merge_transaction(
        session,
        "email",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 12)),
        email_id=3320,
    )
    assert o1 == "created"
    assert o2 == "enriched"
    assert r2.id == r1.id
    assert r2.source == "sms+email"
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1


@pytest.mark.anyio
async def test_bank_duplicate_equal_balance_same_channel_dedups(session: AsyncSession):
    """A bank re-sends an identical SMS (same balance) for one charge. Equal
    balance proves same event even though the SMS slot is already filled →
    enrich (dedup), no second row."""
    o1, r1, _ = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=389,
    )
    o2, r2, diff = await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=395,
    )
    assert o1 == "created"
    assert o2 == "enriched"
    assert r2.id == r1.id
    # No-op duplicate: nothing changed, so enriched_at must stay None.
    assert diff.changed_fields == []
    assert r2.enriched_at is None
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1


@pytest.mark.anyio
async def test_incoming_balance_vs_balanceless_candidate_defers(session: AsyncSession):
    """Incoming carries a balance but the lone candidate has none — a
    presence mismatch. Don't blind-merge; DEFER."""
    await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance=None, transaction_time=time(21, 36, 27)),
        sms_message_id=389,
    )
    outcome, txn, _ = await merge_transaction(
        session,
        "email",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 12)),
        email_id=3320,
    )
    assert outcome == "deferred"
    assert txn is None
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 1


@pytest.mark.anyio
async def test_three_identical_charges_distinct_balances_three_rows(
    session: AsyncSession,
):
    """Three back-to-back identical-amount charges, each a distinct balance,
    produce three rows."""
    for i, bal in enumerate(("100000.00", "95000.00", "90000.00")):
        outcome, _r, _ = await merge_transaction(
            session,
            "sms",
            _icici_spend_sms(balance=bal, transaction_time=time(21, 36, 27 + i)),
            sms_message_id=400 + i,
        )
        assert outcome == "created"
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 3


@pytest.mark.anyio
async def test_worked_trace_all_four_notifications_pair_by_balance(
    session: AsyncSession,
):
    """Two distinct ₹5,000 charges, each reported by SMS + email. All four
    notifications converge to exactly two rows, each paired by balance into
    an sms+email row."""
    # SMS1 (charge A), SMS2 (charge B), Email1 (A), Email2 (B).
    await merge_transaction(
        session,
        "sms",
        _icici_spend_sms(balance="100000.00", transaction_time=time(21, 36, 27)),
        sms_message_id=389,
    )
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

    match = await find_match(
        session,
        {
            "bank": "icici",
            "direction": "debit",
            "amount": Decimal("500"),
            "currency": "INR",
            "reference_number": None,
            "transaction_date": date(2026, 5, 16),
            "transaction_time": time(22, 33, 30),
            "counterparty": "STARBUCKS",
            "balance": Decimal("500.00"),  # differs from candidate
        },
    )
    assert match.action == "insert"


@pytest.mark.anyio
async def test_am_pm_alias_balanceless_candidate_still_matches(session: AsyncSession):
    """A pre-AM/PM-fix row may have balance=None. An alias hit must still
    MATCH it (treat None as merge, not presence-mismatch defer)."""
    existing = Transaction(
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("500"),
        currency="INR",
        transaction_date=date(2026, 5, 16),
        transaction_time=time(10, 33, 11),
        counterparty="STARBUCKS",
        balance=None,
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
            "transaction_time": time(22, 33, 30),
            "counterparty": "STARBUCKS",
            "balance": Decimal("500.00"),
        },
    )
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
