"""Tests that pair an HDFC NEFT SMS with its email.

The SMS has a time and the source account mask but no payee. The email has the
payee and the same mask but no time. The arrival-time fallback gives the email
a time. Both sides then have a time, so find_match uses the timed path and not
the date-only path. The date-only path needs a counterparty that the SMS
cannot supply.

The tests below also examine the limits that keep the fallback safe. The
matcher must never merge a second, different payment with the first row.
"""

import datetime
from decimal import Decimal

import pytest

from financial_dashboard.services.txn_merge import merge_transaction


def _sms_txn(amount: str = "1234.56", time: datetime.time | None = None) -> dict:
    """The SMS side. It has a time and a mask. It has no payee, no
    reference, and no balance."""
    return {
        "bank": "hdfc",
        "email_type": "hdfc_account_neft_debit_alert",
        "direction": "debit",
        "amount": Decimal(amount),
        "currency": "INR",
        "account_mask": "XX0000",
        "channel": "neft",
        "transaction_date": datetime.date(2026, 7, 27),
        "transaction_time": time or datetime.time(1, 3, 51),
        # The parser for this SMS declares message_arrival, so the pipeline
        # sets this for every such SMS.
        "transaction_time_is_received_time": True,
    }


def _email_txn(
    amount: str = "1234.56",
    time: datetime.time | None = None,
    counterparty: str = "Sample Payee",
) -> dict:
    """The email side. It has the payee and the same mask. The fallback
    supplies the time."""
    return {
        "bank": "hdfc",
        "email_type": "hdfc_account_neft_debit_alert",
        "direction": "debit",
        "amount": Decimal(amount),
        "currency": "INR",
        "account_mask": "XX0000",
        "channel": "neft",
        "counterparty": counterparty,
        "transaction_date": datetime.date(2026, 7, 27),
        # Some seconds after the SMS, as in the true pairs.
        "transaction_time": time or datetime.time(1, 3, 53),
        # The arrival time supplied this. Thus it gets the small window.
        "transaction_time_is_received_time": True,
    }


@pytest.mark.anyio
async def test_neft_sms_then_email_merges_and_fills_the_payee(session):
    async with session.begin():
        _outcome, sms_row, _diff = await merge_transaction(
            session, "sms", _sms_txn(), sms_message_id=1
        )
    txn_id = sms_row.id
    assert sms_row.counterparty is None

    async with session.begin():
        outcome, row, diff = await merge_transaction(
            session, "email", _email_txn(), email_id=1
        )

    assert outcome == "enriched"
    assert row.id == txn_id, "the email must add data to the SMS row"
    assert row.source == "sms+email"
    # The payee is the reason for the pair.
    assert row.counterparty == "Sample Payee"
    assert "counterparty" in diff.filled
    # Keep the first time that a message gave for this event.
    assert row.transaction_time == datetime.time(1, 3, 51)


@pytest.mark.anyio
async def test_matcher_does_not_merge_a_second_distinct_neft(session):
    """This test examines the limit on merges. Two NEFT payments of the same
    amount occur some minutes apart, and all messages arrive. The second SMS
    must not merge into the first row.

    This failure caused the refusal of the general time fallback. A different
    payment disappeared. The slot test prevents this.
    """
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", _sms_txn(), sms_message_id=1
        )
    async with session.begin():
        await merge_transaction(session, "email", _email_txn(), email_id=1)

    # A different second NEFT of the same amount, 4 minutes later. This is
    # inside the window of 10 minutes.
    async with session.begin():
        outcome, row, _diff = await merge_transaction(
            session,
            "sms",
            _sms_txn(time=datetime.time(1, 7, 20)),
            sms_message_id=2,
        )

    # The matcher can defer the row for review or make a new row. Both
    # results are correct. A merge into the first row is not correct.
    assert outcome in ("deferred", "created"), (
        f"the matcher must not merge a different payment. Outcome: {outcome!r}"
    )
    if row is not None:
        assert row.id != first_row.id
        assert row.counterparty != "Sample Payee"


@pytest.mark.anyio
async def test_different_source_accounts_never_merge(session):
    """Two HDFC accounts can send the same amount some minutes apart. These
    are different events, even if the amount and the time agree. Thus the
    account mask must divide them. If it does not, the payee and the mask of
    the second event replace those of the first row. One payment is then
    absent from the ledger."""
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", _sms_txn() | {"account_mask": "XX0001"}, sms_message_id=1
        )

    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session,
            "email",
            _email_txn(time=datetime.time(1, 5, 10), counterparty="Other Payee")
            | {"account_mask": "XX0002"},
            email_id=1,
        )

    assert outcome == "created", "a different account is a different event"
    assert row.id != first_row.id
    assert first_row.account_mask == "XX0001", (
        "the code must keep the mask of the first row"
    )


@pytest.mark.anyio
async def test_an_absent_mask_never_splits_a_true_pair(session):
    """An absent mask is not a different mask. Keep a candidate that has no
    mask. If you remove it, the code that follows sees an empty set, calls the
    event new, and makes a duplicate row."""
    sms = _sms_txn()
    del sms["account_mask"]
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", sms, sms_message_id=1
        )

    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "email", _email_txn(), email_id=1
        )

    assert outcome == "enriched"
    assert row.id == first_row.id


@pytest.mark.anyio
async def test_matcher_does_not_merge_a_lone_email_minutes_away(session):
    """This is the worst condition for a supplied time. The email for
    payment A does not arrive, so the email slot of row A stays open. The
    email for a different payment B then arrives, and the SMS for B is also
    absent. No later step can show this loss. Thus the window itself must
    refuse the match.

    An arrival time gives the time of the message and not the time of the
    event. Thus it gets a window of the measured size and not 10 minutes.
    """
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", _sms_txn(), sms_message_id=1
        )

    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session,
            "email",
            _email_txn(time=datetime.time(1, 7, 51), counterparty="Payee B"),
            email_id=1,
        )

    assert outcome == "created", "a payment minutes away is a different event"
    assert row.id != first_row.id
    assert first_row.counterparty is None, "the matcher must not change the first row"


@pytest.mark.anyio
async def test_email_first_row_also_gets_the_tight_window(session):
    """The email can make the row if the SMS is late. The arrival time then
    supplied the stored time also, but no column keeps this fact. If the
    matcher does not know it, a different payment some minutes later arrives
    on the full window. The matcher then merges that payment with the row.
    Calculate the fact from the row.
    """
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "email", _email_txn(counterparty="Payee A"), email_id=1
        )

    # A different payment. Its SMS has no hint. Thus only the supplied time
    # of the stored row can keep the window small.
    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "sms", _sms_txn(time=datetime.time(1, 8, 51)), sms_message_id=2
        )

    assert outcome == "created", "a payment minutes away is a different event"
    assert row.id != first_row.id
    assert first_row.counterparty == "Payee A"


@pytest.mark.anyio
async def test_a_late_uploaded_sms_still_pairs(session):
    """The forwarder can send an SMS to the server many hours or days after
    the phone receives it. The SMS fallback reads the time of receipt at the
    phone and not the time of upload. Thus the delay does not move
    transaction_time, and the small window does not refuse the pair.
    """
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "email", _email_txn(counterparty="Payee A"), email_id=1
        )

    # The phone received this SMS 2 seconds before the email. The forwarder
    # sent it to the server 3 days later.
    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "sms", _sms_txn(time=datetime.time(1, 3, 51)), sms_message_id=1
        )

    assert outcome == "enriched"
    assert row.id == first_row.id
    assert row.source == "sms+email"


@pytest.mark.anyio
async def test_true_pair_still_merges_at_the_worst_measured_jitter(session):
    """The window must stay large enough for a true pair. In 59 pairs from
    production the two messages arrive -5 to +14 seconds apart."""
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", _sms_txn(), sms_message_id=1
        )

    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "email", _email_txn(time=datetime.time(1, 4, 5)), email_id=1
        )

    assert outcome == "enriched"
    assert row.id == first_row.id
    assert row.counterparty == "Sample Payee"


@pytest.mark.anyio
async def test_residual_within_the_tight_window_still_leaves_a_trail(session):
    """In this small window you cannot tell two payments apart. They have the
    same account, the same amount, and a difference of some seconds. Neither
    has a reference or a balance. The small window makes this band shorter but
    cannot remove it. Thus the important limit is this: the matcher must never
    merge the SMS of the second payment with the first row. That SMS goes to the review queue,
    where you can make the true row. The ledger then shows too little, but the
    operator can see the problem.
    """
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", _sms_txn(), sms_message_id=1
        )

    # The email for payment 1 does not arrive. The email for payment 2
    # arrives 9 seconds later.
    async with session.begin():
        lone_outcome, lone_row, _d = await merge_transaction(
            session,
            "email",
            _email_txn(time=datetime.time(1, 4, 0), counterparty="Other Payee"),
            email_id=2,
        )
    assert lone_outcome == "enriched"
    assert lone_row.id == first_row.id

    # The matcher must not merge the SMS for payment 2 with that row.
    outcome, row, _d = await merge_transaction(
        session, "sms", _sms_txn(time=datetime.time(1, 4, 0)), sms_message_id=2
    )
    await session.commit()
    assert outcome == "deferred", "the second payment must stay recoverable"
    assert row is None


@pytest.mark.anyio
async def test_second_email_does_not_overwrite_a_filled_payee(session):
    """After the pair is complete, a second email of the same amount must not
    write a different payee to the same row."""
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", _sms_txn(), sms_message_id=1
        )
    async with session.begin():
        await merge_transaction(session, "email", _email_txn(), email_id=1)

    async with session.begin():
        outcome, row, _diff = await merge_transaction(
            session,
            "email",
            _email_txn(time=datetime.time(1, 7, 22), counterparty="Other Payee"),
            email_id=2,
        )

    assert outcome != "enriched"
    if row is not None and row.id == first_row.id:
        assert row.counterparty == "Sample Payee"


@pytest.mark.anyio
async def test_the_flag_keeps_agreeing_with_the_stored_time(session):
    """The column describes the stored transaction_time. The two must not
    disagree after a merge.

    An SMS does not overwrite a value that came from an email. Thus a stated
    time from an SMS does not replace a supplied time from an email, and the
    column stays correct.
    """
    async with session.begin():
        _o, row, _d = await merge_transaction(
            session, "email", _email_txn(counterparty="Payee A"), email_id=1
        )
    assert row.transaction_time == datetime.time(1, 3, 53)
    assert row.transaction_time_is_received_time is True

    # The SMS states its own time, so its flag is False.
    sms = _sms_txn() | {"transaction_time_is_received_time": False}
    async with session.begin():
        outcome, merged, _d = await merge_transaction(
            session, "sms", sms, sms_message_id=1
        )

    assert outcome == "enriched"
    assert merged.transaction_time == datetime.time(1, 3, 53)
    assert merged.transaction_time_is_received_time is True


@pytest.mark.anyio
async def test_a_row_from_before_the_column_is_still_safe(session):
    """A row that this database held before the column existed has the value
    0, so it claims a stated time that it does not have. The incoming message
    still carries 1, and the small window applies from that side. Thus the
    matcher does not merge a different payment into such a row.
    """
    stale = _email_txn(counterparty="Payee A") | {
        "transaction_time_is_received_time": False
    }
    async with session.begin():
        _o, first_row, _d = await merge_transaction(session, "email", stale, email_id=1)

    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "sms", _sms_txn(time=datetime.time(1, 8, 51)), sms_message_id=2
        )

    assert outcome == "created"
    assert row.id != first_row.id


@pytest.mark.anyio
async def test_enrichment_keeps_the_flag_with_the_time_it_describes(session):
    """A row can hold no time and the default value 0. An email then fills the
    time from the time of arrival. The column must change with the time.

    If it does not, the row holds a supplied time but claims a stated one. The
    matcher then gives the row the wide window of 10 minutes, and a different
    payment can merge into it.
    """
    from financial_dashboard.db import Transaction

    async with session.begin():
        row = Transaction(
            bank="hdfc",
            email_type="hdfc_account_neft_debit_alert",
            direction="debit",
            amount=Decimal("1234.56"),
            currency="INR",
            account_mask="XX0000",
            channel="neft",
            transaction_date=datetime.date(2026, 7, 27),
            transaction_time=None,
            transaction_time_is_received_time=False,
            counterparty="Sample Payee",
            source="email",
        )
        session.add(row)
        await session.flush()

    async with session.begin():
        outcome, merged, diff = await merge_transaction(
            session, "email", _email_txn(), email_id=9
        )

    assert outcome == "enriched"
    assert "transaction_time" in diff.changed_fields
    assert merged.transaction_time == datetime.time(1, 3, 53)
    assert merged.transaction_time_is_received_time is True
