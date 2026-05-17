"""Regression coverage for the ICICI CC AM/PM disambiguation in _process_email_full.

ICICI's icici_cc_transaction_alert emails emit transaction time on a
12-hour clock and strip the AM/PM marker (e.g. '06:37:31' for 6:37 PM).
The dashboard infers the half-of-day from the email's Date header.
"""

from __future__ import annotations

import datetime
from email.message import EmailMessage

from financial_dashboard.services.emails import (
    _disambiguate_am_pm,
    _process_email_full,
)


def _raw_icici_cc(body: str, date_header: str) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Transaction alert for your ICICI Bank Credit Card"
    msg["From"] = "credit_cards@icici.bank.in"
    msg["Date"] = date_header
    msg.set_content(body)
    return msg.as_bytes()


def _body(time_str: str, *, amount: str = "100.00") -> str:
    return (
        f"Your ICICI Bank Credit Card XX0000 has been used for a transaction of "
        f"INR {amount} on May 17, 2026 at {time_str}. Info: TEST MERCHANT. "
        f"Available Credit Limit on your card is INR 1,000.00."
    )


def test_icici_cc_pm_transaction_is_flipped_when_email_arrives_pm():
    """6:37 in the body + Date header at 6:37 PM IST → flip to 18:37."""
    error, txn_data, _hint, parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("06:37:31"), "Sun, 17 May 2026 18:37:43 +0530"),
    )
    assert error is None, error
    assert parsed is not None
    assert txn_data is not None
    assert txn_data["email_type"] == "icici_cc_transaction_alert"
    assert txn_data["transaction_time"] == datetime.time(18, 37, 31)


def test_icici_cc_am_transaction_is_not_flipped_when_email_arrives_am():
    """6:37 in the body + Date header at 6:38 AM IST → leave at 06:37."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("06:37:31"), "Sun, 17 May 2026 06:38:10 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(6, 37, 31)


def test_icici_cc_noon_transaction_resolves_to_12pm_when_email_arrives_at_noon():
    """Body '12:55:20' could be 00:55 (midnight) or 12:55 (noon).
    Email arriving at 12:55 PM IST disambiguates to noon."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("12:55:20"), "Sun, 17 May 2026 12:55:29 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(12, 55, 20)


def test_icici_cc_midnight_transaction_resolves_to_00am():
    """Body '12:01:00' with email arriving just after midnight resolves
    to 00:01:00, not 12:01:00."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("12:01:00"), "Sun, 17 May 2026 00:01:35 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(0, 1, 0)


def test_icici_cc_delayed_am_email_is_not_flipped_to_pm():
    """If an AM transaction's email arrives many hours late but still
    before the PM mirror, AM must stay. Body 06:30, email at 10:00 IST:
    PM candidate (18:30) is in the future and must be rejected; AM
    (06:30) wins even though it's 3.5h before received_at."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("06:30:00"), "Sun, 17 May 2026 10:00:00 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(6, 30, 0)


def test_icici_cc_six_hour_late_am_email_still_picks_am():
    """Body 06:30, email at 12:31 IST (6h+1m late). PM candidate 18:30
    sits 5h59m *after* received_at — rejected by the 5-minute future
    tolerance. AM wins (6h+1m before received_at)."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("06:30:00"), "Sun, 17 May 2026 12:31:00 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(6, 30, 0)


def test_icici_cc_twelve_hour_late_am_email_degenerates_to_pm():
    """Documented limitation: a real ~12h-late AM alert is degenerate.
    Body 06:30, email at 18:35 (12h05m later). PM candidate (18:30) is
    5min *before* received_at — accepted — and wins on the closer-wins
    rule. Per spec ('don't assume 12h skew'), this is acceptable; we
    do not try to recover an extremely-delayed alert."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("06:30:00"), "Sun, 17 May 2026 18:35:00 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(18, 30, 0)


def test_icici_cc_next_day_email_falls_back_to_parsed_time():
    """Pathological: body says May 17 12:01 (ambiguous midnight/noon) but
    the email's Date header is May 18 00:02 IST — both candidates
    anchored to May 17 are >12h before received_at, so both are rejected
    by the past-gap cap. The disambiguator returns the parsed time
    unchanged (12:01:00); the data is unrecoverable from this email
    alone. This is documented behavior — we do not flip to noon and
    pretend we know."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("12:01:00"), "Mon, 18 May 2026 00:02:00 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(12, 1, 0)


def test_disambiguate_passes_hour_zero_through_unchanged():
    """hour=0 can only come from a 24-hour body (ICICI's 12-hour clock
    emits 1..12, never 0). Pass through untouched — do NOT treat it as
    ambiguous and flip it to noon."""
    received = datetime.datetime(2026, 5, 17, 12, 30, 0, tzinfo=datetime.timezone.utc)
    out = _disambiguate_am_pm(datetime.time(0, 15, 0), datetime.date(2026, 5, 17), received)
    assert out == datetime.time(0, 15, 0)


def test_icici_cc_exactly_12h_plus_one_second_boundary():
    """Boundary: body 06:30:00, email at 18:30:01 (12h+1s late). AM
    candidate's past-gap is strictly > 12h → rejected. PM candidate is
    1s before received_at → accepted. PM wins. This pins the strict
    > 12h cap so a refactor to >= 12h would be caught."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("06:30:00"), "Sun, 17 May 2026 18:30:01 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(18, 30, 0)


def test_icici_cc_late_night_midnight_transaction_email_just_after():
    """Body '12:01:00' for a 00:01 midnight txn, email arrives the same
    night at 00:02 IST. Both candidates anchored to the body's date
    (May 17): midnight (00:01) is 1min before received_at; noon (12:01)
    is ~12h in the future — rejected. Pick midnight."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw_icici_cc(_body("12:01:00"), "Sun, 17 May 2026 00:02:00 +0530"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(0, 1, 0)
