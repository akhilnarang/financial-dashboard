"""Regression coverage for the ICICI CC AM/PM disambiguation in _process_email_full.

ICICI's icici_cc_transaction_alert emails emit transaction time on a
12-hour clock and strip the AM/PM marker (e.g. '06:37:31' for 6:37 PM).
The dashboard infers the half-of-day from the email's Date header.
"""

import datetime
from email.message import EmailMessage

import pytest

from financial_dashboard.services.emails import _process_email_full


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


@pytest.mark.parametrize(
    ("body_time", "date_header", "expected"),
    [
        pytest.param(
            "06:37:31", "Sun, 17 May 2026 18:37:43 +0530", (18, 37, 31), id="pm-flip"
        ),
        pytest.param(
            "06:37:31", "Sun, 17 May 2026 06:38:10 +0530", (6, 37, 31), id="am-keep"
        ),
        pytest.param(
            "12:55:20", "Sun, 17 May 2026 12:55:29 +0530", (12, 55, 20), id="noon"
        ),
        pytest.param(
            "12:01:00", "Sun, 17 May 2026 00:01:35 +0530", (0, 1, 0), id="midnight"
        ),
        # The PM candidate lies in the future, so a late AM alert stays AM.
        pytest.param(
            "06:30:00", "Sun, 17 May 2026 12:31:00 +0530", (6, 30, 0), id="late-am"
        ),
        # The AM candidate is more than 12h old, so PM wins.
        pytest.param(
            "06:30:00", "Sun, 17 May 2026 18:30:01 +0530", (18, 30, 0), id="12h-cap"
        ),
        # Both candidates fail, so the parsed time stays unchanged.
        pytest.param(
            "12:01:00", "Mon, 18 May 2026 00:02:00 +0530", (12, 1, 0), id="next-day"
        ),
    ],
)
def test_icici_cc_time_resolves_half_of_day_from_email_date(
    body_time, date_header, expected
):
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici", _raw_icici_cc(_body(body_time), date_header)
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["email_type"] == "icici_cc_transaction_alert"
    assert txn_data["transaction_time"] == datetime.time(*expected)
