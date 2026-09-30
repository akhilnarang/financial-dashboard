"""Card alert emails through ``_process_email_full``."""

import datetime
from decimal import Decimal
from email.message import EmailMessage

import pytest

from financial_dashboard.services.emails import _process_email_full


def _raw(subject: str, sender: str, date_header: str, body: str) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["Date"] = date_header
    msg.set_content(body)
    return msg.as_bytes()


# ICICI CC alerts print a 12-hour time with no AM/PM marker. The email Date
# header decides the half of the day.
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
        # Both candidates fail, so the parsed time stays unchanged.
        pytest.param(
            "12:01:00", "Mon, 18 May 2026 00:02:00 +0530", (12, 1, 0), id="next-day"
        ),
    ],
)
def test_icici_cc_time_resolves_half_of_day_from_email_date(
    body_time, date_header, expected
):
    body = (
        "Your ICICI Bank Credit Card XX0000 has been used for a transaction of "
        f"INR 100.00 on May 17, 2026 at {body_time}. Info: TEST MERCHANT. "
        "Available Credit Limit on your card is INR 1,000.00."
    )
    error, txn_data, _hint, _parsed = _process_email_full(
        "icici",
        _raw(
            "Transaction alert for your ICICI Bank Credit Card",
            "credit_cards@icici.bank.in",
            date_header,
            body,
        ),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["email_type"] == "icici_cc_transaction_alert"
    assert txn_data["transaction_time"] == datetime.time(*expected)


def test_idfc_cc_credit_alert_is_processed_as_credit_transaction():
    error, txn_data, _hint, _parsed = _process_email_full(
        "idfc",
        _raw(
            "Payment received on your IDFC FIRST Credit Card",
            "alerts@idfcfirstbank.com",
            "Fri, 15 May 2026 10:00:00 +0530",
            "Payment of Rs. 1,234.56 was received on your FIRST Wealth "
            "Credit Card ending with XX1234 on 15 May 2026.",
        ),
    )

    assert error is None
    assert txn_data is not None
    assert txn_data["email_type"] == "idfc_cc_credit_alert"
    assert txn_data["direction"] == "credit"
    assert txn_data["amount"] == Decimal("1234.56")
    assert txn_data["card_mask"] == "XX1234"
    assert txn_data["channel"] == "card"
