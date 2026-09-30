"""Tests for the arrival-time fallback in _process_email_full.

The HDFC NEFT email has a payee but no time. The related SMS has a time but no
payee. Without a time, the email goes to the date-only path in find_match.
That path needs the same counterparty on both sides, and the SMS has none.
Thus the two messages made two rows. HDFC sends this email at the moment of
the transaction, so the arrival time is a good substitute.

Only a parser that declares message_arrival gets this fallback. Do not
declare it for all email types.
Some banks send an email many hours after the event. Such an email would get a
wrong time. It could then match a different payment.
"""

import datetime
from email.message import EmailMessage


from financial_dashboard.services.emails import _process_email_full


def _raw_hdfc_neft(date_header: str, *, amount: str = "1234.56") -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "View: Account update for your HDFC Bank A/c"
    msg["From"] = "HDFC Bank InstaAlerts <alerts@hdfcbank.bank.in>"
    msg["Date"] = date_header
    msg.set_content(
        f"Dear Customer, Thank you for banking with HDFC Bank. Rs. {amount} has "
        f"been deducted from your HDFC Bank account ending in XX0000 for a "
        f"transfer to payee Sample Payee via NEFT using HDFC Bank Online "
        f"Banking. Not you? Call 00000000000 from your registered mobile number."
    )
    return msg.as_bytes()


def _raw_hdfc_upi(date_header: str) -> bytes:
    """A shape that is not in the list. It shows that the list controls the
    fallback."""
    msg = EmailMessage()
    msg["Subject"] = "You have done a UPI txn"
    msg["From"] = "HDFC Bank InstaAlerts <alerts@hdfcbank.bank.in>"
    msg["Date"] = date_header
    msg.set_content(
        "Rs.500.00 has been debited from account 0000 to VPA "
        "merchant@upi Sample Merchant on 26-07-26."
    )
    return msg.as_bytes()


def test_neft_email_gets_transaction_time_from_received_time():
    """The date and time come from one IST conversion. 19:33:51 UTC on the
    26th is 01:03:51 IST on the 27th. Two separate conversions would give two
    different moments."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "hdfc", _raw_hdfc_neft("Sun, 26 Jul 2026 19:33:51 +0000")
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["email_type"] == "hdfc_account_neft_debit_alert"
    assert txn_data["transaction_time"] == datetime.time(1, 3, 51)
    assert txn_data["transaction_date"] == datetime.date(2026, 7, 27)
    # The SMS side has no payee. This payee must reach the row.
    assert txn_data["counterparty"] == "Sample Payee"
    # The matcher reads these flags from the stored row later.
    assert txn_data["transaction_time_is_received_time"] is True
    assert txn_data["counterparty_source"] == "user_alias"


def test_fallback_never_moves_a_date_the_body_supplied():
    """A date from the body is correct. If a listed type has a body date but
    no time, the fallback must supply only the time. If it moves the date to
    the day of the arrival time, it moves the event to a different day."""
    import datetime as _dt

    from financial_dashboard.services import emails as emails_mod

    original = emails_mod.parse_email

    def _with_body_date(bank, html):
        parsed = original(bank, html)
        parsed.transaction.transaction_date = _dt.date(2026, 7, 25)
        return parsed

    emails_mod.parse_email = _with_body_date
    try:
        error, txn_data, _hint, _parsed = _process_email_full(
            "hdfc", _raw_hdfc_neft("Sun, 26 Jul 2026 19:33:51 +0000")
        )
    finally:
        emails_mod.parse_email = original

    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_date"] == datetime.date(2026, 7, 25)
    assert txn_data["transaction_time"] == datetime.time(1, 3, 51)


def test_a_body_time_source_keeps_a_null_transaction_time():
    """The declaration gives the safety. A parser that does not declare
    message_arrival must not get a time. A slow email would become a
    candidate for a different event."""
    error, txn_data, _hint, _parsed = _process_email_full(
        "hdfc", _raw_hdfc_upi("Sun, 26 Jul 2026 20:15:42 +0530")
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["email_type"] == "hdfc_upi_alert"
    assert txn_data["transaction_time"] is None


def test_date_fallback_uses_ist_for_every_email_type():
    """A body with no date gets the arrival date in IST. The slice repayment
    email arrived at 01:23 IST. Its UTC date is one day earlier. The SMS for
    the same repayment has the IST date. A UTC date made two rows."""
    msg = EmailMessage()
    msg["Subject"] = "Payment received for your slice credit card"
    msg["From"] = "slice <noreply@slice.bank.in>"
    # 19:53:15 UTC on the 29th == 01:23:15 IST on the 30th.
    msg["Date"] = "Tue, 29 Sep 2026 19:53:15 +0000"
    msg.set_content(
        "<p>We&rsquo;ve received your repayment of &#8377; 1,234.56 for the "
        "slice credit card.</p>",
        subtype="html",
    )
    error, txn_data, _hint, _parsed = _process_email_full("slice", msg.as_bytes())
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_date"] == datetime.date(2026, 9, 30)
    assert txn_data["transaction_time"] is None
