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

import pytest

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
    error, txn_data, _hint, parsed = _process_email_full(
        "hdfc", _raw_hdfc_neft("Sun, 26 Jul 2026 20:15:42 +0530")
    )
    assert error is None, error
    assert parsed is not None
    assert txn_data is not None
    assert txn_data["email_type"] == "hdfc_account_neft_debit_alert"
    assert txn_data["transaction_time"] == datetime.time(20, 15, 42)
    assert txn_data["transaction_date"] == datetime.date(2026, 7, 26)
    # The SMS side has no payee. This payee must reach the row.
    assert txn_data["counterparty"] == "Sample Payee"


def test_neft_date_and_time_come_from_one_ist_conversion():
    """The true pair arrived at 01:03 IST. This is after midnight, so the IST
    date is one day later than the UTC date. Two separate conversions would
    give two different moments. This test keeps them together."""
    error, txn_data, _hint, _parsed = _process_email_full(
        # 19:33:51 UTC on the 26th == 01:03:51 IST on the 27th.
        "hdfc",
        _raw_hdfc_neft("Sun, 26 Jul 2026 19:33:51 +0000"),
    )
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_date"] == datetime.date(2026, 7, 27)
    assert txn_data["transaction_time"] == datetime.time(1, 3, 51)


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


def test_the_declaration_reaches_txn_data_as_a_column_value():
    """The parser declares the fact. The dashboard must record it, because
    the matcher reads it from the stored row later.

    This test uses true parser output and not a dict that you write here. A
    dict that you write can omit the key.
    """
    _error, txn_data, _hint, parsed = _process_email_full(
        "hdfc", _raw_hdfc_neft("Sun, 26 Jul 2026 20:15:42 +0530")
    )
    assert parsed is not None
    assert parsed.event_time_source == "message_arrival"
    assert txn_data is not None
    assert txn_data["transaction_time_is_received_time"] is True


def test_the_name_source_reaches_txn_data_as_a_column_value():
    """HDFC prints the payee label the user saved, not the account holder.

    The parser says so. The dashboard must record it, or every such row
    claims a bank stated the name and a later label can replace it.
    """
    _error, txn_data, _hint, parsed = _process_email_full(
        "hdfc", _raw_hdfc_neft("Sun, 26 Jul 2026 20:15:42 +0530")
    )
    assert parsed is not None
    assert parsed.counterparty_source == "user_alias"
    assert txn_data is not None
    assert txn_data["counterparty"] == "Sample Payee"
    assert txn_data["counterparty_source"] == "user_alias"


def test_am_pm_disambiguation_still_runs_for_its_own_types():
    """The fallback is an elif before the AM/PM branch. Make sure that it did
    not remove the AM/PM step for types that read a true time."""
    msg = EmailMessage()
    msg["Subject"] = "Transaction alert for your ICICI Bank Credit Card"
    msg["From"] = "credit_cards@icici.bank.in"
    msg["Date"] = "Sun, 17 May 2026 18:37:43 +0530"
    msg.set_content(
        "Your ICICI Bank Credit Card XX0000 has been used for a transaction of "
        "INR 100.00 on May 17, 2026 at 06:37:31. Info: TEST MERCHANT. "
        "Available Credit Limit on your card is INR 1,000.00."
    )
    error, txn_data, _hint, _parsed = _process_email_full("icici", msg.as_bytes())
    assert error is None, error
    assert txn_data is not None
    assert txn_data["transaction_time"] == datetime.time(18, 37, 31)


def test_a_new_bank_needs_no_dashboard_change():
    """The parser declares the fact, so the dashboard holds no list of names.
    A parser class that declares message_arrival gets the fallback at once.
    This test builds such a class to show that no code here names a bank.
    """
    from bank_email_parser.models import Money, ParsedEmail, TransactionAlert
    from bank_email_parser.parsers.base import BaseEmailParser, parse_with_parsers

    class _NewBankAlertParser(BaseEmailParser):
        bank = "newbank"
        email_type = "newbank_transfer_debit_alert"
        event_time_source = "message_arrival"

        def parse(self, html: str) -> ParsedEmail:
            return ParsedEmail(
                email_type=self.email_type,
                bank=self.bank,
                transaction=TransactionAlert(
                    direction="debit", amount=Money(amount="1.00")
                ),
            )

    result = parse_with_parsers("newbank", "<p>x</p>", (_NewBankAlertParser(),))
    assert result.event_time_source == "message_arrival"


def test_a_parser_cannot_declare_a_nonsense_time_source():
    """The base class checks the value when you define the class. A typo thus
    fails at import and not at run time."""
    from bank_email_parser.parsers.base import BaseEmailParser

    with pytest.raises(TypeError, match="event_time_source"):

        class _BadParser(BaseEmailParser):
            bank = "newbank"
            email_type = "newbank_typo_alert"
            event_time_source = "recieved"  # codespell:ignore

            def parse(self, html):  # pragma: no cover
                raise NotImplementedError
