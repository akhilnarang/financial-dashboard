"""Regression coverage for IDFC credit-card payment received alerts."""

from __future__ import annotations

from email.message import EmailMessage

from bank_email_fetcher.services.emails import _process_email_full


def _raw_email(body: str) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Payment received on your IDFC FIRST Credit Card"
    msg["From"] = "alerts@idfcfirstbank.com"
    msg["Date"] = "Fri, 15 May 2026 10:00:00 +0530"
    msg.set_content(body)
    return msg.as_bytes()


def test_idfc_cc_credit_alert_is_processed_as_credit_transaction():
    error, txn_data, password_hint, parsed_email = _process_email_full(
        "idfc",
        _raw_email(
            "Payment of Rs. 1,234.56 was received on your FIRST Wealth "
            "Credit Card ending with XX1234 on 15 May 2026."
        ),
    )

    assert error is None
    assert password_hint is None
    assert parsed_email is not None
    assert txn_data is not None
    assert txn_data["bank"] == "idfc"
    assert txn_data["email_type"] == "idfc_cc_credit_alert"
    assert txn_data["direction"] == "credit"
    assert txn_data["amount"] == 1234.56
    assert txn_data["currency"] == "INR"
    assert txn_data["counterparty"] == "Payment received"
    assert txn_data["card_mask"] == "XX1234"
    assert txn_data["account_mask"] is None
    assert txn_data["channel"] == "card"
