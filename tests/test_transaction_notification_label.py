from decimal import Decimal

import pytest

from financial_dashboard.db import Account, Card, Transaction
from financial_dashboard.services.linker import build_link_context, link_transaction
from financial_dashboard.services.sms_pipeline import _notification_payload
from financial_dashboard.services.telegram import format_money

pytestmark = pytest.mark.anyio


async def test_notification_payload_labels_a_fresh_linked_row(session):
    # A fresh row has no loaded relationships. A lazy load raises MissingGreenlet.
    account = Account(bank="HDFC", label="HDFC Credit Card", type="credit_card")
    session.add(account)
    await session.flush()
    session.add(
        Card(account_id=account.id, card_mask="XXXX XXXX XXXX 1234", label="self")
    )
    await session.flush()
    link_ctx = await build_link_context(session)

    txn = Transaction(
        bank="HDFC",
        email_type="hdfc_cc_transaction",
        direction="debit",
        amount=Decimal("100.00"),
        currency="EUR",
        card_mask="XXXX XXXX XXXX 1234",
    )
    session.add(txn)
    await session.flush()
    link_transaction(link_ctx, txn)
    await session.flush()

    payload = await _notification_payload(txn, session)

    assert payload["account_label"] == "HDFC Credit Card - self"
    assert format_money(payload["amount"], payload["currency"]) == "EUR 100.00"
