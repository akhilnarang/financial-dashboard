"""Tests that pair the two messages for one credit card bill payment.

You pay your own card bill, so such a message names no merchant. The
counterparty thus cannot show which payment the message reports. Each parser
declares what does show it:

- ``card_mask``: the bank gives the card mask (IDFC, ICICI).
- ``none``: the bank gives no field at all (IndusInd).

The dashboard reads the declaration. It holds no list of bank names.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db import Transaction
from financial_dashboard.services.txn_merge import merge_transaction


def _idfc_sms(mask: str = "XX0000") -> dict:
    """The IDFC SMS. It gives the card mask and no counterparty."""
    return {
        "bank": "idfc",
        "email_type": "idfc_cc_payment_received_alert",
        "direction": "credit",
        "amount": Decimal("1234.56"),
        "currency": "INR",
        "card_mask": mask,
        "transaction_date": datetime.date(2026, 7, 26),
        "transaction_time": datetime.time(9, 45, 17),
        "identifies_by": "card_mask",
    }


def _idfc_email(mask: str = "XX0000") -> dict:
    """The IDFC email. It gives the card mask and a fixed label. It has no
    time, so find_match uses the date-only path."""
    return {
        "bank": "idfc",
        "email_type": "idfc_cc_credit_alert",
        "direction": "credit",
        "amount": Decimal("1234.56"),
        "currency": "INR",
        "card_mask": mask,
        "counterparty": "Payment received",
        "transaction_date": datetime.date(2026, 7, 26),
        "transaction_time": None,
        "identifies_by": "card_mask",
    }


def _indus_email(amount: str = "2000.00") -> dict:
    """The IndusInd email. The bank gives no card mask at all."""
    return {
        "bank": "indusind",
        "email_type": "indusind_cc_payment_alert",
        "direction": "credit",
        "amount": Decimal(amount),
        "currency": "INR",
        "counterparty": "Payment received",
        "transaction_date": datetime.date(2026, 6, 22),
        "transaction_time": None,
        "identifies_by": "none",
    }


def _indus_sms(amount: str = "2000.00") -> dict:
    """The IndusInd SMS. It writes the same fixed label as the email."""
    return {
        "bank": "indusind",
        "email_type": "indusind_cc_payment_received_alert",
        "direction": "credit",
        "amount": Decimal(amount),
        "currency": "INR",
        "counterparty": "Payment received",
        "transaction_date": datetime.date(2026, 6, 22),
        "transaction_time": datetime.time(11, 39, 21),
        "identifies_by": "none",
    }


@pytest.mark.anyio
async def test_the_idfc_pair_merges_on_the_card_mask(session):
    """One real pair in production. One payment made two rows, because
    the SMS has no counterparty and the email has a fixed label."""
    async with session.begin():
        _o, sms_row, _d = await merge_transaction(
            session, "sms", _idfc_sms(), sms_message_id=1
        )
    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "email", _idfc_email(), email_id=1
        )

    assert outcome == "enriched"
    assert row.id == sms_row.id
    assert row.source == "sms+email"


@pytest.mark.anyio
async def test_a_different_card_never_merges(session):
    """Two cards can take a payment of the same amount on the same day. The
    mask shows that these are different events."""
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", _idfc_sms("XX0000"), sms_message_id=1
        )
    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "email", _idfc_email("XX9999"), email_id=1
        )

    assert outcome == "created"
    assert row.id != first_row.id


@pytest.mark.anyio
async def test_a_mask_that_ends_in_a_wildcard_shows_no_card(session):
    """A mask can show the first digits of a card number and hide the rest.
    Such a mask ends in a wildcard and shows no card. Its last four digits
    still look like a suffix, so a match on digits alone gives a wrong pair.
    """
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "sms", _idfc_sms("543210XXXXXX"), sms_message_id=1
        )
    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "email", _idfc_email("XXXXXXXX3210"), email_id=1
        )

    assert outcome == "created"
    assert row.id != first_row.id


@pytest.mark.anyio
async def test_the_indusind_pair_merges_after_a_statement_rewrite(session):
    """One real pair in production. Both parsers write the same label,
    so the pair merges on that label alone. But a statement import replaces
    the label on the stored row with the narration from the statement. The
    SMS arrived 26 days later, and the two labels then disagreed.

    The label is thus not a property of the event. Do not read it for a shape
    that declares that no field shows the event.
    """
    async with session.begin():
        _o, email_row, _d = await merge_transaction(
            session, "email", _indus_email(), email_id=1
        )
    # The statement import writes the narration over the label.
    async with session.begin():
        email_row.counterparty = "BBPS PAYMENT"

    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "sms", _indus_sms(), sms_message_id=2
        )

    assert outcome == "enriched"
    assert row.id == email_row.id
    assert row.source == "sms+email"


@pytest.mark.anyio
async def test_two_alike_indusind_payments_go_to_review(session):
    """This bank sends no field that shows the card. Two payments of the same
    amount on the same day are thus alike in every field. The matcher must
    not choose between them. It defers, and a person decides.
    """
    async with session.begin():
        await merge_transaction(session, "email", _indus_email(), email_id=1)
    async with session.begin():
        await merge_transaction(
            session, "email", _indus_email(), email_id=2, force_new=True
        )
    rows = (await session.execute(select(Transaction))).scalars().all()
    assert len(rows) == 2

    outcome, row, _d = await merge_transaction(
        session, "sms", _indus_sms(), sms_message_id=2
    )
    await session.commit()

    assert outcome == "deferred"
    assert row is None


@pytest.mark.anyio
async def test_a_different_amount_never_merges(session):
    """The amount stays part of the identity of the event."""
    async with session.begin():
        _o, first_row, _d = await merge_transaction(
            session, "email", _indus_email("2000.00"), email_id=1
        )
    async with session.begin():
        outcome, row, _d = await merge_transaction(
            session, "sms", _indus_sms("3000.00"), sms_message_id=3
        )

    assert outcome == "created"
    assert row.id != first_row.id
