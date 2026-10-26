"""Tests for POST /api/transactions/{id}/exclude and /void.

Each endpoint sets the flag to a given value, not a toggle. A repeated request
lands the same state. The flag drops a row from the cashflow report only. The
row keeps its category and every other view. Each test reads the state back
through the API, not the database.
"""

import datetime
from decimal import Decimal

import pytest

from financial_dashboard.db import Receivable, ReceivableLink, Transaction

pytestmark = pytest.mark.anyio


async def _seed(session) -> int:
    txn = Transaction(
        bank="hdfc",
        email_type="x",
        direction="debit",
        amount=Decimal("24995"),
        currency="INR",
        category="shopping",
        transaction_date=datetime.date(2026, 6, 12),
    )
    session.add(txn)
    await session.commit()
    return txn.id


async def _read_flag(client, txn_id: int, field: str) -> bool:
    """Read the flag back through the list API, the way a client sees it."""
    r = await client.get(f"/api/transactions?transaction_id={txn_id}")
    assert r.status_code == 200, r.text
    return r.json()["items"][0][field]


@pytest.mark.parametrize(
    ("endpoint", "field"), [("exclude", "exclude_from_cashflow"), ("void", "voided")]
)
async def test_set_and_clear_is_observable_and_idempotent(
    client, session, endpoint, field
):
    txn_id = await _seed(session)
    # A fresh row reads as unflagged.
    assert await _read_flag(client, txn_id, field) is False

    # Set it, twice. The repeat lands the same state. It does not toggle.
    for _ in range(2):
        r = await client.post(
            f"/api/transactions/{txn_id}/{endpoint}", json={field: True}
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, field: True}
    assert await _read_flag(client, txn_id, field) is True

    # Clear it.
    r = await client.post(f"/api/transactions/{txn_id}/{endpoint}", json={field: False})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, field: False}
    assert await _read_flag(client, txn_id, field) is False

    missing = await client.post(
        f"/api/transactions/999999/{endpoint}", json={field: True}
    )
    assert missing.status_code == 404


@pytest.mark.parametrize("case", ["credit", "pays_receivable"])
async def test_a_credit_or_a_receivable_payment_cannot_be_voided(client, session, case):
    """A credit can be a card bill payment, and a payment row counts toward a
    receivable. Voiding either would leave that state stale, so the endpoint
    refuses it and the row stays counted."""
    txn_id = await _seed(session)
    if case == "credit":
        txn = await session.get(Transaction, txn_id)
        txn.direction = "credit"
    else:
        session.add(
            Receivable(
                kind="payable",
                person="Friend A",
                amount=Decimal("24995"),
                links=[ReceivableLink(transaction_id=txn_id, role="payment")],
            )
        )
    await session.commit()

    r = await client.post(f"/api/transactions/{txn_id}/void", json={"voided": True})
    assert r.status_code == 400
    assert await _read_flag(client, txn_id, "voided") is False
