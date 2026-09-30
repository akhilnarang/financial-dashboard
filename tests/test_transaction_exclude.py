"""Tests for POST /api/transactions/{id}/exclude.

The endpoint sets the flag to a given value, not a toggle. A repeated request
lands the same state. The flag drops a row from the cashflow report only. The
row keeps its category and every other view. Each test reads the state back
through the API, not the database.
"""

import datetime
from decimal import Decimal

import pytest

from financial_dashboard.db import Transaction

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


async def _read_flag(client, txn_id: int) -> bool:
    """Read the flag back through the list API, the way a client sees it."""
    r = await client.get(f"/api/transactions?transaction_id={txn_id}")
    assert r.status_code == 200, r.text
    return r.json()["items"][0]["exclude_from_cashflow"]


async def test_set_and_clear_is_observable_and_idempotent(client, session):
    txn_id = await _seed(session)
    # A fresh row reads as included.
    assert await _read_flag(client, txn_id) is False

    # Set it, twice. The repeat lands the same state. It does not toggle.
    for _ in range(2):
        r = await client.post(
            f"/api/transactions/{txn_id}/exclude",
            json={"exclude_from_cashflow": True},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "exclude_from_cashflow": True}
    assert await _read_flag(client, txn_id) is True

    # Clear it.
    r = await client.post(
        f"/api/transactions/{txn_id}/exclude",
        json={"exclude_from_cashflow": False},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "exclude_from_cashflow": False}
    assert await _read_flag(client, txn_id) is False

    missing = await client.post(
        "/api/transactions/999999/exclude",
        json={"exclude_from_cashflow": True},
    )
    assert missing.status_code == 404
