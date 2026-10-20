"""Receivables ledger API and page. All names and amounts are synthetic."""

import datetime as dt
from decimal import Decimal

import pytest

from financial_dashboard.db import Transaction

pytestmark = pytest.mark.anyio


async def _row(session, direction: str, amount: str) -> int:
    """Seed one INR transaction and return its id."""
    txn = Transaction(
        bank="testbank",
        email_type="testbank_alert",
        direction=direction,
        amount=Decimal(amount),
        currency="INR",
        transaction_date=dt.date(2030, 3, 4),
    )
    session.add(txn)
    await session.commit()
    return txn.id


async def test_payment_link_lowers_outstanding(client, session):
    charge = await _row(session, "debit", "1500.00")
    payment = await _row(session, "credit", "600.00")
    created = await client.post(
        "/api/receivables",
        json={
            "person": "Friend A",
            "amount": "1500.00",
            "charge_transaction_ids": [charge],
        },
    )
    assert created.status_code == 201
    rid = created.json()["id"]
    assert (created.json()["paid"], created.json()["outstanding"]) == (
        "0.00",
        "1500.00",
    )

    linked = await client.post(
        f"/api/receivables/{rid}/links",
        json={"transaction_id": payment, "role": "payment"},
    )
    reused = await client.post(
        "/api/receivables",
        json={"person": "B", "amount": "600", "payment_transaction_ids": [payment]},
    )
    unlinked = await client.delete(f"/api/receivables/{rid}/links/{payment}")

    assert linked.status_code == 201
    body = linked.json()
    assert (body["paid"], body["outstanding"]) == ("600.00", "900.00")
    assert (body["charge_transaction_ids"], body["payment_transaction_ids"]) == (
        [charge],
        [payment],
    )
    assert reused.status_code == 409
    assert unlinked.json()["outstanding"] == "1500.00"


@pytest.mark.parametrize(
    ("direction", "role"), [(None, "payment"), ("debit", "payment")]
)
async def test_link_rejects_unknown_or_wrong_direction_row(
    client, session, direction, role
):
    txn_id = await _row(session, direction, "100.00") if direction else 9999
    rid = (
        await client.post("/api/receivables", json={"person": "A", "amount": "100"})
    ).json()["id"]

    response = await client.post(
        f"/api/receivables/{rid}/links", json={"transaction_id": txn_id, "role": role}
    )
    created = await client.post(
        "/api/receivables",
        json={"person": "A", "amount": "100", "payment_transaction_ids": [txn_id]},
    )

    assert (response.status_code, created.status_code) == (422, 422)
    listed = (await client.get("/api/receivables")).json()["items"]
    assert [item["payment_transaction_ids"] for item in listed] == [[]]


async def test_status_change_stamps_and_clears_closed_at(client):
    rid = (
        await client.post("/api/receivables", json={"person": "A", "amount": "50"})
    ).json()["id"]

    closed = await client.patch(
        f"/api/receivables/{rid}", json={"status": "written_off"}
    )
    reopened = await client.patch(f"/api/receivables/{rid}", json={"status": "open"})

    assert closed.json()["status"] == "written_off"
    assert closed.json()["closed_at"] is not None
    assert reopened.json()["closed_at"] is None


async def test_list_and_page_filter_by_status(client):
    for person in ("Friend A", "Friend B"):
        await client.post("/api/receivables", json={"person": person, "amount": "75"})
    settled = (await client.get("/api/receivables")).json()["items"][1]["id"]
    await client.patch(f"/api/receivables/{settled}", json={"status": "settled"})

    open_rows = (await client.get("/api/receivables?status=open")).json()["items"]
    page = await client.get("/receivables")

    assert [row["person"] for row in open_rows] == ["Friend A"]
    assert "Friend A" in page.text
    assert "Friend B" not in page.text
