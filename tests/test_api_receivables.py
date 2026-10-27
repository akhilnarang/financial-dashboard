"""Receivables ledger API and page. All names and amounts are synthetic."""

import datetime as dt
from decimal import Decimal

import pytest
from httpx import Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Transaction

pytestmark = pytest.mark.anyio


async def _row(
    session: AsyncSession,
    direction: str,
    amount: str,
    currency: str = "INR",
    **fields: str | bool,
) -> int:
    """Seed one transaction and return its id."""
    txn = Transaction(
        bank="testbank",
        email_type="testbank_alert",
        direction=direction,
        amount=Decimal(amount),
        currency=currency,
        transaction_date=dt.date(2030, 3, 4),
        **fields,
    )
    session.add(txn)
    await session.commit()
    return txn.id


async def test_payment_link_lowers_outstanding(client, session):
    charge = await _row(session, "debit", "1500.00", counterparty="CAFE MERCHANT")
    payment = await _row(
        session,
        "credit",
        "600.00",
        counterparty="FRIEND A",
        raw_description=(
            "UPI/CR/friend.a@okbank/other.b@okbank/9876500001/9876500003/"
            "alerts@bank.example"
        ),
    )
    created = await client.post(
        "/api/receivables",
        json={"person": "Friend A", "amount": "1500.00"},
    )
    assert created.status_code == 201
    rid = created.json()["id"]
    charged = await client.post(
        f"/api/receivables/{rid}/links",
        json={"transaction_id": charge, "role": "charge"},
    )
    assert charged.status_code == 201
    assert (created.json()["paid"], created.json()["outstanding"]) == (
        "0.00",
        "1500.00",
    )

    linked = await client.post(
        f"/api/receivables/{rid}/links",
        json={"transaction_id": payment, "role": "payment"},
    )
    relinked = await client.post(
        f"/api/receivables/{rid}/links",
        json={"transaction_id": charge, "role": "charge"},
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
    assert (relinked.status_code, reused.status_code) == (409, 409)
    assert unlinked.json()["outstanding"] == "1500.00"
    # A payment link teaches the identifiers of its row. A charge does not.
    # Two VPAs or two phones on one row are not taught: one is not the person.
    learned = (await client.get("/api/receivables/identifiers")).json()["items"]
    assert [(i["person"], i["kind"], i["value"]) for i in learned] == [
        ("friend a", "counterparty", "frienda"),
    ]

    # A late unlink must not delete another receivable's link that took the id.
    await client.post(
        f"/api/receivables/{rid}/links",
        json={"transaction_id": payment, "role": "payment"},
    )
    other = await _row(session, "credit", "300.00")
    rid_b = (
        await client.post("/api/receivables", json={"person": "B", "amount": "300"})
    ).json()["id"]
    await session.execute(
        text(
            "UPDATE receivable_links SET receivable_id = :b, transaction_id = :t "
            "WHERE receivable_id = :a AND transaction_id = :p"
        ),
        {"a": rid, "p": payment, "b": rid_b, "t": other},
    )
    await session.commit()

    late = await client.delete(f"/api/receivables/{rid}/links/{payment}")
    session.expire_all()
    listed = (await client.get("/api/receivables")).json()["items"]

    assert late.status_code == 404
    assert [(item["id"], item["paid"]) for item in listed] == [
        (rid, "0.00"),
        (rid_b, "300.00"),
    ]


# A debit cannot pay back a receivable, a credit is not its charge, a payable
# flips both, the ledger is INR only, and a voided row is no money.
@pytest.mark.parametrize(
    ("kind", "row", "role"),
    [
        ("receivable", ("debit", "INR"), "payment"),
        ("receivable", ("credit", "INR"), "charge"),
        ("payable", ("credit", "INR"), "payment"),
        ("receivable", ("credit", "USD"), "payment"),
        ("receivable", 9999, "payment"),
        ("receivable", ("credit", "INR", True), "payment"),
    ],
)
async def test_link_rejects_unknown_or_unfit_row(client, session, kind, row, role):
    txn_id = (
        row
        if isinstance(row, int)
        else await _row(session, row[0], "100", row[1], voided=len(row) > 2)
    )
    rid = (
        await client.post(
            "/api/receivables", json={"kind": kind, "person": "A", "amount": "100"}
        )
    ).json()["id"]

    response = await client.post(
        f"/api/receivables/{rid}/links",
        json={"transaction_id": txn_id, "role": role},
    )
    created = await client.post(
        "/api/receivables",
        json={
            "kind": kind,
            "person": "A",
            "amount": "100",
            f"{role}_transaction_ids": [txn_id],
        },
    )

    assert (response.status_code, created.status_code) == (422, 422)
    listed = (await client.get("/api/receivables")).json()["items"]
    assert [item[f"{role}_transaction_ids"] for item in listed] == [[]]


async def test_patch_changes_only_named_fields(client):
    rid, other = [
        (
            await client.post(
                "/api/receivables",
                json={"person": "A", "amount": "50", "note": "lunch"},
            )
        ).json()["id"]
        for _ in range(2)
    ]
    await client.post("/api/receivables", json={"person": "F", "amount": "10"})
    for person, value in (("A", "AAAA"), ("E", "EEEE")):
        await client.post(
            "/api/receivables/identifiers",
            json={"person": person, "kind": "counterparty", "value": value},
        )

    def pairs(response: Response) -> list[tuple[str, str]]:
        return [(i["person"], i["value"]) for i in response.json()["items"]]

    closed = await client.patch(
        f"/api/receivables/{rid}", json={"status": "written_off"}
    )
    reopened = await client.patch(
        f"/api/receivables/{rid}", json={"status": "open", "note": None}
    )
    nulled = await client.patch(f"/api/receivables/{rid}", json={"amount": None})
    rekinded = await client.patch(f"/api/receivables/{rid}", json={"kind": "payable"})
    renamed = await client.patch(f"/api/receivables/{rid}", json={"person": " B  C "})
    unnamed = await client.patch(f"/api/receivables/{rid}", json={"person": None})
    # Another receivable still uses the old name, so the rename copies.
    copied = pairs(await client.get("/api/receivables/identifiers"))
    await client.patch(f"/api/receivables/{other}", json={"person": "D"})
    moved = pairs(await client.get("/api/receivables/identifiers"))
    # A name with identifiers, or a name another receivable uses, is another
    # person. It gets nothing.
    await client.patch(f"/api/receivables/{rid}", json={"person": "E"})
    await client.patch(f"/api/receivables/{other}", json={"person": "F"})
    kept = pairs(await client.get("/api/receivables/identifiers"))

    assert closed.json()["status"] == "written_off"
    assert closed.json()["closed_at"] is not None
    assert closed.json()["note"] == "lunch"
    assert reopened.json()["closed_at"] is None
    assert reopened.json()["note"] is None
    assert (renamed.json()["person"], renamed.json()["amount"]) == ("B  C", "50.00")
    assert copied == [("a", "aaaa"), ("b c", "aaaa"), ("e", "eeee")]
    assert moved == [("b c", "aaaa"), ("d", "aaaa"), ("e", "eeee")]
    assert kept == moved
    assert (nulled.status_code, rekinded.status_code, unnamed.status_code) == (
        422,
        422,
        422,
    )


async def test_identifier_is_stored_normalized_once(client):
    bodies = [
        {"person": "Friend  A", "kind": "counterparty", "value": "Friend A Joint"},
        {"person": "friend a", "kind": "counterparty", "value": "FRIENDAJOINT"},
        {"person": "Friend A", "kind": "phone", "value": "+91 98765 00001"},
        {"person": "Friend A", "kind": "vpa", "value": "Friend.A@OKBANK"},
        {"person": "Friend A", "kind": "vpa", "value": "friend.a@bank.example"},
        {"person": "Friend A", "kind": "phone", "value": "12345"},
        {"person": "Friend A", "kind": "counterparty", "value": "A B"},
    ]

    codes = [
        (await client.post("/api/receivables/identifiers", json=body)).status_code
        for body in bodies
    ]
    listed = (await client.get("/api/receivables/identifiers")).json()["items"]

    assert codes == [201, 201, 201, 201, 422, 422, 422]
    assert [(i["person"], i["kind"], i["value"]) for i in listed] == [
        ("friend a", "counterparty", "friendajoint"),
        ("friend a", "phone", "9876500001"),
        ("friend a", "vpa", "friend.a@okbank"),
    ]


async def test_list_and_page_filter_by_status(client, session):
    for person in ("Friend A", "Friend B"):
        await client.post("/api/receivables", json={"person": person, "amount": "75"})
    settled = (await client.get("/api/receivables")).json()["items"][1]["id"]
    await client.patch(f"/api/receivables/{settled}", json={"status": "settled"})
    owed = await client.post(
        "/api/receivables",
        json={
            "kind": "payable",
            "person": "Friend C",
            "amount": "300",
            "charge_transaction_ids": [await _row(session, "credit", "300.00")],
        },
    )
    paid = await client.post(
        f"/api/receivables/{owed.json()['id']}/links",
        json={
            "transaction_id": await _row(
                session, "debit", "120.00", counterparty="NEFT TRANSFER"
            ),
            "role": "payment",
        },
    )

    open_rows = (await client.get("/api/receivables?status=open")).json()["items"]
    page = (await client.get("/receivables")).text

    assert (owed.status_code, paid.status_code) == (201, 201)
    assert [(row["person"], row["kind"], row["outstanding"]) for row in open_rows] == [
        ("Friend A", "receivable", "75.00"),
        ("Friend C", "payable", "180.00"),
    ]
    assert page.index("Owed to me") < page.index("Friend A") < page.index("I owe")
    assert page.index("I owe") < page.index("Friend C")
    assert "Friend B" not in page
    # A generic counterparty names no person, so the payment teaches nothing.
    assert (await client.get("/api/receivables/identifiers")).json()["items"] == []

    # Paid in full: a lower amount settles it, and it leaves the open list.
    lowered = await client.patch(
        f"/api/receivables/{owed.json()['id']}", json={"amount": "120"}
    )
    assert (lowered.json()["status"], lowered.json()["outstanding"]) == (
        "settled",
        "0.00",
    )
    assert lowered.json()["closed_at"] is not None
    still_open = (await client.get("/api/receivables?status=open")).json()["items"]
    assert [row["person"] for row in still_open] == ["Friend A"]
    # A named status wins, and an edit that does not change the amount keeps it.
    for body in (
        {"status": "open"},
        {"note": "keep open"},
        {"amount": "120", "status": "open"},
        {"amount": "120"},
    ):
        kept = await client.patch(f"/api/receivables/{owed.json()['id']}", json=body)
        assert kept.json()["status"] == "open", body
    charged = await client.post(
        f"/api/receivables/{owed.json()['id']}/links",
        json={
            "transaction_id": await _row(session, "credit", "10.00"),
            "role": "charge",
        },
    )
    assert (charged.status_code, charged.json()["status"]) == (201, "open")
