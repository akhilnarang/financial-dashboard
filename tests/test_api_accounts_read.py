import datetime
from decimal import Decimal

import pytest
from sqlalchemy import event

from financial_dashboard.db import (
    Account,
    BalanceSnapshot,
    BankStatementUpload,
    Card,
    StatementUpload,
    Transaction,
)

pytestmark = pytest.mark.anyio


async def _account(
    session,
    *,
    bank: str = "synthetic-bank",
    label: str = "Synthetic account",
    account_type: str = "bank_account",
    account_number: str | None = "000000001234",
    active: bool = True,
) -> Account:
    row = Account(
        bank=bank,
        label=label,
        type=account_type,
        account_number=account_number,
        active=active,
    )
    session.add(row)
    await session.flush()
    return row


async def test_account_list_is_stable_bounded_filterable_and_redacted(client, session):
    first = await _account(
        session,
        bank="ExampleBank",
        label="First account",
        account_number="123456789012",
    )
    second = await _account(
        session,
        bank="ExampleBank",
        label="Second account",
        account_type="credit_card",
        account_number="5555444433332222",
    )
    await _account(
        session,
        bank="OtherBank",
        label="Inactive account",
        active=False,
    )
    session.add_all(
        [
            Card(
                account_id=second.id,
                card_mask="4111111111119876",
                label="Primary",
                is_primary=True,
                active=True,
            ),
            Card(
                account_id=second.id,
                card_mask="5500XXXXXXXX4321",
                label="Add-on",
                is_primary=False,
                active=True,
            ),
        ]
    )
    await session.commit()

    response = await client.get(
        "/api/accounts",
        params={
            "bank": "examplebank",
            "active": "true",
            "limit": 1,
            "offset": 1,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["returned_count"] == 1
    assert body["total_count"] == 2
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert [item["id"] for item in body["items"]] == [second.id]
    item = body["items"][0]
    assert item["account_mask"] == "XXXX2222"
    assert [card["label"] for card in item["cards"]] == ["Primary", "Add-on"]
    assert [card["card_mask"] for card in item["cards"]] == [
        "XXXX9876",
        "XXXX4321",
    ]
    assert item["cards_truncated"] is False
    assert str(first.account_number) not in response.text
    assert str(second.account_number) not in response.text
    assert "4111111111119876" not in response.text


async def test_account_list_caps_cards_per_account(client, session):
    account = await _account(session)
    session.add_all(
        [
            Card(
                account_id=account.id,
                card_mask=f"40000000{i:04d}",
                label=f"Card {i}",
                is_primary=i == 0,
                active=True,
            )
            for i in range(51)
        ]
    )
    await session.commit()

    body = (await client.get("/api/accounts")).json()

    assert len(body["items"][0]["cards"]) == 50
    assert body["items"][0]["cards_truncated"] is True
    assert body["items"][0]["cards"][0]["label"] == "Card 0"


async def test_account_detail_has_counts_latest_balances_and_no_secrets(
    client, session
):
    account = await _account(
        session,
        account_number="987654321234",
    )
    account.statement_password = "encrypted-secret-value"
    account.statement_password_hint = "private-hint"
    session.add(
        Card(
            account_id=account.id,
            card_mask="5100000000007788",
            label="Synthetic card",
            is_primary=True,
            active=True,
        )
    )
    session.add_all(
        [
            Transaction(
                account_id=account.id,
                bank="synthetic-bank",
                email_type="synthetic-alert",
                direction="debit",
                amount=Decimal("10.00"),
            ),
            StatementUpload(
                account_id=account.id,
                bank="synthetic-bank",
                filename="synthetic.pdf",
                file_path="/private/statement.pdf",
            ),
            BankStatementUpload(
                account_id=account.id,
                bank="synthetic-bank",
                filename="synthetic-bank.pdf",
                file_path="/private/bank-statement.pdf",
            ),
            BalanceSnapshot(
                account_id=account.id,
                kind="asset",
                category="bank_balance",
                as_of_date=datetime.date(2030, 1, 1),
                value=Decimal("100.00"),
                source="synthetic",
            ),
            BalanceSnapshot(
                account_id=account.id,
                kind="asset",
                category="bank_balance",
                as_of_date=datetime.date(2030, 1, 2),
                value=Decimal("125.00"),
                source="synthetic",
            ),
            BalanceSnapshot(
                account_id=account.id,
                kind="liability",
                category="other_balance",
                as_of_date=datetime.date(2030, 1, 2),
                value=Decimal("5.00"),
                source="synthetic",
            ),
        ]
    )
    await session.commit()

    response = await client.get(f"/api/accounts/{account.id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["transaction_count"] == 1
    assert body["cc_statement_count"] == 1
    assert body["bank_statement_count"] == 1
    assert body["account_mask"] == "XXXX1234"
    assert body["cards"][0]["card_mask"] == "XXXX7788"
    assert body["latest_balance_snapshots"] == [
        {
            "snapshot_id": body["latest_balance_snapshots"][0]["snapshot_id"],
            "category": "bank_balance",
            "as_of_date": "2030-01-02",
            "value": "125.00",
            "currency": "INR",
        },
        {
            "snapshot_id": body["latest_balance_snapshots"][1]["snapshot_id"],
            "category": "other_balance",
            "as_of_date": "2030-01-02",
            "value": "5.00",
            "currency": "INR",
        },
    ]
    for secret in (
        "encrypted-secret-value",
        "private-hint",
        "987654321234",
        "/private/statement.pdf",
        "/private/bank-statement.pdf",
    ):
        assert secret not in response.text


async def test_account_reads_do_not_autoflush_pending_rows(client, session):
    pending = Account(
        bank="pending-bank",
        label="Must stay pending",
        type="bank_account",
        account_number="123456789999",
    )
    session.add(pending)
    statements: list[str] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        response = await client.get("/api/accounts")
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert response.status_code == 200
    assert pending.id is None
    assert not any(statement.startswith("insert") for statement in statements)


async def test_account_detail_returns_404(client):
    response = await client.get("/api/accounts/999999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Account not found"}


@pytest.mark.parametrize("account_id", ["0", "9223372036854775808"])
async def test_account_detail_validates_database_id_range(client, account_id):
    response = await client.get(f"/api/accounts/{account_id}")
    assert response.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"offset": 1_000_001},
        {"bank": ""},
        {"account_type": "x" * 65},
    ],
)
async def test_account_list_validates_bounds(client, params):
    response = await client.get("/api/accounts", params=params)
    assert response.status_code == 422


async def test_account_read_openapi_is_typed(client):
    document = (await client.get("/openapi.json")).json()
    assert document["paths"]["/api/accounts"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/AccountListResponse"}
    assert document["paths"]["/api/accounts/{account_id}"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccountDetailResponse"
    }
