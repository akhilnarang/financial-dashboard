import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import event

from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    Email,
    StatementUpload,
    Transaction,
)

pytestmark = pytest.mark.anyio


async def _seed_statements(session):
    account = Account(
        bank="SyntheticBank",
        label="Synthetic account",
        type="credit_card",
        account_number="123456789012",
    )
    email = Email(
        provider="synthetic",
        message_id="synthetic-statement-message",
        status="parsed",
    )
    session.add_all([account, email])
    await session.flush()
    matched = Transaction(
        account_id=account.id,
        bank="SyntheticBank",
        email_type="synthetic_alert",
        direction="debit",
        amount=Decimal("10.00"),
    )
    session.add(matched)
    await session.flush()
    reconciliation = json.dumps(
        {
            "matched": [
                {
                    "db_txn_id": matched.id,
                    "narration": "Private matched narration",
                }
            ],
            "missing": [
                {
                    "imported": True,
                    "imported_txn_id": 999999,
                    "narration": "Private imported narration",
                },
                {
                    "ambiguous": True,
                    "import_error": "Synthetic import issue",
                },
            ],
        }
    )
    cc = StatementUpload(
        account_id=account.id,
        email_id=email.id,
        bank="SyntheticBank",
        filename="synthetic-cc.pdf",
        file_path="/private/synthetic-cc.pdf",
        source_kind="pdf",
        status="partial_import",
        card_number="4111111111119876",
        statement_name="Synthetic statement holder",
        due_date="2030-01-20",
        total_amount_due="100.00",
        minimum_amount_due="10.00",
        parsed_txn_count=3,
        matched_count=1,
        missing_count=1,
        imported_count=1,
        reconciliation_data=reconciliation,
        error="C" * 1_001,
        payment_status="partially_paid",
        payment_paid_at=datetime.datetime(2030, 1, 5),
        payment_paid_amount=Decimal("25.00"),
        payment_last_reminded_at=datetime.datetime(2030, 1, 4),
        created_at=datetime.datetime(2030, 1, 3),
    )
    bank = BankStatementUpload(
        account_id=account.id,
        email_id=email.id,
        bank="SyntheticBank",
        filename="synthetic-bank.pdf",
        file_path="/private/synthetic-bank.pdf",
        status="imported",
        account_number="987654321234",
        account_holder_name="Synthetic account holder",
        opening_balance="500.00",
        closing_balance="600.00",
        statement_period_start="2030-01-01",
        statement_period_end="2030-01-31",
        parsed_txn_count=2,
        matched_count=1,
        missing_count=0,
        imported_count=1,
        reconciliation_data=reconciliation,
        error="B" * 1_001,
        created_at=datetime.datetime(2030, 1, 3),
    )
    session.add_all([cc, bank])
    await session.flush()
    cc_imported = Transaction(
        account_id=account.id,
        statement_upload_id=cc.id,
        bank="SyntheticBank",
        email_type="cc_statement",
        direction="debit",
        amount=Decimal("20.00"),
    )
    bank_imported = Transaction(
        account_id=account.id,
        bank_statement_upload_id=bank.id,
        bank="SyntheticBank",
        email_type="bank_statement",
        direction="credit",
        amount=Decimal("30.00"),
    )
    session.add_all([cc_imported, bank_imported])
    await session.commit()
    return account, email, matched, cc, bank, cc_imported, bank_imported


async def test_cc_statement_list_filters_bounds_and_redacts(client, session):
    account, email, _, cc, _, _, _ = await _seed_statements(session)

    response = await client.get(
        "/api/statements/cc",
        params={
            "statement_id": cc.id,
            "account_id": account.id,
            "email_id": email.id,
            "bank": "syntheticbank",
            "status": "partial_import",
            "date_from": "2030-01-03",
            "date_to": "2030-01-03",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_count"] == 1
    item = body["items"][0]
    assert item["card_mask"] == "XXXX9876"
    assert item["error"] == "C" * 1_000
    assert item["error_truncated"] is True
    assert item["account"] == {
        "id": account.id,
        "bank": "SyntheticBank",
        "label": "Synthetic account",
        "type": "credit_card",
    }
    for excluded in (
        "/private/synthetic-cc.pdf",
        "4111111111119876",
        "Private matched narration",
        "Private imported narration",
    ):
        assert excluded not in response.text


async def test_statement_lists_sql_bound_oversized_identifiers(client, session):
    _, _, _, cc, bank, _, _ = await _seed_statements(session)
    cc.card_number = "9" * 100_000 + "1234"
    bank.account_number = "8" * 100_000 + "5678"
    await session.commit()

    cc_item = (await client.get("/api/statements/cc")).json()["items"][0]
    bank_item = (await client.get("/api/statements/bank")).json()["items"][0]

    assert cc_item["card_mask"] == "XXXX1234"
    assert bank_item["account_mask"] == "XXXX5678"


async def test_bank_statement_list_filters_bounds_and_redacts(client, session):
    account, email, _, _, bank, _, _ = await _seed_statements(session)

    response = await client.get(
        "/api/statements/bank",
        params={
            "statement_id": bank.id,
            "account_id": account.id,
            "email_id": email.id,
            "bank": "syntheticbank",
            "status": "imported",
        },
    )

    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["account_mask"] == "XXXX1234"
    assert item["opening_balance"] == "500.00"
    assert item["closing_balance"] == "600.00"
    assert item["error"] == "B" * 1_000
    assert "/private/synthetic-bank.pdf" not in response.text
    assert "987654321234" not in response.text
    assert "Private matched narration" not in response.text


async def test_cc_statement_detail_reports_reconciliation_and_payment(client, session):
    _, _, matched, cc, _, imported, _ = await _seed_statements(session)
    cc.error = "X" * 100_001
    await session.commit()

    response = await client.get(f"/api/statements/cc/{cc.id}")

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert len(body["error"]) == 100_000
    assert body["error_truncated"] is True
    assert body["payment_status"] == "partially_paid"
    assert body["payment_paid_amount"] == "25.00"
    assert body["reconciliation"] == {
        "status": "parsed",
        "matched_transaction_ids": [matched.id],
        "matched_transaction_ids_truncated": False,
        "imported_transaction_ids": [imported.id],
        "imported_transaction_ids_truncated": False,
        "ambiguous_entry_count": 1,
        "import_error_entry_count": 1,
    }
    assert "Private matched narration" not in response.text


async def test_bank_statement_detail_reports_reconciliation(client, session):
    _, _, matched, _, bank, _, imported = await _seed_statements(session)

    response = await client.get(f"/api/statements/bank/{bank.id}")

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["reconciliation"] == {
        "status": "parsed",
        "matched_transaction_ids": [matched.id],
        "matched_transaction_ids_truncated": False,
        "imported_transaction_ids": [imported.id],
        "imported_transaction_ids_truncated": False,
        "ambiguous_entry_count": 1,
        "import_error_entry_count": 1,
    }


@pytest.mark.parametrize(
    ("reconciliation_data", "expected_status"),
    [
        (None, "absent"),
        ("not-json", "malformed"),
        ("{}", "malformed"),
        ('{"matched": []}', "malformed"),
        ("X" * 1_000_001, "too_large"),
    ],
)
async def test_statement_detail_handles_unavailable_reconciliation(
    client, session, reconciliation_data, expected_status
):
    _, _, _, cc, _, _, _ = await _seed_statements(session)
    cc.reconciliation_data = reconciliation_data
    await session.commit()

    response = await client.get(f"/api/statements/cc/{cc.id}")

    assert response.status_code == 200
    reconciliation = response.json()["reconciliation"]
    assert reconciliation["status"] == expected_status
    assert reconciliation["matched_transaction_ids"] == []


async def test_statement_lists_are_stable_and_paginated(client, session):
    account, *_ = await _seed_statements(session)
    rows = [
        StatementUpload(
            account_id=account.id,
            bank="SyntheticBank",
            filename=f"synthetic-{index}.pdf",
            file_path=f"/private/{index}.pdf",
            status="parsed",
        )
        for index in range(3)
    ]
    session.add_all(rows)
    await session.commit()

    response = await client.get("/api/statements/cc", params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 4
    assert [item["id"] for item in body["items"]] == [rows[1].id, rows[0].id]


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("/api/statements/cc/batch", "cc"),
        ("/api/statements/bank/batch", "bank"),
    ],
)
async def test_statement_batch_preserves_order_and_missing(client, session, path, kind):
    _, _, _, cc, bank, _, _ = await _seed_statements(session)
    first_id = cc.id if kind == "cc" else bank.id
    model = StatementUpload if kind == "cc" else BankStatementUpload
    second = model(
        account_id=cc.account_id,
        bank="SyntheticBank",
        filename="second.pdf",
        file_path="/private/second.pdf",
        status="parsed",
    )
    session.add(second)
    await session.commit()

    response = await client.post(path, json={"ids": [second.id, 999999, first_id]})

    assert response.status_code == 200, response.text
    assert [item["id"] for item in response.json()["items"]] == [
        second.id,
        first_id,
    ]
    assert response.json()["missing_ids"] == [999999]
    assert "/private/" not in response.text


async def test_statement_reads_do_not_autoflush(client, session):
    pending = Account(
        bank="pending",
        label="pending",
        type="bank_account",
    )
    session.add(pending)
    statements: list[str] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        cc_response = await client.get("/api/statements/cc")
        bank_response = await client.get("/api/statements/bank")
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert cc_response.status_code == 200
    assert bank_response.status_code == 200
    assert pending.id is None
    assert not any(statement.startswith("insert") for statement in statements)


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/api/statements/cc/0", None),
        ("GET", "/api/statements/cc/9223372036854775808", None),
        ("GET", "/api/statements/cc?limit=101", None),
        (
            "GET",
            "/api/statements/cc?date_from=2030-01-03&date_to=2030-01-02",
            None,
        ),
        ("POST", "/api/statements/cc/batch", {"ids": [1, 1]}),
        ("GET", "/api/statements/bank/0", None),
        ("GET", "/api/statements/bank?limit=0", None),
        ("POST", "/api/statements/bank/batch", {"ids": []}),
    ],
)
async def test_statement_reads_validate_bounds(client, method, path, payload):
    response = await client.request(method, path, json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "path", ["/api/statements/cc/999999", "/api/statements/bank/999999"]
)
async def test_statement_details_return_404(client, path):
    response = await client.get(path)
    assert response.status_code == 404


async def test_statement_openapi_is_typed(client):
    document = (await client.get("/openapi.json")).json()
    expected = {
        ("/api/statements/cc", "get"): "CcStatementListResponse",
        ("/api/statements/cc/{statement_id}", "get"): "CcStatementDetailResponse",
        ("/api/statements/cc/batch", "post"): "CcStatementBatchResponse",
        ("/api/statements/bank", "get"): "BankStatementListResponse",
        (
            "/api/statements/bank/{statement_id}",
            "get",
        ): "BankStatementDetailResponse",
        ("/api/statements/bank/batch", "post"): "BankStatementBatchResponse",
    }
    for (path, method), schema_name in expected.items():
        schema = document["paths"][path][method]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert schema == {"$ref": f"#/components/schemas/{schema_name}"}
