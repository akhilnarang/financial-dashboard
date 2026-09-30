import datetime
import json
from decimal import Decimal

import pytest

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


async def test_statement_lists_filter_redact_and_paginate(client, session):
    account, email, _, cc, bank, _, _ = await _seed_statements(session)

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
    assert item["account"]["id"] == account.id
    for excluded in (
        "/private/synthetic-cc.pdf",
        "4111111111119876",
        "Private matched narration",
        "Private imported narration",
    ):
        assert excluded not in response.text

    response = await client.get(
        "/api/statements/bank",
        params={"statement_id": bank.id, "bank": "syntheticbank", "status": "imported"},
    )

    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["account_mask"] == "XXXX1234"
    assert item["opening_balance"] == "500.00"
    assert item["closing_balance"] == "600.00"
    for excluded in ("/private/synthetic-bank.pdf", "987654321234"):
        assert excluded not in response.text

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


async def test_statement_details_report_reconciliation_and_payment(client, session):
    _, _, matched, cc, bank, imported, bank_imported = await _seed_statements(session)
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

    bank_response = await client.get(f"/api/statements/bank/{bank.id}")

    assert bank_response.status_code == 200, bank_response.text
    bank_reconciliation = bank_response.json()["reconciliation"]
    assert bank_reconciliation["matched_transaction_ids"] == [matched.id]
    assert bank_reconciliation["imported_transaction_ids"] == [bank_imported.id]

    for reconciliation_data, expected_status in (
        ("not-json", "malformed"),
        ("X" * 1_000_001, "too_large"),
    ):
        cc.reconciliation_data = reconciliation_data
        await session.commit()
        response = await client.get(f"/api/statements/cc/{cc.id}")
        assert response.status_code == 200
        reconciliation = response.json()["reconciliation"]
        assert reconciliation["status"] == expected_status
        assert reconciliation["matched_transaction_ids"] == []

    assert (await client.get("/api/statements/cc/999999")).status_code == 404


async def test_statement_batch_preserves_order_and_missing(client, session):
    _, _, _, cc, _, _, _ = await _seed_statements(session)
    second = StatementUpload(
        account_id=cc.account_id,
        bank="SyntheticBank",
        filename="second.pdf",
        file_path="/private/second.pdf",
        status="parsed",
    )
    session.add(second)
    await session.commit()

    response = await client.post(
        "/api/statements/cc/batch", json={"ids": [second.id, 999999, cc.id]}
    )

    assert response.status_code == 200, response.text
    assert [item["id"] for item in response.json()["items"]] == [second.id, cc.id]
    assert response.json()["missing_ids"] == [999999]
    assert "/private/" not in response.text


async def test_statement_list_rejects_inverted_date_range(client):
    response = await client.get(
        "/api/statements/cc",
        params={"date_from": "2030-01-03", "date_to": "2030-01-02"},
    )
    assert response.status_code == 422
