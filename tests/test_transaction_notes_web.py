import json
from decimal import Decimal

import pytest

from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    StatementUpload,
    Transaction,
)

pytestmark = pytest.mark.anyio


async def test_note_api_clear_preserves_empty_response_and_stores_null(client, session):
    transaction = Transaction(
        bank="hdfc",
        email_type="test",
        direction="debit",
        amount=Decimal("10.00"),
        note="old note",
    )
    session.add(transaction)
    await session.flush()

    response = await client.post(
        f"/api/transactions/{transaction.id}/note", json={"note": "   "}
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "note": ""}
    await session.refresh(transaction)
    assert transaction.note is None


def _reconciliation(matched_id: int, imported_id: int) -> dict:
    return {
        "matched": [
            {
                "db_txn_id": matched_id,
                "date": "01/08/2026",
                "direction": "debit",
                "amount": "100.00",
                "narration": "Matched statement row",
                "db_counterparty": "Matched merchant",
                "db_reference": None,
                "enriched": False,
                "channel": "upi",
                "stmt_list": "transactions",
            }
        ],
        "missing": [
            {
                "imported": True,
                "imported_txn_id": imported_id,
                "date": "02/08/2026",
                "direction": "debit",
                "amount": "50.00",
                "narration": "Imported statement row",
                "channel": "card",
                "stmt_list": "transactions",
            },
            {
                "imported": False,
                "date": "03/08/2026",
                "direction": "debit",
                "amount": "25.00",
                "narration": "Unimported statement row",
                "stmt_list": "transactions",
            },
        ],
    }


async def _seed_transactions(
    session, account_id: int
) -> tuple[Transaction, Transaction]:
    matched = Transaction(
        account_id=account_id,
        bank="hdfc",
        email_type="test",
        direction="debit",
        amount=Decimal("100.00"),
        category="rent",
        note='<script>alert("matched")</script>',
    )
    imported = Transaction(
        account_id=account_id,
        bank="hdfc",
        email_type="test",
        direction="debit",
        amount=Decimal("50.00"),
        category="unknown",
        note="Imported note & receipt",
    )
    session.add_all([matched, imported])
    await session.flush()
    return matched, imported


async def test_transactions_page_escapes_note(client, session):
    transaction = Transaction(
        bank="hdfc",
        email_type="test",
        direction="debit",
        amount=Decimal("10.00"),
        category="rent",
        note='<script>alert("list")</script>',
    )
    session.add(transaction)
    await session.flush()

    response = await client.get("/transactions")

    assert response.status_code == 200
    assert "&lt;script&gt;alert" in response.text
    assert '<script>alert("list")</script>' not in response.text


async def test_statement_pages_show_current_category_and_escaped_note(client, session):
    upload_kinds = (
        ("credit_card", StatementUpload, "/statements/{}"),
        ("bank_account", BankStatementUpload, "/statements/bank/{}"),
    )
    for account_type, upload_model, path in upload_kinds:
        account = Account(bank="hdfc", label="Test", type=account_type)
        session.add(account)
        await session.flush()
        matched, imported = await _seed_transactions(session, account.id)
        upload = upload_model(
            account_id=account.id,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/tmp/statement.pdf",
            status="parsed",
            parsed_txn_count=3,
            matched_count=1,
            missing_count=2,
            imported_count=1,
            reconciliation_data=json.dumps(_reconciliation(matched.id, imported.id)),
        )
        session.add(upload)
        await session.flush()

        response = await client.get(path.format(upload.id))

        assert response.status_code == 200
        assert "Rent" in response.text
        assert "Uncategorized" in response.text
        assert "&lt;script&gt;alert" in response.text
        assert '<script>alert("matched")</script>' not in response.text
        assert "Imported note &amp; receipt" in response.text
