import datetime
from decimal import Decimal

import pytest
from sqlalchemy import event

from financial_dashboard.db import (
    Account,
    Card,
    Email,
    SmsMessage,
    StatementUpload,
    Transaction,
)

pytestmark = pytest.mark.anyio


async def _seed_transaction(session, *, amount: str = "42.15"):
    account = Account(
        bank="SyntheticBank",
        label="Synthetic checking",
        type="credit_card",
        account_number="123456789012",
    )
    session.add(account)
    await session.flush()
    card = Card(
        account_id=account.id,
        card_mask="4111111111119876",
        label="Synthetic card",
        is_primary=True,
        active=True,
    )
    email = Email(
        provider="synthetic",
        message_id="synthetic-message",
        status="parsed",
        received_at=datetime.datetime(2030, 1, 2, 10, 0),
    )
    sms = SmsMessage(
        bank="SyntheticBank",
        sender="SYNTH",
        body="Synthetic message body",
        received_at=datetime.datetime(2030, 1, 2, 10, 0, 5),
        status="enriched",
    )
    statement = StatementUpload(
        account_id=account.id,
        bank="SyntheticBank",
        filename="synthetic.pdf",
        file_path="/private/synthetic.pdf",
        status="parsed",
    )
    session.add_all([card, email, sms, statement])
    await session.flush()
    transaction = Transaction(
        account_id=account.id,
        card_id=card.id,
        email_id=email.id,
        sms_message_id=sms.id,
        statement_upload_id=statement.id,
        bank="SyntheticBank",
        email_type="synthetic_cc_payment_alert",
        direction="credit",
        amount=Decimal(amount),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        transaction_time=datetime.time(10, 0, 3),
        counterparty="Synthetic Merchant",
        card_mask="4111111111119876",
        account_mask="123456789012",
        reference_number="SYNTHETIC-REF-001",
        channel="online",
        balance=Decimal("900.00"),
        raw_description="Synthetic raw description",
        note="Synthetic note",
        source="sms+email",
        notified_channel="sms",
        category="synthetic_category",
        category_method="manual",
        category_confidence=0.95,
        category_model="synthetic-model",
        category_input_hash="synthetic-hash",
        category_vocab_version=2,
        categorized_at=datetime.datetime(2030, 1, 2, 11, 0),
        review_status="reviewed",
        review_reason="Synthetic review",
        notify_attempts=1,
    )
    session.add(transaction)
    await session.flush()
    sms.transaction_id = transaction.id
    await session.commit()
    return transaction, account, card, email, sms, statement


async def test_transaction_list_is_bounded_filtered_stable_and_redacted(
    client, session
):
    first, account, card, email, sms, statement = await _seed_transaction(session)
    second = Transaction(
        bank="OtherBank",
        email_type="other_alert",
        direction="debit",
        amount=Decimal("5.00"),
        transaction_date=datetime.date(2030, 1, 3),
    )
    session.add(second)
    await session.commit()

    response = await client.get(
        "/api/transactions",
        params={
            "account_id": account.id,
            "card_id": card.id,
            "email_id": email.id,
            "sms_message_id": sms.id,
            "statement_upload_id": statement.id,
            "date_from": "2030-01-01",
            "date_to": "2030-01-02",
            "direction": "credit",
            "amount": "42.15",
            "bank": "syntheticbank",
            "email_type": "synthetic_cc_payment_alert",
            "source": "sms+email",
            "category": "synthetic_category",
            "review_status": "reviewed",
            "reference_number": "SYNTHETIC-REF-001",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["returned_count"] == 1
    assert body["total_count"] == 1
    assert [item["id"] for item in body["items"]] == [first.id]
    item = body["items"][0]
    assert item["card_mask"] == "XXXX9876"
    assert item["account_mask"] == "XXXX9012"
    assert item["reference_number"] == "SYNTHETIC-REF-001"
    assert "4111111111119876" not in response.text
    assert "123456789012" not in response.text
    assert "Synthetic raw description" not in response.text
    assert "Synthetic note" not in response.text
    assert "/private/synthetic.pdf" not in response.text


async def test_transaction_list_orders_newest_first_and_paginates(client, session):
    rows = [
        Transaction(
            bank="synthetic",
            email_type="synthetic",
            direction="debit",
            amount=Decimal(str(index)),
        )
        for index in range(1, 4)
    ]
    session.add_all(rows)
    await session.commit()

    response = await client.get("/api/transactions", params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 3
    assert [item["id"] for item in body["items"]] == [rows[1].id, rows[0].id]


async def test_transaction_detail_returns_provenance_without_file_paths(
    client, session
):
    transaction, account, card, email, sms, statement = await _seed_transaction(session)

    response = await client.get(f"/api/transactions/{transaction.id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["account"] == {
        "id": account.id,
        "bank": "SyntheticBank",
        "label": "Synthetic checking",
        "type": "credit_card",
    }
    assert body["card"] == {
        "id": card.id,
        "label": "Synthetic card",
        "card_mask": "XXXX9876",
        "is_primary": True,
    }
    assert body["email"] == {
        "id": email.id,
        "status": "parsed",
        "timestamp": "2030-01-02T10:00:00",
    }
    assert body["sms"] == {
        "id": sms.id,
        "status": "enriched",
        "timestamp": "2030-01-02T10:00:05",
    }
    assert body["statement"] == {
        "id": statement.id,
        "kind": "cc",
        "status": "parsed",
        "account_id": account.id,
    }
    assert body["raw_description"] == "Synthetic raw description"
    assert body["note"] == "Synthetic note"
    assert body["category_model"] == "synthetic-model"
    assert body["may_affect_cc_payment_state"] is True
    assert "/private/synthetic.pdf" not in response.text


async def test_transaction_detail_bounds_large_text_fields(client, session):
    transaction, *_ = await _seed_transaction(session)
    transaction.raw_description = "R" * 50_001
    transaction.note = "N" * 50_001
    await session.commit()

    body = (await client.get(f"/api/transactions/{transaction.id}")).json()

    assert len(body["raw_description"]) == 50_000
    assert body["raw_description_truncated"] is True
    assert len(body["note"]) == 50_000
    assert body["note_truncated"] is True


async def test_transaction_batch_preserves_requested_order_and_missing_ids(
    client, session
):
    first = Transaction(
        bank="synthetic",
        email_type="synthetic",
        direction="debit",
        amount=Decimal("1.00"),
    )
    second = Transaction(
        bank="synthetic",
        email_type="synthetic",
        direction="debit",
        amount=Decimal("2.00"),
    )
    session.add_all([first, second])
    await session.commit()

    response = await client.post(
        "/api/transactions/batch",
        json={"ids": [second.id, 999999, first.id]},
    )

    assert response.status_code == 200, response.text
    assert [item["id"] for item in response.json()["items"]] == [second.id, first.id]
    assert response.json()["missing_ids"] == [999999]


async def test_transaction_reads_do_not_autoflush(client, session):
    pending = Transaction(
        bank="pending",
        email_type="pending",
        direction="debit",
        amount=Decimal("1.00"),
    )
    session.add(pending)
    statements: list[str] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        response = await client.get("/api/transactions")
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert response.status_code == 200
    assert pending.id is None
    assert not any(statement.startswith("insert") for statement in statements)


@pytest.mark.parametrize(
    ("path", "method", "payload"),
    [
        ("/api/transactions/0", "get", None),
        ("/api/transactions?limit=101", "get", None),
        (
            "/api/transactions?date_from=2030-01-03&date_to=2030-01-02",
            "get",
            None,
        ),
        ("/api/transactions/batch", "post", {"ids": []}),
        ("/api/transactions/batch", "post", {"ids": [1, 1]}),
    ],
)
async def test_transaction_reads_validate_bounds(client, path, method, payload):
    response = await client.request(method, path, json=payload)
    assert response.status_code == 422


async def test_transaction_detail_returns_404(client):
    response = await client.get("/api/transactions/999999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Transaction not found"}


async def test_transaction_read_openapi_is_typed(client):
    document = (await client.get("/openapi.json")).json()
    assert document["paths"]["/api/transactions"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/TransactionListResponse"}
    assert document["paths"]["/api/transactions/{txn_id}"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TransactionDetailResponse"
    }
    assert document["paths"]["/api/transactions/batch"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TransactionBatchResponse"
    }
