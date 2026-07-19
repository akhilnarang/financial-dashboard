import datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event

from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    CasUpload,
    Email,
    FetchRule,
    SmsMessage,
    StatementUpload,
    Transaction,
)

pytestmark = pytest.mark.anyio


async def _seed_sms(session):
    sms = SmsMessage(
        bank="SyntheticBank",
        sender="SYNTH",
        body="Synthetic source body",
        received_at=datetime.datetime(2030, 1, 2, 10, 0),
        status="parsed",
        parse_error="E" * 1_001,
        parsed_at=datetime.datetime(2030, 1, 2, 10, 1),
    )
    session.add(sms)
    await session.flush()
    transaction = Transaction(
        sms_message_id=sms.id,
        bank="SyntheticBank",
        email_type="synthetic_sms_alert",
        direction="debit",
        amount=Decimal("12.00"),
        source="sms",
    )
    session.add(transaction)
    await session.flush()
    sms.transaction_id = transaction.id
    await session.commit()
    return sms, transaction


async def _seed_email(session):
    account = Account(
        bank="SyntheticBank",
        label="Synthetic account",
        type="bank_account",
    )
    rule = FetchRule(
        provider="synthetic",
        sender="synthetic-sender",
        bank="SyntheticBank",
        email_kind="transaction",
        enabled=True,
    )
    session.add_all([account, rule])
    await session.flush()
    email = Email(
        provider="synthetic",
        message_id="synthetic-message-id",
        remote_id="synthetic-remote-id",
        sender="sender@example.invalid",
        subject="Synthetic transaction alert",
        received_at=datetime.datetime(2030, 1, 2, 10, 0),
        fetched_at=datetime.datetime(2030, 1, 2, 10, 1),
        status="parsed",
        error="D" * 1_001,
        rule_id=rule.id,
    )
    session.add(email)
    await session.flush()
    transaction = Transaction(
        account_id=account.id,
        email_id=email.id,
        bank="SyntheticBank",
        email_type="synthetic_email_alert",
        direction="credit",
        amount=Decimal("15.00"),
        source="email",
    )
    cc_statement = StatementUpload(
        account_id=account.id,
        email_id=email.id,
        bank="SyntheticBank",
        filename="synthetic.pdf",
        file_path="/private/synthetic.pdf",
        status="parsed",
    )
    bank_statement = BankStatementUpload(
        account_id=account.id,
        email_id=email.id,
        bank="SyntheticBank",
        filename="synthetic-bank.pdf",
        file_path="/private/synthetic-bank.pdf",
        status="parsed",
    )
    cas_upload = CasUpload(
        email_id=email.id,
        portfolio_key="synthetic-portfolio",
        depository_source="synthetic",
        investor_name="Synthetic Investor",
        statement_date=datetime.date(2030, 1, 2),
        grand_total=Decimal("100.00"),
        portfolio_ok=True,
        raw_holdings_json="{}",
    )
    session.add_all([transaction, cc_statement, bank_statement, cas_upload])
    await session.commit()
    return email, rule, transaction, cc_statement, bank_statement, cas_upload


async def test_sms_list_filters_links_and_excludes_raw_body(client, session):
    sms, transaction = await _seed_sms(session)

    response = await client.get(
        "/api/sms",
        params={
            "sms_id": sms.id,
            "bank": "syntheticbank",
            "status": "parsed",
            "transaction_id": transaction.id,
            "parser_type": "synthetic_sms_alert",
            "date_from": "2030-01-02",
            "date_to": "2030-01-02",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_count"] == 1
    item = body["items"][0]
    assert item["transaction"] == {
        "id": transaction.id,
        "email_type": "synthetic_sms_alert",
        "direction": "debit",
        "source": "sms",
    }
    assert item["parse_error"] == "E" * 1_000
    assert item["parse_error_truncated"] is True
    assert "Synthetic source body" not in response.text


async def test_sms_detail_explicitly_returns_bounded_raw_source(client, session):
    sms, transaction = await _seed_sms(session)
    sms.body = "B" * 100_001
    sms.parse_error = "P" * 100_001
    await session.commit()

    response = await client.get(f"/api/sms/{sms.id}")

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert len(body["body"]) == 100_000
    assert body["body_truncated"] is True
    assert len(body["parse_error"]) == 100_000
    assert body["parse_error_truncated"] is True
    assert body["attached_transaction_ids"] == [transaction.id]
    assert body["attached_transactions_truncated"] is False


async def test_sms_reads_bound_sender_and_bank_metadata(client, session):
    sms, _ = await _seed_sms(session)
    sms.bank = "B" * 1_001
    sms.sender = "S" * 1_001
    await session.commit()

    list_item = (await client.get("/api/sms")).json()["items"][0]
    detail = (await client.get(f"/api/sms/{sms.id}")).json()

    for item in (list_item, detail):
        assert item["bank"] == "B" * 1_000
        assert item["bank_truncated"] is True
        assert item["sender"] == "S" * 1_000
        assert item["sender_truncated"] is True


async def test_sms_batch_preserves_order_and_reports_missing(client, session):
    first, _ = await _seed_sms(session)
    second = SmsMessage(
        bank="SyntheticBank",
        sender="SYNTH2",
        body="Second synthetic body",
        received_at=datetime.datetime(2030, 1, 3, 10, 0),
        status="pending",
    )
    session.add(second)
    await session.commit()

    response = await client.post(
        "/api/sms/batch", json={"ids": [second.id, 999999, first.id]}
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == [
        second.id,
        first.id,
    ]
    assert response.json()["missing_ids"] == [999999]
    assert "Second synthetic body" not in response.text


async def test_email_list_filters_links_and_excludes_raw_identifiers(client, session):
    (
        email,
        rule,
        transaction,
        cc_statement,
        bank_statement,
        cas_upload,
    ) = await _seed_email(session)

    response = await client.get(
        "/api/emails",
        params={
            "email_id": email.id,
            "rule_id": rule.id,
            "provider": "synthetic",
            "status": "parsed",
            "bank": "syntheticbank",
            "email_kind": "transaction",
            "transaction_id": transaction.id,
            "parser_type": "synthetic_email_alert",
            "direction": "credit",
            "date_from": "2030-01-02",
            "date_to": "2030-01-02",
            "q": "transaction",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_count"] == 1
    item = body["items"][0]
    assert item["rule"] == {
        "id": rule.id,
        "bank": "SyntheticBank",
        "email_kind": "transaction",
    }
    assert item["transactions"] == [
        {
            "id": transaction.id,
            "email_type": "synthetic_email_alert",
            "direction": "credit",
            "source": "email",
        }
    ]
    assert item["transactions_truncated"] is False
    assert item["statements"] == [
        {"id": cc_statement.id, "kind": "cc", "status": "parsed"},
        {"id": bank_statement.id, "kind": "bank", "status": "parsed"},
        {"id": cas_upload.id, "kind": "cas", "status": "parsed"},
    ]
    assert item["error"] == "D" * 1_000
    assert item["error_truncated"] is True
    for excluded in (
        "synthetic-message-id",
        "synthetic-remote-id",
        "/private/synthetic.pdf",
        "/private/synthetic-bank.pdf",
    ):
        assert excluded not in response.text


async def test_email_filters_apply_to_one_linked_transaction(client, session):
    email, *_ = await _seed_email(session)
    session.add(
        Transaction(
            email_id=email.id,
            bank="SyntheticBank",
            email_type="synthetic_debit_type",
            direction="debit",
            amount=Decimal("1.00"),
            source="email",
        )
    )
    await session.commit()

    response = await client.get(
        "/api/emails",
        params={"parser_type": "synthetic_debit_type", "direction": "credit"},
    )

    assert response.status_code == 200
    assert response.json()["total_count"] == 0


async def test_email_detail_returns_full_metadata_with_bounded_error(client, session):
    email, *_ = await _seed_email(session)
    email.error = "X" * 100_001
    await session.commit()

    response = await client.get(f"/api/emails/{email.id}")

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["message_id"] == "synthetic-message-id"
    assert body["remote_id"] == "synthetic-remote-id"
    assert len(body["error"]) == 100_000
    assert body["error_truncated"] is True
    assert "/private/" not in response.text


async def test_email_raw_is_explicit_bounded_and_not_cached(
    client, session, monkeypatch
):
    email, *_ = await _seed_email(session)
    raw = b"Content-Type: text/plain; charset=utf-8\r\n\r\n" + ("R" * 100_001).encode()

    async def load_after_connection_release(_email):
        assert not session.in_transaction()
        return raw, None

    loader = AsyncMock(side_effect=load_after_connection_release)
    monkeypatch.setattr(
        "financial_dashboard.services.email_reads.load_or_fetch_raw_email", loader
    )

    response = await client.get(f"/api/emails/{email.id}/raw")

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["content_type"] == "text/plain"
    assert len(body["body"]) == 100_000
    assert body["body_truncated"] is True
    assert body["raw_byte_size"] == len(raw)
    loader.assert_awaited_once()


async def test_email_raw_handles_unknown_mime_charset(client, session, monkeypatch):
    email, *_ = await _seed_email(session)
    raw = b"Content-Type: text/plain; charset=x-does-not-exist\r\n\r\nbody"
    monkeypatch.setattr(
        "financial_dashboard.services.email_reads.load_or_fetch_raw_email",
        AsyncMock(return_value=(raw, None)),
    )

    response = await client.get(f"/api/emails/{email.id}/raw")

    assert response.status_code == 422
    assert response.json() == {"detail": "Raw email has no readable body"}


async def test_email_raw_sanitizes_provider_failures(
    client, session, monkeypatch, caplog
):
    email, *_ = await _seed_email(session)
    monkeypatch.setattr(
        "financial_dashboard.services.email_reads.load_or_fetch_raw_email",
        AsyncMock(return_value=(None, "credential failure containing secret")),
    )

    response = await client.get(f"/api/emails/{email.id}/raw")

    assert response.status_code == 424
    assert response.json() == {"detail": "Raw email source is unavailable"}
    assert "secret" not in response.text
    assert "secret" not in caplog.text
    assert "loader-failed" in caplog.text


async def test_email_raw_sanitizes_loader_exceptions(
    client, session, monkeypatch, caplog
):
    email, *_ = await _seed_email(session)
    monkeypatch.setattr(
        "financial_dashboard.services.email_reads.load_or_fetch_raw_email",
        AsyncMock(side_effect=OSError("private spool path and secret")),
    )

    response = await client.get(f"/api/emails/{email.id}/raw")

    assert response.status_code == 424
    assert response.json() == {"detail": "Raw email source is unavailable"}
    assert "private spool path" not in caplog.text
    assert "secret" not in caplog.text
    assert "loader-exception" in caplog.text


async def test_email_batch_preserves_order_and_reports_missing(client, session):
    first, *_ = await _seed_email(session)
    second = Email(
        provider="synthetic",
        message_id="second-synthetic-message",
        status="pending",
    )
    session.add(second)
    await session.commit()

    response = await client.post(
        "/api/emails/batch", json={"ids": [second.id, 999999, first.id]}
    )

    assert response.status_code == 200, response.text
    assert [item["id"] for item in response.json()["items"]] == [
        second.id,
        first.id,
    ]
    assert response.json()["missing_ids"] == [999999]


async def test_source_lists_do_not_autoflush(client, session):
    pending_sms = SmsMessage(
        bank="pending",
        sender="pending",
        body="pending",
        received_at=datetime.datetime(2030, 1, 1),
        status="pending",
    )
    pending_email = Email(
        provider="pending",
        message_id="pending-message",
        status="pending",
    )
    session.add_all([pending_sms, pending_email])
    statements: list[str] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        sms_response = await client.get("/api/sms")
        email_response = await client.get("/api/emails")
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert sms_response.status_code == 200
    assert email_response.status_code == 200
    assert pending_sms.id is None
    assert pending_email.id is None
    assert not any(statement.startswith("insert") for statement in statements)


@pytest.mark.parametrize(
    ("method", "path", "json"),
    [
        ("GET", "/api/sms/0", None),
        ("GET", "/api/sms?limit=101", None),
        ("GET", "/api/sms?date_from=2030-01-03&date_to=2030-01-02", None),
        ("POST", "/api/sms/batch", {"ids": [1, 1]}),
        ("GET", "/api/emails/0", None),
        ("GET", "/api/emails?limit=101", None),
        ("GET", "/api/emails?date_from=2030-01-03&date_to=2030-01-02", None),
        ("POST", "/api/emails/batch", {"ids": []}),
    ],
)
async def test_source_reads_validate_bounds(client, method, path, json):
    response = await client.request(method, path, json=json)
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/api/sms/999999", "/api/emails/999999"])
async def test_source_detail_returns_404(client, path):
    response = await client.get(path)
    assert response.status_code == 404


async def test_source_read_openapi_is_typed(client):
    document = (await client.get("/openapi.json")).json()
    expected = {
        ("/api/sms", "get"): "SmsListResponse",
        ("/api/sms/{sms_id}", "get"): "SmsDetailResponse",
        ("/api/sms/batch", "post"): "SmsBatchResponse",
        ("/api/emails", "get"): "EmailListResponse",
        ("/api/emails/{email_id}", "get"): "EmailDetailResponse",
        ("/api/emails/{email_id}/raw", "get"): "EmailRawResponse",
        ("/api/emails/batch", "post"): "EmailBatchResponse",
    }
    for (path, method), schema_name in expected.items():
        schema = document["paths"][path][method]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert schema == {"$ref": f"#/components/schemas/{schema_name}"}
