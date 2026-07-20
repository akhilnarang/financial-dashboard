import datetime
from decimal import Decimal

import pytest
from bank_email_parser.models import (
    Money,
    ParsedEmail,
    StatementSummary,
    TransactionAlert,
)
from sqlalchemy import event

from financial_dashboard.db import Email, FetchRule, Transaction
from financial_dashboard.integrations.email.body import RawEmailResult
from financial_dashboard.services.emails import ProcessedEmailParse

pytestmark = pytest.mark.anyio


def _transaction_parse(
    *, direction: str = "debit", reference_number: str | None = None
) -> ProcessedEmailParse:
    parsed = ParsedEmail(
        bank="synthetic-bank",
        email_type="synthetic_transaction_alert",
        transaction=TransactionAlert(
            direction=direction,
            amount=Money(amount=Decimal("12.34"), currency="INR"),
            transaction_date=datetime.date(2030, 1, 2),
            transaction_time=datetime.time(10, 30),
            counterparty="Synthetic Merchant",
            card_mask="4111111111111234",
            reference_number=reference_number,
            channel="card",
        ),
    )
    return ProcessedEmailParse(
        None,
        {
            "bank": parsed.bank,
            "email_type": parsed.email_type,
            "direction": direction,
            "amount": Decimal("12.34"),
            "currency": "INR",
            "transaction_date": datetime.date(2030, 1, 2),
            "transaction_time": datetime.time(10, 30),
            "counterparty": "Synthetic Merchant",
            "card_mask": "4111111111111234",
            "account_mask": None,
            "reference_number": reference_number,
            "channel": "card",
            "balance": None,
            "raw_description": "Synthetic parser context",
        },
        None,
        parsed,
    )


async def _email(
    session,
    *,
    status="failed",
    error="Old parser error",
    email_kind="transaction",
) -> Email:
    rule = FetchRule(
        provider="synthetic",
        sender="synthetic@example.invalid",
        bank="synthetic-bank",
        email_kind=email_kind,
    )
    session.add(rule)
    await session.flush()
    row = Email(
        provider="synthetic",
        message_id=f"synthetic-{rule.id}@example.invalid",
        sender="synthetic@example.invalid",
        subject="Synthetic source",
        received_at=datetime.datetime(2030, 1, 2, 5, 0),
        status=status,
        error=error,
        rule_id=rule.id,
    )
    session.add(row)
    await session.flush()
    return row


def _patch_raw_and_parse(monkeypatch, result):
    async def raw(_email):
        return RawEmailResult(b"Synthetic email", None, "provider")

    monkeypatch.setattr(
        "financial_dashboard.services.parse_previews.load_or_fetch_raw_email",
        raw,
    )
    monkeypatch.setattr(
        "financial_dashboard.services.parse_previews._process_email_full",
        lambda *_args, **_kwargs: result,
    )


async def test_email_parse_preview_projects_insert_without_writes(
    client, session, monkeypatch
):
    email = await _email(session)
    await session.commit()
    _patch_raw_and_parse(monkeypatch, _transaction_parse())
    statements: list[str] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        response = await client.post(f"/api/emails/{email.id}/parse-preview")
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["raw_provenance"] == "provider"
    assert body["parser"]["disposition"] == "transaction"
    assert body["parser"]["transaction"]["card_mask"] == "XXXX1234"
    assert "raw_description" not in body["parser"]["transaction"]
    assert body["merge"]["action"] == "insert"
    assert not any(
        statement.startswith(("insert", "update", "delete")) for statement in statements
    )


async def test_email_parse_preview_projects_linked_refresh(
    client, session, monkeypatch
):
    email = await _email(session)
    transaction = Transaction(
        email_id=email.id,
        bank="synthetic-bank",
        email_type="synthetic_transaction_alert",
        direction="debit",
        amount=Decimal("12.34"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        transaction_time=datetime.time(10, 30),
        counterparty="Old synthetic merchant",
    )
    session.add(transaction)
    await session.commit()
    _patch_raw_and_parse(monkeypatch, _transaction_parse())

    response = await client.post(f"/api/emails/{email.id}/parse-preview")

    assert response.status_code == 200, response.text
    merge = response.json()["merge"]
    assert merge["action"] == "refresh_linked"
    assert merge["target_transaction_id"] == transaction.id
    assert merge["identity_conflicts"] == []
    assert merge["linked_attribution_refresh"] is True
    assert set(merge["changed_fields"]) == {
        "counterparty",
        "card_mask",
        "channel",
        "raw_description",
    }


async def test_email_parse_preview_reports_claimed_reference_conflict(
    client, session, monkeypatch
):
    incoming = await _email(session)
    claimed_source = await _email(session)
    transaction = Transaction(
        email_id=claimed_source.id,
        bank="synthetic-bank",
        email_type="synthetic_transaction_alert",
        direction="debit",
        amount=Decimal("12.34"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        transaction_time=datetime.time(10, 30),
        reference_number="SYNTHETIC-REF",
    )
    session.add(transaction)
    await session.commit()
    _patch_raw_and_parse(
        monkeypatch, _transaction_parse(reference_number="SYNTHETIC-REF")
    )

    response = await client.post(f"/api/emails/{incoming.id}/parse-preview")

    assert response.status_code == 200
    assert response.json()["merge"]["action"] == "conflict"
    assert response.json()["merge"]["target_transaction_id"] == transaction.id
    assert response.json()["merge"]["match_kind"] == "claimed_reference"


async def test_email_parse_preview_preserves_existing_duplicate_defer(
    client, session, monkeypatch
):
    email = await _email(
        session,
        status="skipped",
        error="[dup-defer] synthetic deferred source",
    )
    await session.commit()
    _patch_raw_and_parse(monkeypatch, _transaction_parse())

    response = await client.post(f"/api/emails/{email.id}/parse-preview")

    assert response.status_code == 200
    assert response.json()["merge"] == {
        "action": "defer",
        "target_transaction_id": None,
        "match_kind": "existing_dup_defer",
        "changed_fields": [],
        "identity_conflicts": [],
        "linked_attribution_refresh": False,
        "match_evidence": None,
    }


async def test_email_parse_preview_suppresses_merge_for_statement_rule(
    client, session, monkeypatch
):
    email = await _email(session, email_kind="cc_statement")
    await session.commit()
    _patch_raw_and_parse(monkeypatch, _transaction_parse())

    response = await client.post(f"/api/emails/{email.id}/parse-preview")

    assert response.status_code == 200
    assert response.json()["routing"] == "statement"
    assert response.json()["parser"]["disposition"] == "transaction"
    assert response.json()["merge"]["action"] == "none"
    assert response.json()["merge"]["match_kind"] == "statement_rule"


async def test_email_parse_preview_reports_statement_summary(
    client, session, monkeypatch
):
    email = await _email(session, email_kind="cc_statement")
    await session.commit()
    parsed = ParsedEmail(
        bank="synthetic-bank",
        email_type="synthetic_statement_summary",
        statement=StatementSummary(
            total_amount_due=Money(amount=Decimal("123.45"), currency="INR"),
            due_date=datetime.date(2030, 1, 20),
            card_mask="4111111111111234",
        ),
        password_hint="must-not-be-returned",
    )
    _patch_raw_and_parse(
        monkeypatch, ProcessedEmailParse(None, None, parsed.password_hint, parsed)
    )

    response = await client.post(f"/api/emails/{email.id}/parse-preview")

    assert response.status_code == 200
    parser = response.json()["parser"]
    assert parser["disposition"] == "statement_summary"
    assert parser["password_hint_present"] is True
    assert parser["statement"]["card_mask"] == "XXXX1234"
    assert "must-not-be-returned" not in response.text
    assert response.json()["merge"]["action"] == "none"


async def test_email_parse_preview_returns_404_when_raw_unavailable(
    client, session, monkeypatch
):
    email = await _email(session)
    await session.commit()

    async def unavailable(_email):
        return RawEmailResult(None, "Sensitive loader details", None)

    monkeypatch.setattr(
        "financial_dashboard.services.parse_previews.load_or_fetch_raw_email",
        unavailable,
    )
    response = await client.post(f"/api/emails/{email.id}/parse-preview")

    assert response.status_code == 404
    assert response.json() == {"detail": "Raw email is unavailable"}
    assert "Sensitive" not in response.text


async def test_email_parse_preview_openapi_is_typed(client):
    document = (await client.get("/openapi.json")).json()
    schema = document["paths"]["/api/emails/{email_id}/parse-preview"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/EmailParsePreviewResponse"}
