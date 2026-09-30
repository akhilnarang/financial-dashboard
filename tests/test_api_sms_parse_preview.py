import datetime
from decimal import Decimal

import pytest
from bank_sms_parser.exceptions import ParseError
from bank_sms_parser.models import Money, ParsedSms, SmsTransactionAlert
from sqlalchemy import event

from financial_dashboard.db import SmsMessage, Transaction

pytestmark = pytest.mark.anyio


def _parsed_sms(*, direction: str = "debit", ledger_role: str = "primary"):
    """Build a synthetic parser result without using real source text."""
    return ParsedSms(
        bank="synthetic-bank",
        email_type="synthetic_transaction_alert",
        ledger_role=ledger_role,
        transaction=SmsTransactionAlert(
            direction=direction,
            amount=Money(amount=Decimal("12.34"), currency="INR"),
            transaction_date=datetime.date(2030, 1, 2),
            transaction_time=datetime.time(10, 30),
            counterparty="Synthetic Merchant",
            card_mask="4111111111111234",
            channel="card",
        ),
    )


async def _sms(session, *, transaction_id: int | None = None) -> SmsMessage:
    row = SmsMessage(
        bank="synthetic-bank",
        sender="SYNTH",
        body="Synthetic source text",
        received_at=datetime.datetime(2030, 1, 2, 5, 0),
        status="parsed",
        transaction_id=transaction_id,
    )
    session.add(row)
    await session.flush()
    return row


class _WriteRecorder:
    """Record every write statement that reaches the database."""

    def __init__(self, session):
        self.bind = session.get_bind()
        self.writes: list[str] = []

    def _record(self, _conn, _cursor, statement, _parameters, _context, _many):
        if statement.strip().lower().startswith(("insert", "update", "delete")):
            self.writes.append(statement)

    def __enter__(self):
        event.listen(self.bind, "before_cursor_execute", self._record)
        return self

    def __exit__(self, *_exc):
        event.remove(self.bind, "before_cursor_execute", self._record)


async def test_sms_parse_preview_projects_insert_without_writes(
    client, session, monkeypatch
):
    sms = await _sms(session)
    await session.commit()
    monkeypatch.setattr(
        "financial_dashboard.services.parse_previews.parse_sms",
        lambda *_args, **_kwargs: _parsed_sms(),
    )
    with _WriteRecorder(session) as recorder:
        response = await client.post(f"/api/sms/{sms.id}/parse-preview")

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["current_status"] == "parsed"
    assert body["parser"]["disposition"] == "transaction"
    assert body["parser"]["transaction"]["card_mask"] == "XXXX1234"
    evidence = body["merge"].pop("match_evidence")
    assert body["merge"] == {
        "action": "insert",
        "target_transaction_id": None,
        "match_kind": None,
        "changed_fields": [],
        "identity_conflicts": [],
    }
    assert evidence["candidate_ids"] == []
    assert evidence["reason"] == "alias_no_candidates"
    assert recorder.writes == []

    # A linked row whose identity differs is reported, not merged.
    transaction = Transaction(
        sms_message_id=sms.id,
        bank="synthetic-bank",
        email_type="synthetic_transaction_alert",
        direction="credit",
        amount=Decimal("12.34"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        transaction_time=datetime.time(10, 30),
    )
    session.add(transaction)
    await session.flush()
    sms.transaction_id = transaction.id
    await session.commit()

    response = await client.post(f"/api/sms/{sms.id}/parse-preview")

    assert response.status_code == 200, response.text
    merge = response.json()["merge"]
    assert merge["action"] == "insert"
    assert merge["identity_conflicts"] == ["direction"]


async def test_sms_parse_preview_projects_completion_without_writes(
    client, session, monkeypatch
):
    """A completion leg previews as 'completion' with its target row, not as an
    insert/match/defer from the matcher. The bank name replaces a saved label.
    Once the leg is linked, the preview reports no change, as a reparse does."""
    primary = Transaction(
        bank="synthetic-bank",
        email_type="synthetic_debit_alert",
        direction="debit",
        amount=Decimal("12.34"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        channel="neft",
        account_mask="XX000",
        counterparty="My Saved Payee",
        counterparty_source="user_alias",
        reference_number=None,
        source="sms",
    )
    session.add(primary)
    sms = await _sms(session)
    await session.commit()

    def _completion(*_args, **_kwargs):
        return ParsedSms(
            bank="synthetic-bank",
            email_type="synthetic_neft_completion",
            ledger_role="completion",
            transaction=SmsTransactionAlert(
                direction="debit",
                amount=Money(amount=Decimal("12.34"), currency="INR"),
                transaction_date=datetime.date(2030, 1, 2),
                transaction_time=datetime.time(10, 30),
                counterparty="SAMPLE BENEFICIARY",
                reference_number="INFULLREF0001",
                channel="neft",
            ),
        )

    monkeypatch.setattr(
        "financial_dashboard.services.parse_previews.parse_sms", _completion
    )
    with _WriteRecorder(session) as recorder:
        response = await client.post(f"/api/sms/{sms.id}/parse-preview")

    assert response.status_code == 200, response.text
    merge = response.json()["merge"]
    assert merge["action"] == "completion"
    assert merge["target_transaction_id"] == primary.id
    assert set(merge["changed_fields"]) == {
        "reference_number",
        "counterparty",
        "counterparty_source",
    }
    assert recorder.writes == []

    primary.reference_number = "INFULLREF0001"
    sms.transaction_id = primary.id
    await session.commit()

    response = await client.post(f"/api/sms/{sms.id}/parse-preview")

    assert response.status_code == 200, response.text
    merge = response.json()["merge"]
    assert merge["action"] == "completion"
    assert merge["target_transaction_id"] == primary.id
    assert merge["changed_fields"] == []


async def test_sms_parse_preview_includes_matching_candidate_evidence(
    client, session, monkeypatch
):
    sms = await _sms(session)
    candidate = Transaction(
        bank="synthetic-bank",
        email_type="synthetic_transaction_alert",
        direction="debit",
        amount=Decimal("12.34"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        transaction_time=datetime.time(10, 30),
        counterparty="Synthetic Merchant",
        enriched_at=datetime.datetime(2029, 12, 31, 12, 0),
    )
    session.add(candidate)
    await session.commit()
    monkeypatch.setattr(
        "financial_dashboard.services.parse_previews.parse_sms",
        lambda *_args, **_kwargs: _parsed_sms(),
    )

    with _WriteRecorder(session) as recorder:
        response = await client.post(f"/api/sms/{sms.id}/parse-preview")

    assert response.status_code == 200
    merge = response.json()["merge"]
    assert merge["action"] == "match"
    assert merge["target_transaction_id"] == candidate.id
    assert merge["match_evidence"]["path"] == "fuzzy"
    assert merge["match_evidence"]["candidate_ids"] == [candidate.id]
    assert merge["match_evidence"]["reason"] == "fuzzy_match"
    assert candidate.enriched_at == datetime.datetime(2029, 12, 31, 12, 0)
    assert recorder.writes == []


async def test_sms_parse_preview_reports_parser_error(client, session, monkeypatch):
    sms = await _sms(session)
    await session.commit()

    def fail_parse(*_args, **_kwargs):
        raise ParseError("Synthetic parser failure")

    monkeypatch.setattr(
        "financial_dashboard.services.parse_previews.parse_sms", fail_parse
    )

    response = await client.post(f"/api/sms/{sms.id}/parse-preview")

    assert response.status_code == 200
    assert response.json()["parser"] == {
        "disposition": "error",
        "email_type": None,
        "ledger_role": None,
        "error": "Synthetic parser failure",
        "transaction": None,
    }
    assert response.json()["merge"]["action"] == "none"
