import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from bank_statement_parser.models import BankTransaction, ParsedBankStatement
from cc_parser.parsers.models import Transaction as CcTransaction
from sqlalchemy import event

from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.statement_previews import (
    _could_be_statement_candidate,
    _statement_candidate_index,
)

pytestmark = pytest.mark.anyio


async def _account(session, *, bank="synthetic-bank", account_type="credit_card"):
    account = Account(
        bank=bank,
        type=account_type,
        label="Synthetic account",
        account_number="4111XXXXXXXX1234",
    )
    session.add(account)
    await session.flush()
    return account


def _cc_parsed(rows):
    return SimpleNamespace(
        bank="synthetic-bank",
        name="Synthetic statement",
        card_number="4111111111111234",
        due_date="20/01/2030",
        statement_total_amount_due="12.34",
        transactions=rows,
        payments_refunds=[],
        card_summaries=[],
        possible_adjustment_pairs=[],
        payments_refunds_total="0.00",
        overall_total="12.34",
        overall_reward_points="0",
    )


def _cc_row(*, narration="Synthetic Merchant"):
    return CcTransaction(
        date="02/01/2030",
        narration=narration,
        amount="12.34",
        card_number="4111111111111234",
        transaction_type="debit",
    )


def _bank_parsed(rows):
    return ParsedBankStatement(
        file="synthetic.pdf",
        bank="synthetic-bank",
        account_holder_name="Synthetic Holder",
        account_number="000000001234",
        statement_period_start="01/01/2030",
        statement_period_end="31/01/2030",
        opening_balance="100.00",
        closing_balance="87.66",
        debit_count=len(rows),
        debit_total="12.34",
        credit_total="0.00",
        transactions=rows,
    )


def _bank_row():
    return BankTransaction(
        date="02/01/2030",
        narration="Synthetic transfer",
        amount="12.34",
        transaction_type="debit",
        reference_number="SYNTHETIC-REF",
        channel="imps",
    )


async def test_cc_statement_parse_preview_is_bounded_and_read_only(
    client, session, monkeypatch, tmp_path
):
    account = await _account(session)
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="synthetic.pdf",
        file_path=str(pdf),
        source_kind="pdf",
        status="parsed",
    )
    session.add(upload)
    await session.commit()
    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_statement",
        lambda *_args, **_kwargs: _cc_parsed(
            [_cc_row(narration=f"Synthetic merchant {index}") for index in range(105)]
        ),
    )
    statements: list[str] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        response = await client.post(f"/api/statements/cc/{upload.id}/parse-preview")
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "cc"
    assert body["card_mask"] == "XXXX1234"
    assert body["parsed_row_count"] == 105
    assert len(body["rows"]) == 100
    assert body["rows_truncated"] is True
    assert not any(
        statement.startswith(("insert", "update", "delete")) for statement in statements
    )


async def test_cc_statement_reconcile_preview_classifies_matches_and_missing(
    client, session, monkeypatch, tmp_path
):
    account = await _account(session)
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="synthetic.pdf",
        file_path=str(pdf),
        source_kind="pdf",
        status="parsed",
    )
    transaction = Transaction(
        account_id=account.id,
        bank=account.bank,
        email_type="synthetic_alert",
        direction="debit",
        amount=Decimal("12.34"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        card_mask="1234",
    )
    session.add_all([upload, transaction])
    await session.commit()
    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_statement",
        lambda *_args, **_kwargs: _cc_parsed(
            [_cc_row(), _cc_row(narration="Second synthetic purchase")]
        ),
    )

    response = await client.post(f"/api/statements/cc/{upload.id}/reconcile-preview")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["matched_count"] == 0
    assert body["missing_count"] == 0
    assert body["ambiguous_count"] == 2
    assert {entry["decision_reason"] for entry in body["ambiguous"]} == {
        "candidate_refused_or_consumed",
        "contested_match_demoted",
    }
    assert all(
        entry["candidate_transaction_ids"] == [transaction.id]
        and entry["candidate_count"] == 1
        and "contention" in entry["gates"]
        for entry in body["ambiguous"]
    )
    assert body["extra_count"] == 0
    assert body["extra_transaction_ids"] == []


async def test_bank_statement_parse_and_reconcile_preview(
    client, session, monkeypatch, tmp_path
):
    account = await _account(session, account_type="bank_account")
    pdf = tmp_path / "synthetic-bank.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload = BankStatementUpload(
        account_id=account.id,
        bank="statement-parser-bank",
        filename="synthetic-bank.pdf",
        file_path=str(pdf),
        status="parsed",
    )
    transaction = Transaction(
        account_id=account.id,
        bank=account.bank,
        email_type="synthetic_alert",
        direction="debit",
        amount=Decimal("12.34"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 2),
        reference_number="SYNTHETIC-REF",
        enriched_at=datetime.datetime(2029, 12, 31, 12, 0),
    )
    extra_transaction = Transaction(
        account_id=account.id,
        bank=account.bank,
        email_type="synthetic_alert",
        direction="debit",
        amount=Decimal("99.99"),
        currency="INR",
        transaction_date=datetime.date(2030, 1, 3),
    )
    session.add_all([upload, transaction, extra_transaction])
    await session.commit()
    parser_banks = []

    def parse_bank(_path, bank, _password):
        parser_banks.append(bank)
        return _bank_parsed([_bank_row()])

    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_bank_statement",
        parse_bank,
    )

    parse_response = await client.post(
        f"/api/statements/bank/{upload.id}/parse-preview"
    )
    statements: list[str] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        reconcile_response = await client.post(
            f"/api/statements/bank/{upload.id}/reconcile-preview"
        )
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert parse_response.status_code == 200, parse_response.text
    assert parse_response.json()["account_mask"] == "XXXX1234"
    assert reconcile_response.status_code == 200, reconcile_response.text
    body = reconcile_response.json()
    assert body["matched_count"] == 1
    assert body["matched"][0]["matched_transaction_id"] == transaction.id
    assert body["matched"][0]["candidate_transaction_ids"] == [transaction.id]
    assert body["matched"][0]["decision_reason"] == "matched_reference"
    assert "reference_direction" in body["matched"][0]["gates"]
    assert body["missing_count"] == 0
    assert body["extra_count"] == 1
    assert body["extra_transaction_ids"] == [extra_transaction.id]
    assert parser_banks == ["statement-parser-bank", "statement-parser-bank"]
    assert body["candidate_scope"] == "date_buffer_plus_statement_references"
    assert transaction.enriched_at == datetime.datetime(2029, 12, 31, 12, 0)
    assert not any(
        statement.startswith(("insert", "update", "delete")) for statement in statements
    )


async def test_bank_reconcile_preview_includes_null_dated_reference_match(
    client, session, monkeypatch, tmp_path
):
    account = await _account(session, account_type="bank_account")
    pdf = tmp_path / "synthetic-null-date.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload = BankStatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="synthetic-null-date.pdf",
        file_path=str(pdf),
        status="parsed",
    )
    transaction = Transaction(
        account_id=account.id,
        bank=account.bank,
        email_type="synthetic_alert",
        direction="debit",
        amount=Decimal("12.34"),
        currency="INR",
        transaction_date=None,
        reference_number="SYNTHETIC-REF",
    )
    session.add_all([upload, transaction])
    await session.commit()
    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_bank_statement",
        lambda *_args, **_kwargs: _bank_parsed([_bank_row()]),
    )

    response = await client.post(f"/api/statements/bank/{upload.id}/reconcile-preview")

    assert response.status_code == 200, response.text
    assert response.json()["matched"][0]["matched_transaction_id"] == transaction.id
    assert response.json()["matched"][0]["decision_reason"] == "matched_reference"


async def test_statement_preview_rejects_email_summary(client, session):
    account = await _account(session)
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="",
        file_path="",
        source_kind="email_summary",
        status="parsed",
    )
    session.add(upload)
    await session.commit()

    response = await client.post(f"/api/statements/cc/{upload.id}/parse-preview")

    assert response.status_code == 409
    assert response.json() == {"detail": "Email-summary statement has no PDF"}


async def test_statement_preview_parse_failure_is_sanitized(
    client, session, monkeypatch, tmp_path
):
    account = await _account(session)
    pdf = tmp_path / "private-name.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload = StatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="private-name.pdf",
        file_path=str(pdf),
        source_kind="pdf",
        status="parse_error",
    )
    session.add(upload)
    await session.commit()

    def fail(*_args, **_kwargs):
        raise ValueError(f"failed at sensitive path {pdf}")

    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_statement", fail
    )
    response = await client.post(f"/api/statements/cc/{upload.id}/parse-preview")

    assert response.status_code == 422
    assert response.json() == {"detail": "Statement parse failed"}
    assert str(pdf) not in response.text


async def test_statement_reconciliation_rejects_reversed_period(
    client, session, monkeypatch, tmp_path
):
    account = await _account(session, account_type="bank_account")
    pdf = tmp_path / "synthetic-reversed.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload = BankStatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="synthetic-reversed.pdf",
        file_path=str(pdf),
        status="parsed",
    )
    session.add(upload)
    await session.commit()
    parsed = _bank_parsed([_bank_row()])
    parsed.statement_period_start = "31/01/2030"
    parsed.statement_period_end = "01/01/2030"
    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_bank_statement",
        lambda *_args, **_kwargs: parsed,
    )

    response = await client.post(f"/api/statements/bank/{upload.id}/reconcile-preview")

    assert response.status_code == 422
    assert response.json() == {"detail": "Statement date range is invalid"}


async def test_statement_reconciliation_failure_is_sanitized(
    client, session, monkeypatch, tmp_path
):
    account = await _account(session, account_type="bank_account")
    pdf = tmp_path / "private-bank.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload = BankStatementUpload(
        account_id=account.id,
        bank=account.bank,
        filename="private-bank.pdf",
        file_path=str(pdf),
        status="parsed",
    )
    session.add(upload)
    await session.commit()
    parsed = _bank_parsed([_bank_row()])
    parsed.opening_balance = "not-an-amount"
    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_bank_statement",
        lambda *_args, **_kwargs: parsed,
    )

    response = await client.post(f"/api/statements/bank/{upload.id}/reconcile-preview")

    assert response.status_code == 422
    assert response.json() == {"detail": "Statement reconciliation failed"}


async def test_extra_classification_treats_missing_direction_as_global_uncertainty():
    candidate_index = _statement_candidate_index(
        [{"direction": None, "amount": "12.34", "date": "02/01/2030"}]
    )
    transaction = Transaction(
        direction="debit",
        amount=Decimal("99.99"),
        transaction_date=datetime.date(2030, 1, 2),
    )

    assert candidate_index.uncertain_all_directions is True
    assert _could_be_statement_candidate(
        transaction,
        candidate_index.identities,
        candidate_index.uncertain_directions,
        candidate_index.uncertain_all_directions,
    )


async def test_statement_preview_openapi_routes_are_typed(client):
    document = (await client.get("/openapi.json")).json()
    expected = {
        "/api/statements/cc/{statement_id}/parse-preview": "StatementParsePreviewResponse",
        "/api/statements/bank/{statement_id}/parse-preview": "StatementParsePreviewResponse",
        "/api/statements/cc/{statement_id}/reconcile-preview": "StatementReconciliationPreviewResponse",
        "/api/statements/bank/{statement_id}/reconcile-preview": "StatementReconciliationPreviewResponse",
    }
    for path, model in expected.items():
        schema = document["paths"][path]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert schema == {"$ref": f"#/components/schemas/{model}"}
