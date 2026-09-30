import pytest

from financial_dashboard.db import BankStatementUpload, StatementUpload
from financial_dashboard.schemas.sms import ReparseSmsResponse

pytestmark = pytest.mark.anyio


async def test_api_sms_reparse_forwards_to_canonical_operation(client, monkeypatch):
    calls = []

    async def fake_reparse(sms_id, force_new, session):
        calls.append((sms_id, force_new, session is not None))
        return ReparseSmsResponse(
            message="Synthetic SMS reparse",
            new_status="enriched",
            txn_id=42,
            diff=["counterparty"],
        )

    monkeypatch.setattr("financial_dashboard.api.sms.reparse_sms_service", fake_reparse)

    response = await client.post("/api/sms/7/reparse?force_new=true")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Synthetic SMS reparse",
        "new_status": "enriched",
        "txn_id": 42,
        "diff": ["counterparty"],
    }
    assert calls == [(7, True, True)]


@pytest.mark.parametrize("kind", ["cc", "bank"])
async def test_api_statement_reparse_forwards_password_without_persisting(
    client, session, monkeypatch, kind
):
    if kind == "cc":
        statement = StatementUpload(
            account_id=77,
            bank="synthetic",
            filename="synthetic.pdf",
            file_path="/synthetic/statement.pdf",
            status="parsed",
        )
        operation_path = "financial_dashboard.api.statements.retry_cc_statement_upload"
    else:
        statement = BankStatementUpload(
            account_id=77,
            bank="synthetic",
            filename="synthetic.pdf",
            file_path="/synthetic/statement.pdf",
            status="parsed",
        )
        operation_path = (
            "financial_dashboard.api.statements.retry_bank_statement_upload"
        )
    session.add(statement)
    await session.commit()
    statement_id = statement.id
    calls = []

    async def fake_reparse(reparsed_statement_id, password):
        calls.append((reparsed_statement_id, password))
        return True

    monkeypatch.setattr(operation_path, fake_reparse)

    response = await client.post(
        f"/api/statements/{kind}/{statement_id}/reparse",
        json={"password": "pw-123"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["id"] == statement_id
    assert response.json()["status"] == "parsed"
    assert "pw-123" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert calls == [(statement_id, "pw-123")]


async def test_api_cc_statement_reparse_rejects_summary_without_calling_operation(
    client, session, monkeypatch
):
    statement = StatementUpload(
        account_id=77,
        bank="synthetic",
        filename="synthetic-summary",
        file_path="",
        source_kind="email_summary",
        status="parsed",
    )
    session.add(statement)
    await session.commit()
    calls = []

    async def fake_reparse(statement_id, password):
        calls.append((statement_id, password))
        return True

    monkeypatch.setattr(
        "financial_dashboard.api.statements.retry_cc_statement_upload",
        fake_reparse,
    )

    response = await client.post(
        f"/api/statements/cc/{statement.id}/reparse",
        json={"password": "synthetic"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Email-summary statements have no PDF to reparse"
    }
    assert calls == []


async def test_api_statement_reparse_rejects_long_password(
    client, session, monkeypatch
):
    statement = BankStatementUpload(
        account_id=77,
        bank="synthetic",
        filename="synthetic.pdf",
        file_path="/synthetic/statement.pdf",
        status="password_required",
    )
    session.add(statement)
    await session.commit()
    calls = []

    async def fake_reparse(statement_id, password):
        calls.append((statement_id, password))
        return True

    monkeypatch.setattr(
        "financial_dashboard.api.statements.retry_bank_statement_upload",
        fake_reparse,
    )
    long_password = "x" * 13

    response = await client.post(
        f"/api/statements/bank/{statement.id}/reparse",
        json={"password": long_password},
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert calls == []


async def test_api_bank_statement_reparse_failure_is_sanitized(
    client, session, monkeypatch
):
    statement = BankStatementUpload(
        account_id=77,
        bank="synthetic",
        filename="synthetic.pdf",
        file_path="/synthetic/statement.pdf",
        status="password_required",
        error="Existing synthetic error",
    )
    session.add(statement)
    await session.commit()

    async def fake_reparse(_statement_id, _password):
        return False

    monkeypatch.setattr(
        "financial_dashboard.api.statements.retry_bank_statement_upload",
        fake_reparse,
    )

    response = await client.post(
        f"/api/statements/bank/{statement.id}/reparse",
        json={"password": "wrong-pass"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Bank statement reparse failed"}
    assert "wrong-pass" not in response.text
