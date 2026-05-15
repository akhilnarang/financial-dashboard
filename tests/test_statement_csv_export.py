from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.main import create_app
import financial_dashboard.core.deps as core_deps
from financial_dashboard.db import Account, Base, StatementUpload
from financial_dashboard.web import statements as statement_routes


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()


async def _seed_upload(
    maker,
    *,
    tmp_path: Path,
    account_bank: str = "hdfc",
    upload_bank: str = "hdfc",
    source_kind: str = "pdf",
    file_path: str | None = None,
    status: str = "parsed",
) -> tuple[int, int, Path | None]:
    pdf_path = None
    if file_path is None and source_kind != "email_summary":
        pdf_path = tmp_path / "statement.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        file_path = str(pdf_path)

    async with maker() as session:
        account = Account(
            bank=account_bank,
            label=f"{account_bank.upper()} Test CC",
            type="credit_card",
            active=True,
        )
        session.add(account)
        await session.flush()
        upload = StatementUpload(
            account_id=account.id,
            bank=upload_bank,
            filename="statement.pdf",
            file_path=file_path or "",
            source_kind=source_kind,
            status=status,
            card_number="1234",
            due_date="25/05/2026",
            total_amount_due="100.00",
            parsed_txn_count=1,
            matched_count=0,
            missing_count=1,
            imported_count=0,
        )
        session.add(upload)
        await session.commit()
        return upload.id, account.id, pdf_path


def _client():
    return AsyncClient(
        transport=ASGITransport(app=create_app()),
        base_url="http://test",
        follow_redirects=False,
    )


@pytest.mark.anyio
async def test_statement_csv_download_returns_cc_parser_csv_bytes(
    session_factory, tmp_path, monkeypatch
):
    upload_id, _account_id, pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path
    )
    calls = {}

    def fake_parse_statement(path, password, bank):
        calls["parse"] = {"path": path, "password": password, "bank": bank}
        return SimpleNamespace(marker="parsed")

    def fake_write_transactions_csv(parsed, output_path):
        calls["export"] = {"parsed": parsed, "output_path": output_path}
        output_path.write_text("source,amount\ntransactions,123.45\n", encoding="utf-8")

    monkeypatch.setattr(statement_routes, "parse_statement", fake_parse_statement)
    monkeypatch.setattr(
        statement_routes,
        "write_transactions_csv",
        fake_write_transactions_csv,
        raising=False,
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        response.headers["content-disposition"]
        == f'attachment; filename="statement-{upload_id}.csv"'
    )
    assert response.content == b"source,amount\ntransactions,123.45\n"
    assert calls["parse"] == {
        "path": pdf_path,
        "password": None,
        "bank": "hdfc",
    }
    assert calls["export"]["parsed"].marker == "parsed"


@pytest.mark.anyio
async def test_statement_csv_download_does_not_mutate_upload(
    session_factory, tmp_path, monkeypatch
):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path
    )
    monkeypatch.setattr(
        statement_routes,
        "parse_statement",
        lambda path, password, bank: SimpleNamespace(),
    )
    monkeypatch.setattr(
        statement_routes,
        "write_transactions_csv",
        lambda parsed, output_path: output_path.write_text("x\n", encoding="utf-8"),
        raising=False,
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 200
    async with session_factory() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.status == "parsed"
        assert upload.reconciliation_data is None
        assert upload.imported_count == 0


@pytest.mark.anyio
async def test_statement_csv_download_rejects_email_summary(session_factory, tmp_path):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory,
        tmp_path=tmp_path,
        source_kind="email_summary",
        file_path="",
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/statements/{upload_id}?error=")
    assert "email+body" in response.headers["location"]


@pytest.mark.anyio
async def test_statement_csv_download_rejects_missing_pdf(session_factory, tmp_path):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory,
        tmp_path=tmp_path,
        file_path=str(tmp_path / "missing.pdf"),
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/statements/{upload_id}?error=")
    assert "PDF+file+missing" in response.headers["location"]


@pytest.mark.anyio
async def test_statement_csv_download_rejects_blank_pdf_path(session_factory, tmp_path):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory,
        tmp_path=tmp_path,
        file_path="",
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/statements/{upload_id}?error=")
    assert "PDF+file+missing" in response.headers["location"]


@pytest.mark.anyio
async def test_statement_csv_download_parse_error_redirects(
    session_factory, tmp_path, monkeypatch
):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path
    )

    def fake_parse_statement(path, password, bank):
        raise ValueError("bad pdf")

    monkeypatch.setattr(statement_routes, "parse_statement", fake_parse_statement)

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/statements/{upload_id}?error=")
    assert "CSV+export+failed%3A+bad+pdf" in response.headers["location"]


@pytest.mark.anyio
async def test_statement_csv_download_export_error_redirects(
    session_factory, tmp_path, monkeypatch
):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path
    )
    monkeypatch.setattr(
        statement_routes,
        "parse_statement",
        lambda path, password, bank: SimpleNamespace(),
    )

    def fake_write_transactions_csv(parsed, output_path):
        raise RuntimeError("csv failed")

    monkeypatch.setattr(
        statement_routes,
        "write_transactions_csv",
        fake_write_transactions_csv,
        raising=False,
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/statements/{upload_id}?error=")
    assert "CSV+export+failed%3A+csv+failed" in response.headers["location"]


@pytest.mark.anyio
async def test_statement_csv_download_ignores_bad_saved_password(
    session_factory, tmp_path, monkeypatch
):
    upload_id, account_id, _pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path
    )
    async with session_factory() as session:
        assert (account := await session.get(Account, account_id)) is not None
        account.statement_password = "not-valid-fernet"
        await session.commit()

    calls = {}

    def fake_parse_statement(path, password, bank):
        calls["password"] = password
        return SimpleNamespace()

    monkeypatch.setattr(statement_routes, "parse_statement", fake_parse_statement)
    monkeypatch.setattr(
        statement_routes,
        "write_transactions_csv",
        lambda parsed, output_path: output_path.write_text("x\n", encoding="utf-8"),
        raising=False,
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 200
    assert calls["password"] is None


@pytest.mark.anyio
async def test_statement_csv_download_uses_decrypted_saved_password(
    session_factory, tmp_path, monkeypatch
):
    upload_id, account_id, _pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path
    )
    async with session_factory() as session:
        assert (account := await session.get(Account, account_id)) is not None
        account.statement_password = "dummy-encrypted-value"
        await session.commit()

    calls = {}

    class FakeFernet:
        def decrypt(self, value):
            calls["decrypt_value"] = value
            return b"secretpw"

    def fake_parse_statement(path, password, bank):
        calls["password"] = password
        return SimpleNamespace()

    monkeypatch.setattr(statement_routes, "get_fernet", lambda: FakeFernet())
    monkeypatch.setattr(statement_routes, "parse_statement", fake_parse_statement)
    monkeypatch.setattr(
        statement_routes,
        "write_transactions_csv",
        lambda parsed, output_path: output_path.write_text("x\n", encoding="utf-8"),
        raising=False,
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 200
    assert calls["decrypt_value"] == b"dummy-encrypted-value"
    assert calls["password"] == "secretpw"


@pytest.mark.anyio
async def test_statement_csv_download_prefers_account_bank_when_account_exists(
    session_factory, tmp_path, monkeypatch
):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory,
        tmp_path=tmp_path,
        account_bank="hdfc",
        upload_bank="axis",
    )
    calls = {}

    def fake_parse_statement(path, password, bank):
        calls["bank"] = bank
        return SimpleNamespace()

    monkeypatch.setattr(statement_routes, "parse_statement", fake_parse_statement)
    monkeypatch.setattr(
        statement_routes,
        "write_transactions_csv",
        lambda parsed, output_path: output_path.write_text("x\n", encoding="utf-8"),
        raising=False,
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 200
    assert calls["bank"] == "hdfc"


@pytest.mark.anyio
async def test_statement_csv_download_uses_upload_bank_when_account_missing(
    session_factory, tmp_path, monkeypatch
):
    upload_id, account_id, _pdf_path = await _seed_upload(
        session_factory,
        tmp_path=tmp_path,
        account_bank="hdfc",
        upload_bank="axis",
    )
    async with session_factory() as session:
        assert (account := await session.get(Account, account_id)) is not None
        await session.delete(account)
        await session.commit()

    calls = {}

    def fake_parse_statement(path, password, bank):
        calls["bank"] = bank
        return SimpleNamespace()

    monkeypatch.setattr(statement_routes, "parse_statement", fake_parse_statement)
    monkeypatch.setattr(
        statement_routes,
        "write_transactions_csv",
        lambda parsed, output_path: output_path.write_text("x\n", encoding="utf-8"),
        raising=False,
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}/csv")

    assert response.status_code == 200
    assert calls["bank"] == "axis"


@pytest.mark.anyio
async def test_statement_csv_download_missing_upload_returns_404(session_factory):
    async with _client() as client:
        response = await client.get("/statements/999/csv")

    assert response.status_code == 404
    assert b"Statement not found" in response.content


@pytest.mark.anyio
async def test_statement_detail_shows_csv_link_for_pdf_upload(
    session_factory, tmp_path
):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}")

    assert response.status_code == 200
    assert f'href="/statements/{upload_id}/csv"' in response.text
    assert "Download CSV" in response.text


@pytest.mark.anyio
async def test_statement_detail_omits_csv_link_for_email_summary(
    session_factory, tmp_path
):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory,
        tmp_path=tmp_path,
        source_kind="email_summary",
        file_path="",
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}")

    assert response.status_code == 200
    assert f"/statements/{upload_id}/csv" not in response.text
    assert "Download CSV" not in response.text


@pytest.mark.anyio
async def test_statement_detail_error_banner_uses_generic_action_label(
    session_factory, tmp_path
):
    upload_id, _account_id, _pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path
    )

    async with _client() as client:
        response = await client.get(f"/statements/{upload_id}?error=boom")

    assert response.status_code == 200
    assert "Statement action failed:" in response.text
    assert "Reprocess failed:" not in response.text
