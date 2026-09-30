from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.config import get_fernet
from financial_dashboard.main import create_app
import financial_dashboard.core.deps as core_deps
from financial_dashboard.db import Account, StatementUpload
from financial_dashboard.web import statements as statement_routes
from tests.conftest import new_test_engine


@pytest.fixture
async def session_factory(monkeypatch):
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()
    holder.close()


async def _seed_upload(
    maker,
    *,
    tmp_path: Path,
    upload_bank: str = "hdfc",
    file_path: str | None = None,
) -> tuple[int, int, Path | None]:
    pdf_path = None
    if file_path is None:
        pdf_path = tmp_path / "statement.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        file_path = str(pdf_path)

    async with maker() as session:
        account = Account(
            bank="hdfc",
            label="HDFC Test CC",
            type="credit_card",
            active=True,
        )
        session.add(account)
        await session.flush()
        upload = StatementUpload(
            account_id=account.id,
            bank=upload_bank,
            filename="statement.pdf",
            file_path=file_path,
            source_kind="pdf",
            status="parsed",
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
    upload_id, account_id, pdf_path = await _seed_upload(
        session_factory, tmp_path=tmp_path, upload_bank="axis"
    )
    async with session_factory() as session:
        assert (account := await session.get(Account, account_id)) is not None
        account.statement_password = get_fernet().encrypt(b"secretpw").decode()
        await session.commit()
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
        "password": "secretpw",
        "bank": "hdfc",
    }
    assert calls["export"]["parsed"].marker == "parsed"
    async with session_factory() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.status == "parsed"
        assert upload.reconciliation_data is None
        assert upload.imported_count == 0


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
