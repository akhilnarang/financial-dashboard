"""Tests for HTTP Basic Auth (config validation, security logic, integration)."""

import base64
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBasicCredentials
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from financial_dashboard.config import Settings
from financial_dashboard.core.deps import get_session, verify_credentials
from financial_dashboard.core.security import check_credentials
from financial_dashboard.db import Transaction
from financial_dashboard.api import get_router as get_api_router
from financial_dashboard.config import settings as app_settings


def test_auth_requires_both_or_neither_credential():
    assert not Settings(auth_username="", auth_password=SecretStr("")).auth_enabled
    assert Settings(
        auth_username="admin", auth_password=SecretStr("secret")
    ).auth_enabled
    # One credential alone must fail. Else auth is silently off.
    with pytest.raises(ValueError):
        Settings(auth_username="admin", auth_password=SecretStr(""))
    with pytest.raises(ValueError):
        Settings(auth_username="", auth_password=SecretStr("secret"))


def _make_settings(username: str = "", password: str = "") -> Settings:
    return Settings(auth_username=username, auth_password=SecretStr(password))


def test_non_ascii_credentials_are_accepted():
    with patch(
        "financial_dashboard.core.security.settings",
        _make_settings("ユーザー", "пароль"),
    ):
        check_credentials(HTTPBasicCredentials(username="ユーザー", password="пароль"))


def _build_app():
    """Create a tiny FastAPI app with the same auth wiring as the real one."""
    app = FastAPI(dependencies=[Depends(verify_credentials)])

    @app.get("/", response_class=PlainTextResponse)
    async def root():
        return "ok"

    return app


def _basic_auth_header(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.mark.anyio
async def test_auth_disabled_allows_and_enabled_requires_matching_credentials():
    transport = ASGITransport(app=_build_app())
    with patch("financial_dashboard.core.security.settings", _make_settings()):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            open_response = await client.get("/")
    assert open_response.status_code == 200
    assert open_response.text == "ok"

    # A colon in the password must not break Basic auth user:pass splitting.
    with patch(
        "financial_dashboard.core.security.settings",
        _make_settings("admin", "p:a:s:s"),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            missing = await client.get("/")
            wrong = await client.get("/", headers=_basic_auth_header("admin", "wrong"))
            wrong_user = await client.get(
                "/", headers=_basic_auth_header("wrong", "p:a:s:s")
            )
            allowed = await client.get(
                "/", headers=_basic_auth_header("admin", "p:a:s:s")
            )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Basic"
    assert wrong.status_code == 401
    assert wrong_user.status_code == 401
    assert allowed.status_code == 200
    assert allowed.text == "ok"


@pytest.mark.anyio
async def test_transaction_attachment_route_uses_global_auth(
    session, tmp_path, monkeypatch
):
    monkeypatch.setattr(app_settings, "transaction_attachment_root", str(tmp_path))
    (tmp_path / "receipt.pdf").write_bytes(b"%PDF-1.7\nreceipt")
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="12.00",
        attachment_path="receipt.pdf",
    )
    session.add(transaction)
    await session.commit()

    app = FastAPI(dependencies=[Depends(verify_credentials)])

    async def _override_session():
        yield session

    app.dependency_overrides[get_session] = _override_session
    app.include_router(get_api_router(paisa_enabled=True))
    auth_settings = _make_settings("admin", "pass")
    with patch("financial_dashboard.core.security.settings", auth_settings):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            denied = await client.get(f"/api/transactions/{transaction.id}/attachment")
            allowed = await client.get(
                f"/api/transactions/{transaction.id}/attachment",
                headers=_basic_auth_header("admin", "pass"),
            )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.content.startswith(b"%PDF-")
