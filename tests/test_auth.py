"""Tests for HTTP Basic Auth (config validation, security logic, integration)."""

import base64
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBasicCredentials
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from financial_dashboard.config import Settings
from financial_dashboard.core.deps import verify_credentials
from financial_dashboard.core.security import check_credentials


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_both_unset_is_valid(self):
        s = Settings(auth_username="", auth_password=SecretStr(""))
        assert not s.auth_enabled

    def test_both_set_is_valid(self):
        s = Settings(auth_username="admin", auth_password=SecretStr("secret"))
        assert s.auth_enabled

    def test_username_only_raises(self):
        with pytest.raises(ValueError, match="both be set or both be empty"):
            Settings(auth_username="admin", auth_password=SecretStr(""))

    def test_password_only_raises(self):
        with pytest.raises(ValueError, match="both be set or both be empty"):
            Settings(auth_username="", auth_password=SecretStr("secret"))


# ---------------------------------------------------------------------------
# check_credentials unit tests
# ---------------------------------------------------------------------------


def _make_settings(username: str = "", password: str = "") -> Settings:
    return Settings(auth_username=username, auth_password=SecretStr(password))


def _make_creds(username: str, password: str):
    return HTTPBasicCredentials(username=username, password=password)


class TestCheckCredentials:
    def test_disabled_allows_none(self):
        with patch("financial_dashboard.core.security.settings", _make_settings()):
            check_credentials(None)  # should not raise

    def test_disabled_allows_any_creds(self):
        with patch("financial_dashboard.core.security.settings", _make_settings()):
            check_credentials(_make_creds("whoever", "whatever"))

    def test_enabled_rejects_none(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            with pytest.raises(HTTPException) as exc_info:
                check_credentials(None)
            assert exc_info.value.status_code == 401
            assert exc_info.value.headers is not None
            assert exc_info.value.headers["WWW-Authenticate"] == "Basic"

    def test_enabled_accepts_correct(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            check_credentials(_make_creds("admin", "pass"))

    def test_enabled_rejects_wrong_username(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            with pytest.raises(HTTPException) as exc_info:
                check_credentials(_make_creds("wrong", "pass"))
            assert exc_info.value.status_code == 401

    def test_enabled_rejects_wrong_password(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            with pytest.raises(HTTPException) as exc_info:
                check_credentials(_make_creds("admin", "wrong"))
            assert exc_info.value.status_code == 401

    def test_enabled_rejects_both_wrong(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            with pytest.raises(HTTPException) as exc_info:
                check_credentials(_make_creds("wrong", "wrong"))
            assert exc_info.value.status_code == 401

    def test_password_with_colons(self):
        """Colons in password must not break Basic auth's user:pass splitting."""
        with patch(
            "financial_dashboard.core.security.settings",
            _make_settings("admin", "p:a:s:s"),
        ):
            check_credentials(_make_creds("admin", "p:a:s:s"))

    def test_non_ascii_credentials(self):
        with patch(
            "financial_dashboard.core.security.settings",
            _make_settings("ユーザー", "пароль"),
        ):
            check_credentials(_make_creds("ユーザー", "пароль"))

    def test_non_ascii_wrong_password_rejected(self):
        with patch(
            "financial_dashboard.core.security.settings",
            _make_settings("ユーザー", "пароль"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                check_credentials(_make_creds("ユーザー", "wrong"))
            assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Integration tests — minimal FastAPI app with the auth dependency
# ---------------------------------------------------------------------------


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
class TestAuthIntegration:
    async def test_auth_disabled_no_header(self):
        with patch("financial_dashboard.core.security.settings", _make_settings()):
            async with AsyncClient(
                transport=ASGITransport(app=_build_app()), base_url="http://test"
            ) as client:
                r = await client.get("/")
                assert r.status_code == 200
                assert r.text == "ok"

    async def test_auth_disabled_with_header_still_passes(self):
        """When auth is disabled, a stray Authorization header should not cause errors."""
        with patch("financial_dashboard.core.security.settings", _make_settings()):
            async with AsyncClient(
                transport=ASGITransport(app=_build_app()), base_url="http://test"
            ) as client:
                r = await client.get(
                    "/", headers=_basic_auth_header("whoever", "whatever")
                )
                assert r.status_code == 200
                assert r.text == "ok"

    async def test_auth_enabled_no_header_returns_401(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            async with AsyncClient(
                transport=ASGITransport(app=_build_app()), base_url="http://test"
            ) as client:
                r = await client.get("/")
                assert r.status_code == 401
                assert r.headers["www-authenticate"] == "Basic"

    async def test_auth_enabled_correct_creds(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            async with AsyncClient(
                transport=ASGITransport(app=_build_app()), base_url="http://test"
            ) as client:
                r = await client.get("/", headers=_basic_auth_header("admin", "pass"))
                assert r.status_code == 200
                assert r.text == "ok"

    async def test_auth_enabled_wrong_creds_returns_401(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            async with AsyncClient(
                transport=ASGITransport(app=_build_app()), base_url="http://test"
            ) as client:
                r = await client.get("/", headers=_basic_auth_header("admin", "wrong"))
                assert r.status_code == 401

    async def test_auth_enabled_wrong_username_returns_401(self):
        with patch(
            "financial_dashboard.core.security.settings", _make_settings("admin", "pass")
        ):
            async with AsyncClient(
                transport=ASGITransport(app=_build_app()), base_url="http://test"
            ) as client:
                r = await client.get("/", headers=_basic_auth_header("wrong", "pass"))
                assert r.status_code == 401

    async def test_password_with_colons(self):
        with patch(
            "financial_dashboard.core.security.settings",
            _make_settings("admin", "p:a:s:s"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=_build_app()), base_url="http://test"
            ) as client:
                r = await client.get(
                    "/", headers=_basic_auth_header("admin", "p:a:s:s")
                )
                assert r.status_code == 200
