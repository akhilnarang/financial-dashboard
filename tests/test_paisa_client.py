"""Contracts, security and retry behaviour for the Paisa API client.

Pinned to the v0.7.x shapes defined in ananthakumaran/paisa
``internal/server/server.go``:

* ``GET /api/ping`` → ``{"success": true}``
* ``GET /api/config`` → ``{"config": {readonly, ledger_cli, default_currency, ...}}``
* ``POST /api/sync`` → ``{"success": bool, "message": str}`` (HTTP 200 even on
  failure; readonly fakes success)
* ``GET /api/diagnosis`` → ``{"issues": [{"level","summary","description","details"}]}``
  where ``level == "danger"`` is an error and ``"warning"`` is informational

All network is faked with :class:`httpx.MockTransport`; no real socket is
opened. The tests pin exact request shapes, the URL-validation security
envelope, and the bounded 429 retry.
"""

import datetime
import email.utils
import json

import httpx
import pytest

from financial_dashboard.integrations.paisa import (
    DEFAULT_BASE_URL,
    MAX_429_RETRIES,
    SYNC_PAYLOAD,
    PaisaCapabilities,
    PaisaClient,
    PaisaDiagnosis,
    PaisaError,
    PaisaSyncResult,
    _retry_after_seconds,
    validate_base_url,
)

pytestmark = pytest.mark.anyio

CANONICAL_DEFAULT = "http://127.0.0.1:7500"


def _client(transport: httpx.MockTransport, **kwargs) -> PaisaClient:
    return PaisaClient(
        base_url=kwargs.pop("base_url", DEFAULT_BASE_URL),
        transport=transport,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:7500",
        "https://localhost",
        "http://localhost:9000/",
        "http://127.0.0.1/paisa",  # reverse-proxy subpath allowed
    ],
)
def test_validate_loopback_ok(url):
    v = validate_base_url(url, allow_remote=False)
    assert v.is_loopback is True
    assert v.scheme in ("http", "https")


@pytest.mark.parametrize(
    "url",
    ["http://10.0.0.1:7500", "http://192.168.1.5"],
)
def test_validate_remote_http_rejected_even_with_flag(url):
    # Remote must be HTTPS regardless of allow_remote.
    with pytest.raises(PaisaError) as exc:
        validate_base_url(url, allow_remote=True)
    assert exc.value.code == "remote_requires_https"


def test_validate_remote_https_allowed_with_flag():
    v = validate_base_url("https://paisa.example.com", allow_remote=True)
    assert v.is_loopback is False
    assert v.scheme == "https"


def test_validate_remote_rejected_without_flag():
    with pytest.raises(PaisaError) as exc:
        validate_base_url("https://paisa.example.com", allow_remote=False)
    assert exc.value.code == "remote_not_allowed"


@pytest.mark.parametrize(
    "url, code",
    [
        ("ftp://127.0.0.1", "invalid_scheme"),
        ("file:///etc/passwd", "invalid_scheme"),
        ("http://127.0.0.1/?x=1", "query_in_url"),
        ("http://127.0.0.1/#frag", "fragment_in_url"),
        ("http://user:pass@127.0.0.1", "credentials_in_url"),
        ("http://127.0.0.1/../etc", "path_traversal"),
        ("", "invalid_url"),
        ("   ", "invalid_url"),
        ("http://", "missing_host"),
    ],
)
def test_validate_rejects(url, code):
    with pytest.raises(PaisaError) as exc:
        validate_base_url(url, allow_remote=False)
    assert exc.value.code == code


def test_validate_normalizes_path_prefix():
    v = validate_base_url("http://127.0.0.1/paisa/", allow_remote=False)
    assert v.path_prefix == "/paisa"
    assert v.display == "http://127.0.0.1/paisa"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:not-a-port",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:-1",
    ],
)
def test_validate_malformed_port_is_typed_error(url):
    with pytest.raises(PaisaError) as exc:
        validate_base_url(url, allow_remote=False)
    assert exc.value.code == "invalid_port"


def test_validate_normalizes_host_case_and_default_port():
    a = validate_base_url("HTTP://LOCALHOST:80/", allow_remote=False)
    b = validate_base_url("http://localhost", allow_remote=False)
    assert a.display == b.display == "http://localhost/"


def test_default_base_url_is_canonical_loopback_7500():
    # The official default port is 7500 (cmd/serve.go). A regression here would
    # silently point the client at the wrong port.
    assert DEFAULT_BASE_URL == CANONICAL_DEFAULT


# ---------------------------------------------------------------------------
# Request contracts
# ---------------------------------------------------------------------------


async def test_ping_true_on_2xx():
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"success": True})

    async with _client(httpx.MockTransport(handler)) as c:
        assert await c.ping() is True
    assert seen[0].url.path == "/api/ping"


async def test_ping_false_when_unreachable():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=req)

    async with _client(httpx.MockTransport(handler)) as c:
        assert await c.ping() is False  # unreachable -> False, not raise


async def test_ping_raises_on_redirect():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "/elsewhere"})

    async with _client(httpx.MockTransport(handler)) as c:
        with pytest.raises(PaisaError) as exc:
            await c.ping()
        assert exc.value.code == "redirect_disallowed"


async def test_fetch_config_sanitizes_subset():
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        return httpx.Response(
            200,
            json={
                "config": {
                    "ledger_cli": "ledger",
                    "readonly": False,
                    "default_currency": "INR",
                    "amount_alignment_column": 52,
                    # Plenty of upstream keys we must NOT surface:
                    "db_path": "/home/me/.paisa/db",
                    "user_accounts": [{"username": "x", "password": "sha256:y"}],
                },
                "accounts": [{"name": "Assets:Bank"}],
                "now": None,
            },
        )

    async with _client(httpx.MockTransport(handler)) as c:
        caps = await c.fetch_config()
    assert captured["path"] == "/api/config"
    assert caps == PaisaCapabilities(
        ledger_cli="ledger", readonly=False, default_currency="INR"
    )
    # No raw field leaks through.
    assert not hasattr(caps, "db_path")
    assert not hasattr(caps, "user_accounts")


async def test_fetch_config_readonly_truthy_forms():
    for raw in ["true", "True", "yes", 1, True]:

        def handler(req: httpx.Request, _raw=raw) -> httpx.Response:
            return httpx.Response(200, json={"config": {"readonly": _raw}})

        async with _client(httpx.MockTransport(handler)) as c:
            assert (await c.fetch_config()).readonly is True


async def test_fetch_config_bad_json_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with _client(httpx.MockTransport(handler)) as c:
        with pytest.raises(PaisaError) as exc:
            await c.fetch_config()
        assert exc.value.code == "bad_json"


# ---------------------------------------------------------------------------
# /api/sync: success requires 2xx AND success=true
# ---------------------------------------------------------------------------


async def test_sync_journal_posts_exact_payload():
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["method"] = req.method
        captured["body"] = req.read().decode()
        return httpx.Response(200, json={"success": True})

    async with _client(httpx.MockTransport(handler)) as c:
        result = await c.sync_journal()
    assert captured["path"] == "/api/sync"
    assert captured["method"] == "POST"
    assert json.loads(captured["body"]) == SYNC_PAYLOAD  # type: ignore[arg-type]
    assert SYNC_PAYLOAD == {"journal": True, "prices": False, "portfolios": False}
    assert result.accepted is True
    assert result.reason is None


async def test_sync_200_with_success_false_is_rejected_with_reason():
    # Paisa returns HTTP 200 with {success: false, message} when the journal
    # reload itself fails. accepted must be False and the message preserved.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"success": False, "message": "failed to parse journal"}
        )

    async with _client(httpx.MockTransport(handler)) as c:
        result = await c.sync_journal()
    assert result.accepted is False
    assert result.status_code == 200
    assert result.reason == "failed to parse journal"


async def test_sync_5xx_rejected():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with _client(httpx.MockTransport(handler)) as c:
        result = await c.sync_journal()
    assert result.accepted is False
    assert result.status_code == 500


async def test_sync_non_json_rejected():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>")

    async with _client(httpx.MockTransport(handler)) as c:
        result = await c.sync_journal()
    assert isinstance(result, PaisaSyncResult)
    assert result.accepted is False


# ---------------------------------------------------------------------------
# /api/diagnosis: issues list, danger = error
# ---------------------------------------------------------------------------


async def test_diagnosis_clean_issues_is_ok():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issues": []})

    async with _client(httpx.MockTransport(handler)) as c:
        diag = await c.diagnosis()
    assert diag == PaisaDiagnosis(
        ok=True, danger_count=0, warning_count=0, issues=(), first_message=None
    )


async def test_diagnosis_danger_fails_warning_does_not():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "issues": [
                    {
                        "level": "warning",
                        "summary": "Unit Price Mismatch",
                        "description": "d",
                        "details": "x",
                    },
                    {
                        "level": "danger",
                        "summary": "Negative Balance",
                        "description": "d2",
                        "details": "Assets:Bank went negative",
                    },
                ]
            },
        )

    async with _client(httpx.MockTransport(handler)) as c:
        diag = await c.diagnosis()
    assert diag.ok is False
    assert diag.danger_count == 1
    assert diag.warning_count == 1
    assert diag.first_message == "Negative Balance"
    assert diag.issues[0].level == "warning"
    assert diag.issues[1].level == "danger"


async def test_diagnosis_warning_only_is_ok_but_counted():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "issues": [{"level": "warning", "summary": "Allocation", "details": ""}]
            },
        )

    async with _client(httpx.MockTransport(handler)) as c:
        diag = await c.diagnosis()
    assert diag.ok is True
    assert diag.warning_count == 1
    assert diag.danger_count == 0


async def test_diagnosis_missing_issues_treated_as_clean():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(httpx.MockTransport(handler)) as c:
        diag = await c.diagnosis()
    assert diag.ok is True
    assert diag.issues == ()


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------


async def test_x_auth_header_format_user_colon_pass():
    captured: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req.headers.get("X-Auth", ""))
        return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})

    async with _client(
        httpx.MockTransport(handler),
        base_url=DEFAULT_BASE_URL,
        auth_username="alice",
        auth_password="s3cret",
    ) as c:
        await c.fetch_config()
    # Paisa's TokenAuthMiddleware splits X-Auth on ":" into user:pass.
    assert captured == ["alice:s3cret"]


async def test_no_x_auth_header_when_username_blank():
    captured: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req.headers.get("X-Auth", ""))
        return httpx.Response(200, json={})

    async with _client(httpx.MockTransport(handler)) as c:
        await c.fetch_config()
    assert captured == [""]


# ---------------------------------------------------------------------------
# 429 retry
# ---------------------------------------------------------------------------


async def test_429_retries_then_succeeds_honoring_retry_after(monkeypatch):
    sleeps: list[float] = []
    statuses: list[int] = [429, 429, 200]

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 429:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"success": True})

    async with _client(httpx.MockTransport(handler)) as c:
        assert await c.ping() is True
    assert sleeps == [2.0, 2.0]


async def test_429_without_retry_after_uses_floor(monkeypatch):
    # Paisa's rate limiter returns 429 with NO Retry-After header.
    sleeps: list[float] = []
    statuses = [429, 200]

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 429:
            return httpx.Response(429, json={"error": "Too many requests"})
        return httpx.Response(200, json={"success": True})

    async with _client(httpx.MockTransport(handler)) as c:
        assert await c.ping() is True
    # Floor (0.5s) governs when Retry-After is absent.
    assert sleeps == [0.5]


async def test_429_exhausts_retries(monkeypatch):
    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "0"})

    async with _client(httpx.MockTransport(handler)) as c:
        # ping() treats the final non-2xx as not-alive rather than raising.
        assert await c.ping() is False
    # Initial attempt + MAX_429_RETRIES retries.
    assert calls["n"] == 1 + MAX_429_RETRIES


async def test_429_retry_after_capped(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    statuses = [429, 200]

    def handler(req: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 429:
            return httpx.Response(429, headers={"Retry-After": "3600"})
        return httpx.Response(200, json={"success": True})

    async with _client(httpx.MockTransport(handler)) as c:
        assert await c.ping() is True
    # Capped to MAX_RETRY_AFTER_SECONDS (30), not 3600.
    assert sleeps == [30.0]


def test_retry_after_http_date_parses():
    future = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=5)
    header = email.utils.format_datetime(future.replace(microsecond=0))
    seconds = _retry_after_seconds(header)
    assert 0 < seconds <= 10


def test_retry_after_garbage_is_zero():
    assert _retry_after_seconds(None) == 0.0
    assert _retry_after_seconds("not-a-date") == 0.0
    assert _retry_after_seconds("5") == 5.0


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


async def test_redirect_on_config_is_rejected():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"Location": "http://evil.example/x"})

    async with _client(httpx.MockTransport(handler)) as c:
        with pytest.raises(PaisaError) as exc:
            await c.fetch_config()
        assert exc.value.code == "redirect_disallowed"
