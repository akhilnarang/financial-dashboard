"""Tests for the /extensions and /extensions/paisa HTML surface.

Covers: nav entry, extensions index page, the Paisa configuration page context
(config, account picker, preview, safe link, setup include line), PRG config
save (valid → 303 redirect, invalid → 422 re-render), generate/sync PRG
actions, safe-link gating, and optional-extension failure isolation. Dispatch
is verified by monkeypatching at the service boundary.
"""

import datetime as dt
import html
import re
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.config as config_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db import Base, Setting
from financial_dashboard.db.models import Account, ManualItem, Transaction
from financial_dashboard.integrations.paisa import PaisaClient
from financial_dashboard.services.extensions import ExtensionManager
from financial_dashboard.services.paisa import surface
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.orchestrator import SyncReport
from financial_dashboard.web.extensions import _flash

pytestmark = pytest.mark.anyio

CUTOVER = dt.date(2026, 1, 1)


@pytest.fixture(scope="module", autouse=True)
def _ensure_builtins_registered():
    from financial_dashboard.extensions import register_builtin_extensions

    manager = ExtensionManager()
    register_builtin_extensions(manager.registry)


@pytest.fixture
async def settings_db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(settings_mod, "async_session", maker)
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(config_mod.settings, "email_source_master_key", key)
    monkeypatch.setattr(config_mod, "_fernet_instance", None)
    yield maker
    await engine.dispose()


def _config(**overrides) -> PaisaProjectionConfig:
    base = dict(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=(1,),
        cutover_date=CUTOVER,
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
    )
    base.update(overrides)
    return PaisaProjectionConfig(**base)


async def _seed_bank(session, *, id=1):
    session.add(Account(id=id, bank="hdfc", label="Savings", type="bank_account"))
    await session.flush()


async def _seed_txn(session, account_id):
    session.add(
        Transaction(
            account_id=account_id,
            bank="hdfc",
            email_type="test_account_transaction",
            direction="debit",
            amount=Decimal("10.00"),
            transaction_date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()


# ---------------------------------------------------------------------------
# Nav + index
# ---------------------------------------------------------------------------


async def test_base_nav_has_extensions_entry(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert 'href="/extensions"' in r.text


async def test_extensions_index_lists_paisa(client):
    r = await client.get("/extensions")
    assert r.status_code == 200
    assert "Paisa" in r.text
    assert 'href="/extensions/paisa"' in r.text
    assert 'aria-current="page"' in r.text  # active_page = extensions


# ---------------------------------------------------------------------------
# Paisa page render
# ---------------------------------------------------------------------------


async def test_paisa_page_renders_config_accounts_preview(client, session):
    await _seed_bank(session, id=1)
    session.add(Account(id=2, bank="icici", label="Card", type="credit_card"))
    await session.flush()
    await session.commit()

    r = await client.get("/extensions/paisa")
    assert r.status_code == 200
    text = r.text
    # Connection form
    assert 'name="mode"' in text
    assert 'name="base_url"' in text
    assert 'name="generated_path"' in text
    assert 'name="project_since"' in text
    # Account picker shows both accounts
    assert "hdfc" in text
    assert "icici" in text
    # Mappings editors
    assert 'name="account_mapping_key"' in text
    assert 'name="category_mapping_key"' in text
    # Setup include line + actions + safe-link area
    assert "include " in text
    assert "/extensions/paisa/generate" in text
    assert "/extensions/paisa/sync" in text
    # The password field is present but never carries a value (redacted).
    assert 'name="auth_password"' in text
    assert 'value="s3cret"' not in text


async def test_paisa_page_prominently_names_incomplete_networth_scope(
    client, session, monkeypatch
):
    await _seed_bank(session, id=1)
    session.add(
        ManualItem(
            id=9,
            name="Private Property",
            kind="asset",
            category="real_estate",
            active=True,
        )
    )
    await session.commit()
    monkeypatch.setattr(surface, "load_config", lambda: _config())

    response = await client.get("/extensions/paisa")

    assert response.status_code == 200
    assert "Net-worth projection scope" in response.text
    assert "Incomplete — Paisa is not a full native net-worth view." in response.text
    assert "1 outside projection" in response.text
    assert "9: Private Property" in response.text


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            "ledger",
            'include /tmp/C:\\Users\\Analyst\\new "Q1" <unsafe>&.journal',
        ),
        (
            "hledger",
            'include /tmp/C:\\Users\\Analyst\\new "Q1" <unsafe>&.journal',
        ),
        (
            "beancount",
            'include "/tmp/C:\\\\Users\\\\Analyst\\\\new \\"Q1\\" <unsafe>&.journal"',
        ),
    ],
)
async def test_paisa_setup_include_uses_backend_syntax_and_html_escaping(
    client, monkeypatch, backend, expected
):
    generated_path = '/tmp/C:\\Users\\Analyst\\new "Q1" <unsafe>&.journal'
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(
            mode="disabled", generated_path=generated_path, ledger_cli=backend
        ),
    )

    response = await client.get("/extensions/paisa")

    assert response.status_code == 200
    match = re.search(
        r'<pre class="journal-pre" id="paisa-include-instruction"[^>]*>(.*?)</pre>',
        response.text,
        flags=re.DOTALL,
    )
    assert match is not None
    assert html.unescape(match.group(1)) == expected
    assert "<unsafe>" not in match.group(1)


async def test_paisa_setup_include_switches_with_dom_text_only(client):
    response = await client.get("/extensions/paisa")

    assert response.status_code == 200
    assert "ledgerCli.value === 'beancount'" in response.text
    assert "includeLine.textContent = 'include ' + renderedPath" in response.text
    assert "includeLine.innerHTML" not in response.text
    assert (
        "document.getElementById('paisa-ledger-cli').addEventListener('change', "
        "updateSetupInclude)" in response.text
    )


async def test_paisa_page_shows_selected_accounts_checked(client, session):
    await _seed_bank(session, id=1)
    session.add(Account(id=2, bank="icici", label="Card", type="credit_card"))
    await session.flush()
    settings_mod._cache["paisa.selected_account_ids"] = "[2]"
    await session.commit()

    r = await client.get("/extensions/paisa")
    # Account 2 checkbox is checked; account 1 is not.
    assert 'value="2" checked' in r.text
    assert 'value="1" checked' not in r.text


async def test_paisa_page_renders_existing_mappings(client, session):
    settings_mod._cache["paisa.account_mappings"] = '{"1": "Assets:Bank:HDFC:Main"}'
    settings_mod._cache["paisa.category_mappings"] = '{"groceries": "Expenses:Food"}'
    r = await client.get("/extensions/paisa")
    assert "Assets:Bank:HDFC:Main" in r.text
    assert "Expenses:Food" in r.text


async def test_paisa_page_shows_preview_diagnostics_in_project_mode(
    client, session, monkeypatch
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    monkeypatch.setattr(
        surface, "load_config", lambda: _config(selected_account_ids=(1,))
    )
    r = await client.get("/extensions/paisa")
    assert "Preview Diagnostics" in r.text
    assert "1 entry" in r.text or "1 entries" in r.text


async def test_paisa_page_shows_unavailable_preview_when_disabled(client, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="disabled"))
    r = await client.get("/extensions/paisa")
    assert "Preview unavailable" in r.text
    assert "disabled" in r.text


# ---------------------------------------------------------------------------
# Safe external deep link
# ---------------------------------------------------------------------------


async def test_safe_link_falls_back_to_base_url(monkeypatch):
    # external_url blank → the connection base URL (a valid http URL) is used.
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(external_url="", base_url="http://127.0.0.1:7500"),
    )
    assert surface.safe_link() == "http://127.0.0.1:7500"


async def test_safe_link_returns_external_url(monkeypatch):
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(external_url="https://paisa.example.com/"),
    )
    assert surface.safe_link() == "https://paisa.example.com/"


async def test_safe_link_rejects_javascript(monkeypatch):
    # Both candidates disallowed → no link rendered.
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(
            external_url="javascript:alert(1)", base_url="javascript:alert(2)"
        ),
    )
    assert surface.safe_link() == ""


async def test_paisa_page_renders_open_link_when_safe(client, monkeypatch):
    monkeypatch.setattr(surface, "safe_link", lambda: "https://paisa.example.com/")
    r = await client.get("/extensions/paisa")
    assert 'href="https://paisa.example.com/"' in r.text
    assert 'target="_blank"' in r.text
    assert 'rel="noopener"' in r.text


async def test_paisa_page_omits_open_link_when_unsafe(client, monkeypatch):
    monkeypatch.setattr(surface, "safe_link", lambda: "")
    r = await client.get("/extensions/paisa")
    assert "Open Paisa" not in r.text


# ---------------------------------------------------------------------------
# PRG config save
# ---------------------------------------------------------------------------


async def test_config_save_valid_redirects_with_flash(client, session, settings_db):
    await _seed_bank(session, id=1)
    await session.commit()
    form = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "external_url": "",
        "allow_remote": "",
        "auth_username": "",
        "auth_password": "new-secret",
        "generated_path": "",
        "project_since": "",
        "request_timeout_seconds": "15",
        "non_inr_policy": "skip",
        "selected_account_ids": ["1"],
        "account_mapping_key": [""],
        "account_mapping_value": [""],
        "category_mapping_key": [""],
        "category_mapping_value": [""],
    }
    r = await client.post("/extensions/paisa", data=form, follow_redirects=False)
    assert r.status_code == 303
    assert "saved=1" in r.headers["location"]


async def test_config_save_invalid_rerenders_with_errors(client, session):
    await _seed_bank(session, id=1)
    await session.commit()
    form = {
        "mode": "bogus",
        "base_url": "ftp://nope",
        "request_timeout_seconds": "15",
        "selected_account_ids": [],
        "account_mapping_key": [""],
        "account_mapping_value": [""],
        "category_mapping_key": [""],
        "category_mapping_value": [""],
    }
    r = await client.post("/extensions/paisa", data=form, follow_redirects=False)
    # 422 re-render with validation errors (not a redirect, not a 500).
    assert r.status_code == 422
    assert "Validation errors" in r.text
    assert "Mode" in r.text


async def test_config_save_mappings_parsed_from_rows(client, session, settings_db):
    await _seed_bank(session, id=1)
    await session.commit()
    form = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "request_timeout_seconds": "15",
        "selected_account_ids": ["1"],
        "account_mapping_key": ["1", ""],
        "account_mapping_value": ["Assets:Bank:HDFC:Main", ""],
        "category_mapping_key": ["groceries"],
        "category_mapping_value": ["Expenses:Food"],
    }
    r = await client.post("/extensions/paisa", data=form, follow_redirects=False)
    assert r.status_code == 303
    assert settings_mod._cache.get("paisa.account_mappings") == (
        '{"1": "Assets:Bank:HDFC:Main"}'
    )
    assert (
        settings_mod._cache.get("paisa.category_mappings")
        == '{"groceries": "Expenses:Food"}'
    )


@pytest.mark.parametrize("backend", ["ledger", "hledger"])
async def test_web_save_accepts_ledger_family_mapping_spaces(
    client, settings_db, backend
):
    form = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "request_timeout_seconds": "15",
        "ledger_cli": backend,
        "account_mapping_key": ["1"],
        "account_mapping_value": ["Assets:Bank:Savings Account"],
        "category_mapping_key": ["groceries"],
        "category_mapping_value": ["Expenses:Food And Dining"],
    }

    response = await client.post("/extensions/paisa", data=form, follow_redirects=False)

    assert response.status_code == 303
    assert settings_mod._cache["paisa.ledger_cli"] == backend
    assert settings_mod._cache["paisa.account_mappings"] == (
        '{"1": "Assets:Bank:Savings Account"}'
    )


@pytest.mark.parametrize(
    ("account_name", "error_detail"),
    [
        ("Assets:Bank:Savings Account", "must not contain spaces"),
        ("Asset:Bank:SavingsAccount", "beancount root"),
    ],
)
async def test_web_save_rejects_beancount_invalid_account_mapping(
    client, settings_db, account_name, error_detail
):
    form = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "auth_username": "must-not-save",
        "request_timeout_seconds": "15",
        "ledger_cli": "beancount",
        "account_mapping_key": ["1"],
        "account_mapping_value": [account_name],
        "category_mapping_key": ["groceries"],
        "category_mapping_value": ["Expenses:FoodAndDining"],
    }

    response = await client.post("/extensions/paisa", data=form, follow_redirects=False)

    assert response.status_code == 422
    assert "Account Mappings" in response.text
    assert error_detail in response.text
    assert "paisa.ledger_cli" not in settings_mod._cache
    assert "paisa.auth_username" not in settings_mod._cache


async def test_web_save_accepts_valid_beancount_operator_mappings(client, settings_db):
    form = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "request_timeout_seconds": "15",
        "ledger_cli": "beancount",
        "account_mapping_key": ["1"],
        "account_mapping_value": ["Assets:Bank:SavingsAccount"],
        "category_mapping_key": ["groceries"],
        "category_mapping_value": ["Expenses:FoodAndDining"],
    }

    response = await client.post("/extensions/paisa", data=form, follow_redirects=False)

    assert response.status_code == 303
    assert settings_mod._cache["paisa.ledger_cli"] == "beancount"
    assert settings_mod._cache["paisa.account_mappings"] == (
        '{"1": "Assets:Bank:SavingsAccount"}'
    )
    assert settings_mod._cache["paisa.category_mappings"] == (
        '{"groceries": "Expenses:FoodAndDining"}'
    )


async def test_config_save_project_investments_checkbox(client, session, settings_db):
    """The project_investments checkbox maps to the setting: checked → 'true',
    absent → 'false'."""
    await _seed_bank(session, id=1)
    await session.commit()
    form = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "request_timeout_seconds": "15",
        "selected_account_ids": ["1"],
        "project_investments": "true",
    }
    r = await client.post("/extensions/paisa", data=form, follow_redirects=False)
    assert r.status_code == 303
    assert settings_mod._cache.get("paisa.project_investments") == "true"

    # Absent checkbox → 'false'.
    form.pop("project_investments")
    r2 = await client.post("/extensions/paisa", data=form, follow_redirects=False)
    assert r2.status_code == 303
    assert settings_mod._cache.get("paisa.project_investments") == "false"


async def test_config_save_blank_password_preserves_secret(
    client, session, settings_db
):
    from financial_dashboard.config import get_fernet

    await _seed_bank(session, id=1)
    await session.commit()
    # First save sets a secret.
    form1 = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "request_timeout_seconds": "15",
        "selected_account_ids": ["1"],
        "auth_password": "first-secret",
        "account_mapping_key": [""],
        "account_mapping_value": [""],
        "category_mapping_key": [""],
        "category_mapping_value": [""],
    }
    await client.post("/extensions/paisa", data=form1, follow_redirects=False)
    async with settings_db() as s:
        row1 = await s.get(Setting, "paisa.auth_password")
    assert get_fernet().decrypt(row1.value.encode()).decode() == "first-secret"

    # Second save with blank password + an unrelated change keeps the secret.
    form2 = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "auth_username": "alice",
        "request_timeout_seconds": "15",
        "selected_account_ids": ["1"],
        "auth_password": "",
        "account_mapping_key": [""],
        "account_mapping_value": [""],
        "category_mapping_key": [""],
        "category_mapping_value": [""],
    }
    await client.post("/extensions/paisa", data=form2, follow_redirects=False)
    async with settings_db() as s:
        row2 = await s.get(Setting, "paisa.auth_password")
    assert get_fernet().decrypt(row2.value.encode()).decode() == "first-secret"
    assert settings_mod._cache.get("paisa.auth_username") == "alice"


# ---------------------------------------------------------------------------
# Generate / sync PRG actions
# ---------------------------------------------------------------------------


async def test_generate_action_redirects_with_flash(
    client, session, tmp_path, monkeypatch
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    await session.commit()
    target = tmp_path / "gen.journal"
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(selected_account_ids=(1,), generated_path=str(target)),
    )
    r = await client.post("/extensions/paisa/generate", follow_redirects=False)
    assert r.status_code == 303
    assert "generated=1" in r.headers["location"]
    assert target.exists()


async def test_generate_action_blocked_redirects_with_reason(
    client, session, monkeypatch
):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="connect"))
    r = await client.post("/extensions/paisa/generate", follow_redirects=False)
    assert r.status_code == 303
    assert "error=" in r.headers["location"]


async def test_sync_action_redirects_with_synced(
    client, session, tmp_path, monkeypatch
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    await session.commit()
    target = tmp_path / "gen.journal"
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(selected_account_ids=(1,), generated_path=str(target)),
    )

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    from financial_dashboard.services.paisa import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "_build_client",
        lambda cfg: PaisaClient(
            base_url="http://127.0.0.1:7500", transport=httpx.MockTransport(handler)
        ),
    )
    r = await client.post("/extensions/paisa/sync", follow_redirects=False)
    assert r.status_code == 303
    assert "synced=1" in r.headers["location"]


async def test_sync_action_failure_redirects_with_outcome(client, session, monkeypatch):
    async def fake_sync(session, cfg, *, client=None):
        return SyncReport(
            ok=False,
            outcome="readonly",
            preview=None,
            publish=None,
            diagnosis_ok=None,
            reason="readonly",
        )

    monkeypatch.setattr(surface, "load_config", lambda: _config())
    monkeypatch.setattr(surface, "manual_sync", fake_sync)
    r = await client.post("/extensions/paisa/sync", follow_redirects=False)
    assert r.status_code == 303
    assert "outcome=readonly" in r.headers["location"]


async def test_generate_action_isolation_on_exception(client, session, monkeypatch):
    async def boom(session):
        raise RuntimeError("generate blew up")

    monkeypatch.setattr(surface, "generate_now", boom)
    r = await client.post("/extensions/paisa/generate", follow_redirects=False)
    # Optional-extension isolation: PRG redirect with an error flash, not a 500.
    assert r.status_code == 303
    assert "error=" in r.headers["location"]


async def test_sync_action_isolation_on_exception(client, monkeypatch):
    async def boom(session, *, client=None):
        raise RuntimeError("sync blew up")

    monkeypatch.setattr(surface, "sync_now", boom)
    r = await client.post("/extensions/paisa/sync", follow_redirects=False)
    assert r.status_code == 303
    assert "error=" in r.headers["location"]


# ---------------------------------------------------------------------------
# Flash query URL-encoding
# ---------------------------------------------------------------------------


def test_flash_url_encodes_special_chars_and_unicode():
    """_flash percent-encodes the value so spaces, &, #, and Unicode cannot
    split the Location into extra params or inject a fragment."""
    resp = _flash(
        RedirectResponse(url="/extensions/paisa", status_code=303),
        "error",
        "oops #1 & 2 <script> ñ",
    )
    location = resp.headers["location"]
    assert location.startswith("/extensions/paisa?")
    # Only one '?'; the value's '#' did not start a fragment.
    assert location.count("?") == 1
    assert "#" not in location
    query = location.split("?", 1)[1]
    # Raw special chars are absent from the encoded query.
    assert " " not in query
    assert "&oops" not in query  # the '&' is encoded, not a new param
    assert "<script>" not in query
    assert "ñ" not in query
    # Round-trips through standard query decoding.
    assert parse_qs(urlsplit(location).query)["error"] == ["oops #1 & 2 <script> ñ"]


def test_flash_appends_with_ampersand_when_query_present():
    """A Location that already has a query gets an ``&``-joined flash, still
    URL-encoded."""
    resp = RedirectResponse(url="/extensions/paisa?saved=1", status_code=303)
    _flash(resp, "error", "a&b")
    location = resp.headers["location"]
    assert location.startswith("/extensions/paisa?saved=1&")
    # The '&' inside the value is encoded; only the real join '&' is literal.
    assert location == "/extensions/paisa?saved=1&error=a%26b"
    assert parse_qs(urlsplit(location).query) == {"saved": ["1"], "error": ["a&b"]}


def test_short_error_preserves_unicode_and_round_trips_through_flash():
    """_short_error keeps Unicode (₹, ñ) as real characters via
    ``ensure_ascii=False`` rather than ``\\uXXXX`` escapes, and the value
    round-trips through _flash's URL-encoding — which remains the safety
    boundary that keeps those characters from splitting the Location header."""
    from financial_dashboard.web.extensions import _short_error

    err = _short_error(RuntimeError("sync failed: \u20b9 debit \u00f1"))
    # Unicode survives as real characters, not ASCII escapes.
    assert "\u20b9" in err  # ₹
    assert "\u00f1" in err  # ñ
    assert "\\u" not in err
    # Round-trip through _flash's URL-encoding (the safety boundary).
    resp = _flash(
        RedirectResponse(url="/extensions/paisa", status_code=303), "error", err
    )
    location = resp.headers["location"]
    # No raw Unicode/special chars leak into the Location unencoded.
    assert "\u20b9" not in location
    assert "\u00f1" not in location
    assert "#" not in location
    decoded = parse_qs(urlsplit(location).query)["error"][0]
    assert decoded == err


async def test_generate_action_error_flash_is_url_encoded(client, monkeypatch):
    """An error message containing spaces, &, #, and Unicode must reach the
    Location header fully encoded — never as raw chars that could split params
    or start a fragment."""
    msg = "boom #1 & 2 <img> ñ"

    async def boom(session):
        raise RuntimeError(msg)

    monkeypatch.setattr(surface, "generate_now", boom)
    r = await client.post("/extensions/paisa/generate", follow_redirects=False)
    assert r.status_code == 303
    location = r.headers["location"]
    # The error text did not inject a fragment or a stray param.
    assert location.count("?") == 1
    assert "#" not in location
    query = location.split("?", 1)[1]
    assert "&" not in query.split("error=", 1)[1]  # value's '&' is encoded
    assert " " not in query
    assert "<img>" not in query


# ---------------------------------------------------------------------------
# Status badge: no innerHTML for JSON-sourced content
# ---------------------------------------------------------------------------


async def test_paisa_status_badge_never_uses_innerhtml(client):
    """The status badge is built via DOM APIs, never innerHTML, so a
    misconfigured/hostile upstream string (mode/label/reason) cannot inject
    markup into the page."""
    r = await client.get("/extensions/paisa")
    assert r.status_code == 200
    text = r.text
    # The status badge path assigns via DOM construction, not innerHTML.
    assert "badge.innerHTML" not in text
    # Positive assertions: the safe primitives are present.
    assert "document.createElement" in text
    assert "classList" in text
    assert "replaceChildren" in text
    assert "createTextNode" in text


async def test_paisa_status_script_has_no_markup_injection_surface(client):
    """The status fetch builds text from JSON fields into text nodes / textContent,
    never string-concatenated HTML. The status detail uses textContent (a plain
    string assignment), which cannot parse markup."""
    r = await client.get("/extensions/paisa")
    text = r.text
    # detail (capabilities/reason/diagnosis digest) is a textContent assignment.
    assert "detail.textContent" in text
