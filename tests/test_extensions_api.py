"""Tests for the /api/extensions and /api/extensions/paisa JSON surface.

Covers: extension list shape, redacted config (password never leaks), account
choices with selected flags, config-save validation + password
preserve-on-blank + encryption, all three modes, preview/generate/sync
dispatch and serialization, the no-core-writes guarantee, and optional-
extension failure isolation (monkeypatching at the service boundary, MockTransport).
"""

import datetime as dt
from decimal import Decimal

import httpx
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.config as config_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db import Base, Setting
from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.integrations.paisa import PaisaClient
from financial_dashboard.schemas.extensions import (
    PaisaConfigInput,
    PaisaConfigSaveResponse,
    PaisaGenerateResponse,
    PaisaPreviewResponse,
    PaisaStatusResponse,
    PaisaSyncResponse,
)
from financial_dashboard.services.extensions import ExtensionManager
from financial_dashboard.services.paisa import surface
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.orchestrator import (
    PreviewReport,
    ProbeReport,
    SyncReport,
)
from financial_dashboard.services.paisa.projection import ProjectionReport
from financial_dashboard.services.paisa.publisher import PublishResult
from financial_dashboard.services.paisa.renderer import LedgerDocument

pytestmark = pytest.mark.anyio

CUTOVER = dt.date(2026, 1, 1)


@pytest.fixture(scope="module", autouse=True)
def _ensure_builtins_registered():
    """Mirror the lifespan bootstrap so paisa.* settings are in SETTINGS_REGISTRY."""
    manager = ExtensionManager()
    from financial_dashboard.extensions import register_builtin_extensions

    register_builtin_extensions(manager.registry)


@pytest.fixture
async def settings_db(monkeypatch):
    """Isolated in-memory settings DB + real Fernet key for save round-trips."""
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


async def _seed_txn(session, account_id, *, date=dt.date(2026, 2, 1), amount="10.00"):
    session.add(
        Transaction(
            account_id=account_id,
            bank="hdfc",
            email_type="test_account_transaction",
            direction="debit",
            amount=Decimal(amount),
            transaction_date=date,
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()


def _mock_client(handler) -> PaisaClient:
    return PaisaClient(
        base_url="http://127.0.0.1:7500",
        transport=httpx.MockTransport(handler),
    )


# ---------------------------------------------------------------------------
# Extension list
# ---------------------------------------------------------------------------


async def test_list_extensions_shape(client):
    r = await client.get("/api/extensions")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["extensions"], list)
    ids = [e["id"] for e in body["extensions"]]
    assert "paisa" in ids
    paisa = next(e for e in body["extensions"] if e["id"] == "paisa")
    assert paisa["display_name"] == "Paisa"
    assert "projection" in paisa["capabilities"]


async def test_list_extensions_uses_app_state_manager_when_present(client):
    # The conftest client has no lifespan, so it falls back to BUILTIN_EXTENSIONS.
    # Verify the fallback path still yields paisa.
    r = await client.get("/api/extensions")
    assert r.status_code == 200
    assert any(e["id"] == "paisa" for e in r.json()["extensions"])


# ---------------------------------------------------------------------------
# Config redaction
# ---------------------------------------------------------------------------


async def test_config_never_returns_password(client):
    r = await client.get("/api/extensions/paisa/config")
    assert r.status_code == 200
    body = r.json()
    # The password field must not exist; only auth_password_set is surfaced.
    assert "auth_password" not in body
    assert "auth_password_set" in body
    assert body["auth_password_set"] is False  # default


async def test_config_shape_and_defaults(client):
    r = await client.get("/api/extensions/paisa/config")
    body = r.json()
    assert body["mode"] == "disabled"
    assert body["can_connect"] is False
    assert body["can_project"] is False
    assert body["selected_account_ids"] == []
    assert body["account_mappings"] == {}
    for key in (
        "mode",
        "base_url",
        "external_url",
        "allow_remote",
        "auth_username",
        "auth_password_set",
        "generated_path",
        "selected_account_ids",
        "project_since",
        "account_mappings",
        "category_mappings",
        "non_inr_policy",
        "request_timeout_seconds",
    ):
        assert key in body, key


# ---------------------------------------------------------------------------
# Account choices
# ---------------------------------------------------------------------------


async def test_account_choices_flags_selected(client, session):
    await _seed_bank(session, id=1)
    session.add(Account(id=2, bank="icici", label="Card", type="credit_card"))
    await session.flush()
    # Mark account 1 selected via the settings cache so config_view agrees.
    settings_mod._cache["paisa.selected_account_ids"] = "[1]"
    await session.commit()

    r = await client.get("/api/extensions/paisa/accounts")
    assert r.status_code == 200
    by_id = {a["id"]: a for a in r.json()["accounts"]}
    assert by_id[1]["selected"] is True
    assert by_id[2]["selected"] is False
    assert by_id[1]["bank"] == "hdfc"


# ---------------------------------------------------------------------------
# Config save: validation
# ---------------------------------------------------------------------------


def _valid_input(**overrides) -> PaisaConfigInput:
    base = dict(
        mode="connect",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=[],
        project_since="",
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
    )
    base.update(overrides)
    return PaisaConfigInput(**base)


async def test_save_rejects_unknown_mode(session):
    result = await surface.save_config(session, _valid_input(mode="bogus"))
    assert result.ok is False
    assert any("Mode" in e for e in result.errors)


async def test_save_rejects_invalid_base_url(session):
    result = await surface.save_config(
        session, _valid_input(base_url="ftp://example.com")
    )
    assert result.ok is False
    assert any("Base URL" in e for e in result.errors)


async def test_save_rejects_remote_http_without_https(session):
    result = await surface.save_config(
        session, _valid_input(base_url="http://10.0.0.5:7500", allow_remote=True)
    )
    assert result.ok is False
    assert any("Base URL" in e for e in result.errors)


async def test_save_rejects_non_absolute_generated_path_in_project_mode(
    session, settings_db, tmp_path
):
    target = tmp_path / "gen.journal"
    result = await surface.save_config(
        session,
        _valid_input(
            mode="project",
            generated_path="relative/path.journal",
            project_since="2026-01-01",
            selected_account_ids=[],
        ),
    )
    assert result.ok is False
    assert any("Generated Path" in e for e in result.errors)
    # Parent-missing also rejected; a real path with existing parent accepted.
    result2 = await surface.save_config(
        session,
        _valid_input(
            mode="project",
            generated_path=str(target),
            project_since="2026-01-01",
            selected_account_ids=[],
        ),
    )
    assert result2.ok is True


async def test_save_rejects_missing_cutover_in_project_mode(session, tmp_path):
    result = await surface.save_config(
        session,
        _valid_input(
            mode="project",
            generated_path=str(tmp_path / "g.journal"),
            project_since="",
            selected_account_ids=[],
        ),
    )
    assert result.ok is False
    assert any("Project Since" in e for e in result.errors)


async def test_save_rejects_bad_cutover_date(session, tmp_path):
    result = await surface.save_config(
        session,
        _valid_input(
            mode="project",
            generated_path=str(tmp_path / "g.journal"),
            project_since="not-a-date",
            selected_account_ids=[],
        ),
    )
    assert result.ok is False
    assert any("Project Since" in e for e in result.errors)


async def test_save_rejects_unknown_account_id(session):
    result = await surface.save_config(
        session, _valid_input(selected_account_ids=[9999])
    )
    assert result.ok is False
    assert any("Selected Account IDs" in e for e in result.errors)


async def test_save_rejects_invalid_ledger_name_in_mappings(session):
    result = await surface.save_config(
        session,
        _valid_input(
            account_mappings={"1": "no-colon-leaf"},  # must be a ':' hierarchy
        ),
    )
    assert result.ok is False
    assert any("Account Mappings" in e for e in result.errors)


@pytest.mark.parametrize(
    ("backend", "account_name", "category_name"),
    [
        ("ledger", "Assets:Bank:Savings Account", "Expenses:Food And Dining"),
        ("hledger", "Assets:Bank:Savings Account", "Expenses:Food And Dining"),
        ("beancount", "Assets:Bank:SavingsAccount", "Expenses:FoodAndDining"),
    ],
)
async def test_api_save_accepts_backend_valid_operator_mappings(
    client, settings_db, backend, account_name, category_name
):
    payload = _valid_input(
        ledger_cli=backend,
        account_mappings={"1": account_name},
        category_mappings={"groceries": category_name},
    ).model_dump()

    response = await client.post("/api/extensions/paisa/config", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["config"]["ledger_cli"] == backend
    assert body["config"]["account_mappings"] == {"1": account_name}
    assert body["config"]["category_mappings"] == {"groceries": category_name}


async def test_api_beancount_rejects_ledger_valid_mapping_without_partial_save(
    client, settings_db
):
    ledger_mapping = "Assets:Bank:Savings Account"
    initial = _valid_input(
        ledger_cli="ledger",
        auth_username="before",
        account_mappings={"1": ledger_mapping},
    ).model_dump()
    saved = await client.post("/api/extensions/paisa/config", json=initial)
    assert saved.json()["ok"] is True

    invalid = _valid_input(
        ledger_cli="beancount",
        auth_username="must-not-save",
        account_mappings={"1": ledger_mapping},
    ).model_dump()
    response = await client.post("/api/extensions/paisa/config", json=invalid)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert any(
        "Account Mappings" in error and "must not contain spaces" in error
        for error in body["errors"]
    )
    assert settings_mod._cache["paisa.ledger_cli"] == "ledger"
    assert settings_mod._cache["paisa.auth_username"] == "before"
    assert settings_mod._cache["paisa.account_mappings"] == (
        '{"1": "Assets:Bank:Savings Account"}'
    )


async def test_save_rejects_invalid_external_url_scheme(session):
    result = await surface.save_config(
        session, _valid_input(external_url="javascript:alert(1)")
    )
    assert result.ok is False
    assert any("External URL" in e for e in result.errors)


async def test_save_rejects_zero_timeout(session):
    result = await surface.save_config(
        session,
        _valid_input(request_timeout_seconds=0),
    )
    assert result.ok is False
    assert any("Request Timeout" in e for e in result.errors)


async def test_save_does_not_persist_on_validation_error(session, settings_db):
    before = dict(settings_mod._cache)
    result = await surface.save_config(session, _valid_input(mode="bogus"))
    assert result.ok is False
    # No setting row was written.
    async with settings_db() as s:
        rows = (await s.execute(select(Setting))).scalars().all()
    assert rows == []
    assert settings_mod._cache == before


# ---------------------------------------------------------------------------
# Config save: persistence, redaction, password preserve-on-blank
# ---------------------------------------------------------------------------


async def test_save_persists_and_redacts_password(session, settings_db):
    await _seed_bank(session, id=1)
    result = await surface.save_config(
        session,
        _valid_input(
            mode="connect",
            auth_password="s3cret-paisa",
            selected_account_ids=[1],
            base_url="http://127.0.0.1:7500",
        ),
    )
    assert isinstance(result, PaisaConfigSaveResponse)
    assert result.ok is True
    # Response never carries the plaintext password.
    dumped = result.config.model_dump()
    assert "auth_password" not in dumped
    assert dumped["auth_password_set"] is True

    # At rest it is encrypted.
    async with settings_db() as s:
        row = await s.get(Setting, "paisa.auth_password")
    assert row is not None
    assert row.value != "s3cret-paisa"
    assert row.value != ""


async def test_save_blank_password_preserves_current_secret(session, settings_db):
    await _seed_bank(session, id=1)
    first = await surface.save_config(
        session,
        _valid_input(auth_password="original", selected_account_ids=[1]),
    )
    assert first.ok is True

    # Second save with a blank password and an unrelated change must keep the
    # current secret and apply the other change.
    second = await surface.save_config(
        session,
        _valid_input(
            auth_password="",
            auth_username="alice",
            selected_account_ids=[1],
        ),
    )
    assert second.ok is True
    assert second.config.auth_password_set is True
    assert second.config.auth_username == "alice"

    # The stored (encrypted) value round-trips to the original secret.
    from financial_dashboard.config import get_fernet

    async with settings_db() as s:
        row = await s.get(Setting, "paisa.auth_password")
    assert get_fernet().decrypt(row.value.encode()).decode() == "original"


async def test_save_via_api_route_redacts_and_redirects_cleanly(
    client, session, settings_db
):
    await _seed_bank(session, id=1)
    payload = _valid_input(
        mode="connect",
        auth_password="via-api",
        selected_account_ids=[1],
    ).model_dump()
    r = await client.post("/api/extensions/paisa/config", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "auth_password" not in body["config"]
    assert body["config"]["auth_password_set"] is True


async def test_save_via_api_route_returns_typed_errors(client, session):
    # Invalid mode → typed ok=False body with errors, not a 422/500.
    r = await client.post(
        "/api/extensions/paisa/config",
        json=_valid_input(mode="nope").model_dump(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert isinstance(body["errors"], list) and body["errors"]


# ---------------------------------------------------------------------------
# Status / probe (all modes + failure isolation)
# ---------------------------------------------------------------------------


async def test_status_disabled_mode(client, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="disabled"))
    r = await client.get("/api/extensions/paisa/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["reachable"] is False
    assert body["reason"] == "disabled"
    assert body["can_connect"] is False


async def test_status_connect_mode_serializes_probe(client, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="connect"))

    async def fake_probe(cfg, *, client=None):
        return ProbeReport(
            ok=True,
            reachable=True,
            capabilities=None,
            diagnosis=None,
            reason=None,
        )

    monkeypatch.setattr(surface, "probe", fake_probe)
    r = await client.get("/api/extensions/paisa/status")
    body = r.json()
    assert body["ok"] is True
    assert body["reachable"] is True
    assert body["mode"] == "connect"
    assert body["can_connect"] is True
    assert body["can_project"] is False


async def test_status_probe_with_mock_transport_serializes_capabilities(monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="project"))

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(
                200, json={"config": {"ledger_cli": "ledger", "readonly": False}}
            )
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    status = await surface.probe_status(client=_mock_client(handler))
    assert isinstance(status, PaisaStatusResponse)
    assert status.ok is True
    assert status.capabilities.ledger_cli == "ledger"
    assert status.diagnosis.ok is True


async def test_status_failure_isolation_returns_typed_body(client, monkeypatch):
    async def boom():
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(surface, "probe_status", boom)
    r = await client.get("/api/extensions/paisa/status")
    # Optional-extension isolation: typed JSON body, not a 500.
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "paisa_status_failed"


# ---------------------------------------------------------------------------
# Preview dispatch (mode gating + serialization)
# ---------------------------------------------------------------------------


async def test_preview_disabled_via_route(client, session, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="disabled"))
    r = await client.post("/api/extensions/paisa/preview")
    body = r.json()
    assert body["ok"] is False
    assert body["reason"] == "disabled"
    assert body["mode"] == "disabled"


async def test_preview_connect_blocked_via_route(client, session, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="connect"))
    r = await client.post("/api/extensions/paisa/preview")
    body = r.json()
    assert body["ok"] is False
    assert body["reason"] == "connect_only"


async def test_preview_project_returns_summary(client, session, monkeypatch):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    monkeypatch.setattr(
        surface, "load_config", lambda: _config(selected_account_ids=(1,))
    )
    r = await client.post("/api/extensions/paisa/preview")
    body = r.json()
    assert body["ok"] is True
    assert body["summary"]["emitted_count"] == 1
    assert body["journal"] is not None
    assert "txn:1" in body["journal"]


async def test_preview_failure_isolation(client, monkeypatch):
    async def boom(session):
        raise RuntimeError("preview blew up")

    monkeypatch.setattr(surface, "preview_projection", boom)
    r = await client.post("/api/extensions/paisa/preview")
    assert r.status_code == 503
    body = r.json()
    assert body["detail"]["error"] == "paisa_preview_failed"


# ---------------------------------------------------------------------------
# Generate dispatch (project mode, no core writes)
# ---------------------------------------------------------------------------


async def test_generate_writes_file_and_no_core_writes(
    client, session, tmp_path, monkeypatch
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(selected_account_ids=(1,), generated_path=str(target)),
    )

    txn_before = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    acct_before = [
        a.id for a in (await session.execute(select(Account))).scalars().all()
    ]

    r = await client.post("/api/extensions/paisa/generate")
    body = r.json()
    assert body["ok"] is True
    assert body["publish"]["published"] is True
    assert target.exists()
    assert "; txn:1" in target.read_text()

    txn_after = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    acct_after = [
        a.id for a in (await session.execute(select(Account))).scalars().all()
    ]
    assert txn_before == txn_after
    assert acct_before == acct_after


async def test_generate_connect_blocked(client, session, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="connect"))
    r = await client.post("/api/extensions/paisa/generate")
    body = r.json()
    assert body["ok"] is False
    assert body["reason"] == "connect_only"


async def test_generate_disabled_blocked(client, session, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="disabled"))
    r = await client.post("/api/extensions/paisa/generate")
    body = r.json()
    assert body["ok"] is False
    assert body["reason"] == "disabled"


# ---------------------------------------------------------------------------
# Sync dispatch (project mode, mock transport, no core writes)
# ---------------------------------------------------------------------------


async def test_sync_happy_path_via_mock_transport(
    client, session, tmp_path, monkeypatch
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(selected_account_ids=(1,), generated_path=str(target)),
    )

    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
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
        lambda cfg: _mock_client(handler),
    )

    r = await client.post("/api/extensions/paisa/sync")
    body = r.json()
    assert body["ok"] is True
    assert body["outcome"] == "synced"
    assert body["diagnosis_ok"] is True
    assert target.exists()
    assert seen == ["/api/config", "/api/sync", "/api/diagnosis"]


async def test_sync_never_mutates_core_rows(client, session, tmp_path, monkeypatch):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    monkeypatch.setattr(
        surface,
        "load_config",
        lambda: _config(
            selected_account_ids=(1,), generated_path=str(tmp_path / "g.journal")
        ),
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
        orchestrator, "_build_client", lambda cfg: _mock_client(handler)
    )

    txn_before = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    r = await client.post("/api/extensions/paisa/sync")
    assert r.json()["ok"] is True
    txn_after = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    assert txn_before == txn_after


async def test_sync_connect_blocked(client, session, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="connect"))
    r = await client.post("/api/extensions/paisa/sync")
    body = r.json()
    assert body["ok"] is False
    assert body["outcome"] == "connect_only"


async def test_sync_failure_isolation(client, monkeypatch):
    async def boom(session, *, client=None):
        raise RuntimeError("sync blew up")

    monkeypatch.setattr(surface, "sync_now", boom)
    r = await client.post("/api/extensions/paisa/sync")
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "paisa_sync_failed"


# ---------------------------------------------------------------------------
# Surface serialization unit checks (report → DTO)
# ---------------------------------------------------------------------------


async def test_preview_projection_serializes_not_configured(monkeypatch):
    # No selected accounts → ready_to_project is False → not_configured, no DB.
    monkeypatch.setattr(
        surface, "load_config", lambda: _config(selected_account_ids=())
    )
    out = await surface.preview_projection(session=object())  # type: ignore[arg-type]
    assert isinstance(out, PaisaPreviewResponse)
    assert out.ok is False
    assert out.reason == "not_configured"


async def test_generate_now_serializes_publish(monkeypatch, tmp_path):
    target = tmp_path / "g.journal"
    doc = LedgerDocument(
        cutover_date=CUTOVER, openings=(), entries=(), accounts_declared=()
    )
    report = ProjectionReport(
        journal="",
        document=doc,
        entries=(),
        openings=(),
        emitted_count=0,
        self_transfer_pairs=0,
        card_payments=0,
        card_side_payments=0,
        non_inr_count=0,
        unmatched_count=0,
        unknown_count=0,
        skipped=(),
        cutover_date=CUTOVER,
        account_ids=(),
        cas_portfolio_count=1,
        cas_portfolio_labels=("PAN-SCOPE",),
        cas_investment_scope="excluded",
        manual_asset_count=1,
        manual_asset_labels=("9: Private Property",),
        net_worth_scope_complete=False,
    )

    # generate() is the orchestrator's own; its internals (preview, publish)
    # resolve against the orchestrator module, so monkeypatch them there.
    from financial_dashboard.services.paisa import orchestrator

    async def fake_preview(session, cfg):
        return PreviewReport(ok=True, report=report, reason=None)

    def fake_publish(path, body):
        return PublishResult(
            published=True, path=path, version="1", body_hash="abc", bytes_written=10
        )

    monkeypatch.setattr(
        surface, "load_config", lambda: _config(generated_path=str(target))
    )
    monkeypatch.setattr(orchestrator, "preview", fake_preview)
    monkeypatch.setattr(orchestrator, "publish_journal", fake_publish)
    out = await surface.generate_now(session=object())  # type: ignore[arg-type]
    assert isinstance(out, PaisaGenerateResponse)
    assert out.ok is True
    assert out.publish.skipped is False
    assert out.publish.body_hash == "abc"
    assert out.summary.cas_portfolio_count == 1
    assert out.summary.cas_investment_scope == "excluded"
    assert out.summary.manual_asset_labels == ["9: Private Property"]
    assert out.summary.net_worth_scope_complete is False


async def test_sync_now_serializes_outcome(monkeypatch):
    async def fake_manual_sync(session, cfg, *, client=None):
        return SyncReport(
            ok=False,
            outcome="readonly",
            preview=None,
            publish=None,
            diagnosis_ok=None,
            reason="readonly",
        )

    monkeypatch.setattr(surface, "load_config", lambda: _config())
    monkeypatch.setattr(surface, "manual_sync", fake_manual_sync)
    out = await surface.sync_now(session=object())  # type: ignore[arg-type]
    assert isinstance(out, PaisaSyncResponse)
    assert out.ok is False
    assert out.outcome == "readonly"


# ---------------------------------------------------------------------------
# Manual single-flight lease: busy when held, acquire-then-release otherwise
# ---------------------------------------------------------------------------


async def test_manual_sync_returns_busy_when_lease_held(session, monkeypatch):
    """When the singleton lease is held, a manual sync waits then returns the
    additive ``busy`` outcome instead of overlapping."""
    from financial_dashboard.services.paisa import coordinator as coord_mod
    from financial_dashboard.services.paisa.sync_state import (
        claim_lease,
        ensure_sync_state,
    )

    # Project config so the surface doesn't refuse on mode; sync_now is never
    # reached because the lease wait elapses first.
    monkeypatch.setattr(surface, "load_config", lambda: _config())

    await ensure_sync_state(session)
    await claim_lease(session, owner="other")  # held by someone else
    await session.commit()

    # sync_now must NOT run when busy.
    async def boom(s, *, client=None):
        raise AssertionError("sync_now must not run when lease held")

    monkeypatch.setattr(surface, "sync_now", boom)

    # Shrink the manual wait window so the test is fast.
    original = coord_mod.claim_manual_lease

    async def fast_claim(s, **kw):
        kw["wait_seconds"] = 0.1
        kw["poll_seconds"] = 0.02
        return await original(s, **kw)

    monkeypatch.setattr(surface, "claim_manual_lease", fast_claim)

    out = await surface.sync_now_audited(session)
    assert isinstance(out, PaisaSyncResponse)
    assert out.busy is True
    assert out.outcome == "busy"
    assert out.ok is False
