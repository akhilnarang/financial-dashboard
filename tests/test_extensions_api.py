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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.config as config_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.config import get_fernet
from financial_dashboard.db import Setting
from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.integrations.paisa import PaisaClient
from financial_dashboard.schemas.extensions import (
    PaisaConfigInput,
)
from financial_dashboard.services.extensions import ExtensionManager
from financial_dashboard.services.paisa import surface
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.settings import get_setting, load_all_settings
from tests.conftest import new_test_engine

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
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(settings_mod, "async_session", maker)
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(config_mod.settings, "email_source_master_key", key)
    monkeypatch.setattr(config_mod, "_fernet_instance", None)
    yield maker
    await engine.dispose()
    holder.close()


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
    ids = [e["id"] for e in body["extensions"]]
    assert "paisa" in ids
    paisa = next(e for e in body["extensions"] if e["id"] == "paisa")
    assert paisa["display_name"] == "Paisa"
    assert "projection" in paisa["capabilities"]


# ---------------------------------------------------------------------------
# Config redaction
# ---------------------------------------------------------------------------


async def test_config_redacts_password_and_defaults_to_disabled(client):
    r = await client.get("/api/extensions/paisa/config")
    assert r.status_code == 200
    body = r.json()
    # The password field must not exist; only auth_password_set is surfaced.
    assert "auth_password" not in body
    assert body["auth_password_set"] is False
    assert body["mode"] == "disabled"
    assert body["can_connect"] is False
    assert body["can_project"] is False


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


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"base_url": "http://10.0.0.5:7500", "allow_remote": True}, "Base URL"),
        ({"external_url": "javascript:alert(1)"}, "External URL"),
        (
            {
                "mode": "project",
                "generated_path": "relative/path.journal",
                "project_since": "2026-01-01",
            },
            "Generated Path",
        ),
        (
            {
                "mode": "project",
                "generated_path": "/tmp/g.journal",
                "project_since": "",
            },
            "Project Since",
        ),
        ({"selected_account_ids": [9999]}, "Selected Account IDs"),
        ({"mode": "bogus"}, "Mode"),
    ],
)
async def test_save_rejects_invalid_config_without_persisting(
    session, settings_db, overrides, field
):
    before = dict(settings_mod._cache)
    result = await surface.save_config(session, _valid_input(**overrides))
    assert result.ok is False
    assert any(field in e for e in result.errors)
    async with settings_db() as s:
        rows = (await s.execute(select(Setting))).scalars().all()
    assert rows == []
    assert settings_mod._cache == before


@pytest.mark.parametrize(
    ("backend", "account_name", "category_name"),
    [
        ("hledger", "Assets:Bank:Savings Account", "Expenses:Food And Dining"),
        ("beancount", "Assets:Bank:SavingsAccount", "Expenses:FoodAndDining"),
    ],
)
async def test_api_save_accepts_backend_valid_operator_mappings(
    client, settings_db, tmp_path, backend, account_name, category_name
):
    payload = _valid_input(
        mode="project",
        generated_path=str(tmp_path / "gen.journal"),
        project_since="2026-01-01",
        ledger_cli=backend,
        account_mappings={"1": account_name},
        category_mappings={"groceries": category_name},
    ).model_dump()

    response = await client.post("/api/extensions/paisa/config", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["config"]["mode"] == "project"
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
    assert any("Account Mappings" in error for error in body["errors"])
    assert settings_mod._cache["paisa.ledger_cli"] == "ledger"
    assert settings_mod._cache["paisa.auth_username"] == "before"
    assert settings_mod._cache["paisa.account_mappings"] == (
        '{"1": "Assets:Bank:Savings Account"}'
    )


async def test_save_encrypts_redacts_and_blank_preserves_password(session, settings_db):
    await _seed_bank(session, id=1)
    first = await surface.save_config(
        session,
        _valid_input(auth_password="s3cret-paisa", selected_account_ids=[1]),
    )
    assert first.ok is True
    dumped = first.config.model_dump()
    assert "auth_password" not in dumped
    assert dumped["auth_password_set"] is True

    async with settings_db() as s:
        row = await s.get(Setting, "paisa.auth_password")
    assert row.value not in ("s3cret-paisa", "")
    await load_all_settings()
    assert get_setting("paisa.auth_password") == "s3cret-paisa"

    # A blank password keeps the current secret and applies the other change.
    second = await surface.save_config(
        session,
        _valid_input(auth_password="", auth_username="alice", selected_account_ids=[1]),
    )
    assert second.ok is True
    assert second.config.auth_password_set is True
    assert second.config.auth_username == "alice"
    async with settings_db() as s:
        row = await s.get(Setting, "paisa.auth_password")
    assert get_fernet().decrypt(row.value.encode()).decode() == "s3cret-paisa"


# ---------------------------------------------------------------------------
# Status / probe (all modes + failure isolation)
# ---------------------------------------------------------------------------


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
    assert status.ok is True
    assert status.capabilities.ledger_cli == "ledger"
    assert status.diagnosis.ok is True


async def test_mode_gates_block_actions(client, monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="disabled"))
    status = (await client.get("/api/extensions/paisa/status")).json()
    assert status["ok"] is False
    assert status["reachable"] is False
    assert status["reason"] == "disabled"

    monkeypatch.setattr(surface, "load_config", lambda: _config(mode="connect"))
    for action, key in (
        ("preview", "reason"),
        ("generate", "reason"),
        ("sync", "outcome"),
    ):
        body = (await client.post(f"/api/extensions/paisa/{action}")).json()
        assert body["ok"] is False
        assert body[key] == "connect_only"

    monkeypatch.setattr(
        surface, "load_config", lambda: _config(selected_account_ids=())
    )
    body = (await client.post("/api/extensions/paisa/preview")).json()
    assert body["ok"] is False
    assert body["reason"] == "not_configured"


async def test_route_failures_are_isolated(client, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("paisa blew up")

    monkeypatch.setattr(surface, "probe_status", boom)
    monkeypatch.setattr(surface, "preview_projection", boom)
    monkeypatch.setattr(surface, "sync_now", boom)

    status = await client.get("/api/extensions/paisa/status")
    assert status.status_code == 200
    assert status.json()["ok"] is False
    assert status.json()["error"] == "paisa_status_failed"
    preview = await client.post("/api/extensions/paisa/preview")
    assert preview.status_code == 503
    assert preview.json()["detail"]["error"] == "paisa_preview_failed"
    sync = await client.post("/api/extensions/paisa/sync")
    assert sync.status_code == 503
    assert sync.json()["detail"]["error"] == "paisa_sync_failed"


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
    assert "txn:1" in body["journal"]


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
    assert "; txn:1" in target.read_text()

    txn_after = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    acct_after = [
        a.id for a in (await session.execute(select(Account))).scalars().all()
    ]
    assert txn_before == txn_after
    assert acct_before == acct_after


async def test_sync_happy_path_never_mutates_core_rows(
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

    txn_before = (await session.execute(select(Transaction))).scalars().all()
    r = await client.post("/api/extensions/paisa/sync")
    body = r.json()
    txn_after = (await session.execute(select(Transaction))).scalars().all()
    assert [t.id for t in txn_before] == [t.id for t in txn_after]
    assert body["ok"] is True
    assert body["outcome"] == "synced"
    assert body["diagnosis_ok"] is True
    assert target.exists()
    assert seen == ["/api/config", "/api/sync", "/api/diagnosis"]


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
    assert out.busy is True
    assert out.outcome == "busy"
    assert out.ok is False
