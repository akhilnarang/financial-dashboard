"""Extension framework tests.

Covers: deterministic registration/iteration, duplicate collision rejection
(registry + setting registration), bootstrap idempotency, Paisa setting
defaults/types/visibility, and encrypted Paisa-password behavior.

All values here are synthetic.
"""

import pytest
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.config as config_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db import Setting
from financial_dashboard.extensions import (
    ExtensionManifest,
    ExtensionRegistry,
    register_builtin_extensions,
)
from financial_dashboard.extensions.base import ExtensionRegistrationError
from financial_dashboard.services.extensions import (
    ExtensionManager,
    bootstrap_extensions,
)
from financial_dashboard.services.settings import (
    SETTINGS_REGISTRY,
    get_grouped_settings,
    get_setting,
    load_all_settings,
    parse_form_updates,
    register_setting,
    save_settings,
)
from tests.conftest import new_test_engine

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module", autouse=True)
def _ensure_builtins_registered():
    """Mirror the lifespan bootstrap so paisa settings are present in the global
    SETTINGS_REGISTRY for this module. Idempotent across the session."""
    manager = ExtensionManager()
    register_builtin_extensions(manager.registry)
    return manager


# --------------------------------------------------------------------------- #
# Deterministic registration & iteration
# --------------------------------------------------------------------------- #


def test_registry_preserves_insertion_order():
    reg = ExtensionRegistry()
    a = ExtensionManifest(id="a", display_name="A")
    b = ExtensionManifest(id="b", display_name="B")
    c = ExtensionManifest(id="c", display_name="C")
    reg.register(a)
    reg.register(b)
    reg.register(c)
    assert [m.id for m in reg] == ["a", "b", "c"]
    assert reg.all() == (a, b, c)
    assert len(reg) == 3
    assert reg.get("b") is b
    assert reg.get("missing") is None


# --------------------------------------------------------------------------- #
# Duplicate collision rejection
# --------------------------------------------------------------------------- #


def test_registry_rejects_duplicate_id():
    reg = ExtensionRegistry()
    reg.register(ExtensionManifest(id="dup", display_name="First"))
    with pytest.raises(ExtensionRegistrationError, match="dup"):
        reg.register(ExtensionManifest(id="dup", display_name="Second"))


def test_register_setting_rejects_duplicate_key():
    existing = SETTINGS_REGISTRY["telegram.chat_id"]
    with pytest.raises(ValueError, match="telegram.chat_id"):
        register_setting("telegram.chat_id", existing)


def test_builtin_registration_is_idempotent_for_settings():
    # The autouse fixture already registered paisa settings; re-running with the
    # SAME definitions must not raise on the now-present setting keys.
    reg = ExtensionRegistry()
    register_builtin_extensions(reg)
    assert "paisa" in reg


def test_conflicting_setting_definition_is_rejected(monkeypatch):
    # A key already present with a DIFFERENT defn must raise, not silently skip.
    from financial_dashboard.services.settings import SettingDef

    conflicting = SettingDef(
        default="not-the-real-default",
        data_type="str",
        category="Paisa",
        label="Conflict",
    )
    monkeypatch.setitem(SETTINGS_REGISTRY, "paisa.mode", conflicting)
    reg = ExtensionRegistry()
    with pytest.raises(ExtensionRegistrationError, match="paisa.mode"):
        register_builtin_extensions(reg)


# --------------------------------------------------------------------------- #
# Builtin availability
# --------------------------------------------------------------------------- #


def test_bootstrap_extensions_returns_manager_with_paisa():
    manager = bootstrap_extensions(session_factory=async_sessionmaker())
    assert manager.get("paisa") is not None


# --------------------------------------------------------------------------- #
# Paisa setting defaults / types / visibility
# --------------------------------------------------------------------------- #

PAISA_EXPECTED: dict[str, tuple[str, str]] = {
    "paisa.mode": ("disabled", "str"),
    "paisa.base_url": ("http://localhost:7500", "str"),
    "paisa.external_url": ("", "str"),
    "paisa.allow_remote": ("false", "bool"),
    "paisa.auth_username": ("", "str"),
    "paisa.auth_password": ("", "str"),
    "paisa.generated_path": ("", "str"),
    "paisa.selected_account_ids": ("[]", "json"),
    "paisa.project_since": ("", "str"),
    "paisa.account_mappings": ("{}", "json"),
    "paisa.category_mappings": ("{}", "json"),
    "paisa.non_inr_policy": ("skip", "str"),
    "paisa.request_timeout_seconds": ("15", "int"),
}


def test_paisa_settings_registered_with_defaults_and_types():
    for key, (default, dtype) in PAISA_EXPECTED.items():
        assert key in SETTINGS_REGISTRY, key
        defn = SETTINGS_REGISTRY[key]
        assert defn.default == default, key
        assert defn.data_type == dtype, key
        assert defn.category == "Paisa", key


def test_grouped_settings_exposes_paisa_scalars_not_internal_json():
    grouped = get_grouped_settings()
    rendered = {row["key"] for rows in grouped.values() for row in rows}
    assert "Paisa" in grouped
    assert "paisa.mode" in rendered
    assert "paisa.base_url" in rendered
    assert "paisa.auth_password" in rendered
    for key in (
        "paisa.generated_path",
        "paisa.selected_account_ids",
        "paisa.account_mappings",
        "paisa.category_mappings",
    ):
        assert key not in rendered, key


def test_parse_form_updates_omits_internal_paisa_settings():
    # A form that omits the internal keys must not produce updates or errors.
    updates, errors = parse_form_updates({})
    assert errors == []
    for key in (
        "paisa.generated_path",
        "paisa.selected_account_ids",
        "paisa.account_mappings",
        "paisa.category_mappings",
    ):
        assert key not in updates, key


# --------------------------------------------------------------------------- #
# Encrypted Paisa password behavior
# --------------------------------------------------------------------------- #


@pytest.fixture
async def settings_db(monkeypatch):
    """An isolated in-memory settings DB + a real Fernet key for round-tripping."""
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(settings_mod, "async_session", maker)
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(config_mod.settings, "email_source_master_key", key)
    monkeypatch.setattr(config_mod, "_fernet_instance", None)
    yield maker
    await engine.dispose()
    holder.close()


async def test_paisa_password_encrypted_at_rest_and_round_trips(settings_db):
    changed = await save_settings({"paisa.auth_password": "s3cret-paisa"})
    assert "paisa.auth_password" in changed

    async with settings_db() as session:
        row = await session.get(Setting, "paisa.auth_password")
    assert row is not None
    assert row.value != "s3cret-paisa"  # plaintext is not stored
    assert row.value != ""  # an encrypted token is stored

    from financial_dashboard.config import get_fernet

    assert get_fernet().decrypt(row.value.encode()).decode() == "s3cret-paisa"

    await load_all_settings()
    assert get_setting("paisa.auth_password") == "s3cret-paisa"

    # A blank secret is stored as-is, not as a Fernet token.
    await save_settings({"paisa.auth_password": ""})
    async with settings_db() as session:
        row = await session.get(Setting, "paisa.auth_password")
    assert row is not None
    assert row.value == ""
