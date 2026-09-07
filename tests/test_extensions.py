"""Extension framework tests.

Covers: deterministic registration/iteration, duplicate collision rejection
(registry + setting registration), hardened bootstrap idempotency (identical
re-registration accepted, conflicting definitions rejected), builtin
availability + advertised capabilities, manifest immutability, Paisa setting
defaults/types/visibility, and encrypted Paisa-password behavior.

All values here are synthetic.
"""

from types import MappingProxyType

import pytest
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.config as config_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db import Base, Setting
from financial_dashboard.extensions import (
    BUILTIN_EXTENSIONS,
    ExtensionManifest,
    ExtensionRegistry,
    PAISA_EXTENSION,
    register_builtin_extensions,
)
from financial_dashboard.extensions.base import Capability, ExtensionRegistrationError
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


def test_registry_get_and_contains():
    reg = ExtensionRegistry()
    manifest = ExtensionManifest(id="x", display_name="X")
    reg.register(manifest)
    assert reg.get("x") is manifest
    assert "x" in reg
    assert reg.get("missing") is None
    assert "missing" not in reg


def test_registry_rejects_non_manifest():
    reg = ExtensionRegistry()
    with pytest.raises(ExtensionRegistrationError):
        reg.register("not-a-manifest")  # type: ignore[arg-type]


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


def test_identical_setting_re_registration_accepted():
    # Pre-registering an EQUAL defn, then bootstrapping, must be accepted.
    reg = ExtensionRegistry()
    register_builtin_extensions(reg)


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
# Manifest immutability
# --------------------------------------------------------------------------- #


def test_manifest_is_frozen():
    manifest = ExtensionManifest(
        id="x",
        display_name="X",
        settings={"telegram.chat_id": SETTINGS_REGISTRY["telegram.chat_id"]},
    )
    with pytest.raises(Exception):
        manifest.id = "y"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.settings["nope"] = SETTINGS_REGISTRY["telegram.chat_id"]


# --------------------------------------------------------------------------- #
# Builtin availability
# --------------------------------------------------------------------------- #


def test_builtins_include_paisa():
    assert "paisa" in {m.id for m in BUILTIN_EXTENSIONS}


def test_paisa_manifest_shape():
    assert PAISA_EXTENSION.id == "paisa"
    assert PAISA_EXTENSION.display_name == "Paisa"
    caps = PAISA_EXTENSION.capabilities
    assert Capability.SETTING_CONTRIBUTION in caps
    assert Capability.HTTP_READ in caps
    assert Capability.PROJECTION in caps
    assert Capability.SYNTHETIC_GENERATION not in caps  # not yet active
    assert isinstance(PAISA_EXTENSION.settings, MappingProxyType)


def test_bootstrap_extensions_returns_manager_with_paisa():
    manager = bootstrap_extensions(session_factory=async_sessionmaker())
    assert isinstance(manager, ExtensionManager)
    assert manager.get("paisa") is not None
    assert "paisa" in manager
    assert any(m.id == "paisa" for m in manager.all())
    assert len(manager) >= 1


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


def test_paisa_password_is_secret():
    assert SETTINGS_REGISTRY["paisa.auth_password"].secret is True


def test_paisa_internal_settings_are_not_form_editable():
    # generated_path is a dashboard-owned operational path; the JSON mappings
    # have no generic-form widget. None should surface in the settings form.
    for key in (
        "paisa.generated_path",
        "paisa.selected_account_ids",
        "paisa.account_mappings",
        "paisa.category_mappings",
    ):
        assert SETTINGS_REGISTRY[key].internal is True, key


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


async def test_paisa_password_blank_is_not_encrypted(settings_db):
    # A blank secret is stored as-is (no Fernet token) — consistent with the
    # existing save_settings contract for empty secret values.
    await save_settings({"paisa.auth_password": ""})
    async with settings_db() as session:
        row = await session.get(Setting, "paisa.auth_password")
    assert row is not None
    assert row.value == ""
