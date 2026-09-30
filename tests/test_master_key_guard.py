"""Startup guard: fail fast when EMAIL_SOURCE_MASTER_KEY is unset but
encrypted data already exists.

Without the master key, get_fernet() mints an ephemeral key and any stored
secret becomes undecryptable after a restart. The guard refuses to boot in
that case unless ALLOW_EPHEMERAL_MASTER_KEY downgrades it to a warning.

All values here are fully synthetic.
"""

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import async_sessionmaker

import financial_dashboard.services.settings as settings_mod
from financial_dashboard.config import Settings
from financial_dashboard.db import EmailSource, Setting
from financial_dashboard.services.extensions import bootstrap_extensions
from financial_dashboard.services.settings import (
    SETTINGS_REGISTRY,
    assert_master_key_or_no_secrets,
)

pytestmark = pytest.mark.anyio


def _settings(*, master_key: str = "", allow_ephemeral: bool = False) -> Settings:
    return Settings(
        email_source_master_key=master_key,
        allow_ephemeral_master_key=allow_ephemeral,
        auth_username="",
        auth_password=SecretStr(""),
    )


async def test_credentials_without_key_raises(session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings())
    session.add(
        EmailSource(
            provider="imap",
            label="Synthetic source",
            credentials="encrypted-blob",
        )
    )
    await session.commit()
    with pytest.raises(SystemExit, match="EMAIL_SOURCE_MASTER_KEY"):
        await assert_master_key_or_no_secrets(session)


async def test_secret_setting_without_key_raises(session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings())
    # telegram.bot_token is marked secret in SETTINGS_REGISTRY.
    session.add(Setting(key="telegram.bot_token", value="encrypted-token"))
    await session.commit()
    with pytest.raises(SystemExit, match="EMAIL_SOURCE_MASTER_KEY"):
        await assert_master_key_or_no_secrets(session)


async def test_dormant_paisa_secret_without_key_raises(session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings())
    original_registry = dict(SETTINGS_REGISTRY)
    try:
        bootstrap_extensions(session_factory=async_sessionmaker(), paisa_enabled=False)
        assert "paisa.auth_password" not in SETTINGS_REGISTRY
        session.add(
            Setting(key="paisa.auth_password", value="encrypted-paisa-password")
        )
        await session.commit()
        with pytest.raises(SystemExit, match="EMAIL_SOURCE_MASTER_KEY"):
            await assert_master_key_or_no_secrets(session)
    finally:
        SETTINGS_REGISTRY.clear()
        SETTINGS_REGISTRY.update(original_registry)


async def test_non_secret_setting_without_key_does_not_raise(session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings())
    # telegram.chat_id is not a secret. It must not trip the guard.
    session.add(Setting(key="telegram.chat_id", value="123456"))
    await session.commit()
    await assert_master_key_or_no_secrets(session)


async def test_key_set_never_raises(session, monkeypatch):
    monkeypatch.setattr(
        settings_mod, "settings", _settings(master_key="synthetic-master-key")
    )
    session.add(
        EmailSource(
            provider="imap",
            label="Synthetic source",
            credentials="encrypted-blob",
        )
    )
    session.add(Setting(key="telegram.bot_token", value="encrypted-token"))
    await session.commit()
    await assert_master_key_or_no_secrets(session)


async def test_allow_ephemeral_downgrades_to_warning(session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings(allow_ephemeral=True))
    session.add(
        EmailSource(
            provider="imap",
            label="Synthetic source",
            credentials="encrypted-blob",
        )
    )
    await session.commit()
    # Encrypted data + no key, but escape hatch set → warns, no raise.
    await assert_master_key_or_no_secrets(session)
