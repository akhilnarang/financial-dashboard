"""Startup guard: fail fast when EMAIL_SOURCE_MASTER_KEY is unset but
encrypted data already exists.

Without the master key, get_fernet() mints an ephemeral key and any stored
secret becomes undecryptable after a restart. The guard refuses to boot in
that case unless ALLOW_EPHEMERAL_MASTER_KEY downgrades it to a warning.

All values here are fully synthetic.
"""

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.settings as settings_mod
from financial_dashboard.config import Settings
from financial_dashboard.db import EmailSource, Setting
from financial_dashboard.db.models import Base
from financial_dashboard.services.settings import assert_master_key_or_no_secrets

pytestmark = pytest.mark.anyio


def _settings(*, master_key: str = "", allow_ephemeral: bool = False) -> Settings:
    return Settings(
        email_source_master_key=master_key,
        allow_ephemeral_master_key=allow_ephemeral,
        auth_username="",
        auth_password=SecretStr(""),
    )


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


async def test_empty_db_no_key_does_not_raise(db_session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings())
    # Should not raise — fresh DB, nothing encrypted yet.
    await assert_master_key_or_no_secrets(db_session)


async def test_credentials_without_key_raises(db_session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings())
    db_session.add(
        EmailSource(
            provider="imap",
            label="Synthetic source",
            credentials="encrypted-blob",
        )
    )
    await db_session.commit()
    with pytest.raises(SystemExit, match="EMAIL_SOURCE_MASTER_KEY"):
        await assert_master_key_or_no_secrets(db_session)


async def test_secret_setting_without_key_raises(db_session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings())
    # telegram.bot_token is marked secret in SETTINGS_REGISTRY.
    db_session.add(Setting(key="telegram.bot_token", value="encrypted-token"))
    await db_session.commit()
    with pytest.raises(SystemExit, match="EMAIL_SOURCE_MASTER_KEY"):
        await assert_master_key_or_no_secrets(db_session)


async def test_non_secret_setting_without_key_does_not_raise(db_session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings())
    # telegram.chat_id is NOT a secret — a value here must not trip the guard.
    db_session.add(Setting(key="telegram.chat_id", value="123456"))
    await db_session.commit()
    await assert_master_key_or_no_secrets(db_session)


async def test_key_set_never_raises(db_session, monkeypatch):
    monkeypatch.setattr(
        settings_mod, "settings", _settings(master_key="synthetic-master-key")
    )
    db_session.add(
        EmailSource(
            provider="imap",
            label="Synthetic source",
            credentials="encrypted-blob",
        )
    )
    db_session.add(Setting(key="telegram.bot_token", value="encrypted-token"))
    await db_session.commit()
    await assert_master_key_or_no_secrets(db_session)


async def test_allow_ephemeral_downgrades_to_warning(db_session, monkeypatch):
    monkeypatch.setattr(settings_mod, "settings", _settings(allow_ephemeral=True))
    db_session.add(
        EmailSource(
            provider="imap",
            label="Synthetic source",
            credentials="encrypted-blob",
        )
    )
    await db_session.commit()
    # Encrypted data + no key, but escape hatch set → warns, no raise.
    await assert_master_key_or_no_secrets(db_session)
