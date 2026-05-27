"""Tests for ``load_or_fetch_raw_email`` — the universal raw-bytes loader used
by every reparse path. Spool preferred; provider is the fallback when the
spool file has been evicted by the cleanup cron."""

from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Base, Email, EmailSource
from financial_dashboard.integrations.email import body as fetcher_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(fetcher_module, "async_session", maker)
    yield maker
    await engine.dispose()


@pytest.fixture
def spool_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher_module, "FAILED_SPOOL_DIR", tmp_path)
    return tmp_path


async def _make_email(
    maker,
    *,
    provider: str,
    message_id: str,
    remote_id: str | None,
    source_id: int | None,
) -> Email:
    async with maker() as session:
        em = Email(
            provider=provider,
            message_id=message_id,
            remote_id=remote_id,
            source_id=source_id,
            status="failed",
        )
        session.add(em)
        await session.commit()
        return em


async def _make_source(maker, provider: str) -> int:
    """Seed an EmailSource with dummy encrypted creds. The fetch functions are
    mocked in every test so the creds are never actually decrypted."""
    async with maker() as session:
        src = EmailSource(
            provider=provider,
            label=f"{provider} source",
            account_identifier="user@example.com",
            credentials="dummy-encrypted",
        )
        session.add(src)
        await session.commit()
        return src.id


@pytest.mark.anyio
async def test_loads_from_spool_when_present(session_factory, spool_dir):
    em = await _make_email(
        session_factory,
        provider="fastmail",
        message_id="msg-abc",
        remote_id=None,
        source_id=None,
    )
    (spool_dir / "fastmail_msg-abc.eml").write_bytes(b"spooled content")

    with (
        patch.object(fetcher_module, "_fetch_fastmail_single_sync") as fastmail_fetch,
        patch.object(fetcher_module, "_fetch_gmail_single_sync") as gmail_fetch,
    ):
        raw, err = await fetcher_module.load_or_fetch_raw_email(em)

    assert raw == b"spooled content"
    assert err is None
    fastmail_fetch.assert_not_called()
    gmail_fetch.assert_not_called()


@pytest.mark.anyio
async def test_refetches_from_fastmail_when_spool_missing(session_factory, spool_dir):
    source_id = await _make_source(session_factory, "fastmail")
    em = await _make_email(
        session_factory,
        provider="fastmail",
        message_id="msg-123",
        remote_id="remote-xyz",
        source_id=source_id,
    )

    with (
        patch.object(
            fetcher_module, "decrypt_credentials", return_value={"token": "tok"}
        ),
        patch.object(
            fetcher_module,
            "_fetch_fastmail_single_sync",
            return_value=b"re-fetched bytes",
        ) as fastmail_fetch,
    ):
        raw, err = await fetcher_module.load_or_fetch_raw_email(em)

    assert raw == b"re-fetched bytes"
    assert err is None
    fastmail_fetch.assert_called_once_with("tok", "remote-xyz")


@pytest.mark.anyio
async def test_refetches_from_gmail_when_spool_missing(session_factory, spool_dir):
    source_id = await _make_source(session_factory, "gmail")
    em = await _make_email(
        session_factory,
        provider="gmail",
        message_id="msg-456",
        remote_id="gmsgid-999",
        source_id=source_id,
    )

    with (
        patch.object(
            fetcher_module,
            "decrypt_credentials",
            return_value={"user": "u@example.com", "app_password": "pw"},
        ),
        patch.object(
            fetcher_module, "_fetch_gmail_single_sync", return_value=b"gmail bytes"
        ) as gmail_fetch,
    ):
        raw, err = await fetcher_module.load_or_fetch_raw_email(em)

    assert raw == b"gmail bytes"
    assert err is None
    gmail_fetch.assert_called_once_with("u@example.com", "pw", "gmsgid-999")


@pytest.mark.anyio
async def test_error_when_spool_missing_and_no_remote_id(session_factory, spool_dir):
    em = await _make_email(
        session_factory,
        provider="fastmail",
        message_id="orphan",
        remote_id=None,
        source_id=None,
    )

    raw, err = await fetcher_module.load_or_fetch_raw_email(em)

    assert raw is None
    assert err is not None
    assert "no source" in err.lower() or "remote" in err.lower()


@pytest.mark.anyio
async def test_error_when_provider_returns_nothing(session_factory, spool_dir):
    source_id = await _make_source(session_factory, "fastmail")
    em = await _make_email(
        session_factory,
        provider="fastmail",
        message_id="msg-deleted",
        remote_id="remote-gone",
        source_id=source_id,
    )

    with (
        patch.object(
            fetcher_module, "decrypt_credentials", return_value={"token": "tok"}
        ),
        patch.object(fetcher_module, "_fetch_fastmail_single_sync", return_value=None),
    ):
        raw, err = await fetcher_module.load_or_fetch_raw_email(em)

    assert raw is None
    assert err is not None
    assert "deleted" in err.lower() or "no data" in err.lower()


@pytest.mark.anyio
async def test_error_when_source_row_deleted(session_factory, spool_dir):
    """An Email row pointing at a now-deleted EmailSource should fail cleanly
    rather than crash."""
    em = await _make_email(
        session_factory,
        provider="fastmail",
        message_id="msg-orphaned-source",
        remote_id="remote-xyz",
        source_id=9999,  # doesn't exist
    )

    raw, err = await fetcher_module.load_or_fetch_raw_email(em)

    assert raw is None
    assert err is not None
    assert "source" in err.lower()
