import datetime as dt
from decimal import Decimal
from unittest.mock import patch

import pytest

from financial_dashboard.db.enums import EmailKind
from financial_dashboard.db.models import (
    BalanceSnapshot,
    CasUpload,
    EmailSource,
    FetchRule,
)
from financial_dashboard.services import cas_emails

pytestmark = pytest.mark.anyio


async def _source(session, *, label: str = "Gmail Primary"):
    src = EmailSource(
        provider="gmail",
        label=label,
        account_identifier="me@example.com",
        credentials="",
    )
    session.add(src)
    await session.flush()
    return src


def _set_cas_cache(*, enabled: bool, pan: str):
    from financial_dashboard.services import settings as settings_mod

    settings_mod._cache["cas_auto_fetch_enabled"] = "true" if enabled else "false"
    settings_mod._cache["cas_pan"] = pan


async def test_ensure_disables_when_toggle_off(session):
    src = await _source(session)
    rule = FetchRule(
        provider="gmail",
        source_id=src.id,
        sender=cas_emails.CAS_SENDERS[0].address,
        bank="cas_nsdl",
        email_kind=EmailKind.CAS_STATEMENT.value,
        enabled=True,
        auto_managed=True,
    )
    session.add(rule)
    await session.flush()

    _set_cas_cache(enabled=False, pan="ABCDE1234F")
    await cas_emails.ensure_cas_fetch_rules(session)

    await session.refresh(rule)
    assert rule.enabled is False


async def test_ensure_disables_when_pan_missing(session):
    await _source(session)
    _set_cas_cache(enabled=True, pan="")
    await cas_emails.ensure_cas_fetch_rules(session)

    rules = (await session.execute(FetchRule.__table__.select())).fetchall()
    # Rules are created even if PAN missing? No: we skip creation if PAN missing.
    # Toggle ON but PAN empty just disables existing rules; we don't create new ones.
    assert all(not row.enabled for row in rules)


async def test_ensure_creates_and_enables_per_active_source(session):
    src = await _source(session)
    _set_cas_cache(enabled=True, pan="ABCDE1234F")

    await cas_emails.ensure_cas_fetch_rules(session)
    await session.flush()

    from sqlalchemy import select

    rules = (
        (
            await session.execute(
                select(FetchRule).where(FetchRule.auto_managed.is_(True))
            )
        )
        .scalars()
        .all()
    )
    assert len(rules) == 2
    senders = {rule.sender for rule in rules}
    assert senders == {s.address for s in cas_emails.CAS_SENDERS}
    assert all(rule.enabled for rule in rules)
    assert all(rule.source_id == src.id for rule in rules)
    assert all(rule.email_kind == EmailKind.CAS_STATEMENT.value for rule in rules)


async def test_ensure_is_idempotent(session):
    await _source(session)
    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    await cas_emails.ensure_cas_fetch_rules(session)
    await session.flush()
    await cas_emails.ensure_cas_fetch_rules(session)
    await session.flush()

    from sqlalchemy import select

    rules = (
        (
            await session.execute(
                select(FetchRule).where(FetchRule.auto_managed.is_(True))
            )
        )
        .scalars()
        .all()
    )
    assert len(rules) == 2


async def test_ensure_disables_rules_for_inactive_sources(session):
    src = await _source(session)
    rule = FetchRule(
        provider="gmail",
        source_id=src.id,
        sender=cas_emails.CAS_SENDERS[0].address,
        bank="cas_nsdl",
        email_kind=EmailKind.CAS_STATEMENT.value,
        enabled=True,
        auto_managed=True,
    )
    session.add(rule)
    await session.flush()

    src.active = False
    await session.flush()

    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    await cas_emails.ensure_cas_fetch_rules(session)

    await session.refresh(rule)
    assert rule.enabled is False


async def test_per_source_cooldown_disables_within_24h(session):
    src = await _source(session)
    src.cas_last_polled_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=2)
    await session.flush()

    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    await cas_emails.ensure_cas_fetch_rules(session)
    await session.flush()

    from sqlalchemy import select

    rules = (
        (
            await session.execute(
                select(FetchRule).where(FetchRule.auto_managed.is_(True))
            )
        )
        .scalars()
        .all()
    )
    assert len(rules) == 2
    assert all(rule.enabled is False for rule in rules)


async def test_per_source_cooldown_enables_after_24h(session):
    src = await _source(session)
    src.cas_last_polled_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=25)
    await session.flush()

    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    await cas_emails.ensure_cas_fetch_rules(session)
    await session.flush()

    from sqlalchemy import select

    rules = (
        (
            await session.execute(
                select(FetchRule).where(FetchRule.auto_managed.is_(True))
            )
        )
        .scalars()
        .all()
    )
    assert all(rule.enabled is True for rule in rules)


async def test_process_cas_email_happy_path(session, cas_statement_payload, tmp_path):
    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    src = await _source(session)

    fake_pdf_bytes = b"%PDF-1.4 fake"

    class FakeCasStatement:
        def model_dump(self, mode="json"):
            return cas_statement_payload

    with (
        patch(
            "financial_dashboard.services.statements.cc.extract_pdf_from_email",
            return_value=[("example_cas.pdf", fake_pdf_bytes)],
        ),
        patch(
            "financial_dashboard.integrations.parsers.parse_cas_pdf",
            return_value=FakeCasStatement(),
        ),
        patch(
            "financial_dashboard.services.cas_emails.STATEMENTS_DIR",
            tmp_path,
        ),
    ):
        result, error = await cas_emails.process_cas_email(
            session, b"raw", source_id=src.id, log_ref="msg-1"
        )

    assert error is None
    assert result is not None
    cas_upload_id = result["cas_upload_id"]
    upload = await session.get(CasUpload, cas_upload_id)
    assert upload is not None
    assert upload.grand_total == Decimal("200000.00")

    from sqlalchemy import select

    snapshots = (
        (
            await session.execute(
                select(BalanceSnapshot).where(
                    BalanceSnapshot.cas_upload_id == cas_upload_id
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(snapshots) == 1


async def test_process_cas_email_no_pdf(session):
    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    src = await _source(session)

    with patch(
        "financial_dashboard.services.statements.cc.extract_pdf_from_email",
        return_value=[],
    ):
        result, error = await cas_emails.process_cas_email(
            session, b"raw", source_id=src.id, log_ref="msg-2"
        )

    assert result is None
    assert error is not None
    assert "PDF" in error


async def test_process_cas_email_missing_pan(session):
    _set_cas_cache(enabled=True, pan="")
    src = await _source(session)

    result, error = await cas_emails.process_cas_email(
        session, b"raw", source_id=src.id, log_ref="msg-3"
    )

    assert result is None
    assert error is not None
    assert "PAN" in error


async def test_cas_dispatcher_rolls_back_on_ingest_failure(tmp_path):
    """Failure path in parse_email_by_kind must rollback the CAS session,
    not commit it — otherwise ingest_cas_payload's delete-then-insert would
    drop prior rows with no replacement on any mid-flow error. Verified via
    a mock session so we can assert rollback/commit calls without needing a
    real DB."""
    from unittest.mock import AsyncMock, MagicMock

    from financial_dashboard.db.enums import EmailKind
    from financial_dashboard.services import emails as emails_mod

    fake_session = MagicMock()
    fake_session.commit = AsyncMock()
    fake_session.rollback = AsyncMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    def fake_factory():
        return fake_session

    process_cas_email = AsyncMock(return_value=(None, "ingest exploded"))

    with (
        patch.object(emails_mod, "async_session", fake_factory),
        patch(
            "financial_dashboard.services.cas_emails.process_cas_email",
            process_cas_email,
        ),
    ):
        result = await emails_mod.parse_email_by_kind(
            bank="cas_nsdl",
            email_kind=EmailKind.CAS_STATEMENT.value,
            raw_bytes=b"raw",
            subject="CAS",
            source_id=None,
            log_ref="msg-rb",
        )

    assert result.error == "ingest exploded"
    assert result.stmt_result is None
    fake_session.commit.assert_not_awaited()
    fake_session.rollback.assert_awaited_once()


async def test_cas_dispatcher_commits_on_success():
    """Mirror of the rollback test: when process_cas_email returns a result,
    the dispatcher must commit (not rollback)."""
    from unittest.mock import AsyncMock, MagicMock

    from financial_dashboard.db.enums import EmailKind
    from financial_dashboard.services import emails as emails_mod

    fake_session = MagicMock()
    fake_session.commit = AsyncMock()
    fake_session.rollback = AsyncMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    def fake_factory():
        return fake_session

    process_cas_email = AsyncMock(return_value=({"cas_upload_id": 42}, None))

    with (
        patch.object(emails_mod, "async_session", fake_factory),
        patch(
            "financial_dashboard.services.cas_emails.process_cas_email",
            process_cas_email,
        ),
    ):
        result = await emails_mod.parse_email_by_kind(
            bank="cas_nsdl",
            email_kind=EmailKind.CAS_STATEMENT.value,
            raw_bytes=b"raw",
            subject="CAS",
            source_id=None,
            log_ref="msg-ok",
        )

    assert result.error is None
    assert result.stmt_result == {"cas_upload_id": 42}
    fake_session.commit.assert_awaited_once()
    fake_session.rollback.assert_not_awaited()


async def test_cas_dispatcher_rolls_back_on_raised_exception():
    """If process_cas_email raises (vs. returning None+error), the dispatcher
    must still rollback AND not propagate the exception (matches bank/CC
    behaviour — wouldn't want a CAS bug to crash the entire poll cycle)."""
    from unittest.mock import AsyncMock, MagicMock

    from financial_dashboard.db.enums import EmailKind
    from financial_dashboard.services import emails as emails_mod

    fake_session = MagicMock()
    fake_session.commit = AsyncMock()
    fake_session.rollback = AsyncMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    def fake_factory():
        return fake_session

    process_cas_email = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch.object(emails_mod, "async_session", fake_factory),
        patch(
            "financial_dashboard.services.cas_emails.process_cas_email",
            process_cas_email,
        ),
    ):
        result = await emails_mod.parse_email_by_kind(
            bank="cas_nsdl",
            email_kind=EmailKind.CAS_STATEMENT.value,
            raw_bytes=b"raw",
            subject="CAS",
            source_id=None,
            log_ref="msg-boom",
        )

    assert result.error is not None
    assert "RuntimeError" in result.error and "boom" in result.error
    assert result.stmt_result is None
    fake_session.commit.assert_not_awaited()
    fake_session.rollback.assert_awaited_once()


async def test_cas_cooldown_not_stamped_on_fetch_failure(monkeypatch):
    """A transient IMAP failure should not stamp cas_last_polled_at — otherwise
    a single network blip locks out CAS polling for 24h. The stamp is only set
    when the fetch actually succeeded (fetch_ok=True)."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from financial_dashboard.db.models import Base
    from financial_dashboard.integrations.email import orchestrator

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Patch async_session at every site that the orchestrator opens a session
    # through (it imports the symbol into its own module namespace).
    monkeypatch.setattr(orchestrator, "async_session", maker)

    async with maker() as s:
        src = EmailSource(
            provider="gmail",
            label="Gmail Primary",
            account_identifier="me@example.com",
            credentials="",
        )
        s.add(src)
        await s.flush()
        rule = FetchRule(
            provider="gmail",
            source_id=src.id,
            sender=cas_emails.CAS_SENDERS[0].address,
            bank="cas_nsdl",
            email_kind=EmailKind.CAS_STATEMENT.value,
            enabled=True,
            auto_managed=True,
        )
        s.add(rule)
        await s.commit()
        source_id = src.id

    _set_cas_cache(enabled=True, pan="ABCDE1234F")

    fake_provider = AsyncMock()
    # fetch_ok=False simulates a transient IMAP/network failure.
    # Return shape: (results_by_rule, fetch_ok, backfill_ready_rule_ids).
    fake_provider.fetch_source.return_value = ({}, False, set())

    with patch.object(orchestrator, "get_provider", return_value=fake_provider):
        await orchestrator.poll_all(
            poll_lock=asyncio.Lock(),
            poll_status={
                "state": "idle",
                "started_at": None,
                "finished_at": None,
                "last_stats": None,
                "last_error": None,
                "progress": None,
            },
        )

    async with maker() as s:
        refreshed = (
            await s.execute(select(EmailSource).where(EmailSource.id == source_id))
        ).scalar_one()
        assert refreshed.cas_last_polled_at is None, (
            "cas_last_polled_at must not be stamped when fetch failed; "
            "otherwise a transient IMAP error locks CAS polling for 24h"
        )
        assert refreshed.last_error is not None, (
            "last_error should be recorded on fetch failure (sanity check)"
        )

    await engine.dispose()


async def test_cas_cooldown_stamped_on_fetch_success(monkeypatch):
    """Counterpart to the failure test: a successful fetch (even with zero
    new emails) must stamp cas_last_polled_at so we don't repoll for 24h."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from financial_dashboard.db.models import Base
    from financial_dashboard.integrations.email import orchestrator

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setattr(orchestrator, "async_session", maker)

    async with maker() as s:
        src = EmailSource(
            provider="gmail",
            label="Gmail Primary",
            account_identifier="me@example.com",
            credentials="",
        )
        s.add(src)
        await s.flush()
        rule = FetchRule(
            provider="gmail",
            source_id=src.id,
            sender=cas_emails.CAS_SENDERS[0].address,
            bank="cas_nsdl",
            email_kind=EmailKind.CAS_STATEMENT.value,
            enabled=True,
            auto_managed=True,
        )
        s.add(rule)
        await s.commit()
        source_id = src.id

    _set_cas_cache(enabled=True, pan="ABCDE1234F")

    fake_provider = AsyncMock()
    # Return shape: (results_by_rule, fetch_ok, backfill_ready_rule_ids).
    fake_provider.fetch_source.return_value = ({}, True, set())

    with patch.object(orchestrator, "get_provider", return_value=fake_provider):
        await orchestrator.poll_all(
            poll_lock=asyncio.Lock(),
            poll_status={
                "state": "idle",
                "started_at": None,
                "finished_at": None,
                "last_stats": None,
                "last_error": None,
                "progress": None,
            },
        )

    async with maker() as s:
        refreshed = (
            await s.execute(select(EmailSource).where(EmailSource.id == source_id))
        ).scalar_one()
        assert refreshed.cas_last_polled_at is not None, (
            "cas_last_polled_at should be stamped after a successful fetch"
        )

    await engine.dispose()


async def test_process_cas_email_surfaces_specific_ingest_error(
    session, cas_statement_payload, tmp_path
):
    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    src = await _source(session)
    cas_statement_payload["summary"]["grand_total"] = None  # triggers CasIngestError

    fake_pdf_bytes = b"%PDF-1.4 fake"

    class FakeCasStatement:
        def model_dump(self, mode="json"):
            return cas_statement_payload

    with (
        patch(
            "financial_dashboard.services.statements.cc.extract_pdf_from_email",
            return_value=[("example_cas.pdf", fake_pdf_bytes)],
        ),
        patch(
            "financial_dashboard.integrations.parsers.parse_cas_pdf",
            return_value=FakeCasStatement(),
        ),
        patch(
            "financial_dashboard.services.cas_emails.STATEMENTS_DIR",
            tmp_path,
        ),
    ):
        result, error = await cas_emails.process_cas_email(
            session, b"raw", source_id=src.id, log_ref="msg-4"
        )

    assert result is None
    assert error is not None
    assert "grand_total" in error
