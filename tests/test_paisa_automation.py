"""Paisa automatic-sync runtime tests.

Covers the full gating + outcome matrix without a real Paisa/network by
monkeypatching the public orchestrator entrypoints the runtime calls:

* startup/shutdown are inert (no settings mutation, no network).
* disabled/connect modes are full no-ops (no generate call, no audit row, no fs).
* auto_sync_enabled=false is a no-op even in project mode.
* interval debounce: a recent automatic run suppresses the cycle; an expired
  one proceeds.
* skip-unchanged: an unchanged publisher result skips the remote sync and
  records outcome=skipped_unchanged (no manual_sync call).
* a changed file runs the full remote sync and maps the SyncReport outcome.
* a generate refusal (not_configured) records a skipped audit row.
* an exception in the sync records a sanitized failure row.
* failure notification is only attempted when opted in.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db.models import ExtensionRun, Base
from financial_dashboard.services.paisa import automation
from financial_dashboard.services.paisa.automation import (
    DEFAULT_MIN_INTERVAL_MINUTES,
    PaisaAutomationRuntime,
)
from financial_dashboard.services.paisa.audit import (
    OPERATION_AUTOMATIC,
    recent_runs,
)
from financial_dashboard.services.paisa.orchestrator import GenerateResult, SyncReport
from financial_dashboard.services.paisa.publisher import PublishResult

pytestmark = pytest.mark.anyio

UNUSED_PATH = "/tmp/paisa-auto-test-unused.journal"


def _set_paisa_settings(**overrides):
    """Populate the settings cache for the paisa.* keys the runtime reads.

    Accepts short names (``mode=``) or full keys (``paisa.mode=``).
    """
    defaults = {
        "paisa.mode": "project",
        "paisa.auto_sync_enabled": "true",
        "paisa.auto_sync_min_interval_minutes": str(DEFAULT_MIN_INTERVAL_MINUTES),
        "paisa.notify_sync_failures": "false",
    }
    for key, value in overrides.items():
        defaults[key if key.startswith("paisa.") else f"paisa.{key}"] = value
    settings_mod._cache.update(defaults)


@pytest.fixture
async def factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield maker
    await engine.dispose()


@pytest.fixture
def runtime(factory):
    # Project mode + auto-sync on by default; individual tests override.
    _set_paisa_settings()
    return PaisaAutomationRuntime(session_factory=factory)


def _report(emitted=3, skipped=None):
    return SimpleNamespace(emitted_count=emitted, skipped=skipped or [], journal="...")


def _publish(*, published, body_hash="deadbeef"):
    return PublishResult(
        published=published,
        path=UNUSED_PATH,
        version="1",
        body_hash=body_hash,
        bytes_written=42 if published else 7,
    )


async def _recent(factory):
    async with factory() as s:
        return await recent_runs(s, extension_id="paisa", operation=OPERATION_AUTOMATIC)


# --------------------------------------------------------------------------- #
# Inert lifecycle
# --------------------------------------------------------------------------- #


async def test_startup_and_shutdown_are_noops(runtime, monkeypatch):
    # Patch generate so we'd catch any accidental sync kick at lifecycle time.
    monkeypatch.setattr(
        automation, "generate", AsyncMock(side_effect=AssertionError("no generate"))
    )
    await runtime.startup()
    await runtime.shutdown()
    # No audit rows created by lifecycle.
    rows = await _recent(runtime._session_factory)
    assert rows == []


# --------------------------------------------------------------------------- #
# Gating: disabled / connect / auto_sync off
# --------------------------------------------------------------------------- #


async def test_disabled_mode_is_a_full_noop(factory, monkeypatch):
    _set_paisa_settings(mode="disabled", auto_sync_enabled="true")
    gen = AsyncMock(side_effect=AssertionError("generate must not run when disabled"))
    monkeypatch.setattr(automation, "generate", gen)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    gen.assert_not_called()
    assert await _recent(factory) == []


async def test_connect_mode_is_a_full_noop(factory, monkeypatch):
    _set_paisa_settings(mode="connect", auto_sync_enabled="true")
    gen = AsyncMock(side_effect=AssertionError("generate must not run in connect mode"))
    monkeypatch.setattr(automation, "generate", gen)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    gen.assert_not_called()
    assert await _recent(factory) == []


async def test_auto_sync_disabled_is_a_noop_in_project_mode(factory, monkeypatch):
    _set_paisa_settings(mode="project", auto_sync_enabled="false")
    gen = AsyncMock(
        side_effect=AssertionError("generate must not run when auto-sync off")
    )
    monkeypatch.setattr(automation, "generate", gen)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    gen.assert_not_called()
    assert await _recent(factory) == []


# --------------------------------------------------------------------------- #
# Interval debounce
# --------------------------------------------------------------------------- #


async def test_debounce_within_interval_suppresses_cycle(factory, monkeypatch):
    _set_paisa_settings()
    # Seed a prior automatic run 1 minute ago — well inside the default window.
    async with factory() as s:
        s.add(
            ExtensionRun(
                extension_id="paisa",
                operation=OPERATION_AUTOMATIC,
                status="success",
                started_at=datetime.datetime.now(datetime.UTC)
                - datetime.timedelta(minutes=1),
                completed_at=datetime.datetime.now(datetime.UTC),
            )
        )
        await s.commit()
    gen = AsyncMock(side_effect=AssertionError("must be debounced"))
    monkeypatch.setattr(automation, "generate", gen)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    gen.assert_not_called()
    # No NEW row added.
    rows = await _recent(factory)
    assert len(rows) == 1


async def test_debounce_expired_proceeds(factory, monkeypatch):
    _set_paisa_settings(auto_sync_min_interval_minutes="10")
    # Prior run 1 hour ago — past the 10-minute window.
    async with factory() as s:
        s.add(
            ExtensionRun(
                extension_id="paisa",
                operation=OPERATION_AUTOMATIC,
                status="success",
                started_at=datetime.datetime.now(datetime.UTC)
                - datetime.timedelta(hours=1),
                completed_at=datetime.datetime.now(datetime.UTC),
            )
        )
        await s.commit()
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True, report=_report(), publish=_publish(published=False), reason=None
        )
    )
    sync = AsyncMock(
        return_value=SyncReport(
            ok=True,
            outcome="synced",
            preview=None,
            publish=None,
            diagnosis_ok=True,
            reason=None,
        )
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(automation, "manual_sync", sync)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    gen.assert_awaited_once()


async def test_no_prior_run_proceeds(factory, monkeypatch):
    _set_paisa_settings()
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True, report=_report(), publish=_publish(published=False), reason=None
        )
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(
        automation,
        "manual_sync",
        AsyncMock(
            return_value=SyncReport(
                ok=True,
                outcome="synced",
                preview=None,
                publish=None,
                diagnosis_ok=True,
                reason=None,
            )
        ),
    )
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    gen.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Skip-unchanged
# --------------------------------------------------------------------------- #


async def test_unchanged_content_skips_remote_sync(factory, monkeypatch):
    _set_paisa_settings()
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True,
            report=_report(emitted=5),
            publish=_publish(published=False),
            reason=None,
        )
    )
    sync = AsyncMock(
        side_effect=AssertionError("manual_sync must NOT run when unchanged")
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(automation, "manual_sync", sync)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    sync.assert_not_called()
    rows = await _recent(factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "success"
    assert row.outcome == "skipped_unchanged"
    assert row.emitted_count == 5
    assert row.output_hash == "deadbeef"
    assert row.error is None


# --------------------------------------------------------------------------- #
# Changed content -> full remote sync
# --------------------------------------------------------------------------- #


async def test_changed_content_runs_remote_sync_and_maps_success(factory, monkeypatch):
    _set_paisa_settings()
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True,
            report=_report(emitted=4),
            publish=_publish(published=True),
            reason=None,
        )
    )
    sync = AsyncMock(
        return_value=SyncReport(
            ok=True,
            outcome="synced",
            preview=None,
            publish=None,
            diagnosis_ok=True,
            reason=None,
        )
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(automation, "manual_sync", sync)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    gen.assert_awaited_once()
    sync.assert_awaited_once()
    rows = await _recent(factory)
    row = rows[0]
    assert row.status == "success"
    assert row.outcome == "synced"
    assert row.emitted_count == 4


async def test_changed_content_records_diagnosis_counts(factory, monkeypatch):
    """An auto-sync whose dangers were all expected contra-expense ``Debit
    Entry`` issues records the classified counts in the audit details (so an
    operator sees in the audit log that the sync succeeded *because* the
    dangers were classified as expected, not because there were none)."""
    import json

    _set_paisa_settings()
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True,
            report=_report(emitted=4),
            publish=_publish(published=True),
            reason=None,
        )
    )
    sync = AsyncMock(
        return_value=SyncReport(
            ok=True,
            outcome="synced",
            preview=None,
            publish=None,
            diagnosis_ok=True,
            reason=None,
            diagnosis_expected=3,
            diagnosis_accepted=3,
            diagnosis_fatal=0,
        )
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(automation, "manual_sync", sync)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    rows = await _recent(factory)
    row = rows[0]
    assert row.status == "success"
    assert row.outcome == "synced"
    details = json.loads(row.details) if row.details else {}
    assert details["diagnosis_expected"] == 3
    assert details["diagnosis_accepted"] == 3
    assert details["diagnosis_fatal"] == 0
    # No credentials or raw journal text in the auto-sync details.
    blob = json.dumps(details)
    assert "auth_password" not in blob


async def test_changed_content_maps_failure(factory, monkeypatch):
    _set_paisa_settings()
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True, report=_report(), publish=_publish(published=True), reason=None
        )
    )
    sync = AsyncMock(
        return_value=SyncReport(
            ok=False,
            outcome="unreachable",
            preview=None,
            publish=None,
            diagnosis_ok=None,
            reason="could not reach Paisa: timeout",
        )
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(automation, "manual_sync", sync)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    rows = await _recent(factory)
    row = rows[0]
    assert row.status == "failure"
    assert row.outcome == "unreachable"
    assert row.error is not None
    assert "timeout" in row.error
    assert "\n" not in row.error  # sanitized


# --------------------------------------------------------------------------- #
# Generate refusal / exception
# --------------------------------------------------------------------------- #


async def test_generate_not_configured_records_skipped(factory, monkeypatch):
    _set_paisa_settings()
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=False, report=None, publish=None, reason="not_configured"
        )
    )
    sync = AsyncMock(
        side_effect=AssertionError("manual_sync must not run on generate refusal")
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(automation, "manual_sync", sync)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    rows = await _recent(factory)
    row = rows[0]
    assert row.status == "skipped"
    assert row.outcome == "not_configured"


async def test_exception_in_sync_records_failure_without_secrets(factory, monkeypatch):
    _set_paisa_settings()
    secret = "SUPERSECRET-TOKEN-123"
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True, report=_report(), publish=_publish(published=True), reason=None
        )
    )
    sync = AsyncMock(side_effect=RuntimeError(f"boom {secret}"))
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(automation, "manual_sync", sync)
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    rows = await _recent(factory)
    row = rows[0]
    assert row.status == "failure"
    assert row.outcome == "error"
    # The runtime forwards the exception message as-is (the orchestrator's own
    # errors never carry credentials); the audit path itself adds nothing secret.
    assert "boom" in (row.error or "")


# --------------------------------------------------------------------------- #
# Failure notification
# --------------------------------------------------------------------------- #


async def test_notify_failure_only_when_opted_in(factory, monkeypatch):
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True, report=_report(), publish=_publish(published=True), reason=None
        )
    )
    sync = AsyncMock(
        return_value=SyncReport(
            ok=False,
            outcome="unreachable",
            preview=None,
            publish=None,
            diagnosis_ok=None,
            reason="x",
        )
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(automation, "manual_sync", sync)

    sent = []

    async def fake_send(app, *, chat_id, text):
        sent.append(text)

    fake_tg = SimpleNamespace()
    import financial_dashboard.services.telegram as telegram_service

    monkeypatch.setattr(telegram_service, "tg_app", fake_tg)
    monkeypatch.setattr(telegram_service, "_send_with_retry", fake_send)
    monkeypatch.setattr(
        "financial_dashboard.services.settings.is_telegram_configured", lambda: True
    )
    monkeypatch.setattr(
        "financial_dashboard.services.settings.get_telegram_chat_id", lambda: 123
    )

    async def _clear_runs():
        from sqlalchemy import delete

        async with factory() as s:
            await s.execute(delete(ExtensionRun))
            await s.commit()

    # Opted out: a real failure must NOT notify.
    _set_paisa_settings(notify_sync_failures="false")
    await PaisaAutomationRuntime(session_factory=factory).after_fetch_cycle()
    assert sent == []

    # Opted in (fresh debounce window so the cycle actually runs): one notify.
    await _clear_runs()
    _set_paisa_settings(notify_sync_failures="true")
    await PaisaAutomationRuntime(session_factory=factory).after_fetch_cycle()
    assert len(sent) == 1
    assert "unreachable" in sent[0]


# --------------------------------------------------------------------------- #
# One run per cycle
# --------------------------------------------------------------------------- #


async def test_at_most_one_run_per_cycle(factory, monkeypatch):
    _set_paisa_settings()
    gen = AsyncMock(
        return_value=GenerateResult(
            ok=True, report=_report(), publish=_publish(published=False), reason=None
        )
    )
    monkeypatch.setattr(automation, "generate", gen)
    monkeypatch.setattr(
        automation,
        "manual_sync",
        AsyncMock(side_effect=AssertionError("should not run when unchanged")),
    )
    rt = PaisaAutomationRuntime(session_factory=factory)
    await rt.after_fetch_cycle()
    gen.assert_awaited_once()  # generate called exactly once
    rows = await _recent(factory)
    assert len(rows) == 1  # exactly one audit row


# --------------------------------------------------------------------------- #
# Notification dedupe
# --------------------------------------------------------------------------- #


async def _age_last_run(factory, *, hours=2):
    """Push the most recent automatic run's started_at into the past so the next
    cycle is not debounced, while preserving its persisted notify_fp."""
    async with factory() as s:
        rows = await recent_runs(s, extension_id="paisa", operation=OPERATION_AUTOMATIC)
        if rows:
            rows[0].started_at = datetime.datetime.now(
                datetime.UTC
            ) - datetime.timedelta(hours=hours)
        await s.commit()


async def test_notify_dedupes_repeated_identical_failures(factory, monkeypatch):
    _set_paisa_settings(notify_sync_failures="true", auto_sync_min_interval_minutes="1")
    sent = []

    async def fake_send(app, *, chat_id, text):
        sent.append(text)

    import financial_dashboard.services.telegram as telegram_service

    monkeypatch.setattr(telegram_service, "tg_app", SimpleNamespace())
    monkeypatch.setattr(telegram_service, "_send_with_retry", fake_send)
    monkeypatch.setattr(
        "financial_dashboard.services.settings.is_telegram_configured", lambda: True
    )
    monkeypatch.setattr(
        "financial_dashboard.services.settings.get_telegram_chat_id", lambda: 123
    )

    # Same identical failure for the first two cycles.
    monkeypatch.setattr(
        automation,
        "generate",
        AsyncMock(
            return_value=GenerateResult(
                ok=True, report=_report(), publish=_publish(published=True), reason=None
            )
        ),
    )

    def sync_returning(outcome, reason):
        return AsyncMock(
            return_value=SyncReport(
                ok=False,
                outcome=outcome,
                preview=None,
                publish=None,
                diagnosis_ok=None,
                reason=reason,
            )
        )

    monkeypatch.setattr(
        automation, "manual_sync", sync_returning("unreachable", "timeout")
    )

    await PaisaAutomationRuntime(session_factory=factory).after_fetch_cycle()
    assert len(sent) == 1  # first failure notifies

    await _age_last_run(factory)  # past the 1-min window so cycle 2 actually runs
    await PaisaAutomationRuntime(session_factory=factory).after_fetch_cycle()
    # Identical failure (same outcome + reason) → deduped, no second notify.
    assert len(sent) == 1
    # The second run was still recorded (audit row), just not notified.
    assert len(await _recent(factory)) == 2

    # A DIFFERENT failure (changed outcome) breaks the dedupe → notifies again.
    await _age_last_run(factory)
    monkeypatch.setattr(
        automation, "manual_sync", sync_returning("sync_rejected", "rejected")
    )
    await PaisaAutomationRuntime(session_factory=factory).after_fetch_cycle()
    assert len(sent) == 2
    assert "sync_rejected" in sent[1]


async def test_notify_dedupe_fingerprint_in_details(factory, monkeypatch):
    """The dedupe fingerprint is persisted in audit details and survives a fresh
    runtime instance (no in-memory-only state)."""
    _set_paisa_settings(notify_sync_failures="true", auto_sync_min_interval_minutes="1")
    sent = []

    async def fake_send(app, *, chat_id, text):
        sent.append(text)

    import financial_dashboard.services.telegram as telegram_service

    monkeypatch.setattr(telegram_service, "tg_app", SimpleNamespace())
    monkeypatch.setattr(telegram_service, "_send_with_retry", fake_send)
    monkeypatch.setattr(
        "financial_dashboard.services.settings.is_telegram_configured", lambda: True
    )
    monkeypatch.setattr(
        "financial_dashboard.services.settings.get_telegram_chat_id", lambda: 123
    )
    monkeypatch.setattr(
        automation,
        "generate",
        AsyncMock(
            return_value=GenerateResult(
                ok=True, report=_report(), publish=_publish(published=True), reason=None
            )
        ),
    )
    monkeypatch.setattr(
        automation,
        "manual_sync",
        AsyncMock(
            return_value=SyncReport(
                ok=False,
                outcome="unreachable",
                preview=None,
                publish=None,
                diagnosis_ok=None,
                reason="timeout",
            )
        ),
    )

    await PaisaAutomationRuntime(session_factory=factory).after_fetch_cycle()
    import json

    rows = await _recent(factory)
    details = json.loads(rows[0].details)
    assert "notify_fp" in details
    assert details["notify_sent"] is True
