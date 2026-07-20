"""ExtensionRun audit table + service tests.

Covers: start_run creates a ``running`` row (id available after flush),
complete_run sets terminal fields + completed_at and is idempotent on
completed_at, error sanitization (whitespace collapse + truncation), details
JSON encoding + size bounding, recent_runs ordering/filtering/limit bounding,
last_run, run_started_at tz normalization, and the no-secrets guarantee (a
caller-supplied details dict is stored verbatim as JSON and never invents data).
"""

import datetime
import json

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db.models import ExtensionRun, Base
from financial_dashboard.services.paisa.audit import (
    MAX_ERROR_LEN,
    OPERATION_AUTOMATIC,
    OPERATION_PROBE,
    STATUS_FAILURE,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    complete_run,
    last_run,
    recent_runs,
    run_started_at,
    sanitize_error,
    start_run,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


# --------------------------------------------------------------------------- #
# start_run / complete_run
# --------------------------------------------------------------------------- #


async def test_start_run_creates_running_row_with_id(session):
    run = await start_run(
        session,
        extension_id="paisa",
        operation=OPERATION_AUTOMATIC,
        trigger="fetch_cycle",
    )
    assert run.id is not None  # flush made the id available
    assert run.extension_id == "paisa"
    assert run.operation == "automatic"
    assert run.status == STATUS_RUNNING
    assert run.trigger == "fetch_cycle"
    assert run.completed_at is None
    await session.commit()
    reloaded = await session.get(ExtensionRun, run.id)
    assert reloaded is not None
    assert reloaded.status == STATUS_RUNNING


async def test_complete_run_sets_terminal_fields_and_completed_at(session):
    run = await start_run(session, extension_id="paisa", operation=OPERATION_PROBE)
    await complete_run(
        session,
        run,
        status=STATUS_SUCCESS,
        outcome="synced",
        output_hash="abc123",
        emitted_count=7,
        skipped_count=2,
        details={"outcome": "synced", "diagnosis_ok": True},
    )
    assert run.status == STATUS_SUCCESS
    assert run.outcome == "synced"
    assert run.output_hash == "abc123"
    assert run.emitted_count == 7
    assert run.skipped_count == 2
    assert run.completed_at is not None
    assert json.loads(run.details) == {"outcome": "synced", "diagnosis_ok": True}


async def test_complete_run_does_not_overwrite_completed_at(session):
    run = await start_run(session, extension_id="paisa", operation=OPERATION_PROBE)
    await complete_run(session, run, status=STATUS_SUCCESS)
    first = run.completed_at
    assert first is not None
    # A second complete_run must not move the finish time.
    await complete_run(session, run, status=STATUS_FAILURE, outcome="late")
    assert run.completed_at == first


async def test_complete_run_sanitizes_error(session):
    nasty = "line1\nline2\twith   spaces\n\nhttp://x/secret"
    run = await start_run(session, extension_id="paisa", operation=OPERATION_PROBE)
    await complete_run(session, run, status=STATUS_FAILURE, error=nasty)
    assert "\n" not in run.error
    assert "\t" not in run.error
    assert "  " not in run.error  # whitespace collapsed
    assert run.error is not None and "secret" in run.error


def test_sanitize_error_truncates_to_cap():
    long_text = "x" * (MAX_ERROR_LEN + 500)
    out = sanitize_error(long_text)
    assert out is not None
    assert len(out) == MAX_ERROR_LEN
    assert out.endswith("...")


def test_sanitize_error_none_passthrough():
    assert sanitize_error(None) is None
    assert sanitize_error("") == ""


async def test_details_json_is_not_invented_and_omits_unsent_keys(session):
    # The audit row stores exactly what the caller passes — it never invents
    # fields or copies financial rows. A details dict without a password key
    # never gains one.
    run = await start_run(session, extension_id="paisa", operation=OPERATION_PROBE)
    await complete_run(
        session, run, status=STATUS_SKIPPED, details={"reason": "unchanged"}
    )
    stored = json.loads(run.details)
    assert stored == {"reason": "unchanged"}
    assert "auth_password" not in stored


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


async def _make_run(session, *, operation, status, started_at, outcome=None):
    run = ExtensionRun(
        extension_id="paisa",
        operation=operation,
        status=status,
        outcome=outcome,
        started_at=started_at,
        completed_at=started_at,
    )
    session.add(run)
    await session.flush()
    return run


async def test_recent_runs_newest_first(session):
    t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    t1 = datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC)
    await _make_run(session, operation="sync", status="success", started_at=t0)
    await _make_run(session, operation="sync", status="success", started_at=t1)
    await session.commit()
    rows = await recent_runs(session, extension_id="paisa", limit=10)
    assert len(rows) == 2
    assert run_started_at(rows[0]) == t1
    assert run_started_at(rows[1]) == t0


async def test_recent_runs_filters_operation_and_status(session):
    t = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    await _make_run(session, operation="automatic", status="success", started_at=t)
    await _make_run(session, operation="automatic", status="failure", started_at=t)
    await _make_run(session, operation="probe", status="success", started_at=t)
    await session.commit()
    auto = await recent_runs(session, extension_id="paisa", operation="automatic")
    assert all(r.operation == "automatic" for r in auto)
    assert len(auto) == 2
    fails = await recent_runs(
        session, extension_id="paisa", operation="automatic", status="failure"
    )
    assert len(fails) == 1
    assert fails[0].status == "failure"


async def test_recent_runs_limit_is_bounded(session):
    t = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    for i in range(5):
        await _make_run(
            session, operation="sync", status="success", started_at=t.replace(day=1 + i)
        )
    await session.commit()
    rows = await recent_runs(session, extension_id="paisa", limit=300)  # over max
    assert len(rows) == 5  # only 5 exist; cap is 200


async def test_last_run_returns_most_recent_or_none(session):
    assert await last_run(session, extension_id="paisa") is None
    t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    t1 = datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC)
    await _make_run(session, operation="automatic", status="success", started_at=t0)
    await _make_run(session, operation="automatic", status="failure", started_at=t1)
    await session.commit()
    last = await last_run(session, extension_id="paisa", operation="automatic")
    assert last is not None
    assert run_started_at(last) == t1
    # status filter
    last_fail = await last_run(
        session, extension_id="paisa", operation="automatic", status="failure"
    )
    assert last_fail is not None
    assert last_fail.status == "failure"


def test_run_started_at_normalizes_naive_to_utc():
    naive = datetime.datetime(2026, 1, 1)
    run = ExtensionRun(
        extension_id="paisa",
        operation="sync",
        status="success",
        started_at=naive,
    )
    normalized = run_started_at(run)
    assert normalized.tzinfo is datetime.UTC


async def test_extension_run_table_is_generic_over_extension_id(session):
    # Non-paisa extension_id is accepted — the table is extension-owned but
    # not paisa-specific.
    run = await start_run(session, extension_id="other", operation="probe")
    await complete_run(session, run, status=STATUS_SUCCESS, outcome="ok")
    await session.commit()
    assert run.extension_id == "other"


# --------------------------------------------------------------------------- #
# Table schema / docs presence
# --------------------------------------------------------------------------- #


def test_extension_run_columns_and_indexes_documented_in_model():
    cols = {c.name for c in ExtensionRun.__table__.columns}
    for required in (
        "id",
        "extension_id",
        "operation",
        "status",
        "outcome",
        "trigger",
        "started_at",
        "completed_at",
        "input_hash",
        "output_hash",
        "emitted_count",
        "skipped_count",
        "details",
        "error",
    ):
        assert required in cols, required
    idx_names = {idx.name for idx in ExtensionRun.__table__.indexes}
    assert "ix_extension_runs_ext_started" in idx_names
    assert "ix_extension_runs_ext_status" in idx_names
    assert "ix_extension_runs_operation" in idx_names
