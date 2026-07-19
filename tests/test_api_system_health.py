import asyncio
import logging
import pytest
from sqlalchemy import event, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db.models import Account
from financial_dashboard.services import database as database_service

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _reset_quick_check_cache():
    database_service._reset_quick_check_cache()
    yield
    database_service._reset_quick_check_cache()


class _ScalarResult:
    def __init__(self, value: object):
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


async def test_system_health_real_sqlite_shape_redaction_and_read_only(client, session):
    pending_account = Account(
        bank="synthetic",
        label="Must remain pending",
        type="bank_account",
    )
    session.add(pending_account)

    statements: list[str] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        response = await client.get("/api/system/health")
        cached_response = await client.get("/api/system/health")
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"]["backend"] == "sqlite"
    assert body["database"]["connected"] is True
    assert set(body) == {"status", "database"}
    assert set(body["database"]) == {"backend", "connected", "sqlite"}

    sqlite = body["database"]["sqlite"]
    assert set(sqlite) == {
        "journal_mode",
        "foreign_keys_enabled",
        "busy_timeout_ms",
        "synchronous_mode",
        "quick_check",
        "quick_check_source",
        "diagnostics_complete",
    }
    assert sqlite["journal_mode"] == "memory"
    # Foreign keys are intentionally disabled in the test DB and do not degrade it.
    assert sqlite["foreign_keys_enabled"] is False
    assert isinstance(sqlite["busy_timeout_ms"], int)
    assert sqlite["busy_timeout_ms"] >= 0
    assert sqlite["synchronous_mode"] in {"off", "normal", "full", "extra"}
    assert sqlite["quick_check"] == "ok"
    assert sqlite["quick_check_source"] == "live"
    assert sqlite["diagnostics_complete"] is True

    assert cached_response.status_code == 200
    cached_sqlite = cached_response.json()["database"]["sqlite"]
    assert cached_sqlite["quick_check"] == "ok"
    assert cached_sqlite["quick_check_source"] == "cache"

    normalized_statements = [
        " ".join(statement.split()).lower() for statement in statements
    ]
    assert normalized_statements == [
        "select 1",
        "pragma journal_mode",
        "pragma foreign_keys",
        "pragma busy_timeout",
        "pragma synchronous",
        "pragma quick_check(1)",
        "select 1",
        "pragma journal_mode",
        "pragma foreign_keys",
        "pragma busy_timeout",
        "pragma synchronous",
    ]
    assert normalized_statements.count("pragma quick_check(1)") == 1
    assert not any(
        statement.startswith(("insert", "update", "delete", "alter", "create", "drop"))
        for statement in normalized_statements
    )
    assert inspect(pending_account).pending
    assert pending_account.id is None

    response_text = response.text.lower()
    for forbidden in (
        "db_url",
        "database_url",
        "credential",
        "password",
        "secret",
        "sqlite+aiosqlite",
        "financial_dashboard.db",
    ):
        assert forbidden not in response_text


async def test_system_health_quick_check_is_single_flight_per_engine(
    session, monkeypatch
):
    bind = session.bind
    assert bind is not None
    session_maker = async_sessionmaker(bind, expire_on_commit=False)
    original_quick_check = database_service._execute_quick_check
    original_get_bind = AsyncSession.get_bind
    original_session_execute = AsyncSession.execute
    quick_check_calls = 0
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    second_statements: list[str] = []

    async def delayed_quick_check(health_session):
        nonlocal quick_check_calls
        quick_check_calls += 1
        first_started.set()
        await release_first.wait()
        return await original_quick_check(health_session)

    monkeypatch.setattr(database_service, "_execute_quick_check", delayed_quick_check)

    async with session_maker() as first_session, session_maker() as second_session:

        def observe_get_bind(self, *args, **kwargs):
            if self is second_session:
                second_started.set()
            return original_get_bind(self, *args, **kwargs)

        async def observe_execute(self, statement, *args, **kwargs):
            if self is second_session:
                second_statements.append(str(statement))
            return await original_session_execute(self, statement, *args, **kwargs)

        monkeypatch.setattr(AsyncSession, "get_bind", observe_get_bind)
        monkeypatch.setattr(AsyncSession, "execute", observe_execute)

        first = asyncio.create_task(database_service.get_system_health(first_session))
        await first_started.wait()
        second = asyncio.create_task(database_service.get_system_health(second_session))
        await second_started.wait()
        await asyncio.sleep(0)

        assert not second.done()
        assert not second_session.in_transaction()
        assert second_statements == []

        release_first.set()
        first_result, second_result = await asyncio.gather(first, second)

    assert quick_check_calls == 1
    assert first_result.database.sqlite is not None
    assert first_result.database.sqlite.quick_check == "ok"
    assert first_result.database.sqlite.quick_check_source == "live"
    assert second_result.database.sqlite is not None
    assert second_result.database.sqlite.quick_check == "ok"
    assert second_result.database.sqlite.quick_check_source == "cache"


async def test_system_health_connectivity_failure_is_typed_and_sanitized(
    client, session, monkeypatch, caplog
):
    secret_detail = "sqlite+aiosqlite:////private/operator/health.db?token=secret"
    original_execute = AsyncSession.execute

    async def fail_connectivity(self, statement, *args, **kwargs):
        if str(statement).strip().upper() == "SELECT 1":
            raise SQLAlchemyError(secret_detail)
        return await original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", fail_connectivity)
    with caplog.at_level(logging.WARNING, logger=database_service.__name__):
        response = await client.get("/api/system/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "unavailable",
        "database": {
            "backend": "sqlite",
            "connected": False,
            "sqlite": {
                "journal_mode": None,
                "foreign_keys_enabled": None,
                "busy_timeout_ms": None,
                "synchronous_mode": None,
                "quick_check": "unavailable",
                "quick_check_source": "unavailable",
                "diagnostics_complete": False,
            },
        },
    }
    assert secret_detail not in response.text
    assert any(
        secret_detail in record.getMessage()
        for record in caplog.records
        if record.name == database_service.__name__
    )


async def test_system_health_sqlite_diagnostic_failure_is_fail_fast_and_sanitized(
    client, monkeypatch, caplog
):
    secret_detail = "diagnostic failed at /private/database/path"
    original_execute = AsyncSession.execute
    diagnostic_statements: list[str] = []

    async def poison_after_foreign_keys(self, statement, *args, **kwargs):
        normalized = str(statement).strip().upper()
        if normalized.startswith("PRAGMA"):
            diagnostic_statements.append(normalized)
        if normalized == "PRAGMA FOREIGN_KEYS":
            raise SQLAlchemyError(secret_detail)
        if normalized in {
            "PRAGMA BUSY_TIMEOUT",
            "PRAGMA SYNCHRONOUS",
            "PRAGMA QUICK_CHECK(1)",
        }:
            raise AssertionError("diagnostics continued after a DBAPI-style failure")
        return await original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", poison_after_foreign_keys)
    with caplog.at_level(logging.WARNING, logger=database_service.__name__):
        response = await client.get("/api/system/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"]["connected"] is True
    assert body["database"]["sqlite"] == {
        "journal_mode": "memory",
        "foreign_keys_enabled": None,
        "busy_timeout_ms": None,
        "synchronous_mode": None,
        "quick_check": "unavailable",
        "quick_check_source": "unavailable",
        "diagnostics_complete": False,
    }
    assert diagnostic_statements == ["PRAGMA JOURNAL_MODE", "PRAGMA FOREIGN_KEYS"]
    assert secret_detail not in response.text
    assert any(
        secret_detail in record.getMessage()
        for record in caplog.records
        if record.name == database_service.__name__
    )


async def test_system_health_quick_check_returns_only_failed_summary(
    client, monkeypatch
):
    raw_diagnostic = "corrupt page at /private/database/path with secret detail"
    original_execute = AsyncSession.execute

    async def fail_quick_check(self, statement, *args, **kwargs):
        if str(statement).strip().upper() == "PRAGMA QUICK_CHECK(1)":
            return _ScalarResult(raw_diagnostic)
        return await original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", fail_quick_check)
    response = await client.get("/api/system/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"]["sqlite"]["quick_check"] == "failed"
    assert body["database"]["sqlite"]["quick_check_source"] == "live"
    assert body["database"]["sqlite"]["diagnostics_complete"] is True
    assert raw_diagnostic not in response.text
    assert "corrupt page" not in response.text

    cached_response = await client.get("/api/system/health")
    cached_sqlite = cached_response.json()["database"]["sqlite"]
    assert cached_sqlite["quick_check"] == "failed"
    assert cached_sqlite["quick_check_source"] == "cache"


async def test_system_health_quick_check_execution_failure_is_unavailable_and_retried(
    client, monkeypatch
):
    original_execute = AsyncSession.execute
    quick_check_calls = 0

    async def fail_first_quick_check(self, statement, *args, **kwargs):
        nonlocal quick_check_calls
        if str(statement).strip().upper() == "PRAGMA QUICK_CHECK(1)":
            quick_check_calls += 1
            if quick_check_calls == 1:
                raise SQLAlchemyError("private quick-check execution detail")
        return await original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", fail_first_quick_check)

    failed_response = await client.get("/api/system/health")
    assert failed_response.status_code == 200
    failed_body = failed_response.json()
    assert failed_body["status"] == "degraded"
    assert failed_body["database"]["connected"] is True
    assert failed_body["database"]["sqlite"]["quick_check"] == "unavailable"
    assert failed_body["database"]["sqlite"]["quick_check_source"] == "unavailable"
    assert failed_body["database"]["sqlite"]["diagnostics_complete"] is False
    assert "private quick-check execution detail" not in failed_response.text

    retried_response = await client.get("/api/system/health")
    retried_sqlite = retried_response.json()["database"]["sqlite"]
    assert retried_response.json()["status"] == "ok"
    assert retried_sqlite["quick_check"] == "ok"
    assert retried_sqlite["quick_check_source"] == "live"
    assert quick_check_calls == 2


async def test_system_health_openapi_uses_inferred_typed_response(client):
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    operation = document["paths"]["/api/system/health"]["get"]
    response_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    assert response_schema == {"$ref": "#/components/schemas/SystemHealthResponse"}

    health_schema = document["components"]["schemas"]["SystemHealthResponse"]
    assert set(health_schema["required"]) == {"status", "database"}
    assert set(health_schema["properties"]["status"]["enum"]) == {
        "ok",
        "degraded",
        "unavailable",
    }
