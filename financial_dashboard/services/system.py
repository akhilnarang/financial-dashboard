import asyncio
import logging
import time
from weakref import WeakKeyDictionary

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause

from financial_dashboard.db.models import Setting
from financial_dashboard.schemas import system as system_schemas
from financial_dashboard.services import system_metadata

SCHEMA_VERSION_SETTING = "migrations.schema_version"

_SELECT_CONNECTIVITY = text("SELECT 1")
_SQLITE_JOURNAL_MODE = text("PRAGMA journal_mode")
_SQLITE_FOREIGN_KEYS = text("PRAGMA foreign_keys")
_SQLITE_BUSY_TIMEOUT = text("PRAGMA busy_timeout")
_SQLITE_SYNCHRONOUS = text("PRAGMA synchronous")
_SQLITE_QUICK_CHECK = text("PRAGMA quick_check(1)")
_SQLITE_FOREIGN_KEY_CHECK = text(
    """
    SELECT
        "table" AS child_table,
        rowid AS child_row_id,
        parent AS parent_table,
        fkid AS fk_constraint_index
    FROM pragma_foreign_key_check
    ORDER BY
        "table" COLLATE BINARY ASC,
        rowid ASC,
        parent COLLATE BINARY ASC,
        fkid ASC
    LIMIT :fetch_limit
    """
)
_SQLITE_SCHEMA_NAME_MAX_LENGTH = 256
_QUICK_CHECK_TTL_SECONDS = 5 * 60.0

logger = logging.getLogger(__name__)


class _QuickCheckCacheState:
    def __init__(self) -> None:
        self.result: system_schemas.SQLiteQuickCheck | None = None
        self.expires_at = 0.0
        # The lock is scoped to one Engine, so unrelated databases never block
        # one another. The request that owns it executes with its own session;
        # if that request is cancelled, the lock is released and the next
        # waiter retries instead of inheriting a task tied to a closed session.
        self.lock = asyncio.Lock()


_quick_check_cache: WeakKeyDictionary[Engine, _QuickCheckCacheState] = (
    WeakKeyDictionary()
)


async def _schema_state(session: AsyncSession) -> system_schemas.SchemaState:
    schema_version_row = await session.get(Setting, SCHEMA_VERSION_SETTING)
    schema_version = None
    if schema_version_row is not None and schema_version_row.value:
        schema_version = schema_version_row.value

    applied_markers = (
        (
            await session.execute(
                select(Setting.key)
                .where(
                    Setting.key.like("migrations.%"),
                    Setting.key != SCHEMA_VERSION_SETTING,
                )
                .order_by(Setting.key)
            )
        )
        .scalars()
        .all()
    )

    return system_schemas.SchemaState(
        schema_version=schema_version,
        applied_migration_markers=list(applied_markers),
    )


async def get_system_info(session: AsyncSession) -> system_schemas.SystemInfoResponse:
    runtime_metadata = await asyncio.to_thread(system_metadata.collect_runtime_metadata)
    return system_schemas.SystemInfoResponse(
        package_name=system_metadata.APP_DISTRIBUTION,
        package_version=runtime_metadata.package_version,
        app_revision=runtime_metadata.app_revision.value,
        app_revision_source=runtime_metadata.app_revision.source,
        runtime=runtime_metadata.runtime,
        schema_state=await _schema_state(session),
        parser_packages=runtime_metadata.parser_packages,
    )


def _database_engine(session: AsyncSession) -> Engine:
    bind = session.get_bind()
    if isinstance(bind, Engine):
        return bind
    return bind.engine


def _database_backend(engine: Engine) -> system_schemas.DatabaseBackend:
    return "sqlite" if engine.dialect.name == "sqlite" else "other"


def _empty_foreign_key_check_response(
    *,
    status: system_schemas.ForeignKeyCheckStatus,
    backend: system_schemas.DatabaseBackend,
    limit: int,
) -> system_schemas.ForeignKeyCheckResponse:
    return system_schemas.ForeignKeyCheckResponse(
        status=status,
        backend=backend,
        returned_count=0,
        limit=limit,
        truncated=False,
        violations=[],
    )


def _safe_sqlite_schema_name(value: object) -> str:
    """Bound schema identifiers and replace characters unsafe for JSON output."""
    if not isinstance(value, str) or not value:
        return "[invalid]"

    normalized = "".join(
        character if character.isprintable() else "�" for character in value
    )
    return normalized[:_SQLITE_SCHEMA_NAME_MAX_LENGTH] or "[invalid]"


def _foreign_key_violation(
    row: system_schemas.SQLiteForeignKeyCheckRow,
) -> system_schemas.ForeignKeyViolation | None:
    row_id = row.child_row_id
    if row_id is not None and (not isinstance(row_id, int) or isinstance(row_id, bool)):
        return None

    constraint_index = row.fk_constraint_index
    if (
        not isinstance(constraint_index, int)
        or isinstance(constraint_index, bool)
        or constraint_index < 0
    ):
        return None

    return system_schemas.ForeignKeyViolation(
        child_table=_safe_sqlite_schema_name(row.child_table),
        child_row_id=row_id,
        parent_table=_safe_sqlite_schema_name(row.parent_table),
        fk_constraint_index=constraint_index,
    )


async def get_system_foreign_key_check(
    session: AsyncSession,
    *,
    limit: int,
) -> system_schemas.ForeignKeyCheckResponse:
    """Return a bounded, redacted summary of SQLite foreign-key violations."""
    try:
        backend = _database_backend(_database_engine(session))
    except SQLAlchemyError as exc:
        logger.warning(
            "Could not identify database backend for foreign-key check: %s", exc
        )
        return _empty_foreign_key_check_response(
            status="unavailable",
            backend="other",
            limit=limit,
        )

    if backend != "sqlite":
        return _empty_foreign_key_check_response(
            status="unsupported",
            backend="other",
            limit=limit,
        )

    with session.no_autoflush:
        try:
            result = await session.execute(
                _SQLITE_FOREIGN_KEY_CHECK,
                {"fetch_limit": limit + 1},
                execution_options={"autoflush": False},
            )
            rows = result.mappings().all()
        except SQLAlchemyError as exc:
            logger.warning("SQLite foreign-key check failed: %s", exc)
            return _empty_foreign_key_check_response(
                status="unavailable",
                backend="sqlite",
                limit=limit,
            )

    violations: list[system_schemas.ForeignKeyViolation] = []
    for result_row in rows[:limit]:
        violation = _foreign_key_violation(
            system_schemas.SQLiteForeignKeyCheckRow(
                child_table=result_row["child_table"],
                child_row_id=result_row["child_row_id"],
                parent_table=result_row["parent_table"],
                fk_constraint_index=result_row["fk_constraint_index"],
            )
        )
        if violation is None:
            logger.warning("SQLite foreign-key check returned malformed metadata")
            return _empty_foreign_key_check_response(
                status="unavailable",
                backend="sqlite",
                limit=limit,
            )
        violations.append(violation)

    truncated = len(rows) > limit
    return system_schemas.ForeignKeyCheckResponse(
        status="violations" if violations else "ok",
        backend="sqlite",
        returned_count=len(violations),
        limit=limit,
        truncated=truncated,
        violations=violations,
    )


async def _diagnostic_scalar(
    session: AsyncSession,
    statement: TextClause,
    diagnostic_name: str,
) -> system_schemas.DiagnosticScalarResult:
    try:
        result = await session.execute(
            statement,
            execution_options={"autoflush": False},
        )
        return system_schemas.DiagnosticScalarResult(
            value=result.scalar_one_or_none(),
            succeeded=True,
        )
    except SQLAlchemyError as exc:
        logger.warning("SQLite %s health diagnostic failed: %s", diagnostic_name, exc)
        return system_schemas.DiagnosticScalarResult(value=None, succeeded=False)


def _journal_mode(value: object | None) -> system_schemas.SQLiteJournalMode:
    if not isinstance(value, str):
        return "unknown"
    match value.lower():
        case "delete":
            return "delete"
        case "truncate":
            return "truncate"
        case "persist":
            return "persist"
        case "memory":
            return "memory"
        case "wal":
            return "wal"
        case "off":
            return "off"
        case _:
            return "unknown"


def _synchronous_mode(value: object | None) -> system_schemas.SQLiteSynchronousMode:
    match value:
        case 0:
            return "off"
        case 1:
            return "normal"
        case 2:
            return "full"
        case 3:
            return "extra"
        case _:
            return "unknown"


def _enabled_flag(value: object | None) -> bool | None:
    if value == 0:
        return False
    if value == 1:
        return True
    return None


def _busy_timeout(value: object | None) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _reset_quick_check_cache() -> None:
    """Clear process-local quick-check state for test isolation."""
    _quick_check_cache.clear()


def _quick_check_cache_state(engine: Engine) -> _QuickCheckCacheState:
    state = _quick_check_cache.get(engine)
    if state is None:
        state = _QuickCheckCacheState()
        _quick_check_cache[engine] = state
    return state


def _cached_quick_check(
    state: _QuickCheckCacheState,
) -> system_schemas.QuickCheckDiagnosticResult | None:
    if state.result is None or time.monotonic() >= state.expires_at:
        return None
    return system_schemas.QuickCheckDiagnosticResult(state.result, "cache")


def _cache_quick_check(
    state: _QuickCheckCacheState,
    result: system_schemas.SQLiteQuickCheck,
) -> None:
    state.result = result
    state.expires_at = time.monotonic() + _QUICK_CHECK_TTL_SECONDS


async def _execute_quick_check(
    session: AsyncSession,
) -> system_schemas.QuickCheckDiagnosticResult:
    """Run a live scan directly; the caller owns any required coordination."""
    result = await _diagnostic_scalar(session, _SQLITE_QUICK_CHECK, "quick check")
    if not result.succeeded:
        return system_schemas.QuickCheckDiagnosticResult("unavailable", "unavailable")

    quick_check: system_schemas.SQLiteQuickCheck = (
        "ok"
        if isinstance(result.value, str) and result.value.lower() == "ok"
        else "failed"
    )
    return system_schemas.QuickCheckDiagnosticResult(quick_check, "live")


def _unavailable_sqlite_diagnostics(
    *,
    journal_mode: system_schemas.SQLiteJournalMode | None = None,
    foreign_keys_enabled: bool | None = None,
    busy_timeout_ms: int | None = None,
    synchronous_mode: system_schemas.SQLiteSynchronousMode | None = None,
) -> system_schemas.SQLiteHealthDiagnostics:
    return system_schemas.SQLiteHealthDiagnostics(
        journal_mode=journal_mode,
        foreign_keys_enabled=foreign_keys_enabled,
        busy_timeout_ms=busy_timeout_ms,
        synchronous_mode=synchronous_mode,
        quick_check="unavailable",
        quick_check_source="unavailable",
        diagnostics_complete=False,
    )


async def _sqlite_diagnostics(
    session: AsyncSession,
    *,
    quick_check_result: system_schemas.QuickCheckDiagnosticResult | None,
) -> system_schemas.SQLiteHealthDiagnostics:
    """Run cheap diagnostics, then use a cache snapshot or execute a live scan.

    A ``None`` quick-check result is the direct-refresh path and is only used by
    the request holding the engine's quick-check coordinator.
    """
    journal_result = await _diagnostic_scalar(
        session, _SQLITE_JOURNAL_MODE, "journal mode"
    )
    if not journal_result.succeeded:
        return _unavailable_sqlite_diagnostics()
    journal_mode = _journal_mode(journal_result.value)

    foreign_keys_result = await _diagnostic_scalar(
        session, _SQLITE_FOREIGN_KEYS, "foreign keys"
    )
    if not foreign_keys_result.succeeded:
        return _unavailable_sqlite_diagnostics(journal_mode=journal_mode)
    foreign_keys_enabled = _enabled_flag(foreign_keys_result.value)

    busy_timeout_result = await _diagnostic_scalar(
        session, _SQLITE_BUSY_TIMEOUT, "busy timeout"
    )
    if not busy_timeout_result.succeeded:
        return _unavailable_sqlite_diagnostics(
            journal_mode=journal_mode,
            foreign_keys_enabled=foreign_keys_enabled,
        )
    busy_timeout_ms = _busy_timeout(busy_timeout_result.value)

    synchronous_result = await _diagnostic_scalar(
        session, _SQLITE_SYNCHRONOUS, "synchronous mode"
    )
    if not synchronous_result.succeeded:
        return _unavailable_sqlite_diagnostics(
            journal_mode=journal_mode,
            foreign_keys_enabled=foreign_keys_enabled,
            busy_timeout_ms=busy_timeout_ms,
        )
    synchronous_mode = _synchronous_mode(synchronous_result.value)

    if quick_check_result is None:
        quick_check_result = await _execute_quick_check(session)
    quick_check, quick_check_source = quick_check_result
    diagnostics_complete = (
        journal_mode != "unknown"
        and foreign_keys_enabled is not None
        and busy_timeout_ms is not None
        and synchronous_mode != "unknown"
        and quick_check != "unavailable"
    )
    return system_schemas.SQLiteHealthDiagnostics(
        journal_mode=journal_mode,
        foreign_keys_enabled=foreign_keys_enabled,
        busy_timeout_ms=busy_timeout_ms,
        synchronous_mode=synchronous_mode,
        quick_check=quick_check,
        quick_check_source=quick_check_source,
        diagnostics_complete=diagnostics_complete,
    )


async def _run_system_health_checks(
    session: AsyncSession,
    backend: system_schemas.DatabaseBackend,
    *,
    quick_check_result: system_schemas.QuickCheckDiagnosticResult | None = None,
) -> system_schemas.SystemHealthResponse:
    """Perform DB I/O after the caller has selected quick-check coordination."""
    with session.no_autoflush:
        try:
            connectivity = await session.execute(
                _SELECT_CONNECTIVITY,
                execution_options={"autoflush": False},
            )
            connected = connectivity.scalar_one_or_none() == 1
        except SQLAlchemyError as exc:
            logger.warning("Database connectivity health check failed: %s", exc)
            connected = False

        if not connected:
            sqlite = _unavailable_sqlite_diagnostics() if backend == "sqlite" else None
            return system_schemas.SystemHealthResponse(
                status="unavailable",
                database=system_schemas.DatabaseHealth(
                    backend=backend,
                    connected=False,
                    sqlite=sqlite,
                ),
            )

        if backend != "sqlite":
            return system_schemas.SystemHealthResponse(
                status="ok",
                database=system_schemas.DatabaseHealth(
                    backend="other",
                    connected=True,
                ),
            )

        sqlite = await _sqlite_diagnostics(
            session,
            quick_check_result=quick_check_result,
        )
        status: system_schemas.HealthStatus = (
            "ok"
            if sqlite.diagnostics_complete and sqlite.quick_check == "ok"
            else "degraded"
        )
        return system_schemas.SystemHealthResponse(
            status=status,
            database=system_schemas.DatabaseHealth(
                backend="sqlite",
                connected=True,
                sqlite=sqlite,
            ),
        )


async def get_system_health(
    session: AsyncSession,
) -> system_schemas.SystemHealthResponse:
    # AsyncSession.get_bind() resolves the sync Engine without checking a
    # connection out of the pool or opening a transaction.
    try:
        engine = _database_engine(session)
        backend = _database_backend(engine)
    except SQLAlchemyError as exc:
        logger.warning("Could not identify database backend for health check: %s", exc)
        return await _run_system_health_checks(session, "other")

    if backend != "sqlite":
        return await _run_system_health_checks(session, backend)

    state = _quick_check_cache_state(engine)
    cached_result = _cached_quick_check(state)
    if cached_result is not None:
        # Warm requests take a cache snapshot and never acquire the lock.
        return await _run_system_health_checks(
            session,
            backend,
            quick_check_result=cached_result,
        )

    async with state.lock:
        # Waiting requests have done no DB I/O. If the owner populated the
        # cache, release coordination before running their cheap diagnostics.
        cached_result = _cached_quick_check(state)
        if cached_result is None:
            response = await _run_system_health_checks(session, backend)
            sqlite = response.database.sqlite
            if (
                sqlite is not None
                and sqlite.quick_check != "unavailable"
                and sqlite.quick_check_source == "live"
            ):
                # Cache every completed scan independently of the cheap pragma
                # summaries. An unknown journal/synchronous value should make
                # the response degraded, not force another O(N) scan next time.
                _cache_quick_check(state, sqlite.quick_check)
            return response

    return await _run_system_health_checks(
        session,
        backend,
        quick_check_result=cached_result,
    )
