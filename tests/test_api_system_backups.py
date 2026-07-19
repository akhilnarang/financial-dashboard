import asyncio
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import NamedTuple

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import URL, event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from financial_dashboard.api import router as api_router
from financial_dashboard.core.deps import get_session
from financial_dashboard.services import system_backups

pytestmark = pytest.mark.anyio


@pytest.fixture
async def backup_api(tmp_path):
    database_path = tmp_path / "synthetic-ledger.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    async with engine.begin() as connection:
        await connection.execute(text("PRAGMA journal_mode=WAL"))
        await connection.execute(text("PRAGMA wal_autocheckpoint=0"))
        await connection.execute(text("CREATE TABLE synthetic_rows (value TEXT)"))
        await connection.execute(
            text("INSERT INTO synthetic_rows (value) VALUES ('committed-row')")
        )

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        app = FastAPI()

        async def override_session():
            yield session

        app.dependency_overrides[get_session] = override_session
        app.include_router(api_router)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, session, engine, database_path

    await engine.dispose()


class BackupPaths(NamedTuple):
    backup: Path
    manifest: Path


def _backup_root(database_path: Path) -> Path:
    return system_backups._backup_directory(database_path.resolve())


def _backup_paths(database_path: Path, backup_id: str) -> BackupPaths:
    directory = _backup_root(database_path) / backup_id
    return BackupPaths(
        directory / "backup.sqlite3",
        directory / "manifest.json",
    )


def _assert_no_paths(value: object, database_path: Path) -> None:
    encoded = json.dumps(value)
    assert str(database_path) not in encoded
    assert str(database_path.parent) not in encoded
    assert "sqlite+aiosqlite" not in encoded
    if isinstance(value, dict):
        assert not (
            {"path", "source_path", "destination_path", "database_url"} & value.keys()
        )
        for nested in value.values():
            _assert_no_paths(nested, database_path)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_paths(nested, database_path)


async def test_post_creates_verified_online_backup_with_committed_wal_rows(backup_api):
    client, session, _engine, database_path = backup_api
    wal_path = Path(f"{database_path}-wal")
    assert wal_path.exists()
    assert wal_path.stat().st_size > 0
    assert not session.in_transaction()

    response = await client.post("/api/system/backups")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "created"
    assert body["backend"] == "sqlite"
    metadata = body["backup"]
    assert metadata["status"] == "verified"
    assert len(metadata["backup_id"]) == 39
    assert "/" not in metadata["backup_id"]
    _assert_no_paths(body, database_path)
    assert not session.in_transaction()

    backup_path, manifest_path = _backup_paths(database_path, metadata["backup_id"])
    assert backup_path.is_file()
    assert manifest_path.is_file()
    assert stat.S_IMODE(backup_path.parent.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    backup_bytes = backup_path.read_bytes()
    assert metadata["size_bytes"] == len(backup_bytes)
    assert metadata["sha256"] == hashlib.sha256(backup_bytes).hexdigest()

    manifest = json.loads(manifest_path.read_text())
    assert manifest == {
        "backup_id": metadata["backup_id"],
        "created_at": metadata["created_at"],
        "quick_check": "ok",
        "sha256": metadata["sha256"],
        "size_bytes": metadata["size_bytes"],
        "version": 1,
    }
    with sqlite3.connect(backup_path) as backup_connection:
        assert backup_connection.execute(
            "SELECT value FROM synthetic_rows"
        ).fetchall() == [("committed-row",)]
        assert backup_connection.execute("PRAGMA quick_check(1)").fetchall() == [
            ("ok",)
        ]

    source_rows = await session.execute(text("SELECT value FROM synthetic_rows"))
    assert source_rows.all() == [("committed-row",)]


async def test_relative_sqlite_url_is_resolved_internally(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = create_async_engine("sqlite+aiosqlite:///relative-ledger.db")
    async with engine.begin() as connection:
        await connection.execute(text("CREATE TABLE synthetic_rows (value TEXT)"))
        await connection.execute(
            text("INSERT INTO synthetic_rows (value) VALUES ('relative-row')")
        )

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        app = FastAPI()

        async def override_session():
            yield session

        app.dependency_overrides[get_session] = override_session
        app.include_router(api_router)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            body = (await client.post("/api/system/backups")).json()

    assert body["status"] == "created"
    backup_path, _manifest_path = _backup_paths(
        tmp_path / "relative-ledger.db", body["backup"]["backup_id"]
    )
    with sqlite3.connect(backup_path) as backup_connection:
        assert backup_connection.execute(
            "SELECT value FROM synthetic_rows"
        ).fetchall() == [("relative-row",)]
    await engine.dispose()


async def test_literal_tilde_sqlite_path_is_not_expanded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    literal_directory = tmp_path / "~"
    literal_directory.mkdir()
    database_path = literal_directory / "literal-ledger.db"
    engine = create_async_engine("sqlite+aiosqlite:///~/literal-ledger.db")
    async with engine.begin() as connection:
        await connection.execute(text("CREATE TABLE synthetic_rows (value TEXT)"))
        await connection.execute(
            text("INSERT INTO synthetic_rows (value) VALUES ('literal-tilde-row')")
        )

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        response = await system_backups.create_system_backup(session)

    assert response.status == "created"
    assert response.backup is not None
    backup_path, _manifest_path = _backup_paths(
        database_path, response.backup.backup_id
    )
    with sqlite3.connect(backup_path) as backup_connection:
        assert backup_connection.execute(
            "SELECT value FROM synthetic_rows"
        ).fetchall() == [("literal-tilde-row",)]
    await engine.dispose()


async def test_backups_are_namespaced_per_database_in_shared_directory(
    backup_api, tmp_path
):
    client, _session, _engine, first_database = backup_api
    first_backup = (await client.post("/api/system/backups")).json()["backup"]

    second_database = tmp_path / "second-ledger.db"
    second_engine = create_async_engine(f"sqlite+aiosqlite:///{second_database}")
    async with second_engine.begin() as connection:
        await connection.execute(text("CREATE TABLE synthetic_rows (value TEXT)"))
    second_maker = async_sessionmaker(
        second_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with second_maker() as second_session:
        empty = await system_backups.list_system_backups(second_session, limit=50)
        assert empty.status == "ok"
        assert empty.backups == []
        second_backup = await system_backups.create_system_backup(second_session)
        assert second_backup.status == "created"

    assert _backup_root(first_database) != _backup_root(second_database)
    assert _backup_root(first_database).is_dir()
    assert _backup_root(second_database).is_dir()
    first_listing = (await client.get("/api/system/backups")).json()
    assert [item["backup_id"] for item in first_listing["backups"]] == [
        first_backup["backup_id"]
    ]
    await second_engine.dispose()


async def test_get_is_stable_bounded_and_reports_file_statuses(backup_api):
    client, _session, _engine, database_path = backup_api
    created = [
        (await client.post("/api/system/backups")).json()["backup"] for _ in range(3)
    ]
    expected_ids = sorted(
        (item["backup_id"] for item in created),
        reverse=True,
    )

    first = await client.get("/api/system/backups?limit=2")
    second = await client.get("/api/system/backups?limit=2")

    assert first.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert body["status"] == "ok"
    assert body["returned_count"] == 2
    assert body["limit"] == 2
    assert body["truncated"] is True
    assert [item["backup_id"] for item in body["backups"]] == expected_ids[:2]

    newest_path, _ = _backup_paths(database_path, expected_ids[0])
    next_path, _ = _backup_paths(database_path, expected_ids[1])
    newest_path.unlink()
    next_path.write_bytes(next_path.read_bytes() + b"mismatch")

    statuses = {
        item["backup_id"]: item["status"]
        for item in (await client.get("/api/system/backups")).json()["backups"]
    }
    assert statuses[expected_ids[0]] == "missing"
    assert statuses[expected_ids[1]] == "size_mismatch"
    assert statuses[expected_ids[2]] == "verified"


async def test_post_prunes_oldest_backups_to_retention_bound(backup_api, monkeypatch):
    client, _session, _engine, database_path = backup_api
    monkeypatch.setattr(system_backups, "_MAX_RETAINED_BACKUPS", 2)

    created_ids = [
        (await client.post("/api/system/backups")).json()["backup"]["backup_id"]
        for _ in range(3)
    ]

    backup_root = _backup_root(database_path)
    retained_ids = sorted(
        path.name for path in backup_root.iterdir() if not path.name.startswith(".")
    )
    assert retained_ids == sorted(created_ids[-2:])
    listing = (await client.get("/api/system/backups")).json()
    assert [item["backup_id"] for item in listing["backups"]] == sorted(
        created_ids[-2:], reverse=True
    )


async def test_get_ignores_malformed_manifests_without_following_paths(
    backup_api, caplog
):
    client, _session, _engine, database_path = backup_api
    valid = (await client.post("/api/system/backups")).json()["backup"]
    backup_directory = _backup_root(database_path)
    (backup_directory / "not-an-id.json").write_text("not json")
    malformed_id = "20990101T000000000000Z-0000000000000000"
    malformed_directory = backup_directory / malformed_id
    malformed_directory.mkdir()
    (malformed_directory / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backup_id": malformed_id,
                "created_at": "2099-01-01T00:00:00Z",
                "size_bytes": 1,
                "sha256": "not-a-checksum",
                "quick_check": "ok",
                "path": "/private/never-follow-this",
            }
        )
    )

    with caplog.at_level(logging.WARNING, logger=system_backups.__name__):
        response = await client.get("/api/system/backups")

    assert response.status_code == 200
    assert [item["backup_id"] for item in response.json()["backups"]] == [
        valid["backup_id"]
    ]
    assert "/private/never-follow-this" not in response.text
    assert any(
        "malformed system backup manifest" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("limit", ["0", "101", "not-an-integer"])
async def test_get_validates_limit(backup_api, limit):
    client, *_rest = backup_api

    response = await client.get("/api/system/backups", params={"limit": limit})

    assert response.status_code == 422


async def test_backup_openapi_uses_inferred_typed_responses(backup_api):
    client, *_rest = backup_api

    document = (await client.get("/openapi.json")).json()
    get_operation = document["paths"]["/api/system/backups"]["get"]
    post_operation = document["paths"]["/api/system/backups"]["post"]
    assert get_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/SystemBackupListResponse"}
    assert post_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/SystemBackupCreateResponse"}
    limit_parameter = next(
        parameter
        for parameter in get_operation["parameters"]
        if parameter["name"] == "limit"
    )
    assert limit_parameter["schema"] == {
        "type": "integer",
        "maximum": 100,
        "minimum": 1,
        "default": 50,
        "title": "Limit",
    }


async def test_in_memory_sqlite_is_typed_unsupported_without_file_work(client):
    post = await client.post("/api/system/backups")
    listing = await client.get("/api/system/backups?limit=7")

    assert post.status_code == 200
    assert post.json() == {
        "status": "unsupported",
        "backend": "sqlite",
        "backup": None,
    }
    assert listing.status_code == 200
    assert listing.json() == {
        "status": "unsupported",
        "backend": "sqlite",
        "returned_count": 0,
        "limit": 7,
        "truncated": False,
        "backups": [],
    }


@pytest.mark.parametrize(
    ("database", "query"),
    [
        ("file:", {"uri": "true", "mode": "ro"}),
        ("file::memory:?cache=shared", {"uri": "true"}),
        ("file:temporary?mode=memory", {"uri": "true"}),
        ("file:temporary%3Fmode=memory", {"uri": "true"}),
        ("ledger.db?mode=memory", {}),
        ("file://remote.example/private.db", {"uri": "true"}),
        ("file:ledger.db?mode=ro&vfs=private", {"uri": "true"}),
        ("file:ledger.db", {"uri": "true", "vfs": "private"}),
    ],
)
async def test_sqlite_uri_memory_temp_and_unsupported_shapes_are_rejected(
    database, query
):
    engine = create_async_engine(
        URL.create("sqlite+aiosqlite", database=database, query=query)
    )
    try:
        assert system_backups._sqlite_file_database(engine.sync_engine) is None
    finally:
        await engine.dispose()


async def test_embedded_memory_mode_is_typed_unsupported_via_api():
    engine = create_async_engine(
        URL.create(
            "sqlite+aiosqlite",
            database="file:temporary?mode=memory",
            query={"uri": "true"},
        )
    )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        app = FastAPI()

        async def override_session():
            yield session

        app.dependency_overrides[get_session] = override_session
        app.include_router(api_router)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            post = await client.post("/api/system/backups")
            listing = await client.get("/api/system/backups")

    assert post.json() == {
        "status": "unsupported",
        "backend": "sqlite",
        "backup": None,
    }
    assert listing.json()["status"] == "unsupported"
    await engine.dispose()


async def test_non_sqlite_backend_is_generic(backup_api, monkeypatch):
    client, session, engine, database_path = backup_api
    monkeypatch.setattr(engine.sync_engine.dialect, "name", "private-vendor")

    post = await client.post("/api/system/backups")
    listing = await client.get("/api/system/backups")

    assert post.json() == {
        "status": "unsupported",
        "backend": "other",
        "backup": None,
    }
    assert listing.json() == {
        "status": "unsupported",
        "backend": "other",
        "returned_count": 0,
        "limit": 50,
        "truncated": False,
        "backups": [],
    }
    _assert_no_paths(post.json(), database_path)
    _assert_no_paths(listing.json(), database_path)
    assert not session.in_transaction()


async def test_online_backup_progress_deadline_is_fast_and_closes_connections(
    tmp_path, monkeypatch
):
    calls = []

    class FakeSource:
        closed = False

        def backup(self, destination, *, pages, progress, sleep):
            calls.append((destination, pages, sleep))
            progress(0, 1, 2)

        def close(self):
            self.closed = True

    class FakeDestination:
        closed = False

        def close(self):
            self.closed = True

    source = FakeSource()
    destination = FakeDestination()
    connections = iter((source, destination))
    monotonic_values = iter((10.0, 10.25, 11.5))
    monkeypatch.setattr(
        system_backups.sqlite3,
        "connect",
        lambda *_args, **_kwargs: next(connections),
    )
    monkeypatch.setattr(system_backups, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(system_backups, "_BACKUP_TIMEOUT_SECONDS", 1.0)

    with pytest.raises(system_backups.BackupTimeoutError):
        system_backups._run_online_backup(
            tmp_path / "source.db", tmp_path / "destination.db"
        )

    assert calls == [
        (destination, system_backups._BACKUP_PAGES_PER_STEP, 0.05),
    ]
    assert source.closed is True
    assert destination.closed is True


async def test_backup_timeout_cleans_artifacts_and_sanitizes_response(
    backup_api, monkeypatch
):
    client, _session, _engine, database_path = backup_api

    def time_out(_source, _destination):
        raise system_backups.BackupTimeoutError("private contention detail")

    monkeypatch.setattr(system_backups, "_run_online_backup", time_out)
    response = await client.post("/api/system/backups")

    assert response.json() == {
        "status": "unavailable",
        "backend": "sqlite",
        "backup": None,
    }
    assert "private contention detail" not in response.text
    backup_directory = _backup_root(database_path)
    assert list(backup_directory.iterdir()) == []


async def test_create_scavenges_only_unlocked_stale_temporary_directories(
    backup_api, monkeypatch
):
    client, _session, _engine, database_path = backup_api
    backup_root = _backup_root(database_path)
    system_backups._ensure_private_backup_directory(backup_root)
    stale_id = "20000101T000000000000Z-0000000000000000"
    inactive = backup_root / f".{stale_id}.1111111111111111.tmp"
    active = backup_root / f".{stale_id}.2222222222222222.tmp"
    inactive.mkdir(mode=0o700)
    active.mkdir(mode=0o700)
    (inactive / system_backups._ACTIVE_LOCK_FILE_NAME).write_text("")
    active_lock = active / system_backups._ACTIVE_LOCK_FILE_NAME
    descriptor = os.open(active_lock, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.utime(inactive, (0, 0))
    os.utime(active, (0, 0))
    monkeypatch.setattr(system_backups, "_STALE_TEMP_MIN_AGE_SECONDS", 1.0)

    try:
        first = await client.post("/api/system/backups")
        assert first.json()["status"] == "created"
        assert not inactive.exists()
        assert active.exists()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    os.utime(active, (0, 0))
    second = await client.post("/api/system/backups")
    assert second.json()["status"] == "created"
    assert not active.exists()


async def test_verification_failure_cleans_artifacts_and_sanitizes_response(
    backup_api, monkeypatch, caplog
):
    client, _session, _engine, database_path = backup_api
    secret_detail = "private verification detail"

    def fail_verification(_path):
        raise RuntimeError(secret_detail)

    monkeypatch.setattr(system_backups, "_verify_sqlite_backup", fail_verification)
    with caplog.at_level(logging.ERROR, logger=system_backups.__name__):
        response = await client.post("/api/system/backups")

    assert response.status_code == 200
    assert response.json() == {
        "status": "unavailable",
        "backend": "sqlite",
        "backup": None,
    }
    assert secret_detail not in response.text
    assert str(database_path) not in response.text
    backup_directory = _backup_root(database_path)
    assert list(backup_directory.iterdir()) == []
    assert any(
        secret_detail in record.getMessage()
        for record in caplog.records
        if record.name == system_backups.__name__
    )


async def test_directory_publication_failure_removes_all_temp_artifacts(
    backup_api, monkeypatch
):
    client, _session, _engine, database_path = backup_api

    def fail_directory_rename(_source, _destination):
        raise OSError("synthetic directory publication failure")

    monkeypatch.setattr(system_backups.os, "rename", fail_directory_rename)

    response = await client.post("/api/system/backups")
    listing = await client.get("/api/system/backups")

    assert response.json() == {
        "status": "unavailable",
        "backend": "sqlite",
        "backup": None,
    }
    assert listing.json()["backups"] == []
    backup_directory = _backup_root(database_path)
    assert list(backup_directory.iterdir()) == []


async def test_post_rename_sync_failure_removes_published_directory(
    backup_api, monkeypatch
):
    client, _session, _engine, database_path = backup_api
    backup_root = _backup_root(database_path)
    original_fsync_directory = system_backups._fsync_directory
    failed_publication_sync = False

    def fail_first_root_sync(path):
        nonlocal failed_publication_sync
        if path == backup_root and not failed_publication_sync:
            failed_publication_sync = True
            raise OSError("synthetic publication sync failure")
        return original_fsync_directory(path)

    monkeypatch.setattr(system_backups, "_fsync_directory", fail_first_root_sync)
    response = await client.post("/api/system/backups")

    assert response.json() == {
        "status": "unavailable",
        "backend": "sqlite",
        "backup": None,
    }
    assert failed_publication_sync is True
    assert list(backup_root.iterdir()) == []


async def test_dedicated_executor_serializes_workers_across_event_loops(
    backup_api, monkeypatch
):
    _client, _session, _engine, database_path = backup_api
    original_create = system_backups._create_backup_for_database
    counter_lock = threading.Lock()
    start_barrier = threading.Barrier(2)
    active = 0
    maximum_active = 0
    results = []
    errors = []

    def delayed_create(database, *, is_uri):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.05)
            return original_create(database, is_uri=is_uri)
        finally:
            with counter_lock:
                active -= 1

    def run_in_fresh_event_loop():
        async def submit_backup():
            return await system_backups._submit_backup_worker(
                str(database_path),
                is_uri=False,
            )

        try:
            start_barrier.wait()
            results.append(asyncio.run(submit_backup()))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(system_backups, "_create_backup_for_database", delayed_create)
    threads = [threading.Thread(target=run_in_fresh_event_loop) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        await asyncio.to_thread(thread.join)

    assert errors == []
    assert len(results) == 2
    assert maximum_active == 1


async def test_listing_busy_admission_does_not_queue_workers(backup_api, monkeypatch):
    _client, session, _engine, _database_path = backup_api
    original_list = system_backups._list_backups_for_database
    first_started = threading.Event()
    release_first = threading.Event()
    call_count = 0

    def controlled_list(database, *, is_uri, limit):
        nonlocal call_count
        call_count += 1
        first_started.set()
        release_first.wait()
        return original_list(database, is_uri=is_uri, limit=limit)

    monkeypatch.setattr(system_backups, "_list_backups_for_database", controlled_list)
    first = asyncio.create_task(system_backups.list_system_backups(session, limit=5))
    assert await asyncio.to_thread(first_started.wait, 1.0)

    busy = await system_backups.list_system_backups(session, limit=5)
    assert busy.status == "busy"
    assert busy.backups == []
    assert call_count == 1

    release_first.set()
    assert (await first).status == "ok"


async def test_busy_admission_is_bounded_until_cancelled_worker_finishes(
    backup_api, monkeypatch
):
    _client, session, _engine, _database_path = backup_api
    original_create = system_backups._create_backup_for_database
    first_started = threading.Event()
    release_first = threading.Event()
    calls_lock = threading.Lock()
    call_count = 0

    def controlled_create(database, *, is_uri):
        nonlocal call_count
        with calls_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_started.set()
            release_first.wait()
        return original_create(database, is_uri=is_uri)

    monkeypatch.setattr(
        system_backups, "_create_backup_for_database", controlled_create
    )
    first = asyncio.create_task(system_backups.create_system_backup(session))
    assert await asyncio.to_thread(first_started.wait, 1.0)

    first.cancel()
    busy = await system_backups.create_system_backup(session)
    assert busy.status == "busy"
    assert busy.backup is None
    assert first.done() is False
    assert call_count == 1

    # A second cancellation (for example, shutdown after disconnect) must not
    # release admission while the non-cancellable worker thread is still live.
    first.cancel()
    await asyncio.sleep(0)
    still_busy = await system_backups.create_system_backup(session)
    assert still_busy.status == "busy"
    assert first.done() is False
    assert call_count == 1

    release_first.set()
    with pytest.raises(asyncio.CancelledError):
        await first

    retry = await system_backups.create_system_backup(session)
    assert retry.status == "created"
    assert call_count == 2


async def test_cancelled_failed_worker_preserves_cancellation_and_releases_admission(
    backup_api, monkeypatch
):
    _client, session, _engine, _database_path = backup_api
    original_create = system_backups._create_backup_for_database
    first_started = threading.Event()
    release_first = threading.Event()
    call_count = 0

    def fail_first_create(database, *, is_uri):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_started.set()
            release_first.wait()
            raise OSError("synthetic cancelled worker failure")
        return original_create(database, is_uri=is_uri)

    monkeypatch.setattr(
        system_backups, "_create_backup_for_database", fail_first_create
    )
    first = asyncio.create_task(system_backups.create_system_backup(session))
    assert await asyncio.to_thread(first_started.wait, 1.0)
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done()
    release_first.set()

    with pytest.raises(asyncio.CancelledError):
        await first
    retry = await system_backups.create_system_backup(session)
    assert retry.status == "created"
    assert call_count == 2


async def test_backup_operations_do_not_start_request_session_sql_or_transaction(
    backup_api,
):
    client, session, engine, _database_path = backup_api
    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    assert not session.in_transaction()
    event.listen(engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        post = await client.post("/api/system/backups")
        listing = await client.get("/api/system/backups")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record_statement)

    assert post.json()["status"] == "created"
    assert listing.json()["status"] == "ok"
    assert statements == []
    assert not session.in_transaction()
