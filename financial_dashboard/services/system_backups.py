import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import monotonic, time
from typing import NamedTuple, TypeVar
from urllib.parse import parse_qsl, unquote, urlsplit

from pydantic import ValidationError
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.schemas import system as system_schemas
from financial_dashboard.services.database import database_engine

_BACKUP_DIRECTORY_NAME = "backups"
_BACKUP_FILE_NAME = "backup.sqlite3"
_MANIFEST_FILE_NAME = "manifest.json"
_ACTIVE_LOCK_FILE_NAME = ".active.lock"
_MANIFEST_MAX_BYTES = 16 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_BACKUP_ID_PREFIX_FORMAT = "%Y%m%dT%H%M%S%fZ"
_BACKUP_ID_FORMAT = _BACKUP_ID_PREFIX_FORMAT + "-"  # random suffix follows
_BACKUP_ID_PATTERN = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{16}$")
_TEMP_DIRECTORY_PATTERN = re.compile(
    r"^\.\d{8}T\d{12}Z-[0-9a-f]{16}\.[0-9a-f]{16}\.tmp$"
)
_BACKUP_TIMEOUT_SECONDS = 120.0
_STALE_TEMP_MIN_AGE_SECONDS = 24 * 60 * 60.0
_MAX_RETAINED_BACKUPS = 20
_BACKUP_PAGES_PER_STEP = 256
_BACKUP_SLEEP_SECONDS = 0.05
_SQLITE_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")
_SQLITE_URL_QUERY_KEYS = {
    "uri",
    "mode",
    "cache",
    "timeout",
    "check_same_thread",
}
_SQLITE_URI_QUERY_KEYS = {"mode", "cache"}

logger = logging.getLogger(__name__)
_WorkerResult = TypeVar("_WorkerResult")

# Keep backup scans and their queue isolated from asyncio's shared default
# executor. A single dedicated worker serializes copies across event loops
# without letting backup waiters starve unrelated parser/filesystem work.
_backup_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="system-backup",
)
_backup_list_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="system-backup-list",
)
# Admission is non-blocking and process-wide. At most one API-created backup
# and one listing scan may be running, bounding request retention, disk work,
# and executor shutdown time under retries or an authenticated request flood.
_backup_admission = threading.Lock()
_backup_list_admission = threading.Lock()


class BackupVerificationError(Exception):
    """The copied SQLite database did not pass its bounded verification."""


class BackupTimeoutError(Exception):
    """The SQLite online backup exceeded its contention deadline."""


class SQLiteUriTarget(NamedTuple):
    path_text: str
    query: dict[str, str]


def _scalar_url_query(engine: Engine) -> dict[str, str] | None:
    query: dict[str, str] = {}
    for key, value in engine.url.query.items():
        if not isinstance(value, str):
            return None
        query[key] = value
    return query


def _parse_uri_query(query_text: str) -> dict[str, str]:
    if len(query_text) > _MANIFEST_MAX_BYTES:
        raise ValueError("SQLite URI query is too long")
    try:
        pairs = parse_qsl(
            query_text,
            keep_blank_values=True,
            strict_parsing=True,
        )
    except ValueError as exc:
        raise ValueError("malformed SQLite URI query") from exc

    query: dict[str, str] = {}
    for key, value in pairs:
        if key in query:
            raise ValueError("duplicate SQLite URI query parameter")
        if key not in _SQLITE_URI_QUERY_KEYS:
            raise ValueError("unsupported SQLite URI query parameter")
        query[key] = value
    return query


def _file_uri_path_and_query(database: str) -> SQLiteUriTarget:
    if re.search(r"%3f", database, re.IGNORECASE):
        raise ValueError("ambiguous encoded SQLite URI query")
    parsed = urlsplit(database)
    if parsed.scheme.lower() != "file" or parsed.fragment:
        raise ValueError("unsupported SQLite file URI")
    if parsed.netloc not in ("", "localhost"):
        raise ValueError("unsupported SQLite file URI authority")
    if parsed.netloc == "localhost" and not parsed.path.startswith("/"):
        raise ValueError("unsupported SQLite file URI path")

    path_text = unquote(parsed.path)
    if not path_text or path_text == ":memory:" or "\x00" in path_text:
        raise ValueError("SQLite URI does not identify a disk file")
    return SQLiteUriTarget(
        path_text,
        _parse_uri_query(parsed.query) if parsed.query else {},
    )


def _sqlite_file_database(engine: Engine) -> str | None:
    """Return the database component only when it safely names a disk file."""
    database = engine.url.database
    if database is None or database == "" or database == ":memory:":
        return None

    query = _scalar_url_query(engine)
    if query is None or not set(query).issubset(_SQLITE_URL_QUERY_KEYS):
        return None

    uri_value = query.get("uri")
    if uri_value is not None and uri_value != "true":
        return None
    is_uri = uri_value == "true"

    outer_mode = query.get("mode")
    if outer_mode is not None:
        if outer_mode.lower() == "memory":
            return None
        if outer_mode not in {"ro", "rw", "rwc"}:
            return None
    outer_cache = query.get("cache")
    if outer_cache is not None and outer_cache not in {"private", "shared"}:
        return None

    # URI-only options without uri=true are ambiguous DBAPI arguments. Refuse
    # them instead of accidentally treating their text as part of a filename.
    if not is_uri and ({"mode", "cache"} & query.keys()):
        return None

    try:
        if is_uri:
            if not database.lower().startswith("file:"):
                raise ValueError("SQLite URI database must use the file scheme")
            _path_text, embedded_query = _file_uri_path_and_query(database)
            if set(embedded_query) & query.keys():
                raise ValueError("duplicate split SQLite URI query parameter")
            embedded_mode = embedded_query.get("mode")
            if embedded_mode is not None:
                if embedded_mode.lower() == "memory":
                    return None
                if embedded_mode not in {"ro", "rw", "rwc"}:
                    return None
            embedded_cache = embedded_query.get("cache")
            if embedded_cache is not None and embedded_cache not in {
                "private",
                "shared",
            }:
                return None
        else:
            if database.lower().startswith("file:"):
                raise ValueError("file URI requires uri=true")
            # SQLAlchemy URL objects can be built with a query embedded directly
            # in the database component. Never interpret such a mode as a disk
            # filename, including percent-encoded question marks.
            if "?" in database or re.search(r"%3f", database, re.IGNORECASE):
                raise ValueError("ambiguous SQLite database component")
        _source_path(database, is_uri=is_uri)
    except ValueError:
        return None

    return database


def _source_path(database: str, *, is_uri: bool) -> Path:
    """Resolve SQLAlchemy's SQLite database name without exposing it externally."""
    if is_uri:
        path_text, _query = _file_uri_path_and_query(database)
    else:
        if not database or database == ":memory:":
            raise ValueError("SQLite database does not identify a file")
        if database.lower().startswith("file:") or "?" in database:
            raise ValueError("ambiguous SQLite database component")
        path_text = database

    # Match SQLite/SQLAlchemy semantics exactly: a leading '~' is a literal path
    # component, not shell syntax. Expanding it could silently back up a
    # different database under the service account's home directory.
    return Path(path_text).resolve(strict=False)


def _engine_uses_sqlite_uri(engine: Engine) -> bool:
    return engine.url.query.get("uri") == "true"


def _backup_id_created_at(backup_id: str) -> datetime:
    timestamp, _separator, _random = backup_id.partition("-")
    return datetime.strptime(timestamp, _BACKUP_ID_PREFIX_FORMAT).replace(tzinfo=UTC)


def _new_backup_id(created_at: datetime) -> str:
    return created_at.astimezone(UTC).strftime(_BACKUP_ID_FORMAT) + secrets.token_hex(8)


def _backup_directory(source_path: Path) -> Path:
    # Different SQLite files may legitimately share one parent directory. Keep
    # their manifests and retention domains isolated without exposing a source
    # filename in the public API or filesystem namespace.
    source_identity = hashlib.sha256(os.fsencode(str(source_path))).hexdigest()
    return source_path.parent / _BACKUP_DIRECTORY_NAME / source_identity


def _ensure_private_backup_directory(path: Path) -> None:
    # Create and persist both the shared container and source-specific namespace
    # one level at a time; parents=True would obscure which directory entry still
    # needed an fsync for first-backup durability.
    for directory in (path.parent, path):
        directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        mode = directory.lstat().st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise OSError("backup location is not a private directory")
        os.chmod(directory, 0o700)
        _fsync_directory(directory.parent)


def _sqlite_read_only_uri(path: Path) -> str:
    return f"{path.as_uri()}?mode=ro"


def _run_online_backup(source_path: Path, destination_path: Path) -> None:
    deadline = monotonic() + _BACKUP_TIMEOUT_SECONDS

    def enforce_deadline(_status: int, _remaining: int, _total: int) -> None:
        if monotonic() >= deadline:
            raise BackupTimeoutError("SQLite online backup timed out")

    source: sqlite3.Connection | None = None
    destination: sqlite3.Connection | None = None
    try:
        source = sqlite3.connect(
            _sqlite_read_only_uri(source_path),
            uri=True,
            timeout=30.0,
        )
        destination = sqlite3.connect(destination_path, timeout=30.0)
        if monotonic() >= deadline:
            raise BackupTimeoutError("SQLite online backup timed out")
        source.backup(
            destination,
            pages=_BACKUP_PAGES_PER_STEP,
            progress=enforce_deadline,
            sleep=_BACKUP_SLEEP_SECONDS,
        )
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_sqlite_backup(path: Path) -> None:
    connection = sqlite3.connect(
        _sqlite_read_only_uri(path),
        uri=True,
        timeout=30.0,
    )
    try:
        rows = connection.execute("PRAGMA quick_check(1)").fetchmany(2)
    finally:
        connection.close()

    if rows != [("ok",)]:
        raise BackupVerificationError("SQLite quick check did not return ok")


def _hash_file(path: Path) -> system_schemas.SystemBackupHash:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as backup_file:
        while chunk := backup_file.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size_bytes += len(chunk)
    return system_schemas.SystemBackupHash(digest.hexdigest(), size_bytes)


def _write_manifest(path: Path, manifest: system_schemas.SystemBackupManifest) -> None:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MANIFEST_MAX_BYTES:
        raise ValueError("backup manifest exceeds its internal bound")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as manifest_file:
            descriptor = -1
            manifest_file.write(payload)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _remove_sqlite_sidecars(path: Path) -> None:
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _remove_backup_tree(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _temporary_directory_is_active(path: Path) -> bool:
    lock_path = path / _ACTIVE_LOCK_FILE_NAME
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(lock_path, flags)
    except FileNotFoundError:
        return False
    except OSError:
        # If ownership cannot be established safely, never delete the directory.
        return True

    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return True
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _scavenge_stale_temporary_directories(backup_root: Path) -> None:
    """Remove crashed attempts without touching another process's active copy."""
    try:
        entries = list(backup_root.iterdir())
    except OSError:
        logger.warning("Could not inspect temporary system backups")
        return

    cutoff = time() - _STALE_TEMP_MIN_AGE_SECONDS
    removed = False
    for path in entries:
        if _TEMP_DIRECTORY_PATTERN.fullmatch(path.name) is None:
            continue
        try:
            file_stat = path.lstat()
        except OSError:
            continue
        if (
            not stat.S_ISDIR(file_stat.st_mode)
            or stat.S_ISLNK(file_stat.st_mode)
            or file_stat.st_mtime > cutoff
            or _temporary_directory_is_active(path)
        ):
            continue
        try:
            _remove_backup_tree(path)
            removed = True
        except OSError:
            logger.warning("Could not clean a stale temporary system backup")

    if removed:
        try:
            _fsync_directory(backup_root)
        except OSError:
            logger.warning("Could not sync temporary system backup cleanup")


def _cleanup_failed_backup(
    temporary_directory: Path,
    *,
    final_directory: Path | None,
    backup_root: Path,
) -> None:
    for path in (temporary_directory, final_directory):
        if path is None:
            continue
        try:
            _remove_backup_tree(path)
        except OSError:
            logger.warning("Could not clean a partial system backup directory")
    if final_directory is not None:
        try:
            _fsync_directory(backup_root)
        except OSError:
            logger.warning("Could not sync system backup cleanup")


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _prune_old_backups(backup_root: Path, *, current_backup_id: str) -> None:
    """Best-effort retention after a new backup is durably published."""
    candidates: list[Path] = []
    try:
        entries = list(backup_root.iterdir())
    except OSError:
        logger.warning("Could not inspect system backups for retention")
        return
    for path in entries:
        if path.name == current_backup_id:
            continue
        try:
            mode = path.lstat().st_mode
        except OSError:
            logger.warning("Could not inspect a system backup for retention")
            continue
        if (
            _BACKUP_ID_PATTERN.fullmatch(path.name) is None
            or not stat.S_ISDIR(mode)
            or stat.S_ISLNK(mode)
        ):
            continue
        candidates.append(path)

    # The current backup is always retained even if the host clock moved
    # backwards relative to an existing identifier.
    candidates.sort(key=lambda path: path.name, reverse=True)
    removed = False
    for stale in candidates[_MAX_RETAINED_BACKUPS - 1 :]:
        try:
            _remove_backup_tree(stale)
            removed = True
        except OSError:
            logger.warning("Could not prune an expired system backup")
    if removed:
        try:
            _fsync_directory(backup_root)
        except OSError:
            logger.warning("Could not sync system backup retention changes")


def _create_backup_for_database(
    database: str,
    *,
    is_uri: bool,
) -> system_schemas.SystemBackupMetadata:
    source_path = _source_path(database, is_uri=is_uri)
    if not source_path.is_file():
        raise FileNotFoundError("SQLite source database is unavailable")

    backup_root = _backup_directory(source_path)
    _ensure_private_backup_directory(backup_root)
    _scavenge_stale_temporary_directories(backup_root)

    created_at = datetime.now(UTC)
    backup_id = _new_backup_id(created_at)
    final_directory = backup_root / backup_id
    temporary_token = secrets.token_hex(8)
    temporary_directory = backup_root / f".{backup_id}.{temporary_token}.tmp"
    temporary_backup = temporary_directory / _BACKUP_FILE_NAME
    temporary_manifest = temporary_directory / _MANIFEST_FILE_NAME

    if _path_exists(final_directory):
        raise FileExistsError("generated backup identifier collided")

    temporary_created = False
    published = False
    active_descriptor = -1
    try:
        temporary_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        temporary_created = True
        os.chmod(temporary_directory, 0o700)
        active_descriptor = os.open(
            temporary_directory / _ACTIVE_LOCK_FILE_NAME,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        fcntl.flock(active_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

        descriptor = os.open(
            temporary_backup,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(descriptor)
        _run_online_backup(source_path, temporary_backup)
        _remove_sqlite_sidecars(temporary_backup)
        os.chmod(temporary_backup, 0o600)
        _fsync_file(temporary_backup)
        _verify_sqlite_backup(temporary_backup)
        _remove_sqlite_sidecars(temporary_backup)
        sha256, size_bytes = _hash_file(temporary_backup)
        if size_bytes <= 0:
            raise BackupVerificationError("SQLite backup was empty")

        manifest = system_schemas.SystemBackupManifest(
            version=1,
            backup_id=backup_id,
            created_at=created_at,
            size_bytes=size_bytes,
            sha256=sha256,
            quick_check="ok",
        )
        _write_manifest(temporary_manifest, manifest)
        fcntl.flock(active_descriptor, fcntl.LOCK_UN)
        os.close(active_descriptor)
        active_descriptor = -1
        (temporary_directory / _ACTIVE_LOCK_FILE_NAME).unlink()
        _fsync_directory(temporary_directory)

        # Both fixed children become visible together in one same-filesystem
        # directory rename. The random ID makes a cross-process collision
        # negligible; never remove an existing destination on a precheck clash.
        if _path_exists(final_directory):
            raise FileExistsError("generated backup identifier collided")
        os.rename(temporary_directory, final_directory)
        published = True
        _fsync_directory(backup_root)
        _prune_old_backups(backup_root, current_backup_id=backup_id)
    except Exception:
        if active_descriptor >= 0:
            os.close(active_descriptor)
            active_descriptor = -1
        if temporary_created:
            _cleanup_failed_backup(
                temporary_directory,
                final_directory=final_directory if published else None,
                backup_root=backup_root,
            )
        raise

    return system_schemas.SystemBackupMetadata(
        backup_id=backup_id,
        created_at=created_at,
        size_bytes=size_bytes,
        sha256=sha256,
        status="verified",
    )


def _run_backup_worker(
    database: str,
    *,
    is_uri: bool,
) -> system_schemas.SystemBackupWorkerOutcome:
    try:
        return system_schemas.SystemBackupWorkerOutcome(
            _create_backup_for_database(database, is_uri=is_uri),
            None,
        )
    except Exception as exc:
        # Return failures as data so a shield cancelled by its caller never
        # installs asyncio's "exception in shielded future" callback.
        return system_schemas.SystemBackupWorkerOutcome(None, exc)


def _submit_backup_worker(
    database: str,
    *,
    is_uri: bool,
) -> asyncio.Future[system_schemas.SystemBackupWorkerOutcome]:
    return asyncio.get_running_loop().run_in_executor(
        _backup_executor,
        partial(_run_backup_worker, database, is_uri=is_uri),
    )


def _run_backup_list_worker(
    database: str,
    *,
    is_uri: bool,
    limit: int,
) -> system_schemas.SystemBackupListWorkerOutcome:
    try:
        return system_schemas.SystemBackupListWorkerOutcome(
            _list_backups_for_database(database, is_uri=is_uri, limit=limit),
            None,
        )
    except Exception as exc:
        return system_schemas.SystemBackupListWorkerOutcome(None, exc)


def _submit_backup_list_worker(
    database: str,
    *,
    is_uri: bool,
    limit: int,
) -> asyncio.Future[system_schemas.SystemBackupListWorkerOutcome]:
    return asyncio.get_running_loop().run_in_executor(
        _backup_list_executor,
        partial(
            _run_backup_list_worker,
            database,
            is_uri=is_uri,
            limit=limit,
        ),
    )


async def _drain_cancelled_worker(
    worker: asyncio.Future[_WorkerResult],
) -> _WorkerResult:
    """Keep admission held until a non-cancellable executor job finishes."""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Server shutdown may cancel a disconnected request more than once.
            # Continue draining; the thread itself cannot be cancelled safely.
            continue
    return worker.result()


def _read_manifest(path: Path) -> system_schemas.SystemBackupManifest:
    file_mode = path.lstat().st_mode
    if not stat.S_ISREG(file_mode) or stat.S_ISLNK(file_mode):
        raise OSError("backup manifest is not a regular file")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise OSError("backup manifest is not a regular file")
        if opened_stat.st_size > _MANIFEST_MAX_BYTES:
            raise ValueError("backup manifest exceeds its internal bound")
        with os.fdopen(descriptor, "rb") as manifest_file:
            descriptor = -1
            payload = manifest_file.read(_MANIFEST_MAX_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(payload) > _MANIFEST_MAX_BYTES:
        raise ValueError("backup manifest exceeds its internal bound")
    return system_schemas.SystemBackupManifest.model_validate_json(payload)


def _backup_file_status(
    path: Path,
    *,
    expected_size: int,
) -> system_schemas.BackupFileStatus:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return "missing"
    if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
        return "missing"
    if file_stat.st_size != expected_size:
        return "size_mismatch"
    return "verified"


def _manifest_metadata(
    backup_directory: Path,
) -> system_schemas.SystemBackupMetadata | None:
    try:
        directory_mode = backup_directory.lstat().st_mode
        if not stat.S_ISDIR(directory_mode) or stat.S_ISLNK(directory_mode):
            raise OSError("backup entry is not a directory")
        backup_id = backup_directory.name
        manifest = _read_manifest(backup_directory / _MANIFEST_FILE_NAME)
        if manifest.backup_id != backup_id:
            raise ValueError("manifest identifier does not match its directory")
        if manifest.created_at != _backup_id_created_at(manifest.backup_id):
            raise ValueError("manifest timestamp does not match its identifier")
    except OSError, ValueError, ValidationError:
        logger.warning("Ignoring a malformed system backup manifest")
        return None

    # Child names are fixed by the service. No path from manifest content is
    # ever used, and GET intentionally trusts POST's checksum verification.
    backup_path = backup_directory / _BACKUP_FILE_NAME
    return system_schemas.SystemBackupMetadata(
        backup_id=manifest.backup_id,
        created_at=manifest.created_at,
        size_bytes=manifest.size_bytes,
        sha256=manifest.sha256,
        status=_backup_file_status(
            backup_path,
            expected_size=manifest.size_bytes,
        ),
    )


def _list_backups_for_database(
    database: str,
    *,
    is_uri: bool,
    limit: int,
) -> system_schemas.SystemBackupListResult:
    source_path = _source_path(database, is_uri=is_uri)
    backup_root = _backup_directory(source_path)
    try:
        directory_mode = backup_root.lstat().st_mode
    except FileNotFoundError:
        return system_schemas.SystemBackupListResult([], False)
    if not stat.S_ISDIR(directory_mode) or stat.S_ISLNK(directory_mode):
        raise OSError("backup location is not a directory")

    candidates: list[Path] = []
    for path in backup_root.iterdir():
        try:
            if _BACKUP_ID_PATTERN.fullmatch(path.name) is None:
                if not path.name.startswith("."):
                    logger.warning("Ignoring a malformed system backup directory")
                continue
            _backup_id_created_at(path.name)
        except ValueError, TypeError:
            logger.warning("Ignoring a malformed system backup directory")
            continue
        candidates.append(path)

    candidates.sort(key=lambda path: path.name, reverse=True)
    backups: list[system_schemas.SystemBackupMetadata] = []
    for candidate in candidates:
        metadata = _manifest_metadata(candidate)
        if metadata is None:
            continue
        backups.append(metadata)
        if len(backups) > limit:
            break

    return system_schemas.SystemBackupListResult(
        backups=backups[:limit],
        truncated=len(backups) > limit,
    )


def _empty_list_response(
    *,
    status: system_schemas.BackupListStatus,
    limit: int,
) -> system_schemas.SystemBackupListResponse:
    """Build an empty SQLite backup-list response."""
    return system_schemas.SystemBackupListResponse(
        status=status,
        backend="sqlite",
        returned_count=0,
        limit=limit,
        truncated=False,
        backups=[],
    )


def _empty_create_response(
    status: system_schemas.BackupCreateStatus,
) -> system_schemas.SystemBackupCreateResponse:
    """Build an empty SQLite backup-creation response."""
    return system_schemas.SystemBackupCreateResponse(
        status=status,
        backend="sqlite",
        backup=None,
    )


async def list_system_backups(
    session: AsyncSession,
    *,
    limit: int,
) -> system_schemas.SystemBackupListResponse:
    """List bounded sidecar metadata without opening backup database files."""
    try:
        engine = database_engine(session)
    except Exception as exc:
        logger.exception("Could not identify the database for backup listing: %s", exc)
        return _empty_list_response(status="unavailable", limit=limit)

    database = _sqlite_file_database(engine)
    if database is None:
        return _empty_list_response(status="unsupported", limit=limit)

    if not _backup_list_admission.acquire(blocking=False):
        return _empty_list_response(status="busy", limit=limit)

    try:
        try:
            worker = _submit_backup_list_worker(
                database,
                is_uri=_engine_uses_sqlite_uri(engine),
                limit=limit,
            )
            try:
                outcome = await asyncio.shield(worker)
            except asyncio.CancelledError as cancellation:
                outcome = await _drain_cancelled_worker(worker)
                if outcome.error is not None:
                    logger.error(
                        "Cancelled system backup listing failed: %s",
                        outcome.error,
                        exc_info=(
                            type(outcome.error),
                            outcome.error,
                            outcome.error.__traceback__,
                        ),
                    )
                raise cancellation from None
            if outcome.error is not None:
                raise outcome.error
            assert outcome.result is not None
            result = outcome.result
        except Exception as exc:
            logger.exception("System backup listing failed: %s", exc)
            return _empty_list_response(status="unavailable", limit=limit)

        return system_schemas.SystemBackupListResponse(
            status="ok",
            backend="sqlite",
            returned_count=len(result.backups),
            limit=limit,
            truncated=result.truncated,
            backups=result.backups,
        )
    finally:
        _backup_list_admission.release()


async def create_system_backup(
    session: AsyncSession,
) -> system_schemas.SystemBackupCreateResponse:
    """Create, verify, hash, and atomically publish one online SQLite backup."""
    try:
        engine = database_engine(session)
    except Exception as exc:
        logger.exception("Could not identify the database for backup creation: %s", exc)
        return _empty_create_response("unavailable")

    database = _sqlite_file_database(engine)
    if database is None:
        return _empty_create_response("unsupported")

    if not _backup_admission.acquire(blocking=False):
        return _empty_create_response("busy")

    try:
        try:
            worker = _submit_backup_worker(
                database,
                is_uri=_engine_uses_sqlite_uri(engine),
            )
            try:
                outcome = await asyncio.shield(worker)
            except asyncio.CancelledError as cancellation:
                # A cancelled request cannot stop its worker thread. Observe its
                # completion before releasing admission, so a retry cannot
                # overlap the copy or cleanup still running in the thread.
                outcome = await _drain_cancelled_worker(worker)
                if outcome.error is not None:
                    logger.error(
                        "Cancelled system backup worker failed: %s",
                        outcome.error,
                        exc_info=(
                            type(outcome.error),
                            outcome.error,
                            outcome.error.__traceback__,
                        ),
                    )
                raise cancellation from None
            if outcome.error is not None:
                raise outcome.error
            assert outcome.backup is not None
            backup = outcome.backup
        except Exception as exc:
            logger.exception("System backup creation failed: %s", exc)
            return _empty_create_response("unavailable")

        return system_schemas.SystemBackupCreateResponse(
            status="created",
            backend="sqlite",
            backup=backup,
        )
    finally:
        _backup_admission.release()
