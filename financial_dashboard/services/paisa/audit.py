"""Audit/state persistence for extension operations, over the generic
``extension_runs`` table.

This module is the single writer for :class:`~financial_dashboard.db.models.ExtensionRun`
rows. It is generic over ``extension_id`` (Paisa is the only extension today,
but the table and these helpers are extension-agnostic) so a later route agent
can record/query manual operations for any extension without a new table.

Convention (matches ``services.snapshots``): these helpers take a session the
caller already owns and only ``flush`` — they do NOT commit. A request handler
commits its request session; the automation runtime commits the session it opens.

Security: an audit row records *what an extension did*, never credentials and
never a duplicate of a financial row. ``details`` is JSON-serialized by these
helpers from a plain dict the caller supplies; ``error`` is run through
:func:`sanitize_error` so it is bounded and single-line. Callers must not pass
secrets — the Paisa orchestrator's reasons never contain credentials, so the
automation wrapper simply forwards them.
"""

import datetime
import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import ExtensionRun, utc_now

logger = logging.getLogger(__name__)

#: Canonical operation tokens. Free-form strings are tolerated for forward-compat,
#: but producers SHOULD use one of these so query filters are stable.
OPERATION_MANUAL = "manual"
OPERATION_AUTOMATIC = "automatic"
OPERATION_PROBE = "probe"
OPERATION_GENERATE = "generate"
OPERATION_SYNC = "sync"

#: Canonical status tokens.
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"
STATUS_SKIPPED = "skipped"

#: Cap on stored error text — an unbounded traceback could bloat the table and
#: risk leaking structure we did not intend to persist.
MAX_ERROR_LEN = 2000

#: Cap on the JSON ``details`` blob for the same reason.
MAX_DETAILS_BYTES = 16000


def sanitize_error(text: str | None) -> str | None:
    """Collapse whitespace and truncate an error string for safe storage.

    Newlines/tabs become single spaces so a multi-line traceback fits one row,
    and the length is capped. ``None`` passes through. This does NOT attempt to
    strip arbitrary secret patterns — callers control what reaches the audit row
    and the Paisa orchestrator's reasons never contain credentials.
    """
    if text is None:
        return None
    collapsed = " ".join(str(text).split())
    if len(collapsed) > MAX_ERROR_LEN:
        collapsed = collapsed[: MAX_ERROR_LEN - 3] + "..."
    return collapsed


def _encode_details(details: Any) -> str | None:
    """JSON-encode a details payload, bounding its size.

    Returns ``None`` for empty/``None`` input. Oversized payloads are truncated
    to ``MAX_DETAILS_BYTES`` with a trailing marker rather than raising — an
    audit row must never fail to persist because its summary was too verbose.
    """
    if details is None:
        return None
    if isinstance(details, str):
        # Accept a pre-encoded string but still bound it.
        if not details:
            return None
        return (
            details[:MAX_DETAILS_BYTES] if len(details) > MAX_DETAILS_BYTES else details
        )
    try:
        encoded = json.dumps(details, default=str, sort_keys=True)
    except TypeError, ValueError:
        return None
    if len(encoded) > MAX_DETAILS_BYTES:
        encoded = encoded[: MAX_DETAILS_BYTES - 3] + "..."
    return encoded


async def start_run(
    session: AsyncSession,
    *,
    extension_id: str,
    operation: str,
    trigger: str | None = None,
    input_hash: str | None = None,
) -> ExtensionRun:
    """Create a ``running`` audit row and flush it so its id is usable immediately.

    The caller owns the commit. ``input_hash`` is optional — pass the hash of
    the inputs that produced this run (e.g. the projected journal body hash) so
    two runs over identical inputs are comparable.
    """
    run = ExtensionRun(
        extension_id=extension_id,
        operation=operation,
        status=STATUS_RUNNING,
        trigger=trigger,
        started_at=utc_now(),
        input_hash=input_hash,
    )
    session.add(run)
    await session.flush()
    return run


async def complete_run(
    session: AsyncSession,
    run: ExtensionRun,
    *,
    status: str,
    outcome: str | None = None,
    output_hash: str | None = None,
    emitted_count: int | None = None,
    skipped_count: int | None = None,
    details: Any = None,
    error: str | None = None,
) -> ExtensionRun:
    """Mark a run finished with a terminal status and sanitized summary.

    Idempotent w.r.t. completion: sets ``completed_at`` only if unset so a
    double-complete cannot overwrite the real finish time. The caller commits.
    """
    run.status = status
    if outcome is not None:
        run.outcome = outcome
    if output_hash is not None:
        run.output_hash = output_hash
    if emitted_count is not None:
        run.emitted_count = emitted_count
    if skipped_count is not None:
        run.skipped_count = skipped_count
    if details is not None:
        run.details = _encode_details(details)
    if error is not None:
        run.error = sanitize_error(error)
    if run.completed_at is None:
        run.completed_at = utc_now()
    await session.flush()
    return run


async def recent_runs(
    session: AsyncSession,
    *,
    extension_id: str,
    operation: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[ExtensionRun]:
    """Most recent runs for an extension, newest first.

    ``operation``/``status`` are optional filters. ``limit`` is bounded to a
    sane maximum so a route can't accidentally pull the whole table.
    """
    bounded_limit = max(1, min(limit, 200))
    stmt = select(ExtensionRun).where(ExtensionRun.extension_id == extension_id)
    if operation is not None:
        stmt = stmt.where(ExtensionRun.operation == operation)
    if status is not None:
        stmt = stmt.where(ExtensionRun.status == status)
    stmt = stmt.order_by(ExtensionRun.started_at.desc(), ExtensionRun.id.desc()).limit(
        bounded_limit
    )
    return list((await session.execute(stmt)).scalars().all())


async def last_run(
    session: AsyncSession,
    *,
    extension_id: str,
    operation: str | None = None,
    status: str | None = None,
) -> ExtensionRun | None:
    """The single most recent run matching the filters, or ``None``."""
    rows = await recent_runs(
        session,
        extension_id=extension_id,
        operation=operation,
        status=status,
        limit=1,
    )
    return rows[0] if rows else None


def run_started_at(run: ExtensionRun) -> datetime.datetime:
    """Return the run's ``started_at`` as tz-aware UTC.

    Rows written by this module are always tz-aware, but a value read back from
    SQLite can lose its tz under some drivers; normalize to UTC defensively for
    the debounce comparison so a naive datetime never compares unequal to aware.
    """
    ts = run.started_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.UTC)
    return ts


__all__ = [
    "MAX_ERROR_LEN",
    "OPERATION_AUTOMATIC",
    "OPERATION_GENERATE",
    "OPERATION_MANUAL",
    "OPERATION_PROBE",
    "OPERATION_SYNC",
    "STATUS_FAILURE",
    "STATUS_RUNNING",
    "STATUS_SKIPPED",
    "STATUS_SUCCESS",
    "complete_run",
    "last_run",
    "recent_runs",
    "run_started_at",
    "sanitize_error",
    "start_run",
]
