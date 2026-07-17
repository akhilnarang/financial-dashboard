"""Automatic Paisa sync runtime — the :class:`ExtensionRuntime` for the Paisa
extension, driven exclusively by the existing FetchService poll loop.

This module is a *surface adapter*: it calls the already-public Paisa
orchestrator (:func:`generate` / :func:`manual_sync`) and records outcomes via
the generic audit helpers in :mod:`financial_dashboard.services.paisa.audit`.
It never edits the orchestrator, projection, renderer, publisher, or config
loader, and it never mutates core financial rows.

Design rules (enforced here, not elsewhere):

* **No extra loops.** There is no module-level background task. The only trigger
  is :meth:`PaisaAutomationRuntime.after_fetch_cycle`, invoked once per
  FetchService cycle by the :class:`ExtensionManager`.
* **Startup/shutdown are inert.** Neither starts Paisa, opens the network, nor
  enables auto-sync. Auto-sync is opt-in via ``paisa.auto_sync_enabled`` AND
  requires ``project`` mode.
* **Disabled/connect never reach the network or filesystem.** The hook returns
  before any projection or HTTP call when ``mode`` is not ``project``.
* **One run per fetch cycle, debounced by min interval.** The audit table is the
  single source of truth for "when did we last attempt an automatic sync"; there
  is no separate last-run setting to drift out of sync.
* **Skip-unchanged.** The hook generates first; when the publisher reports the
  on-disk bytes are unchanged (its built-in idempotency), the remote
  ``/api/sync`` reload is skipped entirely — Paisa already has that exact file.
  When the bytes did change, the full remote sync (probe + POST + diagnosis) runs
  in the same hook via :func:`manual_sync`. The last-good file is always retained
  by the publisher's atomic-replace-with-skip behavior.
* **Isolated.** Any exception is caught, recorded as a failure audit row, and
  swallowed — the fetch loop must keep running regardless of an optional
  extension's health.
"""

import datetime
import hashlib
import html
import json
import logging
from typing import Any, NamedTuple

from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_dashboard.db.models import utc_now
from financial_dashboard.services.paisa.audit import (
    OPERATION_AUTOMATIC,
    STATUS_FAILURE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    complete_run,
    last_run,
    run_started_at,
    sanitize_error,
    start_run,
)
from financial_dashboard.services.paisa.config import load_config
from financial_dashboard.services.paisa.orchestrator import generate, manual_sync
from financial_dashboard.services.settings import (
    get_setting_bool,
    get_setting_int,
)

logger = logging.getLogger(__name__)

#: Default debounce window when the setting is unset/malformed. Matches the
#: contributed setting default; kept here so a missing setting can never make
#: the runtime fire every cycle.
DEFAULT_MIN_INTERVAL_MINUTES = 30

#: Sync outcomes that count as a real failure worth notifying about. Config
#: gaps (not_configured / disabled / connect_only) and the happy paths are
#: excluded — they are not transient failures a human needs paged for.
FAILURE_OUTCOMES = frozenset(
    {
        "unreachable",
        "readonly",
        "unsupported_backend",
        "publish_failed",
        "sync_rejected",
        "diagnosis_failed",
        "generate_failed",
        "error",
    }
)


class AutoSyncOutcome(NamedTuple):
    """Resolved result of one automatic sync attempt, ready to persist."""

    status: str
    outcome: str
    output_hash: str | None
    emitted_count: int | None
    skipped_count: int | None
    details: Any
    error: str | None
    notify_failure: bool


def _projection_counts(report: Any) -> tuple[int | None, int | None]:
    """Pull (emitted, skipped) counts from a ProjectionReport, tolerating None."""
    if report is None:
        return (None, None)
    # report is always a ProjectionReport here (None handled above); direct
    # attribute access lets a shape mismatch surface rather than hide as None.
    return (report.emitted_count, len(report.skipped))


def _reason_to_outcome(reason: str | None) -> tuple[str, str]:
    """Classify a generate refusal reason into (status, outcome).

    A readiness/config refusal is a *skip* (the operator has not finished
    configuring projection); anything else is a real failure.
    """
    text = (reason or "").lower()
    if (
        "not_configured" in text
        or "generated_path" in text
        or "cutover" in text
        or "selected account" in text
    ):
        return (STATUS_SKIPPED, "not_configured")
    if "disabled" in text:
        return (STATUS_SKIPPED, "disabled")
    if "connect_only" in text:
        return (STATUS_SKIPPED, "connect_only")
    return (STATUS_FAILURE, "generate_failed")


def _map_sync_report(
    report: Any, body_hash: str | None, emitted: int | None
) -> AutoSyncOutcome:
    """Turn an orchestrator :class:`SyncReport` into an audit-ready outcome."""
    ok = bool(report.ok)
    outcome = str(report.outcome)
    status = STATUS_SUCCESS if ok else STATUS_FAILURE
    error = None if ok else sanitize_error(report.reason or outcome)
    details: dict[str, Any] = {"outcome": outcome, "diagnosis_ok": report.diagnosis_ok}
    # Mirror the classified diagnosis counts the manual surface records (None
    # when diagnosis never ran). Counts only — no raw journal text/credentials.
    diagnosis_expected = report.diagnosis_expected
    if diagnosis_expected is not None:
        details["diagnosis_expected"] = diagnosis_expected
    diagnosis_accepted = report.diagnosis_accepted
    if diagnosis_accepted is not None:
        details["diagnosis_accepted"] = diagnosis_accepted
    diagnosis_fatal = report.diagnosis_fatal
    if diagnosis_fatal is not None:
        details["diagnosis_fatal"] = diagnosis_fatal
    return AutoSyncOutcome(
        status=status,
        outcome=outcome,
        output_hash=body_hash,
        emitted_count=emitted,
        skipped_count=None,
        details=details,
        error=error,
        notify_failure=(not ok) and (outcome in FAILURE_OUTCOMES),
    )


def _failure_fingerprint(outcome: str, error: str | None) -> str:
    """A stable short hash of a failure's identity (outcome + sanitized error).

    Two failures with the same outcome and the same (whitespace-collapsed) error
    text produce the same fingerprint, so repeated identical failures are
    deduped. A changed outcome or error text yields a new fingerprint and
    notifies again. Never includes credentials (the error is already
    orchestrator-sanitized).
    """
    norm_outcome = (outcome or "").strip()
    norm_error = " ".join((error or "").split())
    digest = hashlib.sha256(f"{norm_outcome}\n{norm_error}".encode("utf-8"))
    return digest.hexdigest()[:16]


def _notify_fp_from_run(run: Any) -> str | None:
    """Read the persisted notification fingerprint from a prior run's details.

    Returns ``None`` when the run or its details are absent or malformed, so the
    caller falls back to "not previously notified" (i.e. notify) — failing open
    on notification, never silently suppressing a novel failure.
    """
    if run is None or not getattr(run, "details", None):
        return None
    try:
        decoded = json.loads(run.details)
    except ValueError, TypeError:
        return None
    if isinstance(decoded, dict):
        fp = decoded.get("notify_fp")
        if isinstance(fp, str) and fp:
            return fp
    return None


class PaisaAutomationRuntime:
    """Implements :class:`~financial_dashboard.extensions.base.ExtensionRuntime`.

    Construct with no args in production (it resolves the shared
    ``async_session`` lazily so importing this module is side-effect-free). Tests
    inject an ``async_sessionmaker`` to point at an in-memory engine.
    """

    extension_id = "paisa"

    def __init__(self, session_factory: async_sessionmaker | None = None) -> None:
        self._session_factory: async_sessionmaker | None = session_factory

    def _resolve_session_factory(self) -> async_sessionmaker:
        # Lazy import: keeps module import free of a DB/engine touch and breaks
        # the import cycle with financial_dashboard.db at construction time.
        if self._session_factory is None:
            from financial_dashboard.db import async_session

            self._session_factory = async_session
        return self._session_factory

    # ------------------------------------------------------------------
    # ExtensionRuntime protocol
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """No-op. Startup never starts Paisa or enables auto-sync."""

    async def shutdown(self) -> None:
        """No-op. Nothing was started, so there is nothing to stop."""

    async def after_fetch_cycle(self) -> None:
        """Run at most one automatic sync for this fetch cycle.

        Fully self-isolated: any exception is caught and logged here so the
        FetchService loop is unaffected. Opens its own session (the fetch loop
        does not hand one down) per the background-task session convention.
        """
        try:
            await self._after_fetch_cycle_inner()
        except Exception:
            # Last-resort isolation: even a failure inside the audit path must
            # not propagate to the fetch loop. The detailed failure (if it got
            # far enough) is already recorded as an audit row.
            logger.exception("Paisa automatic sync hook failed")

    async def _after_fetch_cycle_inner(self) -> None:
        config = load_config()

        # Disabled/connect: no projection, no file, no network, no audit row.
        if not config.can_project:
            return
        if not get_setting_bool("paisa.auto_sync_enabled", False):
            return

        min_minutes = max(
            1,
            get_setting_int(
                "paisa.auto_sync_min_interval_minutes", DEFAULT_MIN_INTERVAL_MINUTES
            ),
        )
        now = utc_now()
        factory = self._resolve_session_factory()

        async with factory() as session:
            # Debounce on the audit table (single source of truth). A run that
            # happened inside the min window short-circuits with no new row, so
            # repeated cycles don't spam the table.
            previous = await last_run(
                session, extension_id=self.extension_id, operation=OPERATION_AUTOMATIC
            )
            previous_fp = _notify_fp_from_run(previous)
            if previous is not None:
                elapsed = now - run_started_at(previous)
                if elapsed < datetime.timedelta(minutes=min_minutes):
                    return

            run = await start_run(
                session,
                extension_id=self.extension_id,
                operation=OPERATION_AUTOMATIC,
                trigger="fetch_cycle",
            )
            try:
                outcome = await self._sync_once(session, config)
            except Exception as exc:
                logger.exception("Paisa automatic sync attempt raised")
                outcome = AutoSyncOutcome(
                    status=STATUS_FAILURE,
                    outcome="error",
                    output_hash=None,
                    emitted_count=None,
                    skipped_count=None,
                    details={"exception": type(exc).__name__},
                    error=sanitize_error(f"{type(exc).__name__}: {exc}"),
                    notify_failure=True,
                )

            # Dedupe repeated identical failures: only notify when the failure
            # fingerprint differs from the previous automatic run's. The
            # fingerprint is persisted in the audit details so the dedupe
            # survives restarts (no separate state to drift out of sync).
            fp = _failure_fingerprint(outcome.outcome, outcome.error)
            notify_this_time = outcome.notify_failure and fp != previous_fp
            details = dict(outcome.details) if outcome.details else {}
            if outcome.notify_failure:
                details["notify_fp"] = fp
                details["notify_sent"] = notify_this_time

            await complete_run(
                session,
                run,
                status=outcome.status,
                outcome=outcome.outcome,
                output_hash=outcome.output_hash,
                emitted_count=outcome.emitted_count,
                skipped_count=outcome.skipped_count,
                details=details,
                error=outcome.error,
            )
            await session.commit()

        if notify_this_time:
            await self._maybe_notify_failure(outcome.outcome, outcome.error)

    # ------------------------------------------------------------------
    # Sync attempt
    # ------------------------------------------------------------------

    async def _sync_once(self, session: Any, config: Any) -> AutoSyncOutcome:
        """One attempt: generate (detect unchanged) then, if changed, remote-sync.

        Uses only public orchestrator entrypoints. ``generate`` projects + writes
        the include via the publisher (which skips an identical write and always
        retains the last-good file). When unchanged, the remote reload is skipped
        — Paisa already holds that exact file. When changed, ``manual_sync``
        re-generates (idempotent: the just-written file is identical, so its
        publish is a no-op) then probes, POSTs ``/api/sync``, and verifies.
        """
        generated = await generate(session, config)
        if not generated.ok or generated.report is None:
            status, outcome = _reason_to_outcome(generated.reason)
            return AutoSyncOutcome(
                status=status,
                outcome=outcome,
                output_hash=None,
                emitted_count=None,
                skipped_count=None,
                details={"reason": generated.reason, "stage": "generate"},
                error=None
                if status == STATUS_SKIPPED
                else sanitize_error(generated.reason),
                notify_failure=False,
            )

        emitted, skipped = _projection_counts(generated.report)
        body_hash = generated.publish.body_hash if generated.publish else None

        # Skip-unchanged: the on-disk file is byte-identical to what we'd write,
        # so Paisa already has it. No POST /api/sync, no diagnosis — and thus no
        # network round-trip for the common steady-state cycle.
        if generated.publish is not None and generated.publish.skipped:
            return AutoSyncOutcome(
                status=STATUS_SUCCESS,
                outcome="skipped_unchanged",
                output_hash=body_hash,
                emitted_count=emitted,
                skipped_count=skipped,
                details={"reason": "generated content unchanged; remote sync skipped"},
                error=None,
                notify_failure=False,
            )

        # Content changed: full remote sync in the same hook. manual_sync
        # re-generates internally; since we just published identical bytes, its
        # publish step is a no-op, then it reloads Paisa and runs diagnosis.
        sync_report = await manual_sync(session, config)
        result = _map_sync_report(sync_report, body_hash, emitted)
        # Preserve the projection's skipped-row count for the audit even though
        # _map_sync_report leaves it unset.
        if skipped is not None:
            result = result._replace(skipped_count=skipped)
        return result

    # ------------------------------------------------------------------
    # Failure notification
    # ------------------------------------------------------------------

    async def _maybe_notify_failure(self, outcome: str, reason: str | None) -> None:
        """Send a Telegram message on a real sync failure, if opted in.

        Best-effort and fully isolated: a notification failure is only logged.
        Uses the telegram service's retrying sender directly (no new public API
        is added to that module).
        """
        if not get_setting_bool("paisa.notify_sync_failures", False):
            return
        try:
            from financial_dashboard.services import telegram as telegram_service
            from financial_dashboard.services.settings import (
                get_telegram_chat_id,
                is_telegram_configured,
            )

            if not is_telegram_configured() or telegram_service.tg_app is None:
                return
            text = (
                "\u26a0\ufe0f <b>Paisa auto-sync failed</b>\n"
                f"outcome: {html.escape(outcome)}"
                + (f"\n{html.escape(reason)}" if reason else "")
            )
            await telegram_service._send_with_retry(
                telegram_service.tg_app,
                chat_id=get_telegram_chat_id(),
                text=text,
            )
        except Exception:
            logger.warning("Paisa sync-failure notification failed", exc_info=True)


__all__ = [
    "AutoSyncOutcome",
    "DEFAULT_MIN_INTERVAL_MINUTES",
    "FAILURE_OUTCOMES",
    "PaisaAutomationRuntime",
]
