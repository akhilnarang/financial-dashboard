"""Probe / preview / generate / manual-sync orchestration for the Paisa projection.

The orchestrator is the only place the projection, the publisher and the
client meet a request boundary. Its contract with the rest of the dashboard:

* It is **read-only over core rows.** ``preview``, ``generate`` and
  ``manual_sync`` never touch the transactions/accounts tables; ``manual_sync``
  writes only the generated include file and then asks Paisa to reload. A
  failure at any step leaves the dashboard's data exactly as it was.
* It **never manages Paisa.** Spawning the app, supervising it, or editing its
  config is out of scope — Paisa is assumed to be running and pointed at the
  generated include.
* **Mode gating is enforced here, not at the route layer.** ``disabled`` does
  nothing at all; ``connect`` permits only :func:`probe` (ping/config/diagnosis,
  no writes); ``project`` additionally permits :func:`preview`,
  :func:`generate` and :func:`manual_sync`.
* **readonly is unsyncable.** A readonly Paisa instance acknowledges
  ``/api/sync`` with fake success (it reloads nothing), so a sync attempt is
  refused up front and reported as ``readonly`` rather than pretending it
  worked.
* **Backend must match on manual sync.** The configured ``paisa.ledger_cli``
  must be one of the supported backends (ledger/hledger/beancount), and the
  probed upstream Paisa backend must equal it — projecting ledger output into
  an hledger instance (or vice versa) is rejected before any file is written.
  ``connect``/``probe`` works regardless of backend: a connectivity check never
  implies a write.

Staged API (for a coordinator that drives one projection + one publish per
sync attempt):

* :func:`preflight` — mode/readiness gates, then the remote capability/backend/
  readonly checks, *before* any projection or file write. Session-free, so a
  coordinator can preflight before opening a session to generate.
* :func:`generate` — project + publish exactly once (the only stage that
  projects or writes the include).
* :func:`sync_remote` — POST ``/api/sync`` + diagnosis against an
  already-generated/already-published :class:`ProjectionReport`. Neither
  projects nor publishes; it reports the POST outcome (``post_accepted``) and
  the diagnosis outcome separately so a coordinator can stamp the remote hash /
  advance ``applied_revision`` on an accepted POST even when diagnosis is fatal
  or could not run.
* :func:`manual_sync` — composes the three into the same :class:`SyncReport`
  the route layer has always consumed (one projection, one publish).

The functions are service-first (no HTTP) so routes can be layered on later.
"""

from typing import Literal, NamedTuple

from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.integrations.paisa import (
    DEFAULT_TIMEOUT_SECONDS,
    PaisaClient,
    PaisaCapabilities,
    PaisaDiagnosis,
    PaisaError,
)
from financial_dashboard.services.paisa.config import (
    PaisaProjectionConfig,
)
from financial_dashboard.services.paisa.diagnosis import classify_diagnosis
from financial_dashboard.services.paisa.projection import (
    ProjectionError,
    ProjectionReport,
    project,
)
from financial_dashboard.services.paisa.publisher import (
    PublishError,
    PublishResult,
    publish_journal,
)
from financial_dashboard.services.paisa.renderers import (
    SUPPORTED_BACKENDS,
    validate_backend,
)

SyncOutcome = Literal[
    "synced",  # journal written, POST success=true, diagnosis clean
    "readonly",  # upstream is readonly — would report fake success
    "unsupported_backend",  # ledger_cli is not ledger
    "unreachable",  # network error probing the API
    "publish_failed",  # the include file could not be written
    "sync_rejected",  # POST /api/sync returned non-2xx or success=false
    "diagnosis_failed",  # post-sync diagnosis reported danger issues
    "not_configured",  # projection cannot run (no cutover / no accounts / no path)
    "connect_only",  # mode == connect: writes not allowed
    "disabled",  # mode == disabled
]


class ProbeReport(NamedTuple):
    """Result of a read-only probe. Allowed in ``connect`` and ``project``."""

    ok: bool
    reachable: bool
    capabilities: PaisaCapabilities | None
    diagnosis: PaisaDiagnosis | None
    reason: str | None


class PreviewReport(NamedTuple):
    """Result of a preview: the projection report, or the reason it did not run."""

    ok: bool
    report: ProjectionReport | None
    reason: str | None


class GenerateResult(NamedTuple):
    """Result of a generate: the projection plus the publish outcome."""

    ok: bool
    report: ProjectionReport | None
    publish: PublishResult | None
    reason: str | None


class SyncReport(NamedTuple):
    """Result of a manual sync. Core rows are never mutated regardless of outcome.

    ``diagnosis_expected``/``diagnosis_accepted``/``diagnosis_fatal`` classify
    the post-sync diagnosis (see :mod:`financial_dashboard.services.paisa.diagnosis`):
    contra-expense ``Debit Entry`` dangers the projection provably generated are
    *accepted* and do not fail the sync; any unmatched ``Debit Entry`` or any
    other danger kind (e.g. ``Negative Balance``) counts as *fatal* and fails
    it. ``None`` when diagnosis never ran (an earlier step failed).
    """

    ok: bool
    outcome: SyncOutcome
    preview: ProjectionReport | None
    publish: PublishResult | None
    diagnosis_ok: bool | None
    reason: str | None
    diagnosis_expected: int | None = None
    diagnosis_accepted: int | None = None
    diagnosis_fatal: int | None = None


class PreflightReport(NamedTuple):
    """Result of the pre-publish preflight: mode/readiness gates, then the
    remote capability/backend/readonly checks — with no projection and no file
    write.

    Runs *before* :func:`generate`. A non-ok preflight carries the terminal
    ``outcome`` (disabled/connect_only/not_configured/readonly/
    unsupported_backend/unreachable); an ok preflight reports ``outcome=None``
    (preflight is a pre-stage, not a sync outcome). ``capabilities`` is
    populated whenever the probe reached ``/api/config``.
    """

    ok: bool
    outcome: SyncOutcome | None
    capabilities: PaisaCapabilities | None
    reason: str | None


class RemoteSyncReport(NamedTuple):
    """Result of the remote POST + diagnosis stage (no projection, no publish).

    The file is already on disk from the caller's :func:`generate`; this stage
    POSTs ``/api/sync`` (Paisa reloads that include) and then verifies with
    diagnosis. ``post_accepted`` and ``diagnosis_ok`` are reported separately so
    a coordinator can record an accepted POST — advance ``applied_revision`` and
    stamp ``last_remote_hash`` — even when the diagnosis that follows is fatal
    (``diagnosis_ok=False``) or could not run (``diagnosis_ok=None``). ``ok`` is
    True only when the POST was accepted AND diagnosis came back healthy.
    """

    ok: bool
    outcome: SyncOutcome
    post_accepted: bool
    diagnosis_ok: bool | None
    reason: str | None
    diagnosis_expected: int | None = None
    diagnosis_accepted: int | None = None
    diagnosis_fatal: int | None = None


# ---------------------------------------------------------------------------
# Probe (connect or project; no writes)
# ---------------------------------------------------------------------------


async def probe(
    config: PaisaProjectionConfig,
    *,
    client: PaisaClient | None = None,
) -> ProbeReport:
    """Ping, fetch config and diagnosis — read-only network checks.

    Permitted in ``connect`` and ``project`` modes. Never writes. Useful for a
    connectivity/status route that must not imply projection is enabled.
    """
    if not config.can_connect:
        return ProbeReport(
            ok=False,
            reachable=False,
            capabilities=None,
            diagnosis=None,
            reason="disabled",
        )

    owns_client = client is None
    if client is None:
        client = _build_client(config)
    try:
        try:
            capabilities = await client.fetch_config()
        except PaisaError as exc:
            return ProbeReport(
                ok=False,
                reachable=exc.code != "unreachable" and exc.code != "http_error",
                capabilities=None,
                diagnosis=None,
                reason=f"config probe failed ({exc.code}): {exc.message}",
            )
        try:
            diagnosis = await client.diagnosis()
        except PaisaError as exc:
            return ProbeReport(
                ok=True,
                reachable=True,
                capabilities=capabilities,
                diagnosis=None,
                reason=f"diagnosis failed ({exc.code}): {exc.message}",
            )
        return ProbeReport(
            ok=True,
            reachable=True,
            capabilities=capabilities,
            diagnosis=diagnosis,
            reason=None,
        )
    finally:
        if owns_client:
            await client.aclose()


# ---------------------------------------------------------------------------
# Preview & generate (local only — no network)
# ---------------------------------------------------------------------------


async def preview(
    session: AsyncSession, config: PaisaProjectionConfig
) -> PreviewReport:
    """Project without writing anything or touching the network."""
    if not config.can_project:
        return PreviewReport(
            ok=False, report=None, reason=_write_blocked_reason(config)
        )
    if not config.ready_to_project:
        return PreviewReport(ok=False, report=None, reason="not_configured")
    try:
        report = await project(session, config)
    except ProjectionError as exc:
        return PreviewReport(ok=False, report=None, reason=str(exc))
    return PreviewReport(ok=True, report=report, reason=None)


async def generate(
    session: AsyncSession, config: PaisaProjectionConfig
) -> GenerateResult:
    """Project and atomically publish the generated include file.

    Requires ``project`` mode. No network: this is the "write what we have"
    step. A subsequent manual sync reloads Paisa; generate alone leaves the file
    on disk for Paisa to pick up on its next natural reload too.
    """
    previewed = await preview(session, config)
    if not previewed.ok or previewed.report is None:
        return GenerateResult(
            ok=False, report=None, publish=None, reason=previewed.reason
        )
    if not config.generated_path:
        return GenerateResult(
            ok=False,
            report=previewed.report,
            publish=None,
            reason="generated_path not configured",
        )
    try:
        result = publish_journal(config.generated_path, previewed.report.journal)
    except PublishError as exc:
        return GenerateResult(
            ok=False,
            report=previewed.report,
            publish=None,
            reason=f"publish_failed: {exc}",
        )
    return GenerateResult(ok=True, report=previewed.report, publish=result, reason=None)


# ---------------------------------------------------------------------------
# Client builder + mode-gate helper
# ---------------------------------------------------------------------------


def _build_client(config: PaisaProjectionConfig) -> PaisaClient:
    return PaisaClient(
        base_url=config.base_url,
        allow_remote=config.allow_remote,
        auth_username=config.auth_username,
        auth_password=config.auth_password,
        timeout_seconds=float(
            config.request_timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        ),
    )


def _write_blocked_reason(config: PaisaProjectionConfig) -> str:
    if config.mode == "disabled":
        return "disabled"
    return "connect_only"


# ---------------------------------------------------------------------------
# Preflight — pre-publish remote checks (no projection, no file write)
# ---------------------------------------------------------------------------


async def preflight(
    config: PaisaProjectionConfig,
    *,
    client: PaisaClient | None = None,
) -> PreflightReport:
    """Pre-publish remote preflight: mode/readiness gates, then the remote
    capability / backend / readonly checks — with no projection and no file
    write.

    This is the gate a coordinator runs *before* generating/publishing: a
    readonly Paisa would acknowledge ``/api/sync`` with fake success, and a
    backend mismatch (ledger output into an hledger instance) cannot be
    corrected after the fact, so both are caught here, upstream of any write.
    The local gates (mode / readiness / generated_path) run first and never
    touch the network.

    Session-free, so a coordinator can preflight before opening a session to
    generate. When ``client`` is omitted a fresh client is built from the config
    and closed when the call returns; when supplied the caller owns it — a
    coordinator reuses one client across :func:`preflight` and
    :func:`sync_remote`. Never returns a live client.
    """
    if not config.can_project:
        outcome: SyncOutcome = (
            "disabled" if config.mode == "disabled" else "connect_only"
        )
        return PreflightReport(
            ok=False,
            outcome=outcome,
            capabilities=None,
            reason=_write_blocked_reason(config),
        )
    if not config.ready_to_project:
        return PreflightReport(
            ok=False,
            outcome="not_configured",
            capabilities=None,
            reason="cutover date or selected accounts missing",
        )
    if not config.generated_path:
        return PreflightReport(
            ok=False,
            outcome="not_configured",
            capabilities=None,
            reason="generated_path not configured",
        )

    owns_client = client is None
    if client is None:
        client = _build_client(config)
    try:
        # Probe capabilities FIRST. A readonly instance would fake-sync, and an
        # hledger backend cannot consume our ledger output — both must be caught
        # before we write a file or POST anything.
        try:
            capabilities = await client.fetch_config()
        except PaisaError as exc:
            if exc.code in (
                "unreachable",
                "http_error",
                "bad_json",
                "redirect_disallowed",
            ):
                return PreflightReport(
                    ok=False,
                    outcome="unreachable",
                    capabilities=None,
                    reason=f"could not reach Paisa: {exc.message}",
                )
            return PreflightReport(
                ok=False,
                outcome="unreachable",
                capabilities=None,
                reason=f"config probe failed ({exc.code}): {exc.message}",
            )

        if capabilities.readonly:
            return PreflightReport(
                ok=False,
                outcome="readonly",
                capabilities=capabilities,
                reason="Paisa is readonly; /api/sync would report fake success.",
            )
        # The configured backend must be supported, and the upstream Paisa
        # instance must actually be that backend — ledger output does not parse
        # as hledger/beancount, and vice versa. A missing upstream ledger_cli
        # (older Paisa, or a version that does not report it) is tolerated so
        # connectivity still works; a *present but differing* backend is a hard
        # mismatch.
        configured_backend = validate_backend(config.ledger_cli)
        upstream = capabilities.ledger_cli
        if upstream and upstream.strip().lower() != configured_backend:
            return PreflightReport(
                ok=False,
                outcome="unsupported_backend",
                capabilities=capabilities,
                reason=(
                    f"Paisa ledger_cli={upstream!r} does not match configured "
                    f"{configured_backend!r}; only {list(SUPPORTED_BACKENDS)} are "
                    f"supported and the upstream must equal the configured backend."
                ),
            )

        return PreflightReport(
            ok=True,
            outcome=None,
            capabilities=capabilities,
            reason=None,
        )
    finally:
        if owns_client:
            await client.aclose()


# ---------------------------------------------------------------------------
# Remote stage — POST /api/sync + diagnosis (no projection, no publish)
# ---------------------------------------------------------------------------


async def sync_remote(
    report: ProjectionReport,
    config: PaisaProjectionConfig,
    *,
    client: PaisaClient | None = None,
) -> RemoteSyncReport:
    """Remote POST + diagnosis stage against an already-generated /
    already-published report.

    The caller's :func:`generate` has already projected and written the include
    file; this stage POSTs ``/api/sync`` (Paisa reloads that include) and then
    verifies with diagnosis, classifying contra-expense ``Debit Entry`` dangers
    the projection provably generated as expected. Neither projects nor
    publishes — call it with the same :class:`ProjectionReport` ``generate``
    returned (``gen.report``) so the diagnosis is classified against exactly the
    postings on disk.

    The POST result (``post_accepted``) and the diagnosis result
    (``diagnosis_ok``) are reported separately so a coordinator can record an
    accepted POST — advance ``applied_revision`` and stamp ``last_remote_hash``
    — even when the diagnosis that follows is fatal or could not run. ``ok`` is
    True only when the POST was accepted AND diagnosis came back healthy.

    When ``client`` is omitted a fresh client is built and closed on return;
    when supplied the caller owns it (reuse one client across :func:`preflight`
    and this stage). Never returns a live client.
    """
    owns_client = client is None
    if client is None:
        client = _build_client(config)
    try:
        return await _do_sync_remote(report, client)
    finally:
        if owns_client:
            await client.aclose()


async def _do_sync_remote(
    report: ProjectionReport,
    client: PaisaClient,
) -> RemoteSyncReport:
    # POST the exact sync payload. Accepted requires 2xx AND success=true.
    try:
        sync_result = await client.sync_journal()
    except PaisaError as exc:
        return RemoteSyncReport(
            ok=False,
            outcome="unreachable" if exc.code == "unreachable" else "sync_rejected",
            post_accepted=False,
            diagnosis_ok=None,
            reason=f"sync failed ({exc.code}): {exc.message}",
        )
    if not sync_result.accepted:
        reason = sync_result.reason or f"HTTP {sync_result.status_code}"
        return RemoteSyncReport(
            ok=False,
            outcome="sync_rejected",
            post_accepted=False,
            diagnosis_ok=None,
            reason=f"/api/sync rejected: {reason}",
        )

    # Verify with diagnosis. Contra-expense ``Debit Entry`` dangers the
    # projection provably generated are *expected* and downgraded (our canonical
    # semantics post refunds/cashback/reversals as negative Expenses so they
    # net). Anything unmatched — an extra/unknown Debit Entry, an
    # operator-authored negative Expenses posting, or any other danger kind
    # (Negative Balance, …) — stays fatal. The probe path is never touched: it
    # still surfaces raw diagnosis. The POST was accepted regardless of the
    # diagnosis outcome, so a coordinator may stamp the remote hash first.
    try:
        diagnosis = await client.diagnosis()
    except PaisaError as exc:
        # The sync was accepted but we could not verify. The file is written;
        # report a soft failure rather than claiming success. post_accepted
        # stays True so the coordinator can still record the accepted POST.
        return RemoteSyncReport(
            ok=False,
            outcome="diagnosis_failed",
            post_accepted=True,
            diagnosis_ok=None,
            reason=f"diagnosis failed ({exc.code}): {exc.message}",
        )
    classified = classify_diagnosis(diagnosis, report.entries)
    if classified.fatal_count > 0:
        return RemoteSyncReport(
            ok=False,
            outcome="diagnosis_failed",
            post_accepted=True,
            diagnosis_ok=False,
            diagnosis_expected=classified.expected_count,
            diagnosis_accepted=classified.accepted_count,
            diagnosis_fatal=classified.fatal_count,
            reason=classified.first_fatal_message
            or f"{classified.fatal_count} unresolved diagnosis danger(s)",
        )

    return RemoteSyncReport(
        ok=True,
        outcome="synced",
        post_accepted=True,
        diagnosis_ok=True,
        diagnosis_expected=classified.expected_count,
        diagnosis_accepted=classified.accepted_count,
        diagnosis_fatal=0,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Manual sync (network, requires project mode) — composes the stages
# ---------------------------------------------------------------------------


async def manual_sync(
    session: AsyncSession,
    config: PaisaProjectionConfig,
    *,
    client: PaisaClient | None = None,
) -> SyncReport:
    """Probe, write, POST sync, and verify — without ever mutating core rows.

    Composes the staged API: :func:`preflight` (remote capability/backend/
    readonly, plus the mode/readiness gates — no file write on failure), then
    :func:`generate` (project + publish exactly once), then :func:`sync_remote`
    (POST + diagnosis using that same generated report). The result is shaped
    into the same :class:`SyncReport` the route layer has always consumed, so
    public response semantics are unchanged: one projection, one publish.

    Requires ``project`` mode. ``client`` is injectable for tests (e.g. an
    :class:`httpx.MockTransport`-backed client). When omitted, a fresh client is
    built from the config and closed when the call returns.
    """
    owns_client = client is None
    if client is None:
        client = _build_client(config)

    try:
        pre = await preflight(config, client=client)
        if not pre.ok:
            # Every failed preflight carries a terminal outcome (None only on
            # the ok path); capabilities are not part of the SyncReport surface.
            assert pre.outcome is not None
            return SyncReport(
                ok=False,
                outcome=pre.outcome,
                preview=None,
                publish=None,
                diagnosis_ok=None,
                reason=pre.reason,
            )

        generated = await generate(session, config)
        if not generated.ok or generated.report is None:
            outcome: SyncOutcome = (
                "publish_failed" if generated.publish is None else "sync_rejected"
            )
            return SyncReport(
                ok=False,
                outcome=outcome,
                preview=generated.report,
                publish=generated.publish,
                diagnosis_ok=None,
                reason=generated.reason,
            )

        remote = await sync_remote(generated.report, config, client=client)
        return SyncReport(
            ok=remote.ok,
            outcome=remote.outcome,
            preview=generated.report,
            publish=generated.publish,
            diagnosis_ok=remote.diagnosis_ok,
            reason=remote.reason,
            diagnosis_expected=remote.diagnosis_expected,
            diagnosis_accepted=remote.diagnosis_accepted,
            diagnosis_fatal=remote.diagnosis_fatal,
        )
    finally:
        if owns_client:
            await client.aclose()
