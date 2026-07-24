"""Dashboard-facing surface for the Paisa extension.

This module is the adapter between the core ``services.paisa.*`` (config,
orchestrator, projection, publisher, renderer — left untouched) and the
API/web routes under ``/api/extensions/paisa`` and ``/extensions/paisa``.

Responsibilities, all dashboard-owned (none of which belong in the core
projection):

* :func:`config_view` — read the ``paisa.*`` settings via the existing
  accessors and shape them into a *redacted* DTO (the password never leaves).
* :func:`account_choices` — list native dashboard accounts with their
  selected state, so the HTML picker and the JSON API agree on the choices.
* :func:`save_config` — validate an input model and persist it through the
  shared ``save_settings`` path (which handles encryption for the password),
  preserving the current secret when a blank password is submitted.
* :func:`probe_status` / :func:`preview_projection` / :func:`generate_now` /
  :func:`sync_now` — dispatch to the orchestrator and serialize its
  NamedTuple reports into the JSON DTOs, catching optional-extension failures
  into typed responses so a core route is never affected.

It never mutates core rows (the orchestrator already guarantees that), never
manages the Paisa process or its config, and never creates directories — the
generated-path validator mirrors the publisher's safety check without
materializing anything.
"""

import asyncio
import datetime
import json
import logging
import urllib.parse
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any, Awaitable, Callable, NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.core.dates import parse_date
from financial_dashboard.db.models import Account, ExtensionRun
from financial_dashboard.extensions import ExtensionManifest
from financial_dashboard.integrations.paisa import (
    PaisaClient,
    PaisaError,
    validate_base_url,
)
from financial_dashboard.schemas.extensions import (
    ExtensionAuditResponse,
    ExtensionInfo,
    ExtensionRunInfo,
    PaisaAccountChoice,
    PaisaAccountChoicesResponse,
    PaisaCapabilitiesInfo,
    PaisaConfig,
    PaisaConfigInput,
    PaisaConfigSaveResponse,
    PaisaDiagnosisInfo,
    PaisaDiagnosisIssueInfo,
    PaisaFxRateRow,
    PaisaGenerateResponse,
    PaisaPreviewResponse,
    PaisaProjectionSummary,
    PaisaPublishInfo,
    PaisaReconcileResponse,
    PaisaReportSummary,
    PaisaSkippedRowInfo,
    PaisaStatusResponse,
    PaisaSyncResponse,
)
from financial_dashboard.services.paisa.audit import (
    OPERATION_GENERATE,
    OPERATION_PROBE,
    OPERATION_SYNC,
    STATUS_FAILURE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    complete_run,
    recent_runs,
    run_started_at,
    sanitize_error,
    start_run,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig, load_config
from financial_dashboard.services.paisa.coordinator import claim_manual_lease
from financial_dashboard.services.paisa.orchestrator import (
    GenerateResult,
    PreviewReport,
    SyncReport,
    generate,
    manual_sync,
    preview,
    probe,
)
from financial_dashboard.services.paisa.reconciliation import build_reconciliation
from financial_dashboard.services.paisa.reporting import (
    DTO_REPORT_KINDS,
    PaisaReportService,
    ReportCacheKey,
    fetch_report as _fetch_report_impl,
    report_cache_key,
    report_to_dto,
)
from financial_dashboard.services.paisa.renderer import (
    InvalidAccountName,
    validate_account_name,
    validate_backend,
)
from financial_dashboard.services.paisa.renderers import SUPPORTED_BACKENDS
from financial_dashboard.services.paisa.report_cache import get_report_cache
from financial_dashboard.services.paisa.sync_state import (
    DEFAULT_LEASE_TTL_SECONDS,
    DIAGNOSIS_ACCEPTED,
    DIAGNOSIS_FATAL,
    DIAGNOSIS_HEALTHY,
    DIAGNOSIS_UNKNOWN,
    LeaseStaleError,
    capture_target,
    heartbeat_lease,
    record_accepted_post,
    record_diagnosis,
    record_pre_post_failure,
    record_published_hash,
    release_lease,
)
from financial_dashboard.services.settings import (
    get_setting_bool,
    get_setting_int,
    save_settings,
)

logger = logging.getLogger(__name__)

#: The only mode values the surface accepts. The config loader coerces anything
#: else back to ``disabled`` (fail-safe inactive), but the save path rejects an
#: unknown value outright so the operator is told rather than silently downgraded.
VALID_MODES = ("disabled", "connect", "project")

#: Non-INR policy is ``skip`` or ``priced``; ``priced`` requires configured FX
#: rates. The config loader coerces anything else back to ``skip``; the save
#: path accepts both explicit values.
SUPPORTED_NON_INR_POLICY = ("skip", "priced")

#: Manual-lease heartbeat: a manual generate/sync claims the singleton lease
#: (TTL 90s) for single-flight exclusion with the automatic coordinator. A
#: long manual operation (large journal + slow remote POST) could exceed the
#: TTL; without a heartbeat the lease would expire, the coordinator could
#: reclaim it mid-operation, and the manual op's token-guarded state writes
#: would silently fail (issue: remote sync appears coordinated while state
#: stays stale). The heartbeat extends the lease every interval so a manual
#: op cannot outlive its lease. The interval is well under the TTL so a
#: single missed beat does not lose the lease.
MANUAL_HEARTBEAT_INTERVAL_SECONDS = 20.0
MANUAL_LEASE_TTL_SECONDS = DEFAULT_LEASE_TTL_SECONDS  # 90s

#: FX/UI helpers --------------------------------------------------------------

_Q4 = Decimal("0.0001")


def _fx_rates_to_rows(cfg: PaisaProjectionConfig) -> list[PaisaFxRateRow]:
    """Flatten the config's nested ``{ccy: [FxRate]}`` into sorted UI rows.

    Currency is uppercased and rates are quantized to match the stored form, so
    a save round-trip is byte-stable. Deterministic ordering (currency, then
    date) keeps the editor diff-stable.
    """
    rows: list[PaisaFxRateRow] = []
    for ccy, rates in sorted(cfg.fx_rates.items()):
        for rate in rates:
            rows.append(
                PaisaFxRateRow(
                    currency=ccy,
                    date=rate.date.isoformat(),
                    rate=str(rate.rate),
                )
            )
    return rows


class PaisaSaveError(Exception):
    """Carries the list of human-readable validation errors from :func:`save_config`."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


# ---------------------------------------------------------------------------
# Manifests → DTOs
# ---------------------------------------------------------------------------


def extension_info(manifest: ExtensionManifest) -> ExtensionInfo:
    return ExtensionInfo(
        id=manifest.id,
        display_name=manifest.display_name,
        description=manifest.description,
        capabilities=sorted(manifest.capabilities),
    )


# ---------------------------------------------------------------------------
# Config (redacted read)
# ---------------------------------------------------------------------------


def config_view() -> PaisaConfig:
    """Load the Paisa config and return a redacted DTO.

    The auth password is never returned — only whether one is set. Everything
    else is surfaced typed so the API and the HTML form render from one shape.
    """
    cfg = load_config()
    return _config_to_dto(cfg)


def _config_to_dto(cfg: PaisaProjectionConfig) -> PaisaConfig:
    return PaisaConfig(
        mode=cfg.mode,
        base_url=cfg.base_url,
        external_url=cfg.external_url,
        allow_remote=cfg.allow_remote,
        auth_username=cfg.auth_username,
        auth_password_set=bool(cfg.auth_password),
        generated_path=cfg.generated_path,
        selected_account_ids=list(cfg.selected_account_ids),
        project_since=cfg.cutover_date.isoformat() if cfg.cutover_date else "",
        account_mappings=dict(cfg.account_mappings),
        category_mappings=dict(cfg.category_mappings),
        non_inr_policy=cfg.non_inr_policy,
        request_timeout_seconds=cfg.request_timeout_seconds,
        ledger_cli=cfg.ledger_cli,
        fx_rates=_fx_rates_to_rows(cfg),
        report_cache_ttl_seconds=max(
            0, get_setting_int("paisa.report_cache_ttl_seconds", 60)
        ),
        auto_sync_enabled=get_setting_bool("paisa.auto_sync_enabled", False),
        auto_sync_min_interval_minutes=max(
            1, get_setting_int("paisa.auto_sync_min_interval_minutes", 1)
        ),
        notify_sync_failures=get_setting_bool("paisa.notify_sync_failures", False),
        project_investments=cfg.project_investments,
        can_connect=cfg.can_connect,
        can_project=cfg.can_project,
    )


# ---------------------------------------------------------------------------
# Account choices
# ---------------------------------------------------------------------------


async def account_choices(session: AsyncSession) -> PaisaAccountChoicesResponse:
    """Native dashboard accounts, flagged with their projection-selected state.

    Ordered by id for a stable picker; the projection only consumes accounts
    whose id is in ``selected_account_ids``, so the flag is a pure read of the
    config.
    """
    cfg = load_config()
    selected = set(cfg.selected_account_ids)
    rows = (await session.execute(select(Account).order_by(Account.id))).scalars().all()
    return PaisaAccountChoicesResponse(
        accounts=[
            PaisaAccountChoice(
                id=a.id,
                bank=a.bank,
                label=a.label,
                type=a.type,
                selected=a.id in selected,
            )
            for a in rows
        ]
    )


# ---------------------------------------------------------------------------
# Config save (shared validation + persistence)
# ---------------------------------------------------------------------------


def _validate_external_url(url: str) -> str:
    """A safe deep-link target must be a well-formed http/https URL.

    Unlike the connection base URL, the external URL is only linked to (never
    connected to server-side), so the loopback/remote-https rules do not apply
    — but it must still be http/https so the rendered ``<a href>`` can never
    smuggle a ``javascript:`` or other disallowed scheme.
    """
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("must be an http or https URL")
    if not parsed.hostname:
        raise ValueError("must have a host")
    return url.strip()


def _validate_generated_path(path: str) -> str:
    """Mirror the publisher's path safety without creating anything.

    The target must be absolute, must not traverse with ``..``, and its parent
    directory must already exist. Directories are never materialized here —
    silently creating a tree on a typo'd path is the kind of surprise a
    financial file must not allow.
    """
    if not path:
        raise ValueError("is required for project mode")
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise ValueError(f"must be an absolute path, got {path!r}")
    if ".." in target.parts:
        raise ValueError(f"must not contain '..': {path!r}")
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError(f"parent directory does not exist: {str(parent)!r}")
    return path


def _validate_mappings(raw: dict[str, str], label: str, backend: str) -> dict[str, str]:
    """Validate a {key: ledger-account-name} mapping.

    Keys are stringified; values must pass :func:`validate_account_name` so a
    malformed override can never reach the journal. Validation uses the
    renderer backend being saved, exactly as projection will later validate an
    operator override. An invalid name raises :class:`PaisaSaveError` with a
    field-prefixed message.
    """
    out: dict[str, str] = {}
    for key, value in raw.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if not key_text or not value_text:
            continue
        try:
            validate_account_name(value_text, backend)
        except InvalidAccountName as exc:
            raise PaisaSaveError(
                [f"{label}: {key_text!r} → {value_text!r}: {exc}"]
            ) from exc
        out[key_text] = value_text
    return out


def _validate_fx_rates(rows: list[PaisaFxRateRow]) -> str:
    """Validate deterministic currency/date/rate rows and serialize them to the
    exact nested JSON the projection config reads.

    * Currency must be a 1–6 char uppercase token (normalized); duplicates of the
      same ``(currency, date)`` are collapsed, last-wins.
    * Date must parse as ISO ``YYYY-MM-DD``.
    * Rate must be a positive ``Decimal``; quantized to 4 dp to match the
      projection's stored precision.

    Rows with an unparseable/missing field are dropped (never raise) so a single
    bad cell cannot block saving the rest. Returns the JSON string. Raises
    :class:`PaisaSaveError` only for a structurally-wrong row shape (a negative
    rate is a hard error, not a silent drop, so an operator never prices a
    transaction with a wrong-signed rate).
    """
    bucket: dict[str, dict[str, str]] = {}
    for row in rows or []:
        ccy = (row.currency or "").strip().upper()
        date_text = (row.date or "").strip()
        rate_text = (row.rate or "").strip()
        if not ccy or not date_text or not rate_text:
            continue
        if not all(c.isalnum() for c in ccy) or not (1 <= len(ccy) <= 6):
            raise PaisaSaveError([f"FX Rates: invalid currency {ccy!r}"])
        try:
            date = parse_date(date_text)
        except ValueError, OverflowError:
            raise PaisaSaveError(
                [f"FX Rates: {ccy} date {date_text!r} is not a valid date"]
            )
        if date is None:
            raise PaisaSaveError(
                [f"FX Rates: {ccy} date {date_text!r} is not a valid date"]
            )
        try:
            rate = Decimal(rate_text)
        except DecimalException, OverflowError, ValueError:
            raise PaisaSaveError(
                [f"FX Rates: {ccy} {date_text} rate {rate_text!r} is not a number"]
            )
        if not rate.is_finite() or rate <= 0:
            raise PaisaSaveError(
                [
                    f"FX Rates: {ccy} {date_text} rate must be positive, got {rate_text!r}"
                ]
            )
        try:
            quantized = rate.quantize(_Q4)
        except DecimalException, OverflowError:
            raise PaisaSaveError(
                [
                    f"FX Rates: {ccy} {date_text} rate {rate_text!r} "
                    "is outside the supported range"
                ]
            )
        if not quantized.is_finite() or quantized <= 0:
            raise PaisaSaveError(
                [
                    f"FX Rates: {ccy} {date_text} rate {rate_text!r} "
                    "is outside the supported range"
                ]
            )
        bucket.setdefault(ccy, {})[date.isoformat()] = str(quantized)

    nested: dict[str, list[dict[str, str]]] = {}
    for ccy, by_date in sorted(bucket.items()):
        nested[ccy] = [{"date": d, "rate": by_date[d]} for d in sorted(by_date.keys())]
    return json.dumps(nested)


async def save_config(
    session: AsyncSession, data: PaisaConfigInput
) -> PaisaConfigSaveResponse:
    """Validate ``data`` and persist it through the shared settings path.

    Validation runs fully before any write: if any field is invalid the
    response carries ``ok=False`` and the human-readable ``errors``, and
    nothing is saved. The auth password is preserved when the submitted value
    is blank (omitted from the updates dict) so an operator saving unrelated
    fields does not wipe the credential.
    """
    errors: list[str] = []
    updates: dict[str, str] = {}

    # --- mode -------------------------------------------------------------
    mode = (data.mode or "").strip().lower()
    if mode not in VALID_MODES:
        errors.append("Mode must be one of: disabled, connect, project.")
    else:
        updates["paisa.mode"] = mode

    # --- base URL + allow_remote (validated together) ---------------------
    allow_remote = bool(data.allow_remote)
    base_url = (data.base_url or "").strip()
    try:
        validate_base_url(base_url, allow_remote=allow_remote)
    except PaisaError as exc:
        errors.append(f"Base URL: {exc.message}")
    else:
        updates["paisa.base_url"] = base_url
        updates["paisa.allow_remote"] = "true" if allow_remote else "false"

    # --- external URL (deep-link target only) -----------------------------
    try:
        updates["paisa.external_url"] = _validate_external_url(data.external_url or "")
    except ValueError as exc:
        errors.append(f"External URL: {exc}")

    # --- auth -------------------------------------------------------------
    updates["paisa.auth_username"] = (data.auth_username or "").strip()
    password = data.auth_password or ""
    if password:
        updates["paisa.auth_password"] = password
    # A blank password is intentionally omitted: save_settings leaves the row
    # untouched when a key is absent from the updates dict, preserving the
    # current encrypted secret.

    # --- generated path (required for project mode) -----------------------
    generated_path = (data.generated_path or "").strip()
    if mode == "project":
        try:
            updates["paisa.generated_path"] = _validate_generated_path(generated_path)
        except ValueError as exc:
            errors.append(f"Generated Path: {exc}")
    else:
        updates["paisa.generated_path"] = generated_path

    # --- selected account ids (must exist in the DB) ----------------------
    raw_ids = list(data.selected_account_ids or [])
    ids = _dedupe_ints(raw_ids)
    if ids:
        existing = set(
            (await session.execute(select(Account.id).where(Account.id.in_(ids))))
            .scalars()
            .all()
        )
        missing = sorted(set(ids) - existing)
        if missing:
            errors.append(f"Selected Account IDs not found: {missing}")
        else:
            updates["paisa.selected_account_ids"] = json.dumps(ids)
    else:
        updates["paisa.selected_account_ids"] = "[]"

    # --- cutover date -----------------------------------------------------
    project_since = (data.project_since or "").strip()
    cutover: datetime.date | None = None
    parse_failed = False
    if project_since:
        try:
            cutover = parse_date(project_since)
        except ValueError, OverflowError:
            parse_failed = True
    if project_since and (parse_failed or cutover is None):
        errors.append("Project Since: not a valid date (use YYYY-MM-DD).")
    elif mode == "project" and not project_since:
        errors.append("Project Since: required for project mode.")
    else:
        updates["paisa.project_since"] = project_since

    # --- ledger backend ---------------------------------------------------
    # validate_backend coerces an unknown value to ledger at load time; the save
    # path rejects an unknown value outright so the operator is told rather than
    # silently downgraded, and a manual sync's upstream-backend match check has a
    # trustworthy configured value to compare against. Resolve this before
    # mappings so valid inputs are checked with the exact renderer grammar that
    # projection will use; an invalid backend never falls through to Ledger's
    # default grammar and produce misleading mapping results.
    raw_ledger_cli = (data.ledger_cli or "").strip()
    ledger_cli = validate_backend(raw_ledger_cli)
    backend_valid = raw_ledger_cli.lower() in SUPPORTED_BACKENDS
    if not backend_valid:
        errors.append(
            f"Ledger CLI Backend must be one of: {', '.join(SUPPORTED_BACKENDS)}."
        )
    else:
        updates["paisa.ledger_cli"] = ledger_cli

    # --- mappings ---------------------------------------------------------
    # Do not validate against validate_backend's Ledger fallback when the raw
    # backend itself is invalid. The backend field error above is authoritative,
    # and no settings are written when any validation error exists.
    if backend_valid:
        try:
            account_mappings = _validate_mappings(
                dict(data.account_mappings or {}), "Account Mappings", ledger_cli
            )
        except PaisaSaveError as exc:
            errors.extend(exc.errors)
            account_mappings = {}
        updates["paisa.account_mappings"] = json.dumps(account_mappings)

        try:
            category_mappings = _validate_mappings(
                dict(data.category_mappings or {}), "Category Mappings", ledger_cli
            )
        except PaisaSaveError as exc:
            errors.extend(exc.errors)
            category_mappings = {}
        updates["paisa.category_mappings"] = json.dumps(category_mappings)

    # --- non-INR policy (skip | priced) ----------------------------------
    policy = (data.non_inr_policy or "skip").strip().lower()
    if policy not in SUPPORTED_NON_INR_POLICY:
        errors.append("Non-INR Policy must be one of: skip, priced.")
    else:
        updates["paisa.non_inr_policy"] = policy

    # --- FX rates (priced policy) ----------------------------------------
    # Deterministic currency/date/rate rows; only positive Decimal rates are
    # accepted. Serialized to the exact nested JSON the projection config reads
    # ({"CCY": [{"date": "YYYY-MM-DD", "rate": "<quantized>"}]}).
    try:
        updates["paisa.fx_rates"] = _validate_fx_rates(data.fx_rates or [])
    except PaisaSaveError as exc:
        errors.extend(exc.errors)
        # Still set a safe empty payload so a later field error does not leave a
        # stale value; validation runs fully before any write, so this never
        # reaches save_settings when there is at least one error.
        updates["paisa.fx_rates"] = "{}"

    # --- request timeout --------------------------------------------------
    # Pydantic guarantees an int on the API path; the HTML form path parses its
    # own int, so by the time we see this it is already an int. The < 1 check
    # still rejects 0/negative, which Pydantic's ``int`` does not.
    timeout = data.request_timeout_seconds
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        errors.append("Request Timeout: must be a positive integer (>= 1).")
    else:
        updates["paisa.request_timeout_seconds"] = str(timeout)

    # --- report cache TTL -------------------------------------------------
    ttl = data.report_cache_ttl_seconds
    if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl < 0:
        errors.append("Report Cache TTL: must be a non-negative integer (>= 0).")
    else:
        updates["paisa.report_cache_ttl_seconds"] = str(ttl)

    # --- automation settings ---------------------------------------------
    # Auto-sync additionally requires project mode at runtime (the automation
    # hook no-ops otherwise); saving the flags is always allowed so an operator
    # can pre-configure them, but the UI labels the project-mode dependency.
    updates["paisa.auto_sync_enabled"] = (
        "true" if bool(data.auto_sync_enabled) else "false"
    )
    min_interval = data.auto_sync_min_interval_minutes
    if (
        not isinstance(min_interval, int)
        or isinstance(min_interval, bool)
        or min_interval < 1
    ):
        errors.append("Auto Sync Min Interval: must be a positive integer (>= 1).")
    else:
        updates["paisa.auto_sync_min_interval_minutes"] = str(min_interval)
    updates["paisa.notify_sync_failures"] = (
        "true" if bool(data.notify_sync_failures) else "false"
    )
    # Investment-lot projection is a user-facing bool. It only takes effect in
    # project mode (the projection gate no-ops otherwise) and never mutates
    # investment rows; saving the flag is always allowed so an operator can
    # pre-set it. No secret content.
    updates["paisa.project_investments"] = (
        "true" if bool(data.project_investments) else "false"
    )

    if errors:
        return PaisaConfigSaveResponse(ok=False, errors=errors)

    await save_settings(updates)
    return PaisaConfigSaveResponse(ok=True, errors=[], config=config_view())


def _dedupe_ints(values: list[int]) -> list[int]:
    """Order-preserving dedupe, rejecting bools (int subclass) explicitly."""
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        if isinstance(v, bool):
            continue
        if not isinstance(v, int):
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Safe deep link
# ---------------------------------------------------------------------------


def safe_link() -> str:
    """Return a safe http/https URL for the Paisa deep link, or ''.

    Prefers the configured external URL (the public-facing address) and falls
    back to the connection base URL. Both are re-validated as http/https here
    so a tampered or stale stored value can never produce a ``javascript:``
    href on the page.
    """
    cfg = load_config()
    for candidate in (cfg.external_url, cfg.base_url):
        if not candidate:
            continue
        parsed = urllib.parse.urlsplit(candidate.strip())
        if parsed.scheme.lower() in ("http", "https") and parsed.hostname:
            return candidate.strip()
    return ""


# ---------------------------------------------------------------------------
# Orchestrator dispatch → DTOs
# ---------------------------------------------------------------------------


def _capabilities_to_dto(cap) -> PaisaCapabilitiesInfo | None:
    if cap is None:
        return None
    return PaisaCapabilitiesInfo(
        ledger_cli=cap.ledger_cli,
        readonly=cap.readonly,
        default_currency=cap.default_currency,
    )


def _diagnosis_to_dto(diag) -> PaisaDiagnosisInfo | None:
    if diag is None:
        return None
    return PaisaDiagnosisInfo(
        ok=diag.ok,
        danger_count=diag.danger_count,
        warning_count=diag.warning_count,
        issues=[
            PaisaDiagnosisIssueInfo(
                level=issue.level, summary=issue.summary, details=issue.details
            )
            for issue in diag.issues
        ],
        first_message=diag.first_message,
    )


def _projection_summary(report) -> PaisaProjectionSummary | None:
    if report is None:
        return None
    return PaisaProjectionSummary(
        emitted_count=report.emitted_count,
        self_transfer_pairs=report.self_transfer_pairs,
        card_payments=report.card_payments,
        card_side_payments=report.card_side_payments,
        non_inr_count=report.non_inr_count,
        unmatched_count=report.unmatched_count,
        unknown_count=report.unknown_count,
        cutover_date=report.cutover_date.isoformat() if report.cutover_date else None,
        account_ids=list(report.account_ids),
        skipped=[
            PaisaSkippedRowInfo(txn_id=row.txn_id, reason=row.reason, detail=row.detail)
            for row in report.skipped
        ],
        # Computed diagnostics — read directly off the typed ProjectionReport
        # (every field is always present on the NamedTuple).
        imprecise_count=report.imprecise_count,
        card_payments_resolved=report.card_payments_resolved,
        card_payments_unresolved=report.card_payments_unresolved,
        card_payments_ambiguous_mask=report.card_payments_ambiguous_mask,
        investment_lot_count=report.investment_lot_count,
        investment_funding_remapped=report.investment_funding_remapped,
        investment_funding_unresolved=list(report.investment_funding_unresolved),
        investment_current_valuation_count=report.investment_current_valuation_count,
        investment_market_price_count=report.investment_market_price_count,
        investment_market_price_conflicts=list(
            report.investment_market_price_conflicts
        ),
        investment_value_only_count=report.investment_value_only_count,
        investment_quantity_mismatch_count=(report.investment_quantity_mismatch_count),
        investment_missing_market_price_count=(
            report.investment_missing_market_price_count
        ),
        investment_valuation_sources=list(report.investment_valuation_sources),
        cas_portfolio_count=report.cas_portfolio_count,
        cas_portfolio_labels=list(report.cas_portfolio_labels),
        cas_investment_scope=report.cas_investment_scope,
        manual_asset_count=report.manual_asset_count,
        manual_asset_labels=list(report.manual_asset_labels),
        manual_liability_count=report.manual_liability_count,
        manual_liability_labels=list(report.manual_liability_labels),
        net_worth_scope_complete=report.net_worth_scope_complete,
        cas_investment_coverage=report.cas_investment_coverage,
        investment_cost_basis_portfolios=list(report.investment_cost_basis_portfolios),
        investment_valuation_portfolios=list(report.investment_valuation_portfolios),
        investment_valuation_entry_count=report.investment_valuation_entry_count,
        investment_valuation_total=report.investment_valuation_total,
        investment_valuation_unrepresented=list(
            report.investment_valuation_unrepresented
        ),
        investment_unresolved_purchases=report.investment_unresolved_purchases,
        investment_unresolved_redemptions=report.investment_unresolved_redemptions,
        net_worth_sources_complete=report.net_worth_sources_complete,
        kind_counts=dict(report.kind_counts),
        projected_foreign_count=report.projected_foreign_count,
        missing_fx_rate_count=report.missing_fx_rate_count,
        source_currencies=list(report.source_currencies),
    )


def _publish_to_dto(pub) -> PaisaPublishInfo | None:
    if pub is None:
        return None
    return PaisaPublishInfo(
        published=pub.published,
        skipped=pub.skipped,
        path=pub.path,
        version=pub.version,
        body_hash=pub.body_hash,
        bytes_written=pub.bytes_written,
    )


async def probe_status(*, client: PaisaClient | None = None) -> PaisaStatusResponse:
    """Probe (connect/project only) and serialize the typed report."""
    cfg = load_config()
    report = await probe(cfg, client=client)
    return PaisaStatusResponse(
        ok=report.ok,
        reachable=report.reachable,
        mode=cfg.mode,
        can_connect=cfg.can_connect,
        can_project=cfg.can_project,
        capabilities=_capabilities_to_dto(report.capabilities),
        diagnosis=_diagnosis_to_dto(report.diagnosis),
        reason=report.reason,
    )


async def preview_projection(
    session: AsyncSession,
) -> PaisaPreviewResponse:
    """Preview (project mode only) without writing or touching the network."""
    cfg = load_config()
    report: PreviewReport = await preview(session, cfg)
    return PaisaPreviewResponse(
        ok=report.ok,
        mode=cfg.mode,
        journal=report.report.journal if report.report else None,
        summary=_projection_summary(report.report),
        reason=report.reason,
    )


async def generate_now(session: AsyncSession) -> PaisaGenerateResponse:
    """Generate (project mode only) the include file and serialize the result."""
    cfg = load_config()
    result: GenerateResult = await generate(session, cfg)
    return PaisaGenerateResponse(
        ok=result.ok,
        mode=cfg.mode,
        summary=_projection_summary(result.report),
        publish=_publish_to_dto(result.publish),
        reason=result.reason,
    )


async def sync_now(
    session: AsyncSession, *, client: PaisaClient | None = None
) -> PaisaSyncResponse:
    """Manual sync (project mode only) and serialize the report.

    Core rows are never mutated regardless of outcome — the orchestrator owns
    that guarantee; this layer only shapes the report.
    """
    cfg = load_config()
    report: SyncReport = await manual_sync(session, cfg, client=client)
    return PaisaSyncResponse(
        ok=report.ok,
        mode=cfg.mode,
        outcome=report.outcome,
        summary=_projection_summary(report.preview),
        publish=_publish_to_dto(report.publish),
        diagnosis_ok=report.diagnosis_ok,
        reason=report.reason,
        diagnosis_expected=report.diagnosis_expected,
        diagnosis_accepted=report.diagnosis_accepted,
        diagnosis_fatal=report.diagnosis_fatal,
    )


# ---------------------------------------------------------------------------
# Audited manual operations
# ---------------------------------------------------------------------------


def _safe_outcome(reason: str | None, fallback: str) -> str:
    """A short machine token for an audit ``outcome`` derived from a reason."""
    text = (reason or "").strip().lower()
    return text or fallback


#: Outcomes that represent a pre-flight mode/readiness guard rather than a real
#: attempt. They are recorded as STATUS_SKIPPED with no error so they never
#: surface in ``audit_view``'s ``last_error`` — mirroring the automatic
#: runtime's classification so manual and automatic runs agree.
GUARD_OUTCOMES = frozenset({"disabled", "connect_only", "not_configured"})


def _guard_outcome(outcome: str | None, reason: str | None) -> str | None:
    """Classify a failed manual result as a mode/readiness guard.

    Returns the guard outcome token (disabled|connect_only|not_configured) when
    the run never attempted real work, else ``None``. Mirrors the coordinator's
    guard classification so an identical guard condition classifies the same
    way whether triggered manually or automatically. A guard is recorded as
    STATUS_SKIPPED with no error.
    """
    text = (outcome or reason or "").strip().lower()
    if not text:
        return None
    if "disabled" in text:
        return "disabled"
    if "connect_only" in text:
        return "connect_only"
    if (
        "not_configured" in text
        or "generated_path" in text
        or "cutover" in text
        or "selected account" in text
    ):
        return "not_configured"
    return None


async def _audited(
    session: AsyncSession,
    *,
    operation: str,
    trigger: str,
    run: Callable[[], Awaitable["_AuditResult"]],
) -> Any:
    """Wrap a manual operation with a start/complete audit row and commit.

    The running row is persisted in a **short transaction** (``start_run`` +
    commit) *before* ``run`` is invoked. This is mandatory for SQLite writer
    lock hygiene: ``start_run`` flushes an INSERT, so without an intervening
    commit the writer lock would be held across all of ``run``'s network,
    projection, and file I/O — a manual probe would block every concurrent
    core write for the duration of the upstream call. Committing the running
    row up front also means a crash during ``run`` leaves an observable
    ``running`` row (not a silently lost one).

    After ``run`` returns (success or exception), the row is reloaded by id
    and finalized in a separate transaction. ``run`` returns an
    :class:`_AuditResult` carrying the ok flag, outcome token, reason,
    emitted/skipped counts, output hash, the safe details payload, and the
    result DTO. The audit row is completed even when ``run`` raises
    (status=failure, outcome=error) and the exception is re-raised after the
    finalize+commit so the route's failure-isolation path still runs. No
    credentials and no raw journal text are ever placed in ``details`` — only
    sanitized summary counts.

    A failed result that is actually a mode/readiness guard (disabled /
    connect_only / not_configured) or a transient single-flight ``busy`` (the
    lease was held) is recorded as STATUS_SKIPPED with no error so it never
    appears in ``last_error`` — matching the automatic runtime.
    """
    # 1. Persist the running row in a SHORT transaction so it (a) survives a
    #    crash mid-run (observable audit) and (b) does NOT hold a writer lock
    #    during run()'s network/projection/file I/O. Capture the id so the row
    #    can be reloaded/finalized in a separate transaction after run().
    audit_run = await start_run(
        session,
        extension_id="paisa",
        operation=operation,
        trigger=trigger,
    )
    run_id = audit_run.id
    await session.commit()

    # 2. Run the operation with no audit transaction held. Any session state
    #    mutations run() makes (e.g. a manual sync's record_accepted_post) land
    #    in a fresh transaction and commit together with the finalize below.
    try:
        res = await run()
    except Exception as exc:
        # 3a. Failure: roll back any partial state from run() so the
        #     audit-failure commit persists only the audit row's terminal
        #     fields (state mutations from a partially-completed run would be
        #     ambiguous). Then finalize the audit row by id in a fresh
        #     transaction and re-raise so the route's isolation still runs.
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback before audit-failure finalize failed", exc_info=True)
        await _finalize_run_by_id(
            session,
            run_id,
            status=STATUS_FAILURE,
            outcome="error",
            error=sanitize_error(f"{type(exc).__name__}: {exc}"),
        )
        raise

    # 3b. Success: classify the result and finalize the audit row.
    guard = None if res.ok else _guard_outcome(res.outcome, res.reason)
    outcome_token = res.outcome or guard or _safe_outcome(res.reason, operation)
    if res.ok:
        status = STATUS_SUCCESS
        error = None
    elif guard is not None or res.outcome == "busy":
        # A mode/readiness guard or a transient single-flight "busy" is not a
        # real failure — record it as skipped with no error.
        status = STATUS_SKIPPED
        error = None
    else:
        status = STATUS_FAILURE
        error = sanitize_error(res.reason)
    await _finalize_run_by_id(
        session,
        run_id,
        status=status,
        outcome=outcome_token,
        emitted_count=res.emitted_count,
        skipped_count=res.skipped_count,
        output_hash=res.output_hash,
        details=res.details,
        error=error,
    )
    return res.result


async def _finalize_run_by_id(
    session: AsyncSession,
    run_id: int,
    *,
    status: str,
    outcome: str | None = None,
    emitted_count: int | None = None,
    skipped_count: int | None = None,
    output_hash: str | None = None,
    details: Any = None,
    error: str | None = None,
) -> None:
    """Reload an audit run by id and finalize it, then commit.

    The running row was committed in a prior short transaction (``start_run``
    + commit) so it survives a crash mid-run. Reloading by id (rather than
    holding the ORM object across ``run``) keeps the finalize robust to
    intervening commit/expiry/refresh of the original object — only the
    persisted id matters. ``complete_run`` is idempotent on ``completed_at``
    so a redundant finalize is a no-op w.r.t. the finish time. Pending state
    mutations in the caller's session (e.g. a manual sync's
    ``record_accepted_post``) commit atomically with the audit finalization.
    """
    run = await session.get(ExtensionRun, run_id)
    if run is None:
        # Should not happen unless the row was deleted externally; log and
        # swallow so the audit path never masks the original operation result.
        logger.error("audit run id=%s vanished before finalization", run_id)
        return
    await complete_run(
        session,
        run,
        status=status,
        outcome=outcome,
        emitted_count=emitted_count,
        skipped_count=skipped_count,
        output_hash=output_hash,
        details=details,
        error=error,
    )
    await session.commit()


class _AuditResult(NamedTuple):
    ok: bool
    reason: str | None
    emitted_count: int | None
    skipped_count: int | None
    output_hash: str | None
    details: Any
    result: Any
    #: The authoritative outcome token (e.g. a sync SyncOutcome). When None the
    #: outcome is derived from the reason/guard classification.
    outcome: str | None = None


async def probe_status_audited(
    session: AsyncSession, *, trigger: str = "api"
) -> PaisaStatusResponse:
    """Probe wrapped with an audit row (manual probe only)."""

    async def run() -> _AuditResult:
        result = await probe_status()
        details = {
            "reachable": result.reachable,
            "capabilities_ledger_cli": result.capabilities.ledger_cli
            if result.capabilities
            else None,
            "capabilities_readonly": result.capabilities.readonly
            if result.capabilities
            else None,
            "diagnosis_ok": result.diagnosis.ok if result.diagnosis else None,
        }
        return _AuditResult(
            ok=result.ok,
            reason=result.reason,
            emitted_count=None,
            skipped_count=None,
            output_hash=None,
            details=details,
            result=result,
        )

    return await _audited(session, operation=OPERATION_PROBE, trigger=trigger, run=run)


async def _safe_release(session: AsyncSession, token: str | None) -> None:
    """Best-effort release of a held manual lease.

    The lease has a TTL so a failed release is self-healing; a release failure
    must never break the manual operation's audit/commit path.
    """
    if token is None:
        return
    try:
        await release_lease(session, token=token)
    except Exception:
        logger.debug("manual lease release failed", exc_info=True)


async def _safe_release_after_exception(
    session: AsyncSession, token: str | None
) -> None:
    """Roll back failed request state, then release the lease durably.

    ``_audited`` persists its running row before invoking the manual operation.
    Consequently a generate/sync exception reaches this helper with a lease
    claim that was already committed, while the request session may contain
    partial state from the failed operation. Releasing on that same session is
    unsafe: ``_audited`` must roll the partial state back, which would also roll
    the release back and strand the lease until its 90-second TTL.

    End the failed request transaction first, then release with a fresh session
    and an explicit short commit. Both steps are best-effort so cleanup never
    masks the operation's original exception; the TTL remains the final crash
    safety fallback.
    """
    if token is None:
        return
    try:
        await session.rollback()
    except Exception:
        logger.debug("rollback before manual lease release failed", exc_info=True)
    factory = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    try:
        async with factory() as release_session:
            await release_lease(release_session, token=token)
            await release_session.commit()
    except Exception:
        logger.debug("committed manual lease release failed", exc_info=True)


async def _manual_heartbeat_loop(
    bind,
    token: str,
    *,
    interval: float = MANUAL_HEARTBEAT_INTERVAL_SECONDS,
    ttl_seconds: int = MANUAL_LEASE_TTL_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Extend a held manual lease every ``interval`` until cancelled.

    Uses a fresh session per beat (derived from the caller's engine ``bind``)
    so it never contends with the caller's open request-session transaction.
    A failed beat is logged and swallowed; the loop retries on the next
    interval. If the lease was reclaimed (stale token) the heartbeat is a
    no-op (the UPDATE matches no rows) — the caller's subsequent token-guarded
    state writes will surface the staleness as :class:`LeaseStaleError`.

    Cancel the task (``task.cancel()``) when the manual operation finishes;
    the loop has no other termination condition by design.
    """
    factory = async_sessionmaker(bind, class_=AsyncSession, expire_on_commit=False)
    try:
        while True:
            await sleep(interval)
            try:
                async with factory() as hb_session:
                    await heartbeat_lease(
                        hb_session, token=token, ttl_seconds=ttl_seconds
                    )
                    await hb_session.commit()
            except Exception:
                logger.debug("manual lease heartbeat failed", exc_info=True)
    except asyncio.CancelledError:
        raise


def _start_manual_heartbeat(session: AsyncSession, token: str) -> asyncio.Task:
    """Start the heartbeat background task for a manual lease claim.

    The returned task must be cancelled (and awaited) when the manual
    operation finishes — typically in a ``finally`` block — so the heartbeat
    does not outlive the operation.
    """
    return asyncio.create_task(_manual_heartbeat_loop(session.bind, token))


async def _stop_manual_heartbeat(task: asyncio.Task) -> None:
    """Cancel and await a manual heartbeat task, swallowing CancelledError."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("manual heartbeat task teardown failed", exc_info=True)


def _manual_sync_post_accepted(outcome: str) -> bool:
    """Whether a manual SyncReport's POST was accepted, inferred from outcome.

    The SyncReport does not expose ``post_accepted`` directly; the orchestrator
    semantics map ``synced``/``diagnosis_failed`` to an accepted POST, and every
    other non-guard outcome to a pre-POST/ambiguous failure.
    """
    return outcome in ("synced", "diagnosis_failed")


async def _apply_manual_sync_state(
    session: AsyncSession,
    result: PaisaSyncResponse,
    token: str,
    *,
    target_revision: int,
) -> None:
    """Update persisted sync-state consistently for a manual sync.

    Manual sync forces a remote reload (always POSTs) using one projection, and
    advances ``applied`` only through the revision captured before projection /
    generation / POST, then stamps the remote + diagnosis state. A core write
    committed while the sync is in flight therefore remains dirty for a
    follow-up reconcile. Guards (disabled/connect_only/not_configured) record no
    state — no remote attempt happened.
    """
    from financial_dashboard.db.models import utc_now

    body_hash = result.publish.body_hash if result.publish else None
    if body_hash is not None:
        await record_published_hash(session, body_hash=body_hash, token=token)
    outcome = result.outcome
    if outcome in GUARD_OUTCOMES:
        return
    if outcome == "synced":
        await record_accepted_post(
            session,
            target_revision=target_revision,
            remote_hash=body_hash or "",
            token=token,
        )
        state = (
            DIAGNOSIS_ACCEPTED
            if (result.diagnosis_accepted or 0) > 0
            else DIAGNOSIS_HEALTHY
        )
        await record_diagnosis(
            session, state=state, healthy_hash=body_hash, token=token
        )
    elif outcome == "diagnosis_failed":
        await record_accepted_post(
            session,
            target_revision=target_revision,
            remote_hash=body_hash or "",
            token=token,
        )
        state = DIAGNOSIS_FATAL if result.diagnosis_ok is False else DIAGNOSIS_UNKNOWN
        await record_diagnosis(session, state=state, token=token)
    else:
        # unreachable/sync_rejected/readonly/unsupported_backend/publish_failed:
        # the row stays dirty; arm backoff.
        await record_pre_post_failure(session, token=token, now=utc_now())


async def generate_now_audited(
    session: AsyncSession, *, trigger: str = "api"
) -> PaisaGenerateResponse:
    """Generate wrapped with an audit row + the persisted lease single-flight.

    Waits up to 30s for the singleton lease (so a manual generate never
    overlaps a coordinator reconcile or another manual op). If the wait elapses,
    returns a typed ``busy`` outcome. Records ``last_published_hash`` only (not
    remote/applied) — a generate does not imply a remote reload.

    A heartbeat task extends the lease every 20s so a long generate (huge
    journal) cannot lose the lease mid-operation. A stale-lease state-write
    failure is surfaced in the audit details (``state_recorded: false``) rather
    than swallowed, so the audit is truthful.
    """

    async def run() -> _AuditResult:
        claim = await claim_manual_lease(session, owner="manual-generate")
        if not claim.claimed or claim.token is None:
            cfg = load_config()
            return _AuditResult(
                ok=False,
                reason="lease busy",
                emitted_count=None,
                skipped_count=None,
                output_hash=None,
                details={"reason": "lease busy"},
                result=PaisaGenerateResponse(
                    ok=False, mode=cfg.mode, reason="busy", busy=True
                ),
                outcome="busy",
            )
        token = claim.token
        hb_task = _start_manual_heartbeat(session, token)
        state_recorded = True
        failed = False
        try:
            result = await generate_now(session)
            body_hash = result.publish.body_hash if result.publish else None
            if result.ok and body_hash is not None:
                try:
                    await record_published_hash(
                        session, body_hash=body_hash, token=token
                    )
                except LeaseStaleError:
                    state_recorded = False
                    logger.warning("manual generate hash record failed: lease stale")
                except Exception:
                    state_recorded = False
                    logger.warning("manual generate hash record failed", exc_info=True)
            summary = result.summary
            emitted = summary.emitted_count if summary else None
            skipped = len(summary.skipped) if summary else None
            details = {
                "emitted_count": emitted,
                "skipped_count": skipped,
                "published": result.publish.published if result.publish else None,
                "output_hash": body_hash,
            }
            if not state_recorded:
                details["state_recorded"] = False
                details["state_error"] = "lease_stale_or_write_failed"
            return _AuditResult(
                ok=result.ok,
                reason=result.reason,
                emitted_count=emitted,
                skipped_count=skipped,
                output_hash=body_hash,
                details=details,
                result=result,
            )
        except Exception:
            failed = True
            raise
        finally:
            await _stop_manual_heartbeat(hb_task)
            if failed:
                await _safe_release_after_exception(session, token)
            else:
                await _safe_release(session, token)

    return await _audited(
        session, operation=OPERATION_GENERATE, trigger=trigger, run=run
    )


async def sync_now_audited(
    session: AsyncSession, *, trigger: str = "api"
) -> PaisaSyncResponse:
    """Manual sync wrapped with an audit row + the persisted lease single-flight.

    Waits up to 30s for the singleton lease (no overlap with the coordinator or
    another manual op); returns a typed ``busy`` outcome if the wait elapses.
    Forces a remote reload (always POSTs), uses one projection, and updates
    ``last_published_hash`` / ``last_remote_hash`` / ``applied`` / diagnosis
    state consistently so the automatic worker does not redundantly reload.

    A heartbeat task extends the lease every 20s so a long sync (large journal
    + slow remote POST) cannot lose the lease mid-operation and let the
    coordinator reclaim into overlap. A stale-lease state-write failure is
    surfaced in the audit details (``state_recorded: false``) rather than
    swallowed, so the audit never claims a coordinated sync while state
    remains stale.
    """

    async def run() -> _AuditResult:
        claim = await claim_manual_lease(session, owner="manual-sync")
        if not claim.claimed or claim.token is None:
            cfg = load_config()
            return _AuditResult(
                ok=False,
                reason="lease busy",
                emitted_count=None,
                skipped_count=None,
                output_hash=None,
                details={"reason": "lease busy"},
                result=PaisaSyncResponse(
                    ok=False, mode=cfg.mode, outcome="busy", busy=True
                ),
                outcome="busy",
            )
        token = claim.token
        hb_task = _start_manual_heartbeat(session, token)
        state_recorded = True
        failed = False
        try:
            # Capture the exact revision this manual run is about to project
            # before any generation/file/remote work. End the read transaction
            # immediately: SQLite must not retain a snapshot across the slow
            # stage, and completion must never absorb a concurrent dirty bump.
            target = await capture_target(session)
            R = target.target_revision
            await session.commit()
            result = await sync_now(session)
            try:
                await _apply_manual_sync_state(
                    session, result, token, target_revision=R
                )
            except LeaseStaleError:
                state_recorded = False
                logger.warning(
                    "manual sync state record failed: lease stale; "
                    "remote POST may have been accepted but local state "
                    "was not advanced"
                )
            except Exception:
                state_recorded = False
                logger.warning("manual sync state record failed", exc_info=True)
            summary = result.summary
            emitted = summary.emitted_count if summary else None
            skipped = len(summary.skipped) if summary else None
            output_hash = result.publish.body_hash if result.publish else None
            details = {
                "outcome": result.outcome,
                "emitted_count": emitted,
                "skipped_count": skipped,
                "diagnosis_ok": result.diagnosis_ok,
                "diagnosis_expected": result.diagnosis_expected,
                "diagnosis_accepted": result.diagnosis_accepted,
                "diagnosis_fatal": result.diagnosis_fatal,
                "published": result.publish.published if result.publish else None,
                "output_hash": output_hash,
            }
            if not state_recorded:
                details["state_recorded"] = False
                details["state_error"] = "lease_stale_or_write_failed"
            return _AuditResult(
                ok=result.ok,
                reason=result.reason,
                emitted_count=emitted,
                skipped_count=skipped,
                output_hash=output_hash,
                details=details,
                result=result,
                outcome=result.outcome,
            )
        except Exception:
            failed = True
            raise
        finally:
            await _stop_manual_heartbeat(hb_task)
            if failed:
                await _safe_release_after_exception(session, token)
            else:
                await _safe_release(session, token)

    return await _audited(session, operation=OPERATION_SYNC, trigger=trigger, run=run)


# ---------------------------------------------------------------------------
# Audit query
# ---------------------------------------------------------------------------


def _run_to_info(run: ExtensionRun) -> ExtensionRunInfo:
    started = run_started_at(run)
    completed = run.completed_at
    duration: float | None = None
    if completed is not None:
        end = completed
        if end.tzinfo is None:
            end = end.replace(tzinfo=datetime.UTC)
        duration = (end - started).total_seconds()
    details: dict | None = None
    if run.details:
        try:
            decoded = json.loads(run.details)
            if isinstance(decoded, dict):
                details = decoded
        except ValueError, TypeError:
            details = None
    return ExtensionRunInfo(
        id=run.id,
        extension_id=run.extension_id,
        operation=run.operation,
        status=run.status,
        outcome=run.outcome,
        trigger=run.trigger,
        started_at=started.isoformat(),
        completed_at=completed.replace(tzinfo=datetime.UTC).isoformat()
        if completed is not None
        else None,
        input_hash=run.input_hash,
        output_hash=run.output_hash,
        emitted_count=run.emitted_count,
        skipped_count=run.skipped_count,
        details=details,
        error=run.error,
        duration_seconds=duration,
    )


async def audit_view(
    session: AsyncSession, *, limit: int = 20
) -> ExtensionAuditResponse:
    """Recent ExtensionRun rows for the Paisa extension + last success/error.

    Generic over extension_id in the table, but the surface is Paisa-owned (the
    only builtin today). ``limit`` is bounded by ``recent_runs``.
    """
    runs = await recent_runs(session, extension_id="paisa", limit=limit)
    infos = [_run_to_info(r) for r in runs]
    last_success = next((i for i in infos if i.status == STATUS_SUCCESS), None)
    last_error = next((i for i in infos if i.status == STATUS_FAILURE), None)
    return ExtensionAuditResponse(
        runs=infos, last_success=last_success, last_error=last_error
    )


# ---------------------------------------------------------------------------
# Curated reports + reconciliation dispatch
# ---------------------------------------------------------------------------


def _ttl_seconds() -> int:
    return max(0, get_setting_int("paisa.report_cache_ttl_seconds", 60))


_DTO_REPORT_KINDS = DTO_REPORT_KINDS


def _report_cache_key(config: PaisaProjectionConfig, report: str) -> ReportCacheKey:
    """Compatibility wrapper for the former surface-private helper."""
    return report_cache_key(config, report)


def _report_to_dto(report: str, payload: Any) -> PaisaReportSummary:
    """Compatibility wrapper for the former surface-private DTO adapter."""
    return report_to_dto(report, payload)


async def _fetch_report(config: PaisaProjectionConfig, report: str):
    """Compatibility seam for tests and callers that patch report fetching."""
    return await _fetch_report_impl(config, report)


async def report_summary(app_state: Any, report: str) -> PaisaReportSummary:
    """Fetch one curated report through the injected application service."""
    service = PaisaReportService(
        get_report_cache(app_state),
        fetch_report_fn=_fetch_report,
        reconciliation_builder=build_reconciliation,
    )
    return await service.report_summary(
        load_config(), report, ttl_seconds=_ttl_seconds()
    )


async def reconciliation_view(
    session: AsyncSession, app_state: Any
) -> PaisaReconcileResponse:
    """Build the read-only reconciliation view through the report service."""
    service = PaisaReportService(
        get_report_cache(app_state),
        fetch_report_fn=_fetch_report,
        reconciliation_builder=build_reconciliation,
    )
    return await service.reconciliation_view(
        session,
        load_config(),
        ttl_seconds=_ttl_seconds(),
    )


__all__ = [
    "MANUAL_HEARTBEAT_INTERVAL_SECONDS",
    "MANUAL_LEASE_TTL_SECONDS",
    "PaisaSaveError",
    "SUPPORTED_NON_INR_POLICY",
    "VALID_MODES",
    "account_choices",
    "audit_view",
    "config_view",
    "extension_info",
    "generate_now",
    "generate_now_audited",
    "preview_projection",
    "probe_status",
    "probe_status_audited",
    "reconciliation_view",
    "report_summary",
    "safe_link",
    "save_config",
    "sync_now",
    "sync_now_audited",
]
