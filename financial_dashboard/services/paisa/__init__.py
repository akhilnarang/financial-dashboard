"""Paisa integration: project the dashboard's ledger onto Paisa over a loopback
API, atomically publish a generated journal include, and sync on demand.

The package is layered so each concern is independently testable:

* :mod:`financial_dashboard.services.paisa.config` — reads the ``paisa.*``
  settings via the existing settings accessors and returns a typed
  :class:`PaisaProjectionConfig`.
* :mod:`financial_dashboard.services.paisa.renderer` — pure typed Ledger-CLI
  renderer; no DB, no I/O.
* :mod:`financial_dashboard.services.paisa.projection` — read-only projection
  over the ORM, scoped to selected accounts, producing a stable journal plus a
  report of what was emitted / skipped / unmatched.
* :mod:`financial_dashboard.services.paisa.publisher` — atomic publisher for
  the single generated include file.
* :mod:`financial_dashboard.services.paisa.orchestrator` — probe / preview /
  generate / manual-sync wiring that never mutates core rows and never manages
  Paisa.
"""

from financial_dashboard.services.paisa.config import (
    FxRate,
    PaisaProjectionConfig,
    NonInrPolicy,
    load_config,
)
from financial_dashboard.services.paisa.orchestrator import (
    GenerateResult,
    PreflightReport,
    PreviewReport,
    ProbeReport,
    RemoteSyncReport,
    SyncOutcome,
    SyncReport,
    generate,
    manual_sync,
    preflight,
    preview,
    probe,
    sync_remote,
)
from financial_dashboard.services.paisa.projection import (
    FxDecision,
    OpeningBalance,
    ProjectedEntry,
    ProjectionError,
    ProjectionReport,
    SkippedRow,
    project,
)
from financial_dashboard.services.paisa.publisher import (
    PublishResult,
    publish_journal,
)
from financial_dashboard.services.paisa.renderer import (
    LedgerAccount,
    LedgerDocument,
    LedgerPosting,
    PriceDirective,
    render_document,
    render_document_for_backend,
)
from financial_dashboard.services.paisa.renderers import (
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    validate_backend,
)
from financial_dashboard.services.paisa.sync_state import (
    BACKOFF_MINUTES,
    DIAGNOSIS_FATAL,
    DIAGNOSIS_HEALTHY,
    DIAGNOSIS_UNKNOWN,
    EXTENSION_PAISA,
    LeaseStaleError,
    SyncStateSnapshot,
    backoff_delta,
    capture_target,
    claim_lease,
    ensure_sync_state,
    heartbeat_lease,
    read_sync_state,
    record_accepted_post,
    record_diagnosis,
    record_hash_noop,
    record_pre_post_failure,
    record_published_hash,
    reconcile_eligibility,
    release_lease,
    reset_sync_state,
    set_force_reload,
    should_skip_remote_post,
    to_utc,
)

__all__ = [
    "BACKOFF_MINUTES",
    "DEFAULT_BACKEND",
    "DIAGNOSIS_FATAL",
    "DIAGNOSIS_HEALTHY",
    "DIAGNOSIS_UNKNOWN",
    "EXTENSION_PAISA",
    "FxDecision",
    "FxRate",
    "GenerateResult",
    "LedgerAccount",
    "LedgerDocument",
    "LedgerPosting",
    "LeaseStaleError",
    "NonInrPolicy",
    "OpeningBalance",
    "PriceDirective",
    "ProjectedEntry",
    "ProjectionError",
    "ProjectionReport",
    "PublishResult",
    "PaisaProjectionConfig",
    "PreflightReport",
    "PreviewReport",
    "ProbeReport",
    "RemoteSyncReport",
    "SUPPORTED_BACKENDS",
    "SkippedRow",
    "SyncOutcome",
    "SyncReport",
    "SyncStateSnapshot",
    "backoff_delta",
    "capture_target",
    "claim_lease",
    "ensure_sync_state",
    "generate",
    "heartbeat_lease",
    "load_config",
    "manual_sync",
    "project",
    "preview",
    "preflight",
    "probe",
    "publish_journal",
    "read_sync_state",
    "record_accepted_post",
    "record_diagnosis",
    "record_hash_noop",
    "record_pre_post_failure",
    "record_published_hash",
    "reconcile_eligibility",
    "release_lease",
    "render_document",
    "render_document_for_backend",
    "reset_sync_state",
    "set_force_reload",
    "should_skip_remote_post",
    "sync_remote",
    "to_utc",
    "validate_backend",
]
