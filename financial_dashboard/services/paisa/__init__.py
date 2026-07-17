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
    PreviewReport,
    ProbeReport,
    SyncOutcome,
    SyncReport,
    generate,
    manual_sync,
    preview,
    probe,
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

__all__ = [
    "DEFAULT_BACKEND",
    "FxDecision",
    "FxRate",
    "GenerateResult",
    "LedgerAccount",
    "LedgerDocument",
    "LedgerPosting",
    "NonInrPolicy",
    "OpeningBalance",
    "PriceDirective",
    "ProjectedEntry",
    "ProjectionError",
    "ProjectionReport",
    "PublishResult",
    "PaisaProjectionConfig",
    "PreviewReport",
    "ProbeReport",
    "SUPPORTED_BACKENDS",
    "SkippedRow",
    "SyncOutcome",
    "SyncReport",
    "generate",
    "load_config",
    "manual_sync",
    "project",
    "preview",
    "probe",
    "publish_journal",
    "render_document",
    "render_document_for_backend",
    "validate_backend",
]
