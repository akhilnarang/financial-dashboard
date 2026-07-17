"""Pydantic DTOs for the extension framework and the Paisa extension surface.

These models are the API/web boundary for everything under ``/api/extensions``
and ``/extensions``. They are deliberately dashboard-owned and decoupled from
the core ``services.paisa.*`` types (which are NamedTuples): the routes never
hand a raw projection/config object to FastAPI, and a core type change cannot
silently reshape a JSON response. The adapter that turns the core types into
these DTOs lives in ``services.paisa.surface``.

The stored Paisa password is never serialized here — ``PaisaConfig`` carries
only ``auth_password_set`` so an operator can see whether a secret is held
without the secret itself ever leaving the server.
"""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Extension list
# ---------------------------------------------------------------------------


class ExtensionInfo(BaseModel):
    id: str
    display_name: str
    description: str
    capabilities: list[str]


class ExtensionListResponse(BaseModel):
    extensions: list[ExtensionInfo]


# ---------------------------------------------------------------------------
# Paisa config (redacted, read)
# ---------------------------------------------------------------------------


class PaisaFxRateRow(BaseModel):
    """One historical FX quote as the UI/API edits it: rate is INR per unit,
    positive. Serializes to/from the nested ``{ccy: [{date, rate}]}`` JSON the
    projection config consumes."""

    currency: str
    date: str
    rate: str


class PaisaConfig(BaseModel):
    """Redacted, typed view of the ``paisa.*`` settings.

    ``auth_password`` is intentionally absent: only ``auth_password_set`` is
    surfaced, so the stored secret can never leave the server through this API.
    ``fx_rates`` is surfaced as a flat list of {currency, date, rate} rows
    (sorted) so the API and the HTML editor render from one shape; the stored
    nested-JSON shape is reconstructed on save.
    """

    mode: str
    base_url: str
    external_url: str
    allow_remote: bool
    auth_username: str
    auth_password_set: bool
    generated_path: str
    selected_account_ids: list[int]
    project_since: str
    account_mappings: dict[str, str]
    category_mappings: dict[str, str]
    non_inr_policy: str
    request_timeout_seconds: int
    ledger_cli: str = "ledger"
    fx_rates: list[PaisaFxRateRow] = []
    report_cache_ttl_seconds: int = 60
    auto_sync_enabled: bool = False
    auto_sync_min_interval_minutes: int = 30
    notify_sync_failures: bool = False
    project_investments: bool = False
    can_connect: bool
    can_project: bool


# ---------------------------------------------------------------------------
# Paisa account choices (native dashboard accounts)
# ---------------------------------------------------------------------------


class PaisaAccountChoice(BaseModel):
    id: int
    bank: str
    label: str
    type: str
    selected: bool


class PaisaAccountChoicesResponse(BaseModel):
    accounts: list[PaisaAccountChoice]


# ---------------------------------------------------------------------------
# Paisa config save (write)
# ---------------------------------------------------------------------------


class PaisaConfigInput(BaseModel):
    """Submission model for a config save.

    A blank ``auth_password`` means "preserve the current secret"; a non-blank
    value replaces it. Validation (mode, URLs, cutover, path, mappings, ledger
    names, account existence, backend, FX rates) happens in
    ``services.paisa.surface.save_config`` so the API and the HTML form share
    one validator.
    """

    mode: str
    base_url: str = ""
    external_url: str = ""
    allow_remote: bool = False
    auth_username: str = ""
    auth_password: str = ""
    generated_path: str = ""
    selected_account_ids: list[int] = []
    project_since: str = ""
    account_mappings: dict[str, str] = {}
    category_mappings: dict[str, str] = {}
    non_inr_policy: str = "skip"
    request_timeout_seconds: int = 15
    ledger_cli: str = "ledger"
    fx_rates: list[PaisaFxRateRow] = []
    report_cache_ttl_seconds: int = 60
    auto_sync_enabled: bool = False
    auto_sync_min_interval_minutes: int = 30
    notify_sync_failures: bool = False
    project_investments: bool = False


class PaisaConfigSaveResponse(BaseModel):
    ok: bool
    errors: list[str] = []
    config: PaisaConfig | None = None


# ---------------------------------------------------------------------------
# Paisa status / probe
# ---------------------------------------------------------------------------


class PaisaCapabilitiesInfo(BaseModel):
    ledger_cli: str | None
    readonly: bool
    default_currency: str | None


class PaisaDiagnosisIssueInfo(BaseModel):
    level: str
    summary: str
    details: str


class PaisaDiagnosisInfo(BaseModel):
    ok: bool
    danger_count: int
    warning_count: int
    issues: list[PaisaDiagnosisIssueInfo]
    first_message: str | None


class PaisaStatusResponse(BaseModel):
    ok: bool
    reachable: bool
    mode: str
    can_connect: bool
    can_project: bool
    capabilities: PaisaCapabilitiesInfo | None = None
    diagnosis: PaisaDiagnosisInfo | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# Paisa preview / generate / sync
# ---------------------------------------------------------------------------


class PaisaSkippedRowInfo(BaseModel):
    txn_id: int | None
    reason: str
    detail: str


class PaisaProjectionSummary(BaseModel):
    emitted_count: int
    self_transfer_pairs: int
    card_payments: int
    card_side_payments: int
    non_inr_count: int
    unmatched_count: int
    unknown_count: int
    cutover_date: str | None
    account_ids: list[int]
    skipped: list[PaisaSkippedRowInfo]
    #: Computed diagnostics surfaced from the core ProjectionReport. All
    #: default so an older client (and the test fixtures that build a partial
    #: summary) keep deserializing — they are additive fields, never a reshape
    #: of an existing one.
    imprecise_count: int = 0
    card_payments_resolved: int = 0
    card_payments_unresolved: int = 0
    investment_lot_count: int = 0
    investment_funding_remapped: int = 0
    investment_funding_unresolved: list[str] = []
    kind_counts: dict[str, int] = {}
    projected_foreign_count: int = 0
    missing_fx_rate_count: int = 0
    source_currencies: list[str] = []


class PaisaPreviewResponse(BaseModel):
    ok: bool
    mode: str
    journal: str | None = None
    summary: PaisaProjectionSummary | None = None
    reason: str | None = None


class PaisaPublishInfo(BaseModel):
    published: bool
    skipped: bool
    path: str
    version: str
    body_hash: str
    bytes_written: int


class PaisaGenerateResponse(BaseModel):
    ok: bool
    mode: str
    summary: PaisaProjectionSummary | None = None
    publish: PaisaPublishInfo | None = None
    reason: str | None = None


class PaisaSyncResponse(BaseModel):
    ok: bool
    mode: str
    outcome: str
    summary: PaisaProjectionSummary | None = None
    publish: PaisaPublishInfo | None = None
    diagnosis_ok: bool | None = None
    reason: str | None = None
    #: Classified diagnosis counts (see services.paisa.diagnosis).
    #: ``diagnosis_expected`` is how many ``Debit Entry`` dangers the projection
    #: should produce (negative contra-expense postings); ``diagnosis_accepted``
    #: how many upstream dangers matched one and were downgraded; ``diagnosis_fatal``
    #: how many remain fatal (unmatched Debit Entry, Negative Balance, …). Sync
    #: fails iff ``diagnosis_fatal`` > 0. ``None`` when diagnosis did not run.
    #: All default so older clients keep deserializing.
    diagnosis_expected: int | None = None
    diagnosis_accepted: int | None = None
    diagnosis_fatal: int | None = None


# ---------------------------------------------------------------------------
# Typed failure envelope for optional-extension isolation
# ---------------------------------------------------------------------------


class ExtensionErrorResponse(BaseModel):
    """Returned when an optional extension route fails unexpectedly, so a core
    route is never affected and the caller always gets a typed JSON body."""

    ok: bool = False
    error: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Extension audit
# ---------------------------------------------------------------------------


class ExtensionRunInfo(BaseModel):
    """One ExtensionRun row as the audit UI/API surfaces it. No secrets and no
    raw journal text — only sanitized summary fields (see services.paisa.audit).
    """

    id: int
    extension_id: str
    operation: str
    status: str
    outcome: str | None = None
    trigger: str | None = None
    started_at: str
    completed_at: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    emitted_count: int | None = None
    skipped_count: int | None = None
    details: dict | None = None
    error: str | None = None
    duration_seconds: float | None = None


class ExtensionAuditResponse(BaseModel):
    runs: list[ExtensionRunInfo]
    last_success: ExtensionRunInfo | None = None
    last_error: ExtensionRunInfo | None = None


# ---------------------------------------------------------------------------
# Paisa curated reports (typed normalization of upstream v0.7.4 responses)
# ---------------------------------------------------------------------------


class PaisaMoneyLine(BaseModel):
    """An {account/label → amount} entry from a Paisa report section."""

    account: str
    amount: str


class PaisaBudgetMonth(BaseModel):
    """Normalized /api/budget entry for one month."""

    month: str
    forecast: str
    actual: str
    available_this_month: str
    end_of_month_balance: str


class PaisaBudgetReport(BaseModel):
    months: list[PaisaBudgetMonth]
    checking_balance: str
    available_for_budgeting: str


class PaisaAllocationTarget(BaseModel):
    name: str
    target_percent: str
    current_percent: str


class PaisaAllocationReport(BaseModel):
    """Normalized /api/allocation. ``targets`` is the curated summary; the full
    aggregate timeline is intentionally not proxied."""

    targets: list[PaisaAllocationTarget]
    aggregate_accounts: list[PaisaMoneyLine]


class PaisaRecurringSequence(BaseModel):
    """Normalized /api/recurring transaction_sequences entry."""

    key: str
    period: str | None = None
    interval_days: int | None = None
    count: int


class PaisaRecurringReport(BaseModel):
    sequences: list[PaisaRecurringSequence]


class PaisaIncomeStatementPeriod(BaseModel):
    """Normalized /api/income_statement yearly entry. Each section is a list of
    {account, amount} rows (amounts are kept as exact strings)."""

    period: str
    starting_balance: str
    ending_balance: str
    income: list[PaisaMoneyLine]
    interest: list[PaisaMoneyLine]
    expenses: list[PaisaMoneyLine]
    tax: list[PaisaMoneyLine]
    pnl: list[PaisaMoneyLine]


class PaisaIncomeStatementReport(BaseModel):
    periods: list[PaisaIncomeStatementPeriod]


class PaisaLiabilityBreakdown(BaseModel):
    group: str
    drawn_amount: str
    repaid_amount: str
    interest_amount: str
    balance_amount: str
    apr: str | None = None


class PaisaLiabilitiesReport(BaseModel):
    """Normalized /api/liabilities/balance."""

    breakdowns: list[PaisaLiabilityBreakdown]


class PaisaReportSummary(BaseModel):
    """Typed envelope returned by every curated report route. ``ok=False`` with a
    reason (never a 500) when the upstream is unreachable/disabled."""

    ok: bool
    report: str  # one of budget|allocation|recurring|income_statement|liabilities
    cached: bool = False
    reason: str | None = None
    budget: PaisaBudgetReport | None = None
    allocation: PaisaAllocationReport | None = None
    recurring: PaisaRecurringReport | None = None
    income_statement: PaisaIncomeStatementReport | None = None
    liabilities: PaisaLiabilitiesReport | None = None


# ---------------------------------------------------------------------------
# Paisa reconciliation
# ---------------------------------------------------------------------------


class PaisaReconcileProjectionDiag(BaseModel):
    """Counts lifted from the local projection report (unknown/unmatched/FX).

    The ``investment_*`` fields surface investment-lot projection diagnostics so
    an operator can see, in the reconciliation view, when lots were suppressed
    for an unresolvable disposal (``disposal_history_unresolved``) — never
    overstating holdings. The ``card_payments_*``/``imprecise_count``/
    ``kind_counts``/``projected_foreign_count``/``source_currencies`` fields
    mirror the projection summary so an operator reads the same diagnostics in
    the reconciliation view as on the projection surface. All the added fields
    default so the model stays backward-compatible with older callers.
    """

    emitted_count: int
    unknown_count: int
    unmatched_count: int
    non_inr_count: int
    missing_fx_rate_count: int
    card_side_payments: int
    skipped_reason_counts: dict[str, int]
    investment_lot_count: int = 0
    investment_excluded: list[str] = []
    investment_disposal_unresolved_count: int = 0
    imprecise_count: int = 0
    card_payments: int = 0
    card_payments_resolved: int = 0
    card_payments_unresolved: int = 0
    investment_funding_remapped: int = 0
    investment_funding_unresolved: list[str] = []
    kind_counts: dict[str, int] = {}
    projected_foreign_count: int = 0
    source_currencies: list[str] = []


class PaisaReconcileAccountRow(BaseModel):
    """Per-account reconciliation row.

    native_balance is the latest dashboard-owned BalanceSnapshot (or None);
    projected_balance is the projection's computed ending balance (or None when
    projection is unavailable); paisa_balance is the curated upstream balance
    joined *only* through an explicit account mapping (None when no mapping or
    no reliable upstream endpoint). No fuzzy matching is ever performed.

    Opening-data diagnostics: ``opening_available``/``opening_source``/
    ``opening_as_of`` describe the pre-cutover opening the projection struck
    (from a snapshot or, failing that, a transaction running balance). When no
    reliable opening exists the projected balance starts from zero and the
    limitation is surfaced in ``note`` — no balance is ever invented.
    """

    account_id: int
    bank: str
    label: str
    type: str
    mapped_to: str | None = None
    native_balance: str | None = None
    native_as_of: str | None = None
    native_stale: bool | None = None
    projected_balance: str | None = None
    projected_available: bool = False
    paisa_balance: str | None = None
    paisa_available: bool = False
    delta: str | None = None
    note: str | None = None
    opening_available: bool = False
    opening_source: str | None = None
    opening_as_of: str | None = None


class PaisaReconcileMappingSuggestion(BaseModel):
    """Preview-only deterministic default mapping. Never written without an
    explicit accept through the normal config-save path."""

    account_id: int
    bank: str
    label: str
    suggested_mapping: str


class PaisaReconcileResponse(BaseModel):
    ok: bool
    mode: str
    can_connect: bool
    can_project: bool
    projection: PaisaReconcileProjectionDiag | None = None
    accounts: list[PaisaReconcileAccountRow] = []
    suggestions: list[PaisaReconcileMappingSuggestion] = []
    upstream_available: bool = False
    reason: str | None = None


# Re-exported for adapters that build DTOs from core NamedTuples.
__all__ = [
    "ExtensionAuditResponse",
    "ExtensionErrorResponse",
    "ExtensionInfo",
    "ExtensionListResponse",
    "ExtensionRunInfo",
    "PaisaAccountChoice",
    "PaisaAccountChoicesResponse",
    "PaisaAllocationReport",
    "PaisaAllocationTarget",
    "PaisaBudgetMonth",
    "PaisaBudgetReport",
    "PaisaCapabilitiesInfo",
    "PaisaConfig",
    "PaisaConfigInput",
    "PaisaConfigSaveResponse",
    "PaisaDiagnosisInfo",
    "PaisaDiagnosisIssueInfo",
    "PaisaFxRateRow",
    "PaisaGenerateResponse",
    "PaisaIncomeStatementPeriod",
    "PaisaIncomeStatementReport",
    "PaisaLiabilitiesReport",
    "PaisaLiabilityBreakdown",
    "PaisaMoneyLine",
    "PaisaPreviewResponse",
    "PaisaProjectionSummary",
    "PaisaPublishInfo",
    "PaisaReconcileAccountRow",
    "PaisaReconcileMappingSuggestion",
    "PaisaReconcileProjectionDiag",
    "PaisaReconcileResponse",
    "PaisaRecurringReport",
    "PaisaRecurringSequence",
    "PaisaReportSummary",
    "PaisaSkippedRowInfo",
    "PaisaStatusResponse",
    "PaisaSyncResponse",
]
