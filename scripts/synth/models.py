"""NamedTuple containers for the synthetic scenario graph.

These are pure data — no ORM, no database. The loader maps them onto the
dashboard's SQLAlchemy models. Multi-field containers use ``typing.NamedTuple``
per the repo convention (positional unpacking + named attribute access).
"""

import datetime
import datetime as _dt
from decimal import Decimal
from typing import NamedTuple


class SynthAccount(NamedTuple):
    stable_id: str
    pk: int
    bank: str
    label: str
    type: str
    account_number: str | None
    statement_password: str | None
    statement_password_hint: str | None
    active: bool


class SynthCard(NamedTuple):
    stable_id: str
    pk: int
    account_pk: int
    card_mask: str
    label: str | None
    is_primary: bool
    active: bool


class SynthCategory(NamedTuple):
    slug: str
    active: bool


class SynthEmailSource(NamedTuple):
    stable_id: str
    pk: int
    provider: str
    label: str
    account_identifier: str | None
    active: bool


class SynthFetchRule(NamedTuple):
    stable_id: str
    pk: int
    provider: str
    source_pk: int | None
    sender: str
    subject: str | None
    bank: str
    email_kind: str | None
    enabled: bool


class SynthEmail(NamedTuple):
    stable_id: str
    pk: int
    provider: str
    message_id: str
    source_pk: int | None
    sender: str
    subject: str
    received_at: _dt.datetime
    status: str
    bank: str


class SynthSms(NamedTuple):
    stable_id: str
    pk: int
    bank: str
    sender: str
    body: str
    received_at: _dt.datetime
    status: str


class SynthTransaction(NamedTuple):
    stable_id: str
    bank: str
    email_type: str
    direction: str
    amount: Decimal
    currency: str
    transaction_date: _dt.date
    transaction_time: _dt.time | None
    counterparty: str | None
    card_mask: str | None
    account_mask: str | None
    reference_number: str | None
    channel: str | None
    balance: Decimal | None
    raw_description: str | None
    category: str | None
    source: str
    account_pk: int | None
    card_pk: int | None
    email_pk: int | None
    sms_pk: int | None
    #: When two records (email + SMS) describe the same event, both carry the
    #: same ``dedup_group`` so a test can confirm the loader merged them.
    dedup_group: str | None
    #: Ledger projection hint: the dashboard account name this row maps to.
    ledger_account: str | None
    #: Ledger projection hint: the income/expense counterpart account.
    ledger_counterpart: str | None
    #: Optional manual-review status stamped on a few rows so the review queue
    #: surface has non-empty states (``reviewed`` / ``flagged``). ``None`` everywhere
    #: else so the bulk of the corpus is uncategorized-by-review.
    review_status: str | None = None
    #: Link to a CC statement upload so the reconciliation counts
    #: (matched/imported) can be non-zero without bypassing the truthful model.
    statement_upload_id: int | None = None
    #: Link to a bank statement upload for the same reason.
    bank_statement_upload_id: int | None = None
    #: Categorization provenance — varied across the corpus so the
    #: category_method axis (manual/rule/llm/pending_llm/synthetic) is exercised.
    #: ``None`` falls back to ``synthetic`` at load time.
    category_method: str | None = None
    #: LLM confidence (0..1) for ``category_method='llm'`` rows; None otherwise.
    category_confidence: float | None = None
    #: Model identifier for ``category_method='llm'`` rows; None otherwise.
    category_model: str | None = None
    #: Free-text review reason for ``review_status='resolved'``/``'notified'``.
    review_reason: str | None = None


class SynthManualItem(NamedTuple):
    stable_id: str
    pk: int
    name: str
    kind: str
    category: str
    active: bool
    notes: str | None
    as_of_date: _dt.date
    value: Decimal


class SynthCasUpload(NamedTuple):
    stable_id: str
    pk: int
    portfolio_key: str
    depository_source: str
    investor_name: str
    statement_date: _dt.date
    grand_total: Decimal
    portfolio_ok: bool
    portfolio_delta: Decimal | None
    holdings: tuple[tuple[str, Decimal], ...]
    raw_payload: dict


class SynthStatementUpload(NamedTuple):
    stable_id: str
    pk: int
    account_pk: int
    email_pk: int | None
    bank: str
    filename: str
    file_path: str
    source_kind: str
    status: str
    card_number: str | None
    statement_name: str | None
    due_date: str | None
    total_amount_due: str | None
    minimum_amount_due: str | None
    payment_status: str | None
    closing_balance: str | None
    statement_period_end: str | None
    #: Reconciliation counts — the number of transactions truthfully linked to
    #: this statement (``statement_upload_id`` / ``bank_statement_upload_id``).
    #: Zero by default; non-zero only where the scenario actually linked rows.
    parsed_txn_count: int = 0
    matched_count: int = 0
    imported_count: int = 0
    #: Optional reconciliation payload produced by the real
    #: :func:`reconcile_statement` / :func:`reconcile_bank_statement` service
    #: against scenario transactions. Carried on the scenario so a test can
    #: assert the production reconcile path was exercised offline (the loader
    #: does not persist this blob — it is a fidelity-bound artifact).
    reconciliation_data: dict | None = None
    #: Statement password hint for password-protected statements (no secret —
    #: the synthetic hint is a literal string).
    password_hint: str | None = None


class SynthFxRate(NamedTuple):
    """One historical FX quote: ``rate`` INR per 1 unit of ``currency``, effective
    as of ``date``. Mirrors ``services.paisa.config.FxRate`` so a projection
    integration test can build a ``PaisaProjectionConfig.fx_rates`` map straight
    from the scenario without re-deriving it. Pure data — no production import."""

    date: _dt.date
    currency: str
    rate: Decimal


class SynthAccountSnapshot(NamedTuple):
    """A realistic opening/current balance snapshot for an active account,
    derived from the scenario's tracked running balance (not a hardcoded patch).

    ``opening`` is the balance the account started the scenario with; ``current``
    is its final tracked balance at ``as_of``. For a credit-card account
    ``current`` is the outstanding (purchases net of payments) and ``kind`` is
    ``liability``; for a bank account it is the available balance and ``kind``
    is ``asset``. The loader turns these into ``BalanceSnapshot`` rows.
    """

    account_pk: int
    kind: str
    opening: Decimal
    current: Decimal
    as_of: _dt.date
    #: Optional free-text note used by coverage detection (e.g. ``complete_lot``).
    note: str | None = None
    #: Snapshot currency; non-INR snapshots are excluded from net worth, so the
    #: scenario can carry one to exercise that exclusion path. Defaults to INR.
    currency: str = "INR"


class SynthOrphanEmail(NamedTuple):
    """A standalone email not tied to any transaction — a pending, failed or
    skipped source row that the review/error surfaces must be able to show.

    Loaded structurally by id (its ``message_id`` is the natural key), so it
    never collides with the transaction-linked emails the fidelity/bulk lanes
    create. ``pk`` is kept clear of the transaction email-pk range (100+)."""

    stable_id: str
    pk: int
    provider: str
    message_id: str
    source_pk: int | None
    sender: str
    subject: str
    received_at: _dt.datetime
    status: str
    bank: str
    error: str | None = None


class Scenario(NamedTuple):
    seed: int
    as_of: _dt.date
    profile: str
    accounts: tuple[SynthAccount, ...]
    cards: tuple[SynthCard, ...]
    categories: tuple[SynthCategory, ...]
    email_sources: tuple[SynthEmailSource, ...]
    fetch_rules: tuple[SynthFetchRule, ...]
    emails: tuple[SynthEmail, ...]
    sms: tuple[SynthSms, ...]
    transactions: tuple[SynthTransaction, ...]
    manual_items: tuple[SynthManualItem, ...]
    cas_uploads: tuple[SynthCasUpload, ...]
    statement_uploads: tuple[SynthStatementUpload, ...]
    #: Historical FX rates the projection's ``priced`` non-INR policy consumes.
    #: NOT a DB table — deliberately excluded from :meth:`counts` so the count
    #: dict stays aligned 1:1 with the loader's per-table DB counts.
    fx_rates: tuple[SynthFxRate, ...] = ()
    #: Expected number of ``investment_lots`` rows the loader will persist from
    #: the CAS uploads (one per complete MF acquisition fact, via the real
    #: ``ingest_cas_payload`` → ``create_investment_lots`` path). Computed by the
    #: scenario builder from the same extraction rule the loader uses, so a
    #: regression that drops or duplicates lots is caught by manifest verify.
    investment_lots: int = 0
    #: Balance-derived opening/current snapshots for active accounts. NOT counted
    #: in :meth:`counts` directly (the loader emits them via the snapshot
    #: service); kept on the scenario so the loader and tests share one source.
    account_snapshots: tuple[SynthAccountSnapshot, ...] = ()
    #: Standalone pending/failed/skipped emails (no transaction). Counted under
    #: the ``emails`` key alongside the transaction-linked emails so manifest
    #: verify tracks the full ``emails`` table population.
    orphan_emails: tuple[SynthOrphanEmail, ...] = ()
    #: Explicit scenario-branch coverage — the set of canonical edge ids this
    #: scenario exercises, derived by
    #: :func:`scripts.synth.coverage.compute_coverage`. Recorded in the manifest
    #: and policed by ``verify_manifest`` so a regression that drops a branch
    #: fails verification. ``None`` until :func:`build_scenario` populates it.
    coverage: frozenset[str] | None = None

    def counts(self) -> dict[str, int]:
        return {
            "accounts": len(self.accounts),
            "cards": len(self.cards),
            "categories": len(self.categories),
            "email_sources": len(self.email_sources),
            "fetch_rules": len(self.fetch_rules),
            "emails": len(self.emails) + len(self.orphan_emails),
            "sms_messages": len(self.sms),
            "transactions": len(self.transactions),
            "manual_items": len(self.manual_items),
            "cas_uploads": len(self.cas_uploads),
            "statement_uploads": len(self.statement_uploads),
            "investment_lots": self.investment_lots,
            # The loader never runs an extension operation, so no ExtensionRun
            # rows exist after a load. Pinning zero lets manifest verify catch a
            # regression that accidentally records extension runs at load time.
            "extension_runs": 0,
        }


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)
