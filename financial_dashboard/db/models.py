"""SQLAlchemy ORM models."""

import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Time,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from financial_dashboard.db.enums import PaymentStatus


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class EmailSource(Base):
    __tablename__ = "email_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    account_identifier: Mapped[str | None] = mapped_column(String)
    credentials: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool | None] = mapped_column(Boolean, default=True)
    sync_cursor: Mapped[str | None] = mapped_column(String)
    last_synced_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    last_error: Mapped[str | None] = mapped_column(String)
    cas_last_polled_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bank: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    account_number: Mapped[str | None] = mapped_column(String)
    statement_password: Mapped[str | None] = mapped_column(String)
    statement_password_hint: Mapped[str | None] = mapped_column(String)
    active: Mapped[bool | None] = mapped_column(Boolean, default=True)

    cards: Mapped[list["Card"]] = relationship(
        lazy="selectin", order_by="Card.is_primary.desc(), Card.id"
    )


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=False
    )
    card_mask: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str | None] = mapped_column(String)
    is_primary: Mapped[bool | None] = mapped_column(Boolean, default=False)
    active: Mapped[bool | None] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("account_id", "card_mask", name="uq_card_account_mask"),
        Index("ix_cards_card_mask", "card_mask"),
    )


class FetchRule(Base):
    __tablename__ = "fetch_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("email_sources.id"), nullable=True
    )
    sender: Mapped[str | None] = mapped_column(String)
    subject: Mapped[str | None] = mapped_column(String)
    bank: Mapped[str] = mapped_column(String, nullable=False)
    folder: Mapped[str | None] = mapped_column(String)
    email_kind: Mapped[str | None] = mapped_column(String)
    enabled: Mapped[bool | None] = mapped_column(Boolean, default=True)
    initial_backfill_done_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    auto_managed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )

    source: Mapped["EmailSource | None"] = relationship(lazy="joined")

    __table_args__ = (
        Index(
            "uq_fetch_rule_auto_managed",
            "source_id",
            "sender",
            unique=True,
            sqlite_where=text("auto_managed = 1"),
        ),
    )


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    message_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    source_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("email_sources.id"), nullable=True
    )
    remote_id: Mapped[str | None] = mapped_column(String)
    sender: Mapped[str | None] = mapped_column(String)
    subject: Mapped[str | None] = mapped_column(String)
    received_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    fetched_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )
    status: Mapped[str | None] = mapped_column(String, default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    rule_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("fetch_rules.id"))

    __table_args__ = (
        Index("ix_emails_fetched_at", "fetched_at"),
        UniqueConstraint("source_id", "remote_id", name="uq_email_source_remote"),
    )


class StatementUpload(Base):
    __tablename__ = "statement_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=False, index=True
    )
    email_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("emails.id"), nullable=True, index=True
    )
    bank: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    source_kind: Mapped[str] = mapped_column(
        String, nullable=False, server_default="pdf", default="pdf"
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="parsed")
    card_number: Mapped[str | None] = mapped_column(String)
    statement_name: Mapped[str | None] = mapped_column(String)
    due_date: Mapped[str | None] = mapped_column(String)
    total_amount_due: Mapped[str | None] = mapped_column(String)
    minimum_amount_due: Mapped[str | None] = mapped_column(String)
    parsed_txn_count: Mapped[int | None] = mapped_column(Integer, default=0)
    matched_count: Mapped[int | None] = mapped_column(Integer, default=0)
    missing_count: Mapped[int | None] = mapped_column(Integer, default=0)
    imported_count: Mapped[int | None] = mapped_column(Integer, default=0)
    reconciliation_data: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )
    payment_status: Mapped[PaymentStatus | None] = mapped_column(String)
    payment_sent_offsets: Mapped[str | None] = mapped_column(Text, default="[]")
    payment_last_reminded_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    payment_paid_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    payment_paid_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=12, scale=2), default=0
    )

    account: Mapped["Account"] = relationship(lazy="joined")


class BankStatementUpload(Base):
    __tablename__ = "bank_statement_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=False, index=True
    )
    email_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("emails.id"), nullable=True, index=True
    )
    bank: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="parsed")
    account_number: Mapped[str | None] = mapped_column(String)
    account_holder_name: Mapped[str | None] = mapped_column(String)
    opening_balance: Mapped[str | None] = mapped_column(String)
    closing_balance: Mapped[str | None] = mapped_column(String)
    statement_period_start: Mapped[str | None] = mapped_column(String)
    statement_period_end: Mapped[str | None] = mapped_column(String)
    parsed_txn_count: Mapped[int | None] = mapped_column(Integer, default=0)
    matched_count: Mapped[int | None] = mapped_column(Integer, default=0)
    missing_count: Mapped[int | None] = mapped_column(Integer, default=0)
    imported_count: Mapped[int | None] = mapped_column(Integer, default=0)
    reconciliation_data: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )

    account: Mapped["Account"] = relationship(lazy="joined")


class CasUpload(Base):
    __tablename__ = "cas_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("emails.id"), nullable=True, index=True
    )
    portfolio_key: Mapped[str] = mapped_column(String, nullable=False)
    depository_source: Mapped[str] = mapped_column(String, nullable=False)
    investor_name: Mapped[str | None] = mapped_column(String, nullable=True)
    statement_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    grand_total: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    portfolio_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    portfolio_delta: Mapped[Decimal | None] = mapped_column(
        Numeric(16, 2), nullable=True
    )
    raw_holdings_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)

    __table_args__ = (
        Index("uq_cas_portfolio_date", "portfolio_key", "statement_date", unique=True),
    )


class ManualItem(Base):
    __tablename__ = "manual_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)


class BalanceSnapshot(Base):
    __tablename__ = "balance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=True
    )
    cas_upload_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cas_uploads.id"), nullable=True
    )
    manual_item_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("manual_items.id"), nullable=True
    )
    portfolio_key: Mapped[str | None] = mapped_column(String, nullable=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    as_of_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="INR", nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)

    account: Mapped["Account | None"] = relationship(lazy="joined")
    cas_upload: Mapped["CasUpload | None"] = relationship(lazy="joined")
    manual_item: Mapped["ManualItem | None"] = relationship(lazy="joined")
    # ORM-level cascade only — protects future `session.delete(snapshot)`
    # callers from orphaning holdings. DB-level ON DELETE CASCADE would be
    # a no-op on this SQLite deployment since PRAGMA foreign_keys is not
    # enabled. The current cas_ingestion delete path uses Core delete() and
    # bypasses this cascade, so its manual delete chain (holdings →
    # snapshots → upload) stays load-bearing.
    holdings: Mapped[list["SnapshotHolding"]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "(account_id IS NOT NULL) + (cas_upload_id IS NOT NULL) + "
            "(manual_item_id IS NOT NULL) = 1",
            name="ck_balance_snapshot_exactly_one_source",
        ),
        Index(
            "uq_snap_account",
            "account_id",
            "category",
            "as_of_date",
            unique=True,
            sqlite_where=text("account_id IS NOT NULL"),
        ),
        Index(
            "uq_snap_investment",
            "portfolio_key",
            "category",
            "as_of_date",
            unique=True,
            sqlite_where=text("portfolio_key IS NOT NULL"),
        ),
        Index(
            "uq_snap_manual",
            "manual_item_id",
            "as_of_date",
            unique=True,
            sqlite_where=text("manual_item_id IS NOT NULL"),
        ),
    )


class SnapshotHolding(Base):
    __tablename__ = "snapshot_holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("balance_snapshots.id"), nullable=False
    )
    asset_class: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)

    # Optional investment detail a CAS payload carries for a single priced
    # instrument. The current CAS ingestion still aggregates holdings by asset
    # class for the net-worth breakdown (so these stay NULL on those rows); the
    # fields exist so an instrument-level holding can record the facts the CAS
    # explicitly states without fabricating any. None of them is ever derived
    # from ``value`` — ``acquired_on``/``cost_basis`` are only set when the
    # source states them, and a complete cost-basis lot is normalized into a
    # separate :class:`InvestmentLot` row instead.
    instrument_id: Mapped[str | None] = mapped_column(String, nullable=True)
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    unit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    cost_basis: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    acquired_on: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)

    snapshot: Mapped["BalanceSnapshot"] = relationship(back_populates="holdings")


class InvestmentLot(Base):
    """A complete, capital-gains-eligible investment lot normalized from an
    explicit CAS acquisition fact.

    A row exists ONLY when the source states every field a lot needs to be
    projected truthfully — an instrument id, a quantity, a per-unit cost, a
    cost basis, a currency and an acquisition date — and they are mutually
    consistent (``quantity * unit_cost`` agrees with ``cost_basis`` to the
    penny). Today the only CAS source that satisfies this is a mutual-fund
    purchase/switch_in transaction whose ``units`` + ``nav`` + ``amount`` +
    ``date`` + ``isin`` are all present. Value-only holdings and demat
    movements (CAS carries no cost for them) are excluded and reported as
    diagnostics by :mod:`financial_dashboard.services.investments`; they are
    never fabricated into a lot here. Acquisition dates and cost basis are
    never derived from a current market value.

    Re-ingestion of the same CAS period deletes the prior upload (and its lots)
    first. ``source_occurrence`` preserves source multiplicity when two
    otherwise identical acquisition facts occur in one statement; the unique
    key prevents retrying normalization from inserting that occurrence twice.
    Cross-upload overlap is intentionally retained here as source provenance
    and canonicalized by :mod:`financial_dashboard.services.investments` at
    read time.
    """

    __tablename__ = "investment_lots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cas_upload_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cas_uploads.id"), nullable=False
    )
    instrument_id: Mapped[str] = mapped_column(String, nullable=False)
    instrument_name: Mapped[str] = mapped_column(String, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    unit_cost: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    cost_basis: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    acquired_on: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    source_ref: Mapped[str] = mapped_column(String, nullable=False)
    transaction_type: Mapped[str | None] = mapped_column(String, nullable=True)
    reference: Mapped[str | None] = mapped_column(String, nullable=True)
    source_occurrence: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)

    __table_args__ = (
        UniqueConstraint(
            "cas_upload_id",
            "source_ref",
            "instrument_id",
            "acquired_on",
            "reference",
            "source_occurrence",
            name="uq_investment_lot_natural",
        ),
        Index("ix_investment_lots_upload", "cas_upload_id"),
        Index("ix_investment_lots_instrument", "instrument_id"),
    )


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now
    )


class ExtensionSyncState(Base):
    """Per-extension persistent sync-state row (generic; Paisa today).

    A singleton keyed by ``extension_id`` that records where an extension's
    reconciled state is relative to the core data it derives from. SQLite
    triggers installed by :mod:`financial_dashboard.db.init_db` atomically bump
    ``desired_revision`` and maintain ``first_dirty_at``/``last_dirty_at``
    whenever any of the core tables the Paisa projection reads changes, so a
    coordinator can detect drift cheaply (``desired_revision >
    applied_revision``) without re-deriving anything. The coordinator owns the
    remaining fields — it writes ``applied_revision`` plus the hash / retry /
    diagnosis / lease fields as it reconciles.

    No foreign keys to extension manifests: ``extension_id`` is a plain
    string (today the literal ``"paisa"``), so this row is decoupled from the
    in-memory manifest registry and survives a manifest being renamed or
    removed without orphaning a FK. ``extension_runs`` is the per-operation
    audit log; this row is the per-extension *current* state — exactly one
    row per extension, never growing with operation count.

    desired_revision       monotonically incremented by triggers; the tip
                           revision of the core data
    applied_revision       last revision the coordinator successfully pushed
                           to / verified against the remote
    first_dirty_at         when the current dirty window opened (cleared by
                           the coordinator on a successful reconcile); NULL
                           while the row is in sync
    last_dirty_at          last time a dirtying change landed
    last_published_hash    hash of the last body this dashboard generated
                           and wrote to its owned include file
    last_remote_hash       hash the remote Paisa reported the last time we
                           asked it what it currently has loaded
    last_healthy_hash      hash at which the remote last passed a health/
                           diagnosis check (the last known-good)
    last_remote_attempt_at when the coordinator last tried to talk to the
                           remote (success or failure)
    next_attempt_at        earliest next attempt, raised on failure to
                           implement exponential backoff
    failure_count          consecutive remote failures since the last success
    diagnosis_state        short machine token summarizing the last diagnosis
                           outcome (``healthy`` | ``accepted`` | ``fatal`` …);
                           free-form so the diagnosis layer can extend it
    force_reload           one-shot forcing flag — set by migration on first
                           boot so the next enabled coordinator reconciles
                           even with no observable drift; cleared by the
                           coordinator after a successful reconcile
    lease_owner/lease_token/lease_expires_at
                           distributed-coordination lease for the singleton
                           (single coordinator owns it while unexpired)
    created_at/updated_at  row bookkeeping
    """

    __tablename__ = "extension_sync_state"

    extension_id: Mapped[str] = mapped_column(String, primary_key=True)
    desired_revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    applied_revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    first_dirty_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    last_dirty_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    last_published_hash: Mapped[str | None] = mapped_column(String)
    last_remote_hash: Mapped[str | None] = mapped_column(String)
    last_healthy_hash: Mapped[str | None] = mapped_column(String)
    last_remote_attempt_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    next_attempt_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    failure_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    diagnosis_state: Mapped[str | None] = mapped_column(String)
    force_reload: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )
    lease_owner: Mapped[str | None] = mapped_column(String)
    lease_token: Mapped[str | None] = mapped_column(String)
    lease_expires_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now
    )

    __table_args__ = (
        CheckConstraint(
            "desired_revision >= 0", name="ck_extension_sync_state_desired_nonneg"
        ),
        CheckConstraint(
            "applied_revision >= 0", name="ck_extension_sync_state_applied_nonneg"
        ),
        CheckConstraint(
            "failure_count >= 0", name="ck_extension_sync_state_failure_nonneg"
        ),
        CheckConstraint(
            "desired_revision >= applied_revision",
            name="ck_extension_sync_state_desired_gte_applied",
        ),
        # Both support the coordinator's hot lookups: "which extensions are
        # due for retry" and "which leased rows have expired". Single-row
        # cardinality today (one Paisa), but the indexes keep those queries
        # seek-shaped as more extensions arrive.
        Index("ix_extension_sync_state_next_attempt", "next_attempt_at"),
        Index("ix_extension_sync_state_lease_expires", "lease_expires_at"),
    )


class ExtensionRun(Base):
    """Audit/state row for a single extension operation (generic, not Paisa-specific).

    One row per attempted extension operation (a probe, a generate, an automatic
    or manual sync). It is the single source of truth for "what did the extension
    do, when, and how did it end" — it stores NO credentials and never duplicates
    financial rows; ``details``/``error`` carry sanitized summary text only.

    operation   ``manual`` | ``automatic`` | ``probe`` | ``generate`` | ``sync``
    status      ``running`` | ``success`` | ``failure`` | ``skipped``
    outcome     free-form machine token (``synced`` | ``skipped_unchanged`` |
                ``disabled`` | ``connect_only`` | ``not_configured`` | …) mirroring
                the orchestrator's outcomes where applicable
    trigger     what started it (``fetch_cycle`` | ``api`` | ``startup`` | …)
    started/completed_at are tz-aware UTC (see utc_now)
    input_hash  hash of the inputs that produced this run (e.g. the projected
                journal body hash) so two runs over identical inputs are comparable
    output_hash hash of the emitted output (the published file body hash) so an
                operator can verify the on-disk file matches the audit row
    emitted/skipped_count  projection counts when the operation produced them
    details     JSON-encoded text blob of safe, non-secret summary fields
    error       sanitized error text (truncated, no credentials)
    """

    __tablename__ = "extension_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    extension_id: Mapped[str] = mapped_column(String, nullable=False)
    operation: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    outcome: Mapped[str | None] = mapped_column(String)
    trigger: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)
    completed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    input_hash: Mapped[str | None] = mapped_column(String)
    output_hash: Mapped[str | None] = mapped_column(String)
    emitted_count: Mapped[int | None] = mapped_column(Integer)
    skipped_count: Mapped[int | None] = mapped_column(Integer)
    details: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        # The hot query is "most recent run(s) for this extension", often filtered
        # to an operation/status — a compound index on (extension_id, started_at)
        # serves both the recent-list and the debounce lookup, and a per-status
        # index serves a future "show me failures" filter.
        Index("ix_extension_runs_ext_started", "extension_id", "started_at"),
        Index("ix_extension_runs_ext_status", "extension_id", "status"),
        Index("ix_extension_runs_operation", "operation"),
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("emails.id"), index=True
    )
    account_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=True
    )
    card_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cards.id"), nullable=True
    )
    statement_upload_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("statement_uploads.id"), nullable=True, index=True
    )
    bank_statement_upload_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("bank_statement_uploads.id"), nullable=True, index=True
    )

    account: Mapped["Account | None"] = relationship(lazy="joined")
    card: Mapped["Card | None"] = relationship(lazy="joined")

    bank: Mapped[str] = mapped_column(String, nullable=False)
    email_type: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=12, scale=2), nullable=False
    )
    currency: Mapped[str | None] = mapped_column(String, default="INR")
    transaction_date: Mapped[datetime.date | None] = mapped_column(Date)
    transaction_time: Mapped[datetime.time | None] = mapped_column(Time)
    counterparty: Mapped[str | None] = mapped_column(String)
    card_mask: Mapped[str | None] = mapped_column(String)
    account_mask: Mapped[str | None] = mapped_column(String)
    reference_number: Mapped[str | None] = mapped_column(String)
    channel: Mapped[str | None] = mapped_column(String)
    balance: Mapped[Decimal | None] = mapped_column(Numeric(precision=12, scale=2))
    raw_description: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String)
    category_method: Mapped[str | None] = mapped_column(String)
    category_confidence: Mapped[float | None] = mapped_column(Float)
    category_model: Mapped[str | None] = mapped_column(String)
    category_input_hash: Mapped[str | None] = mapped_column(String)
    category_vocab_version: Mapped[int | None] = mapped_column(Integer)
    categorized_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    review_status: Mapped[str | None] = mapped_column(String)
    review_reason: Mapped[str | None] = mapped_column(Text)
    last_notified_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    notify_attempts: Mapped[int | None] = mapped_column(Integer, default=0)
    sms_message_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sms_messages.id"), nullable=True, index=True
    )
    source: Mapped[str | None] = mapped_column(String)
    notified_channel: Mapped[str | None] = mapped_column(String)
    enriched_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )

    __table_args__ = (
        Index("ix_transactions_transaction_date", "transaction_date"),
        Index("ix_transactions_bank", "bank"),
        # Category-first, date-second. A drill-through that carries dates was
        # already served by the date index alone; the case this exists for is a
        # bare ?category= with no bounds, whose row query scanned the whole table
        # through the date index and whose paging count() scanned it outright. The
        # date column rides along so a dated category filter uses both terms of one
        # index instead of narrowing on dates and re-checking the category per row.
        Index("ix_transactions_category_date", "category", "transaction_date"),
        Index(
            "ix_transactions_reference_number",
            "reference_number",
            sqlite_where=text("reference_number IS NOT NULL"),
        ),
        Index(
            "uq_transactions_ref",
            "bank",
            "reference_number",
            "direction",
            unique=True,
            sqlite_where=text("reference_number IS NOT NULL"),
        ),
    )


class SmsMessage(Base):
    __tablename__ = "sms_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bank: Mapped[str] = mapped_column(String, nullable=False)
    sender: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=utc_now
    )
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending", server_default="pending"
    )
    transaction_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("transactions.id"), nullable=True, index=True
    )
    parse_error: Mapped[str | None] = mapped_column(Text)
    parsed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)

    __table_args__ = (
        UniqueConstraint(
            "sender", "received_at", "body", name="uq_sms_sender_received_body"
        ),
    )


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )


class MerchantRule(Base):
    __tablename__ = "merchant_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pattern: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )
