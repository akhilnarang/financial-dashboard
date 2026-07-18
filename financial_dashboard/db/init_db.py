"""Database initialization and inline migrations."""

from sqlalchemy import text

from financial_dashboard.db.models import Base


#: Core tables whose changes dirty an extension's reconciled projection.
#: Every INSERT/UPDATE/DELETE on each bumps ``extension_sync_state.desired_revision``
#: so a coordinator can detect drift without re-deriving the projection.
_EXTENSION_SYNC_TRIGGER_TABLES = (
    "transactions",
    "accounts",
    "cards",
    "balance_snapshots",
    "investment_lots",
    "cas_uploads",
)


def _extension_sync_bump_body(*, reset_backoff: bool) -> str:
    """SQL fragment that bumps desired_revision + manages dirty timestamps.

    ``reset_backoff`` additionally clears the retry fields so a Paisa config
    change is retried immediately rather than waiting out a stale backoff.
    """
    fragment = (
        "UPDATE extension_sync_state "
        "SET desired_revision = desired_revision + 1, "
        "first_dirty_at = COALESCE(first_dirty_at, CURRENT_TIMESTAMP), "
        "last_dirty_at = CURRENT_TIMESTAMP, "
        "updated_at = CURRENT_TIMESTAMP"
    )
    if reset_backoff:
        fragment += (
            ", failure_count = 0, next_attempt_at = NULL, last_remote_attempt_at = NULL"
        )
    return fragment


def _build_extension_sync_state_trigger_ddl() -> list[str]:
    """Return the CREATE TRIGGER statements that keep extension_sync_state dirty.

    Each statement is ``CREATE TRIGGER IF NOT EXISTS`` so re-running ``init_db``
    is safe (re-boot is the common path). Every trigger is an AFTER row trigger,
    so:

    * the bump runs inside the triggering statement's transaction and rolls
      back with it (and with any enclosing SAVEPOINT) — a rolled-back write
      leaves ``desired_revision`` untouched;
    * ``INSERT ... ON CONFLICT DO NOTHING`` / ``INSERT OR IGNORE`` that inserts
      no row does not fire the AFTER trigger, so a conflict-no-op never bumps;
    * the singleton row is updated in-place; if it is missing the UPDATE
      matches zero rows (no failure, no recursion).

    No trigger is installed on ``extension_sync_state`` or ``extension_runs``
    themselves, so a coordinator's direct writes to this row never recurse.
    The settings trigger additionally filters to ``paisa.%`` keys via a
    NEW/OLD reference, so non-Paisa settings writes never dirty the state —
    and a settings change also resets the retry backoff so a config fix is
    retried immediately. SQLite only: on other dialects the table still
    exists (built by ``create_all``) but no triggers are installed.
    """
    statements: list[str] = []
    bump = _extension_sync_bump_body(reset_backoff=False)
    for table in _EXTENSION_SYNC_TRIGGER_TABLES:
        for verb in ("INSERT", "UPDATE", "DELETE"):
            statements.append(
                f"CREATE TRIGGER IF NOT EXISTS ext_sync_dirty_{table}_{verb.lower()} "
                f"AFTER {verb} ON {table} BEGIN {bump} WHERE extension_id = 'paisa'; END"
            )
    bump_settings = _extension_sync_bump_body(reset_backoff=True)
    for verb, ref in (("INSERT", "NEW"), ("UPDATE", "NEW"), ("DELETE", "OLD")):
        statements.append(
            f"CREATE TRIGGER IF NOT EXISTS ext_sync_dirty_settings_{verb.lower()} "
            f"AFTER {verb} ON settings BEGIN {bump_settings} "
            f"WHERE extension_id = 'paisa' AND {ref}.key LIKE 'paisa.%'; END"
        )
    return statements


async def init_db(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.begin() as conn:
        try:
            await conn.execute(text("SELECT note FROM transactions LIMIT 0"))
        except Exception:
            await conn.execute(text("ALTER TABLE transactions ADD COLUMN note TEXT"))
        try:
            await conn.execute(
                text("SELECT statement_upload_id FROM transactions LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE transactions ADD COLUMN statement_upload_id INTEGER REFERENCES statement_uploads(id)"
                )
            )
        try:
            await conn.execute(text("SELECT statement_password FROM accounts LIMIT 0"))
        except Exception:
            await conn.execute(
                text("ALTER TABLE accounts ADD COLUMN statement_password VARCHAR")
            )
        try:
            await conn.execute(
                text("SELECT initial_backfill_done_at FROM fetch_rules LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE fetch_rules ADD COLUMN initial_backfill_done_at DATETIME"
                )
            )
            await conn.execute(
                text(
                    "UPDATE fetch_rules SET initial_backfill_done_at = CURRENT_TIMESTAMP "
                    "WHERE id IN (SELECT DISTINCT rule_id FROM emails WHERE rule_id IS NOT NULL)"
                )
            )
        try:
            await conn.execute(text("SELECT email_id FROM statement_uploads LIMIT 0"))
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE statement_uploads ADD COLUMN email_id INTEGER REFERENCES emails(id)"
                )
            )
        for col in (
            "payment_status",
            "payment_sent_offsets",
            "payment_last_reminded_at",
            "payment_paid_at",
        ):
            try:
                await conn.execute(text(f"SELECT {col} FROM statement_uploads LIMIT 0"))
            except Exception:
                default = " DEFAULT '[]'" if col == "payment_sent_offsets" else ""
                await conn.execute(
                    text(
                        f"ALTER TABLE statement_uploads ADD COLUMN {col} {'TEXT' if 'offsets' in col else 'VARCHAR' if col == 'payment_status' else 'DATETIME'}{default}"
                    )
                )
        try:
            await conn.execute(
                text("SELECT payment_paid_amount FROM statement_uploads LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE statement_uploads ADD COLUMN payment_paid_amount NUMERIC(12,2) DEFAULT 0"
                )
            )
        await conn.execute(
            text(
                "UPDATE statement_uploads SET payment_status = 'unpaid' "
                "WHERE due_date IS NOT NULL AND due_date != '' "
                "AND total_amount_due IS NOT NULL AND total_amount_due != '' "
                "AND payment_status IS NULL "
                "AND created_at >= date('now', 'start of month')"
            )
        )
        try:
            await conn.execute(
                text("SELECT bank_statement_upload_id FROM transactions LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE transactions ADD COLUMN bank_statement_upload_id INTEGER REFERENCES bank_statement_uploads(id)"
                )
            )
        try:
            await conn.execute(text("SELECT email_kind FROM fetch_rules LIMIT 0"))
        except Exception:
            await conn.execute(
                text("ALTER TABLE fetch_rules ADD COLUMN email_kind VARCHAR")
            )
        try:
            await conn.execute(
                text("SELECT statement_password_hint FROM accounts LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text("ALTER TABLE accounts ADD COLUMN statement_password_hint VARCHAR")
            )
        try:
            await conn.execute(text("SELECT category FROM transactions LIMIT 0"))
        except Exception:
            await conn.execute(
                text("ALTER TABLE transactions ADD COLUMN category VARCHAR")
            )
        for _col, _type in (
            ("category_method", "VARCHAR"),
            ("category_confidence", "REAL"),
            ("category_model", "VARCHAR"),
            ("category_input_hash", "VARCHAR"),
            ("category_vocab_version", "INTEGER"),
            ("categorized_at", "DATETIME"),
            ("review_status", "VARCHAR"),
            ("review_reason", "TEXT"),
            ("last_notified_at", "DATETIME"),
            ("notify_attempts", "INTEGER"),
        ):
            try:
                await conn.execute(text(f"SELECT {_col} FROM transactions LIMIT 0"))
            except Exception:
                await conn.execute(
                    text(f"ALTER TABLE transactions ADD COLUMN {_col} {_type}")
                )
        # One-time backfill: rows that already had a hand-entered category before
        # this feature must be treated as user-set (manual), or the rule/LLM
        # sweeps would see category_method IS NULL and OVERWRITE them on first run.
        _method_marker = (
            await conn.execute(
                text(
                    "SELECT 1 FROM settings WHERE key = "
                    "'migrations.category_method_backfill'"
                )
            )
        ).first()
        if not _method_marker:
            await conn.execute(
                text(
                    "UPDATE transactions SET category_method = 'manual' "
                    "WHERE category IS NOT NULL AND category != '' "
                    "AND category_method IS NULL"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) VALUES "
                    "('migrations.category_method_backfill', '1')"
                )
            )
        from financial_dashboard.services.categorization.vocabulary import (
            SEED_CATEGORIES,
        )

        _cat_before = (
            await conn.execute(text("SELECT COUNT(*) FROM categories"))
        ).scalar() or 0
        for _slug in SEED_CATEGORIES:
            await conn.execute(
                text(
                    "INSERT OR IGNORE INTO categories (slug, active) VALUES (:slug, 1)"
                ),
                {"slug": _slug},
            )
        _cat_after = (
            await conn.execute(text("SELECT COUNT(*) FROM categories"))
        ).scalar() or 0
        if _cat_after > _cat_before:
            # New seed slugs landed → bump the vocab version so non-manual rows
            # get reconsidered against the expanded vocabulary.
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) VALUES "
                    "('category_vocab_version', '2') "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
                )
            )

        # One-time migration: the categorization name settings were renamed
        # (self_names → self_identifiers, redact_names → hidden_identifiers).
        # Without this, a DB that persisted the old keys would silently stop
        # redacting/self-detecting those names — leaking them to the LLM.
        _rename_marker = (
            await conn.execute(
                text(
                    "SELECT 1 FROM settings WHERE key = "
                    "'migrations.categorization_identifier_rename'"
                )
            )
        ).first()
        if not _rename_marker:
            for _old, _new in (
                ("categorization.self_names", "categorization.self_identifiers"),
                ("categorization.redact_names", "categorization.hidden_identifiers"),
            ):
                _old_val = (
                    await conn.execute(
                        text("SELECT value FROM settings WHERE key = :k"), {"k": _old}
                    )
                ).scalar()
                if not _old_val:
                    continue
                _new_val = (
                    await conn.execute(
                        text("SELECT value FROM settings WHERE key = :k"), {"k": _new}
                    )
                ).scalar()
                if not _new_val:  # only fill if the new key is missing/blank
                    await conn.execute(
                        text(
                            "INSERT INTO settings (key, value) VALUES (:k, :v) "
                            "ON CONFLICT(key) DO UPDATE SET value = :v"
                        ),
                        {"k": _new, "v": _old_val},
                    )
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) VALUES "
                    "('migrations.categorization_identifier_rename', '1')"
                )
            )
        try:
            await conn.execute(
                text("SELECT source_kind FROM statement_uploads LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE statement_uploads ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'pdf'"
                )
            )
        try:
            await conn.execute(
                text("SELECT minimum_amount_due FROM statement_uploads LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text("ALTER TABLE statement_uploads ADD COLUMN minimum_amount_due TEXT")
            )
        ref_index_marker = (
            await conn.execute(
                text(
                    "SELECT 1 FROM settings WHERE key = 'migrations.uq_ref_includes_direction'"
                )
            )
        ).first()
        if not ref_index_marker:
            await conn.execute(text("DROP INDEX IF EXISTS uq_transactions_ref"))
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX uq_transactions_ref "
                    "ON transactions (bank, reference_number, direction) "
                    "WHERE reference_number IS NOT NULL"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) VALUES "
                    "('migrations.uq_ref_includes_direction', '1')"
                )
            )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_transactions_reference_number "
                "ON transactions (reference_number) "
                "WHERE reference_number IS NOT NULL"
            )
        )

        # An existing database does not get the index from the model's table_args
        # — create_all only builds indexes for tables it creates — so it is added
        # here. Category-first so a filter with no date bounds can seek instead of
        # scanning the table; transaction_date second so a dated category filter
        # uses both terms of one index. The marker keeps the ANALYZE off every
        # subsequent boot; the index itself is created idempotently regardless, so
        # a database that somehow lost it still gets it back.
        category_index_marker = (
            await conn.execute(
                text(
                    "SELECT 1 FROM settings WHERE key = 'migrations.ix_transactions_category_date'"
                )
            )
        ).first()
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_transactions_category_date "
                "ON transactions (category, transaction_date)"
            )
        )
        if not category_index_marker:
            # Without fresh stats the planner keeps its old shape for the queries
            # this index exists for.
            await conn.execute(text("ANALYZE transactions"))
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) VALUES "
                    "('migrations.ix_transactions_category_date', '1')"
                )
            )

        nach_marker = (
            await conn.execute(
                text(
                    "SELECT 1 FROM settings WHERE key = 'migrations.nach_ref_nullified'"
                )
            )
        ).first()
        if not nach_marker:
            await conn.execute(
                text(
                    "UPDATE transactions SET reference_number = NULL "
                    "WHERE reference_number IS NOT NULL "
                    "AND (channel = 'nach' OR email_type LIKE '%nach%')"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO settings (key, value) VALUES "
                    "('migrations.nach_ref_nullified', '1')"
                )
            )

        # --- SMS pipeline columns ---
        try:
            await conn.execute(text("SELECT sms_message_id FROM transactions LIMIT 0"))
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE transactions ADD COLUMN sms_message_id INTEGER "
                    "REFERENCES sms_messages(id)"
                )
            )
        try:
            await conn.execute(text("SELECT source FROM transactions LIMIT 0"))
        except Exception:
            await conn.execute(text("ALTER TABLE transactions ADD COLUMN source TEXT"))
            # Backfill: every existing transaction was created by the email
            # path (the only path before this spec). Runs only on the same
            # code path that adds the column, so the UPDATE fires once.
            await conn.execute(
                text("UPDATE transactions SET source = 'email' WHERE source IS NULL")
            )
        try:
            await conn.execute(
                text("SELECT notified_channel FROM transactions LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text("ALTER TABLE transactions ADD COLUMN notified_channel TEXT")
            )
            await conn.execute(
                text(
                    "UPDATE transactions SET notified_channel = 'email' "
                    "WHERE notified_channel IS NULL"
                )
            )
        try:
            await conn.execute(text("SELECT enriched_at FROM transactions LIMIT 0"))
        except Exception:
            await conn.execute(
                text("ALTER TABLE transactions ADD COLUMN enriched_at DATETIME")
            )
        try:
            await conn.execute(text("SELECT status FROM sms_messages LIMIT 0"))
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE sms_messages ADD COLUMN status TEXT NOT NULL "
                    "DEFAULT 'pending'"
                )
            )
        try:
            await conn.execute(text("SELECT transaction_id FROM sms_messages LIMIT 0"))
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE sms_messages ADD COLUMN transaction_id INTEGER "
                    "REFERENCES transactions(id)"
                )
            )
        try:
            await conn.execute(text("SELECT parse_error FROM sms_messages LIMIT 0"))
        except Exception:
            await conn.execute(
                text("ALTER TABLE sms_messages ADD COLUMN parse_error TEXT")
            )
        try:
            await conn.execute(text("SELECT parsed_at FROM sms_messages LIMIT 0"))
        except Exception:
            await conn.execute(
                text("ALTER TABLE sms_messages ADD COLUMN parsed_at DATETIME")
            )
        try:
            await conn.execute(
                text("SELECT cas_last_polled_at FROM email_sources LIMIT 0")
            )
        except Exception:
            await conn.execute(
                text("ALTER TABLE email_sources ADD COLUMN cas_last_polled_at DATETIME")
            )
        try:
            await conn.execute(text("SELECT auto_managed FROM fetch_rules LIMIT 0"))
        except Exception:
            await conn.execute(
                text(
                    "ALTER TABLE fetch_rules ADD COLUMN auto_managed BOOLEAN "
                    "NOT NULL DEFAULT 0"
                )
            )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_fetch_rule_auto_managed "
                "ON fetch_rules (source_id, sender) WHERE auto_managed = 1"
            )
        )

        # --- Investment-grade holding detail (Phase 4) ---
        # SnapshotHolding gains optional instrument-level columns a CAS payload
        # can state for a priced instrument. They stay NULL on the existing
        # per-asset-class aggregated rows; the columns are added so a future
        # instrument-level holding can record them. The new investment_lots
        # table is created by ``create_all`` (it only creates missing tables),
        # so it needs no ALTER here.
        for _col, _type in (
            ("instrument_id", "VARCHAR"),
            ("quantity", "NUMERIC(20,6)"),
            ("unit_price", "NUMERIC(20,6)"),
            ("currency", "VARCHAR(3)"),
            ("cost_basis", "NUMERIC(18,4)"),
            ("acquired_on", "DATE"),
        ):
            try:
                await conn.execute(
                    text(f"SELECT {_col} FROM snapshot_holdings LIMIT 0")
                )
            except Exception:
                await conn.execute(
                    text(f"ALTER TABLE snapshot_holdings ADD COLUMN {_col} {_type}")
                )

        # --- Extension sync-state: Paisa singleton + dirty-revision triggers ---
        # ``create_all`` builds extension_sync_state on fresh + legacy DBs (it
        # only creates missing tables). Idempotently seed the Paisa singleton
        # row dirty (desired_revision=1, applied_revision=0) and force_reload=1
        # so the next enabled coordinator reconciles once, even with no
        # observable drift yet. ON CONFLICT DO NOTHING leaves a coordinator-
        # owned row authoritative — migration never overwrites a row a
        # coordinator has already touched (so re-boot does not re-force a
        # reconcile after the first one succeeded).
        await conn.execute(
            text(
                "INSERT INTO extension_sync_state "
                "(extension_id, desired_revision, applied_revision, "
                " first_dirty_at, last_dirty_at, force_reload, failure_count, "
                " created_at, updated_at) "
                "VALUES ('paisa', 1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, "
                "        1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                "ON CONFLICT(extension_id) DO NOTHING"
            )
        )

        # Install the SQLite triggers LAST in this block, after every column
        # ALTER and data backfill above has run un-tracked. That keeps the
        # migration's own UPDATEs (e.g. category_method backfill) from firing
        # them, so a legacy DB does not accumulate spurious bumps during its
        # first migration pass — only writes after this point dirty the state.
        # Non-SQLite: the table exists, triggers are a SQLite implementation
        # detail and are skipped (coordinators fall back to always-reconcile).
        if conn.dialect.name == "sqlite":
            for _ddl in _build_extension_sync_state_trigger_ddl():
                await conn.execute(text(_ddl))

    async with engine.begin() as conn:
        # Seed the generic built-in merchant rules (INSERT OR IGNORE, idempotent).
        # Personal/local overrides are loaded separately via the CLI from the
        # untracked merchant_seed_data.py — the runtime never depends on that.
        from financial_dashboard.services.categorization.merchant_defaults import (
            DEFAULT_MERCHANT_RULES,
        )

        for _category, _patterns in DEFAULT_MERCHANT_RULES.items():
            for _pattern in _patterns:
                await conn.execute(
                    text(
                        "INSERT OR IGNORE INTO merchant_rules "
                        "(pattern, category, active, priority) "
                        "VALUES (:pattern, :category, 1, 100)"
                    ),
                    {"pattern": _pattern, "category": _category},
                )

    # function-local: breaks cycle with services.settings (settings imports db at top)
    from financial_dashboard.services.settings import load_all_settings

    await load_all_settings()

    # function-local: breaks cycle with categorization (merchant_rules imports db at top)
    from financial_dashboard.services.categorization.merchant_rules import (
        load_merchant_rules,
    )

    await load_merchant_rules()
