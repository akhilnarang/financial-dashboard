"""Database initialization and inline migrations."""

from sqlalchemy import text

from financial_dashboard.db.models import Base


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
                text("INSERT OR IGNORE INTO categories (slug, active) VALUES (:slug, 1)"),
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
