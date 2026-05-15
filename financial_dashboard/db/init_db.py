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
            await conn.execute(
                text("SELECT sms_message_id FROM transactions LIMIT 0")
            )
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
            await conn.execute(
                text("ALTER TABLE transactions ADD COLUMN source TEXT")
            )
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
            await conn.execute(
                text("SELECT transaction_id FROM sms_messages LIMIT 0")
            )
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

    # function-local: breaks cycle with services.settings (settings imports db at top)
    from financial_dashboard.services.settings import load_all_settings

    await load_all_settings()
