from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from financial_dashboard.db.init_db import init_db


@pytest.mark.anyio
async def test_assistant_schema_builds_and_migrates_idempotently(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        "financial_dashboard.services.settings.load_all_settings",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "financial_dashboard.services.categorization.merchant_rules.load_merchant_rules",
        AsyncMock(),
    )
    database = tmp_path / "assistant-schema.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    try:
        await init_db(engine, paisa_enabled=False)
        await init_db(engine, paisa_enabled=False)
        async with engine.begin() as connection:
            tables, transaction_columns, outbound_indexes = await connection.run_sync(
                lambda sync: (
                    set(inspect(sync).get_table_names()),
                    {
                        column["name"]
                        for column in inspect(sync).get_columns("transactions")
                    },
                    {
                        index["name"]
                        for index in inspect(sync).get_indexes(
                            "telegram_outbound_deliveries"
                        )
                    },
                )
            )
        assert {
            "telegram_conversations",
            "telegram_message_contexts",
            "audit_interactions",
            "telegram_outbound_deliveries",
            "audit_actions",
            "category_review_decisions",
        } <= tables
        assert "attachment_path" in transaction_columns
        assert {
            "uq_telegram_outbound_interaction_ordinal",
            "uq_telegram_outbound_decision_ordinal",
        } <= outbound_indexes
    finally:
        await engine.dispose()
