import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import Base

pytestmark = pytest.mark.anyio

_INDEXES = {
    "ix_transactions_email_id": "transactions",
    "ix_transactions_sms_message_id": "transactions",
    "ix_statement_uploads_email_id": "statement_uploads",
    "ix_bank_statement_uploads_email_id": "bank_statement_uploads",
    "ix_sms_messages_transaction_id": "sms_messages",
    "ix_cas_uploads_email_id": "cas_uploads",
}


async def _index_names(session: AsyncSession) -> set[str]:
    rows = (
        await session.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'index'")
        )
    ).all()
    return {row[0] for row in rows}


async def test_source_link_indexes_are_in_fresh_schema(session):
    assert _INDEXES.keys() <= await _index_names(session)


async def test_init_adds_source_link_indexes_to_existing_database(
    tmp_path, monkeypatch
):
    from financial_dashboard.services import settings as settings_mod
    from financial_dashboard.services.categorization import merchant_rules

    async def noop():
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", noop)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            for index_name in _INDEXES:
                await connection.execute(text(f"DROP INDEX IF EXISTS {index_name}"))

        async with maker() as session:
            assert not (_INDEXES.keys() & await _index_names(session))

        await init_db(engine)
        await init_db(engine)

        async with maker() as session:
            names = await _index_names(session)
            assert _INDEXES.keys() <= names
            for index_name, table_name in _INDEXES.items():
                count = (
                    await session.execute(
                        text(
                            "SELECT count(*) FROM sqlite_master "
                            "WHERE type = 'index' AND name = :name "
                            "AND tbl_name = :table"
                        ),
                        {"name": index_name, "table": table_name},
                    )
                ).scalar_one()
                assert count == 1
    finally:
        await engine.dispose()
