"""The index behind the category drill-through, and the migration that adds it.

A cashflow figure links to the rows it counted, and those links carry dates — the
date index already served them. The case that did not have an index is the bare
``/transactions?category=X``: its row query walked the whole table through the
date index, and the count() its pager needs scanned the table outright. These
tests pin the plan, so a lost index shows up as a failure rather than as a page
that is merely slower.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import Base

pytestmark = pytest.mark.anyio

INDEX = "ix_transactions_category_date"
MARKER = "migrations.ix_transactions_category_date"


async def _plan(session: AsyncSession, sql: str) -> str:
    rows = (await session.execute(text("EXPLAIN QUERY PLAN " + sql))).all()
    return " | ".join(str(row[3]) for row in rows)


async def _indexes(session: AsyncSession) -> list[str]:
    rows = (
        await session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'transactions'"
            )
        )
    ).all()
    return [row[0] for row in rows]


async def test_migration_adds_the_index_to_an_existing_database_and_is_idempotent(
    tmp_path, monkeypatch
):
    """An existing deployment has no index from the model — the migration is what adds it.

    ``create_all`` only builds indexes for tables it is creating, so a database that
    already has ``transactions`` would never see this one. Running the migration
    twice must leave exactly one index and one marker: it is a boot path, and it
    runs on every start.
    """
    from financial_dashboard.services import settings as settings_mod
    from financial_dashboard.services.categorization import merchant_rules

    # init_db ends by warming two caches through the *application's* engine; this
    # test drives an engine of its own, so those steps are stubbed rather than
    # pointed at a real database. The schema and migration work above them, which is
    # what is under test here, runs against the engine it is handed.
    async def _noop():
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", _noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", _noop)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
    try:
        # A database as it exists today: the table, without the new index.
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text(f"DROP INDEX IF EXISTS {INDEX}"))

        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as s:
            assert INDEX not in await _indexes(s)

        await init_db(engine)
        await init_db(engine)

        async with maker() as s:
            assert (await _indexes(s)).count(INDEX) == 1
            markers = (
                await s.execute(
                    text("SELECT count(*) FROM settings WHERE key = :k"), {"k": MARKER}
                )
            ).scalar_one()
            assert markers == 1
            # The bare filter and its pager count() both seek on the index.
            for sql in (
                "SELECT * FROM transactions WHERE category = 'groceries' "
                "ORDER BY transaction_date DESC LIMIT 50",
                "SELECT count(*) FROM "
                "(SELECT * FROM transactions WHERE category = 'groceries')",
            ):
                plan = await _plan(s, sql)
                assert INDEX in plan and "SEARCH" in plan, sql
    finally:
        await engine.dispose()
