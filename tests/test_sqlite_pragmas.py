"""SQLite connect-time PRAGMA listener: WAL journaling + busy_timeout.

WAL must be asserted against a FILE-backed DB (a :memory: database reports
journal_mode=memory regardless of the PRAGMA). All values here are synthetic.
"""

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from financial_dashboard.db import _set_sqlite_pragmas

pytestmark = pytest.mark.anyio


async def test_wal_and_busy_timeout_applied(tmp_path):
    db_path = tmp_path / "pragma_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    event.listen(engine.sync_engine, "connect", _set_sqlite_pragmas)
    try:
        async with engine.connect() as conn:
            journal_mode = (
                await conn.execute(text("PRAGMA journal_mode"))
            ).scalar_one()
            busy_timeout = (
                await conn.execute(text("PRAGMA busy_timeout"))
            ).scalar_one()
        assert journal_mode == "wal"
        assert busy_timeout == 5000
    finally:
        await engine.dispose()
