# tests/test_categorization_sweep.py
from decimal import Decimal

import pytest

from financial_dashboard.db.models import Base, Transaction
from financial_dashboard.services.categorization import sweep

pytestmark = pytest.mark.anyio


@pytest.fixture
async def memdb(monkeypatch):
    from sqlalchemy.ext.asyncio import (
        async_sessionmaker,
        create_async_engine,
        AsyncSession,
    )

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(
        "financial_dashboard.services.categorization.sweep.async_session", maker
    )
    yield maker
    await engine.dispose()


async def test_rule_sweep_categorizes_interest_rows(memdb):
    async with memdb() as s:
        s.add(
            Transaction(
                bank="testbank",
                email_type="x",
                direction="credit",
                amount=Decimal("10"),
                channel="interest",
            )
        )
        await s.commit()

    n = await sweep.run_rule_sweep()
    assert n == 1

    async with memdb() as s:
        from sqlalchemy import select

        row = (await s.execute(select(Transaction))).scalars().one()
        assert row.category == "interest"
        assert row.category_method == "rule"


async def test_rule_sweep_marks_unmatched_pending_and_terminates(memdb):
    # An unmatched row becomes 'pending_llm' after the rule sweep, and a second
    # sweep finds zero never-touched rows (returns 0) — this is what lets the
    # backfill loop terminate with full coverage instead of re-evaluating forever.
    async with memdb() as s:
        s.add(
            Transaction(
                bank="testbank",
                email_type="x",
                direction="debit",
                amount=Decimal("99"),
                counterparty="ACME STORE",
                raw_description="ACME STORE MUMBAI",
            )
        )
        await s.commit()

    first = await sweep.run_rule_sweep()
    assert first == 1  # one row processed

    second = await sweep.run_rule_sweep()
    assert second == 0  # nothing left untouched → backfill loop would terminate

    async with memdb() as s:
        from sqlalchemy import select

        row = (await s.execute(select(Transaction))).scalars().one()
        assert row.category_method == "pending_llm"
        assert row.category is None
