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


def test_needs_llm_eligibility_guard():
    """The re-check that closes the select→process window: eligible for the LLM
    pass on never-evaluated / pending_llm / prior-'unknown' rows, but never on a
    manual (or already-finalised) row — so a manual set mid-batch isn't clobbered."""
    from financial_dashboard.db.models import Transaction

    def txn(method, category=None):
        return Transaction(
            bank="b", email_type="x", direction="debit", amount=Decimal("1"),
            category_method=method, category=category,
        )

    assert sweep._needs_llm(txn(None)) is True
    assert sweep._needs_llm(txn("pending_llm")) is True
    assert sweep._needs_llm(txn("llm", "unknown")) is True  # stale-unknown reprocess
    # authoritative / finalised → never touched
    assert sweep._needs_llm(txn("manual", "gift")) is False
    assert sweep._needs_llm(txn("rule", "interest")) is False
    assert sweep._needs_llm(txn("llm", "groceries")) is False
