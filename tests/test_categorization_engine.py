from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization import engine as eng
from financial_dashboard.services.categorization import gemini as gem

pytestmark = pytest.mark.anyio


async def test_rule_hit_sets_method_rule_without_llm(session: AsyncSession):
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="credit",
        amount=Decimal("10"),
        channel="interest",
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=False)
    assert method == "rule"
    assert txn.category == "interest"
    assert txn.category_method == "rule"
    assert txn.category_input_hash is not None


async def test_rule_pass_no_match_marks_pending_llm(session: AsyncSession):
    # A row the rules don't match must not stay method=NULL (that would make the
    # rule pass re-evaluate it forever and break backfill termination); it becomes
    # 'pending_llm' so the LLM pass picks it up.
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("42"),
        counterparty="ACME STORE",
        raw_description="ACME STORE MUMBAI",
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=False)
    assert method == "skip"
    assert txn.category_method == "pending_llm"
    assert txn.category is None


async def test_llm_low_confidence_routes_to_review(session: AsyncSession, monkeypatch):
    # seed an active category so the slug is valid
    from financial_dashboard.services.categorization.vocabulary import ensure_category

    await ensure_category(session, "groceries")

    async def fake_classify(**kwargs):
        return gem.GeminiResult("groceries", 0.10, "unsure")

    monkeypatch.setattr(eng, "_llm_classify", fake_classify)

    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("99"),
        counterparty="MYSTERY MERCHANT",
        raw_description="MYSTERY MERCHANT",
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=True)
    assert method == "llm"
    assert (
        txn.category == "expense"
    )  # debit + low-confidence 'unknown' -> direction default
    assert txn.review_status == "pending"
    assert txn.review_reason == "unsure"
