"""End-to-end categorization lanes through the engine, with the LLM mocked at
the model boundary.

The engine is the one place the rule path, the polarity guard, the
self-transfer rule and the LLM all meet: it picks the path a row takes and
writes the result. These tests pin the lanes by going through the engine,
mocking only ``engine._llm_classify`` (the thin indirection between the engine
and the provider) so a real network call is never made — the model boundary
itself is the seam.
"""

import json
from decimal import Decimal

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

import financial_dashboard.services.categorization.engine as eng
from financial_dashboard.db.models import (
    Account,
    AuditAction,
    AuditInteraction,
    CategoryReviewDecision,
    Transaction,
)
from financial_dashboard.services.categorization import llm
from financial_dashboard.services.categorization.merchant_rules import (
    load_merchant_rules,
)
from financial_dashboard.services.categorization.vocabulary import ensure_category

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Merchant rule lane
# ---------------------------------------------------------------------------


async def test_merchant_rule_fires_via_engine_without_touching_the_llm(
    session: AsyncSession, monkeypatch
):
    """A seeded merchant rule wins via the rule path; the LLM classifier is
    never called. The seam is ``engine._llm_classify`` — if it fires, the test
    fails loudly, because the rule path is meant to short-circuit before it."""
    await ensure_category(session, "dining")
    # Seed a merchant rule and load the cache the rule pass reads from.
    from financial_dashboard.db.models import MerchantRule

    session.add(MerchantRule(pattern="dinerco", category="dining", priority=100))
    await session.flush()
    await load_merchant_rules(_session=session)

    def fail_if_called(**kwargs):
        raise AssertionError("LLM must not be called when a merchant rule fires")

    monkeypatch.setattr(eng, "_llm_classify", fail_if_called)

    account = Account(bank="testbank", label="Savings", type="bank_account")
    session.add(account)
    await session.flush()
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("50"),
        counterparty="DINERCO",
        raw_description="DINERCO MUMBAI",
        account_id=account.id,
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=True)
    assert method == "rule"
    assert txn.category == "dining"
    assert txn.category_method == "rule"
    assert txn.category_model == "rules-v1"


@pytest.mark.parametrize("manual_edit", [False, True])
async def test_merchant_lookup_audits_category_and_preserves_intervening_manual_edit(
    session: AsyncSession, monkeypatch, manual_edit
):
    """A lookup must audit its category, but must not overwrite an intervening edit.
    ``_llm_classify`` is the seam: a real provider is
    never contacted."""
    await ensure_category(session, "groceries")

    async def fake_classify(**kwargs):
        if manual_edit:
            async with AsyncSession(bind=session.bind) as other:
                await other.execute(
                    update(Transaction)
                    .where(Transaction.id == txn.id)
                    .values(
                        category="shopping",
                        category_method="manual",
                        category_confidence=1.0,
                    )
                )
                await other.commit()
        return llm.LlmResult(
            "groceries",
            0.95,
            "grocery store",
            merchant_search={
                "status": "completed",
                "model": "gpt-5.6-luna",
                "merchant": "ACME GROCERS",
                "city": "",
                "description": "Grocery store",
                "sources": [{"url": "https://example.com/acme", "title": "ACME"}],
            },
        )

    monkeypatch.setattr(eng, "_llm_classify", fake_classify)

    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("50"),
        counterparty="ACME GROCERS",
        raw_description="ACME GROCERS",
    )
    session.add(txn)
    await session.commit()

    method = await eng.categorize_one(session, txn, use_llm=True)
    assert method == ("skip" if manual_edit else "llm")
    assert txn.category == ("shopping" if manual_edit else "groceries")
    assert txn.category_confidence == (1.0 if manual_edit else 0.95)
    assert txn.review_status is None
    await session.commit()
    audit = await session.scalar(
        select(AuditInteraction).where(AuditInteraction.transaction_id == txn.id)
    )
    assert audit.status == "completed"
    assert audit.inbound_chat_id == 0 and audit.trigger == "merchant_lookup"
    output = json.loads(audit.model_output_json)
    assert output["merchant_search"]["sources"][0]["url"] == "https://example.com/acme"
    action = await session.scalar(
        select(AuditAction).where(AuditAction.interaction_id == audit.id)
    )
    if manual_edit:
        assert txn.category_method == "manual"
        assert output["applied_category"] is None
        assert "changed" in output["gate_reason"]
        assert action is None
        assert audit.model == "gpt-5.6-luna"
        return
    assert audit.outcome == "classified"
    assert output["applied_category"] == "groceries" and output["gate_reason"] is None
    assert action.action_type == "categorize_transaction" and action.undo_status is None
    assert json.loads(action.before_json) == output["before"]
    assert json.loads(action.after_json) == {
        "category": "groceries",
        "confidence": 0.95,
        "method": "llm",
    }


# ---------------------------------------------------------------------------
# LLM low confidence
# ---------------------------------------------------------------------------


async def test_llm_low_confidence_routes_to_review_at_the_model_boundary(
    session: AsyncSession, monkeypatch
):
    """A low-confidence LLM answer routes the row to review_status='pending',
    with the model call intercepted at ``_llm_classify`` (the engine's only
    seam with the provider). No real network call is made; the provider is
    what would have decided the slug/confidence, and the test stands in for it
    deterministically."""
    await ensure_category(session, "groceries")

    captured: dict = {}

    async def fake_classify(*, fields, examples, active_slugs):
        captured["fields"] = fields
        captured["active_slugs"] = active_slugs
        return llm.LlmResult(
            "groceries",
            0.10,
            "unsure",
            merchant_search={
                "status": "completed",
                "merchant": "MYSTERY MERCHANT",
                "city": "",
                "description": "Ambiguous merchant",
                "sources": [{"url": "https://example.com/mystery", "title": "Mystery"}],
            },
        )

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
    # Low confidence -> 'unknown' -> direction default (debit -> expense).
    assert txn.category == "expense"
    assert txn.review_status == "pending"
    assert txn.review_reason.startswith("unsure\nMerchant lookup: completed")
    assert "https://example.com/mystery" in txn.review_reason
    await session.commit()
    audit = await session.scalar(
        select(AuditInteraction).where(AuditInteraction.transaction_id == txn.id)
    )
    assert audit.outcome == "needs_review"
    output = json.loads(audit.model_output_json)
    assert output["applied_category"] == "expense" and "below" in output["gate_reason"]
    decision = await session.scalar(
        select(CategoryReviewDecision).where(
            CategoryReviewDecision.transaction_id == txn.id
        )
    )
    assert decision.source_interaction_id is None and decision.status == "active"
    # The model boundary really was the seam: the call saw the engine's
    # fields dict and the active-slug list (with 'self_transfer' filtered out).
    assert captured["fields"]["counterparty"] == "MYSTERY MERCHANT"
    assert "self_transfer" not in captured["active_slugs"]
    assert "groceries" in captured["active_slugs"]


async def test_llm_invalid_slug_preserves_review_gate_reason(session, monkeypatch):
    async def fake_classify(**kwargs):
        return llm.parse_result(
            {"category": "made_up", "confidence": 0.9, "reason": "bad output"},
            ["groceries"],
        )

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

    await eng.categorize_one(session, txn, use_llm=True)
    # The NEEDS_REVIEW sentinel is never stored: the direction default is.
    assert txn.category == "expense"
    assert txn.review_status == "pending"
    assert txn.review_reason == "invalid model category slug: made_up"
    decision = await session.scalar(
        select(CategoryReviewDecision).where(
            CategoryReviewDecision.transaction_id == txn.id
        )
    )
    assert decision.gate_reason == "invalid model category slug: made_up"


async def test_empty_input_skips_the_llm_call(session: AsyncSession, monkeypatch):
    """A row with neither counterparty nor raw_description does not spend an
    LLM call — the engine short-circuits to method='llm'/unknown so a
    stale-vocab requeue can reconsider it once enrichment populates text. The
    model seam raises if called."""

    def fail_if_called(**kwargs):
        raise AssertionError("LLM must not be called for an empty-input row")

    monkeypatch.setattr(eng, "_llm_classify", fail_if_called)

    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("99"),
        counterparty=None,
        raw_description=None,
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=True)
    assert method == "llm"
    assert txn.category == "unknown"
    assert txn.category_model == "empty-input"
    assert txn.category_confidence == 0.0
    assert txn.review_status is None
