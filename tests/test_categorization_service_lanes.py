"""End-to-end categorization lanes through the engine, with the LLM mocked at
the model boundary.

The engine is the one place the rule path, the polarity guard, the
self-transfer rule and the LLM all meet: it picks the path a row takes and
writes the result. These tests pin the lanes by going through the engine,
mocking only ``engine._llm_classify`` (the thin indirection between the engine
and the provider) so a real network call is never made — the model boundary
itself is the seam.

Each lane is asserted at the layer it belongs to:

* **Merchant rule** — a seeded merchant rule fires via the rule path, never
  reaching the LLM.
* **Polarity** — a directionally-impossible slug from the LLM is coerced AND
  queued for review, never stored silently.
* **Self-transfer refusal/success** — the reference-pair rule refuses a same-
  account/same-direction pair and succeeds on a different-account opposite.
* **Manual category persistence** — a manual override survives a subsequent
  sweep's eligibility check (the ``_needs_llm`` guard).
* **pending_llm / LLM-low-confidence** — the rule pass marks a no-match row
  'pending_llm' (so the rule pass never re-evaluates it and the backfill
  terminates); a low-confidence LLM answer routes the row to review, with the
  model call intercepted at ``_llm_classify`` so no provider is contacted.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import financial_dashboard.services.categorization.engine as eng
import financial_dashboard.services.categorization.sweep as sweep
from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.services.categorization import gemini as gem
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


# ---------------------------------------------------------------------------
# Polarity lane
# ---------------------------------------------------------------------------


async def test_polarity_flips_directionally_impossible_llm_slug_and_queues_for_review(
    session: AsyncSession, monkeypatch
):
    """A confident LLM answer that is directionally impossible (a debit row
    categorised as 'refund', an income slug) is coerced to the debit default
    AND queued for review at capped confidence — never stored silently."""
    await ensure_category(session, "refund")

    async def fake_classify(**kwargs):
        return gem.GeminiResult("refund", 0.95, "looks like a refund")

    monkeypatch.setattr(eng, "_llm_classify", fake_classify)

    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("99"),
        counterparty="SOMEWHERE",
        raw_description="SOMEWHERE",
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=True)
    assert method == "llm"
    assert txn.category == "expense"  # debit + income slug -> DEBIT_DEFAULT
    assert txn.review_status == "pending"
    assert txn.category_confidence <= 0.4
    assert "refund" in (txn.review_reason or "")


async def test_polarity_keeps_directionally_consistent_llm_slug(
    session: AsyncSession, monkeypatch
):
    """A confident, directionally-consistent LLM answer is stored unchanged,
    with no review queueing. ``_llm_classify`` is the seam: a real provider is
    never contacted."""
    await ensure_category(session, "groceries")

    async def fake_classify(**kwargs):
        return gem.GeminiResult("groceries", 0.95, "grocery store")

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
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=True)
    assert method == "llm"
    assert txn.category == "groceries"
    assert txn.category_confidence == 0.95
    assert txn.review_status is None


# ---------------------------------------------------------------------------
# Self-transfer refusal / success
# ---------------------------------------------------------------------------


async def test_self_transfer_refuses_same_account_pair(session: AsyncSession):
    """A reference shared by two opposite-direction legs on the SAME account is
    not a self-transfer — it is a charge and its refund. The rule refuses to
    pair them and the row falls through to the LLM/unknown path."""
    account = Account(bank="icici", label="Card", type="credit_card")
    session.add(account)
    await session.flush()

    charge = Transaction(
        bank="icici",
        email_type="x",
        direction="debit",
        amount=Decimal("100"),
        reference_number="REF-SAME-ACCT",
        account_id=account.id,
    )
    refund = Transaction(
        bank="icici",
        email_type="x",
        direction="credit",
        amount=Decimal("100"),
        reference_number="REF-SAME-ACCT",
        account_id=account.id,
    )
    session.add_all([charge, refund])
    await session.flush()

    from financial_dashboard.services.categorization.self_transfer import (
        apply_reference_self_transfer_rule,
    )

    paired = await apply_reference_self_transfer_rule(session, refund)
    assert paired is False
    assert charge.category is None
    assert refund.category is None


async def test_self_transfer_succeeds_on_different_accounts(session: AsyncSession):
    """A reference shared by two opposite-direction legs on DIFFERENT accounts
    is a self-transfer: both legs are marked authoritative."""
    a = Account(bank="hdfc", label="HDFC", type="bank_account")
    b = Account(bank="icici", label="ICICI", type="bank_account")
    session.add_all([a, b])
    await session.flush()

    debit = Transaction(
        bank="hdfc",
        email_type="x",
        direction="debit",
        amount=Decimal("1000"),
        reference_number="REF-DIFF-ACCT",
        account_id=a.id,
    )
    credit = Transaction(
        bank="icici",
        email_type="x",
        direction="credit",
        amount=Decimal("1000"),
        reference_number="REF-DIFF-ACCT",
        account_id=b.id,
    )
    session.add_all([debit, credit])
    await session.flush()

    from financial_dashboard.services.categorization.self_transfer import (
        REFERENCE_PAIR_RULESET_VERSION,
        apply_reference_self_transfer_rule,
    )

    paired = await apply_reference_self_transfer_rule(session, credit)
    assert paired is True
    for txn in (debit, credit):
        assert txn.category == "self_transfer"
        assert txn.category_method == "rule"
        assert txn.category_confidence == 1.0
        assert txn.category_model == REFERENCE_PAIR_RULESET_VERSION
        assert txn.review_status is None


async def test_self_transfer_rule_short_circuits_the_engine(session: AsyncSession):
    """Through the engine, a paired self-transfer leg returns ``'rule'``
    without consulting the LLM. The LLM seam raises if called."""
    a = Account(bank="hdfc", label="HDFC", type="bank_account")
    b = Account(bank="icici", label="ICICI", type="bank_account")
    session.add_all([a, b])
    await session.flush()

    debit = Transaction(
        bank="hdfc",
        email_type="x",
        direction="debit",
        amount=Decimal("1000"),
        reference_number="REF-ENGINE-ST",
        account_id=a.id,
    )
    credit = Transaction(
        bank="icici",
        email_type="x",
        direction="credit",
        amount=Decimal("1000"),
        reference_number="REF-ENGINE-ST",
        account_id=b.id,
    )
    session.add_all([debit, credit])
    await session.flush()

    def fail_if_called(**kwargs):
        raise AssertionError("LLM must not be called for a self-transfer pair")

    monkeypatch_target = pytest.importorskip(
        "financial_dashboard.services.categorization.engine"
    )
    monkeypatch_target._llm_classify = fail_if_called  # noqa: SLF001

    method = await eng.categorize_one(session, credit, use_llm=True)
    assert method == "rule"
    assert credit.category == "self_transfer"
    assert debit.category == "self_transfer"


# ---------------------------------------------------------------------------
# Manual category persistence
# ---------------------------------------------------------------------------


async def test_manual_category_persists_through_a_subsequent_sweep_check(
    session: AsyncSession,
):
    """A manual override lands as ``category_method='manual'`` / confidence 1.0
    / review_status='resolved', and the sweep's ``_needs_llm`` guard refuses to
    re-evaluate it — so a later poll cannot overwrite the human's decision."""
    from financial_dashboard.services.categorization.manual import (
        assign_category_manual,
    )

    await ensure_category(session, "groceries")
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("50"),
    )
    session.add(txn)
    await session.flush()

    ok, slug = await assign_category_manual(session, txn.id, "Groceries")
    assert ok is True
    assert slug == "groceries"
    # The provenance fields a sweep's guard reads.
    assert txn.category_method == "manual"
    assert txn.category_confidence == 1.0
    assert txn.review_status == "resolved"
    # And the guard refuses it.
    assert sweep._needs_llm(txn) is False


# ---------------------------------------------------------------------------
# pending_llm / LLM-low-confidence
# ---------------------------------------------------------------------------


async def test_rule_pass_marks_unmatched_row_pending_llm_so_backfill_terminates(
    session: AsyncSession,
):
    """The rule pass with ``use_llm=False`` marks a no-match row
    ``pending_llm``, NOT NULL — so the next rule pass does not re-evaluate it
    and a backfill loop terminates at zero never-touched rows."""
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("99"),
        counterparty="ACME STORE",
        raw_description="ACME STORE MUMBAI",
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=False)
    assert method == "skip"
    assert txn.category_method == "pending_llm"
    assert txn.category is None
    assert txn.category_input_hash is not None
    # And a sweep eligibility check still picks it up for the LLM pass.
    assert sweep._needs_llm(txn) is True


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
    # Low confidence -> 'unknown' -> direction default (debit -> expense).
    assert txn.category == "expense"
    assert txn.review_status == "pending"
    assert txn.review_reason == "unsure"
    # The model boundary really was the seam: the call saw the engine's
    # fields dict and the active-slug list (with 'self_transfer' filtered out).
    assert captured["fields"]["counterparty"] == "MYSTERY MERCHANT"
    assert "self_transfer" not in captured["active_slugs"]
    assert "groceries" in captured["active_slugs"]


async def test_llm_needs_review_slug_routes_to_review(
    session: AsyncSession, monkeypatch
):
    """The provider's ``NEEDS_REVIEW`` sentinel is itself a low-confidence
    path: the row is routed to review with the direction default as its
    category, and the engine never stores the sentinel slug itself."""

    async def fake_classify(**kwargs):
        return gem.GeminiResult(gem.NEEDS_REVIEW, 0.0, "no fit")

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
    assert txn.category == "expense"  # debit + unknown default
    assert txn.review_status == "pending"
    assert txn.category != gem.NEEDS_REVIEW


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
