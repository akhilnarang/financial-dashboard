"""Categorization-direction semantics across the rules layer and the polarity guard.

* Refund, cashback and reimbursement are contra-expense credits. A credit
  ``fees_charges`` lands only through the evidence-gated fee-reversal rule.
  The polarity guard flips any other credit ``fees_charges``.
* A debit on the interest channel is interest paid, not income.
* A credit-card payment and a self-transfer are direction-neutral.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.services.cashflow.buckets import CONTRA_EXPENSE_SLUGS
from financial_dashboard.services.categorization import engine as eng
from financial_dashboard.services.categorization import llm
from financial_dashboard.services.categorization.polarity import (
    EXPENSE_SLUGS,
    resolve_direction,
)
from financial_dashboard.services.categorization.rules import (
    default_rule_config,
    match_rules,
)
from financial_dashboard.services.categorization.slugs import (
    CREDIT_CARD_ACCOUNT_TYPE,
    CREDIT_CARD_PAYMENT_SLUG,
    REPAYMENT_SLUG,
)
from financial_dashboard.services.categorization.vocabulary import ensure_category

pytestmark = pytest.mark.anyio

CFG = default_rule_config()


def _f(raw=None, direction="debit", account_type=None, channel=None):
    return {
        "counterparty": None,
        "raw_description": raw,
        "channel": channel,
        "direction": direction,
        "account_type": account_type,
    }


# ---------------------------------------------------------------------------
# Contra-expense polarity: refund / cashback / fee reversal
# ---------------------------------------------------------------------------


def test_polarity_guard_on_credits():
    """Contra-expense credits survive on a bank. Every other expense slug,
    fees_charges included, flips on a credit and stays on a debit. The
    fee-reversal rule is the explicit exception, not a blanket allowance."""
    for slug in CONTRA_EXPENSE_SLUGS:
        assert resolve_direction(slug, "credit", "bank_account") == (slug, False)
    ordinary = EXPENSE_SLUGS - CONTRA_EXPENSE_SLUGS
    assert "fees_charges" in ordinary
    for slug in ordinary:
        assert resolve_direction(slug, "credit", "bank_account") == (
            REPAYMENT_SLUG,
            True,
        ), slug
        assert resolve_direction(slug, "debit", "bank_account") == (slug, False)
    # A card cannot receive income, so the unexplained credit is a bill payment.
    assert resolve_direction("fees_charges", "credit", CREDIT_CARD_ACCOUNT_TYPE) == (
        CREDIT_CARD_PAYMENT_SLUG,
        True,
    )
    assert resolve_direction("fees_charges", "credit", None) == (REPAYMENT_SLUG, True)


def test_fee_reversal_rule_fires_on_a_credit():
    # The fee-reversal check runs ahead of the card-credit block. So a card
    # "reversal" or "payment received" narration stays a fee reversal.
    for raw, account_type in (
        ("Annual fee reversal credited", "bank_account"),
        ("Late fee waived", CREDIT_CARD_ACCOUNT_TYPE),
        ("Annual fee reversal payment received", CREDIT_CARD_ACCOUNT_TYPE),
    ):
        r = match_rules(_f(raw=raw, direction="credit", account_type=account_type), CFG)
        assert r is not None and r.slug == "fees_charges", raw
        assert r.confidence == 0.9


def test_fee_reversal_rule_needs_a_credit_and_a_whole_marker():
    # A debit with the marker is the original fee. A bare "fee" is not a marker.
    assert match_rules(_f(raw="ACME BANK annual fee reversal"), CFG) is None
    for direction in ("credit", "debit"):
        assert (
            match_rules(_f(raw="ACME BANK late fee charged", direction=direction), CFG)
            is None
        )


# ---------------------------------------------------------------------------
# Interest paid vs earned
# ---------------------------------------------------------------------------


def test_interest_channel_debit_falls_through_to_spend():
    # A debit on the interest channel is interest PAID. The rule does not fire,
    # and the polarity guard flips an LLM "interest" on a debit.
    assert match_rules(_f(channel="interest", direction="debit"), CFG) is None
    assert resolve_direction("interest", "debit", "bank_account") == ("expense", True)


# ---------------------------------------------------------------------------
# Neutral slugs: card payment and self-transfer
# ---------------------------------------------------------------------------


def test_neutral_slugs_survive_the_polarity_guard():
    # The bank-side debit and the card-side credit are the same event.
    for slug in (CREDIT_CARD_PAYMENT_SLUG, "self_transfer"):
        for direction in ("debit", "credit"):
            for account_type in ("bank_account", CREDIT_CARD_ACCOUNT_TYPE, None):
                assert resolve_direction(slug, direction, account_type) == (
                    slug,
                    False,
                )


# ---------------------------------------------------------------------------
# Engine integration: the rule path bypasses the polarity guard
# ---------------------------------------------------------------------------


async def test_engine_fee_reversal_credit_lands_as_fees_charges(session: AsyncSession):
    """End-to-end: a credit fee-reversal row is stored as fees_charges via the
    rule path, even though the polarity guard would otherwise flip a credit
    fees_charges. The rule path writes the slug directly and never calls
    resolve_direction, so the explicit evidence is what is stored."""
    account = Account(bank="testbank", label="Savings", type="bank_account")
    session.add(account)
    await session.flush()
    txn = Transaction(
        bank="testbank",
        email_type="testbank_misc_alert",
        direction="credit",
        amount=Decimal("500"),
        counterparty="ACME BANK",
        raw_description="Annual fee reversal credited",
        account_id=account.id,
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=False)
    assert method == "rule"
    assert txn.category == "fees_charges"
    assert txn.category_method == "rule"
    assert txn.category_confidence == 0.9
    assert txn.review_status is None


async def test_engine_llm_fee_charges_on_credit_is_flipped_to_repayment(
    session: AsyncSession, monkeypatch
):
    """The polarity path the rule deliberately does NOT take: an LLM that
    returns 'fees_charges' on a credit, with no narration evidence, is flipped
    to repayment and queued for review. The fee-reversal rule is the explicit
    exception; this is the rule that proves the guard was not weakened."""
    await ensure_category(session, "fees_charges")
    account = Account(bank="testbank", label="Savings", type="bank_account")
    session.add(account)
    await session.flush()

    async def fake_classify(**kwargs):
        return llm.LlmResult("fees_charges", 0.95, "looks like a fee credit")

    monkeypatch.setattr(eng, "_llm_classify", fake_classify)

    txn = Transaction(
        bank="testbank",
        email_type="testbank_misc_alert",
        direction="credit",
        amount=Decimal("50"),
        counterparty="ACME BANK",
        raw_description="ACME BANK something",  # no fee-reversal narration
        account_id=account.id,
    )
    session.add(txn)
    await session.flush()

    method = await eng.categorize_one(session, txn, use_llm=True)
    assert method == "llm"
    # The polarity guard flipped the credit fees_charges — without weakening,
    # it lands on the credit default for a bank account.
    assert txn.category == REPAYMENT_SLUG
    assert txn.review_status == "pending"
    assert "fees_charges" in (txn.review_reason or "")
