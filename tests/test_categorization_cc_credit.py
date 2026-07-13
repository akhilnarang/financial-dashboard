"""A credit on a credit card is never inbound money.

On a bank account an unexplained credit can plausibly be somebody paying you
back, so it defaults to 'repayment'. On a card there is no such thing: the only
credits a card can receive are a merchant refund/reversal or a payment against
the bill. These tests pin that distinction at every layer — the rules pass, the
polarity guard, and the LLM path through the engine.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.services.categorization import engine as eng
from financial_dashboard.services.categorization import gemini as gem
from financial_dashboard.services.categorization.merchant_defaults import (
    DEFAULT_MERCHANT_RULES,
)
from financial_dashboard.services.categorization.polarity import resolve_direction
from financial_dashboard.services.categorization.rules import (
    default_rule_config,
    match_rules,
)
from financial_dashboard.services.categorization.vocabulary import ensure_category

CFG = default_rule_config()._replace(
    self_name_tokens=("alex", "doe"),
    merchant_rules=(("dinerco", "dining"),),
)

# Real-world card-credit narrations, sanitized.
CARD_BILL_PAYMENTS = (
    "BPPY CC PAYMENT 000000",
    "BBPS PMT VIA UPI",
    "Payment/HDFC BANK LTD",
    "CC PAYMENT VIA PayZapp",
    "CREDIT CARD PAYMENT Net Banking",
    "Bill repayment",
)


def _f(cp=None, raw=None, direction="credit", account_type="credit_card"):
    return {
        "counterparty": cp,
        "raw_description": raw,
        "channel": None,
        "direction": direction,
        "account_type": account_type,
    }


@pytest.mark.parametrize("raw", CARD_BILL_PAYMENTS)
def test_card_credit_bill_payment_narrations(raw):
    r = match_rules(_f(raw=raw), CFG)
    assert r is not None and r.slug == "credit_card_payment", raw


def test_card_credit_refund_narration():
    r = match_rules(_f(cp="DINERCO", raw="Refund for order 991"), CFG)
    assert r is not None and r.slug == "refund"


def test_card_credit_reversal_narration():
    r = match_rules(_f(raw="TXN REVERSAL 8891"), CFG)
    assert r is not None and r.slug == "refund"


def test_card_credit_unexplained_is_card_payment_never_repayment():
    r = match_rules(_f(cp="SOMETHING ODD", raw="SOMETHING ODD REF 12"), CFG)
    assert r is not None and r.slug == "credit_card_payment"


def test_bank_credit_unexplained_still_falls_back_to_repayment():
    # The whole point of the account-type split: a bank credit keeps the old
    # behavior. No rule fires, and the direction guard defaults it to repayment.
    r = match_rules(_f(cp="SOMEONE", raw="SOMEONE PAID ME", account_type="bank"), CFG)
    assert r is None
    slug, changed = resolve_direction("unknown", "credit", "bank")
    assert slug == "repayment" and changed is True


def test_no_account_type_still_falls_back_to_repayment():
    r = match_rules(_f(cp="SOMEONE", raw="SOMEONE PAID ME", account_type=None), CFG)
    assert r is None
    slug, _ = resolve_direction("unknown", "credit", None)
    assert slug == "repayment"


def test_card_credit_debit_direction_unaffected():
    # A DEBIT on a card is a purchase — the card-credit rule must not touch it.
    r = match_rules(_f(cp="DINERCO", raw="DINERCO", direction="debit"), CFG)
    assert r is not None and r.slug == "dining"


@pytest.mark.parametrize("slug", ["repayment", "unknown", "shopping", "salary"])
def test_polarity_guard_card_credit_never_repayment(slug):
    resolved, changed = resolve_direction(slug, "credit", "credit_card")
    assert resolved != "repayment"
    assert resolved == "credit_card_payment"
    assert changed is True


def test_polarity_guard_keeps_valid_card_credits():
    for slug in ("refund", "cashback_rewards", "credit_card_payment"):
        resolved, changed = resolve_direction(slug, "credit", "credit_card")
        assert (resolved, changed) == (slug, False)


@pytest.mark.parametrize(
    "pattern",
    [
        "bppy cc payment",
        "cc payment",
        "credit card payment",
        "bill repayment",
        "bbps pmt",
    ],
)
def test_merchant_defaults_cover_card_payment_narrations(pattern):
    assert pattern in DEFAULT_MERCHANT_RULES["credit_card_payment"]


def test_merchant_default_resolves_on_rules_only_pass():
    # Rules-only (no account_type known, e.g. the bank-side leg of the payment):
    # the merchant default still catches it.
    cfg = default_rule_config()._replace(
        merchant_rules=tuple(
            (p, "credit_card_payment")
            for p in DEFAULT_MERCHANT_RULES["credit_card_payment"]
        )
    )
    r = match_rules(
        _f(raw="BPPY CC PAYMENT 000000", direction="debit", account_type=None), cfg
    )
    assert r is not None and r.slug == "credit_card_payment"


pytestmark = pytest.mark.anyio


async def _card_txn(session: AsyncSession, raw: str) -> Transaction:
    account = Account(bank="testbank", label="Card", type="credit_card")
    session.add(account)
    await session.flush()
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="credit",
        amount=Decimal("5000"),
        counterparty=raw,
        raw_description=raw,
        account_id=account.id,
    )
    session.add(txn)
    await session.flush()
    return txn


async def test_engine_card_credit_llm_cannot_produce_repayment(
    session: AsyncSession, monkeypatch
):
    # Even if the LLM confidently says 'repayment' (exactly what happened in
    # prod), a card credit must not be stored as inbound money.
    await ensure_category(session, "repayment")

    async def fake_classify(**kwargs):
        return gem.GeminiResult("repayment", 0.95, "looks like money coming back")

    monkeypatch.setattr(eng, "_llm_classify", fake_classify)

    txn = await _card_txn(session, "NEFT CR SOMEBODY")
    await eng.categorize_one(session, txn, use_llm=True)
    assert txn.category != "repayment"
    assert txn.category in ("credit_card_payment", "refund")


async def test_engine_card_credit_rules_only(session: AsyncSession):
    txn = await _card_txn(session, "BPPY CC PAYMENT 000000")
    method = await eng.categorize_one(session, txn, use_llm=False)
    assert method == "rule"
    assert txn.category == "credit_card_payment"


async def test_engine_bank_credit_unexplained_still_repayment(
    session: AsyncSession, monkeypatch
):
    await ensure_category(session, "repayment")
    account = Account(bank="testbank", label="Savings", type="bank")
    session.add(account)
    await session.flush()

    async def fake_classify(**kwargs):
        return gem.GeminiResult("repayment", 0.95, "friend paid me back")

    monkeypatch.setattr(eng, "_llm_classify", fake_classify)

    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="credit",
        amount=Decimal("500"),
        counterparty="A FRIEND",
        raw_description="UPI/A FRIEND",
        account_id=account.id,
    )
    session.add(txn)
    await session.flush()

    await eng.categorize_one(session, txn, use_llm=True)
    assert txn.category == "repayment"
