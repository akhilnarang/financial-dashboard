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

from financial_dashboard.db.models import Account, Card, Transaction
from financial_dashboard.services import settings as settings_mod
from financial_dashboard.services.categorization import engine as eng
from financial_dashboard.services.categorization import llm
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


def _f(cp=None, raw=None, direction="credit", account_type="credit_card"):
    return {
        "counterparty": cp,
        "raw_description": raw,
        "channel": None,
        "direction": direction,
        "account_type": account_type,
    }


def test_card_credit_bill_payment_narration():
    # The own-name counterparty would read as self_transfer without the marker.
    r = match_rules(_f(cp="ALEX DOE", raw="BBPS PMT VIA UPI"), CFG)
    assert r is not None and r.slug == "credit_card_payment"


def test_card_credit_leading_payment_label_is_identified():
    # The bare card-credit default gives 0.7. A "Payment/..." label is evidence.
    r = match_rules(_f(raw="Payment/HDFC BANK LTD"), CFG)
    assert r is not None and r.slug == "credit_card_payment"
    assert r.confidence > 0.7


def test_card_credit_refund_narration():
    r = match_rules(_f(cp="DINERCO", raw="Refund for order 991"), CFG)
    assert r is not None and r.slug == "refund"


def test_card_credit_unexplained_is_card_payment_never_repayment():
    r = match_rules(_f(cp="SOMETHING ODD", raw="SOMETHING ODD REF 12"), CFG)
    assert r is not None and r.slug == "credit_card_payment"


@pytest.mark.parametrize("slug", ["repayment", "unknown", "shopping"])
def test_polarity_guard_card_credit_never_repayment(slug):
    resolved, changed = resolve_direction(slug, "credit", "credit_card")
    assert resolved != "repayment"
    assert resolved == "credit_card_payment"
    assert changed is True


def test_polarity_guard_keeps_valid_card_credits():
    for slug in ("refund", "cashback_rewards", "credit_card_payment"):
        resolved, changed = resolve_direction(slug, "credit", "credit_card")
        assert (resolved, changed) == (slug, False)


pytestmark = pytest.mark.anyio


async def test_engine_bank_credit_unexplained_still_repayment(
    session: AsyncSession, monkeypatch
):
    await ensure_category(session, "repayment")
    account = Account(bank="testbank", label="Savings", type="bank")
    session.add(account)
    await session.flush()

    async def fake_classify(**kwargs):
        return llm.LlmResult("repayment", 0.95, "friend paid me back")

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


async def _family_account(session: AsyncSession, client, monkeypatch) -> Account:
    """Seed a card account with a primary card and an add-on flagged by the form."""
    monkeypatch.setitem(
        settings_mod._cache, "categorization.family_card_holders", "Jane Roe, Sam Roe"
    )
    account = Account(bank="testbank", label="Card", type="credit_card")
    session.add(account)
    await session.flush()
    primary = Card(account_id=account.id, card_mask="XXXX1111", label="Alex Doe")
    addon = Card(account_id=account.id, card_mask="XXXX2222", label="Kid")
    session.add_all([primary, addon])
    await session.commit()
    response = await client.post(
        f"/accounts/{account.id}/cards/{addon.id}/edit",
        data={"label": "Kid", "is_family": "1"},
    )
    assert response.status_code == 303
    return account


@pytest.mark.parametrize(
    ("card_mask", "holder", "direction", "raw", "expected"),
    [
        # An alert names the add-on by its own mask only.
        ("XX2222", None, "debit", "AMAZON", "family"),
        # Two digits are too few to name a card, even when only one matches.
        ("XX22", None, "debit", "AMAZON", None),
        # The issuer prints the primary mask, and only the holder differs.
        ("XX1111", "JANE  ROE", "credit", "REFUND AMAZON", "family"),
        ("XX1111", "Alex Doe", "debit", "AMAZON", None),
    ],
)
async def test_family_addon_card_rows_are_family(
    session: AsyncSession,
    client,
    monkeypatch,
    card_mask,
    holder,
    direction,
    raw,
    expected,
):
    account = await _family_account(session, client, monkeypatch)
    txn = Transaction(
        bank="testbank",
        email_type="cc_txn",
        direction=direction,
        amount=Decimal("100"),
        raw_description=raw,
        account_id=account.id,
        card_mask=card_mask,
        card_holder=holder,
    )
    session.add(txn)
    await session.flush()

    await eng.categorize_one(session, txn, use_llm=False)

    assert txn.category == expected
    if expected:
        assert (txn.category_method, txn.category_confidence) == ("rule", 1.0)


async def test_enrichment_requeues_a_family_row_but_keeps_a_manual_category(
    session: AsyncSession, client, monkeypatch
):
    account = await _family_account(session, client, monkeypatch)
    rows = [
        Transaction(
            bank="testbank",
            email_type="cc_txn",
            direction="debit",
            amount=Decimal("100"),
            raw_description="AMAZON",
            account_id=account.id,
            card_mask="XX1111",
            card_holder="Jane Roe",
            category="shopping",
            category_method=method,
        )
        for method in ("llm", "manual")
    ]
    session.add_all(rows)
    await session.flush()

    for txn in rows:
        await eng.requeue_after_enrichment(session, txn)
        if txn.category_method is None:
            await eng.categorize_one(session, txn, use_llm=False)

    assert [(t.category, t.category_method) for t in rows] == [
        ("family", "rule"),
        ("shopping", "manual"),
    ]
