"""Tests for the DB-backed merchant-rules layer."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import financial_dashboard.services.categorization.merchant_rules as mr_mod
from financial_dashboard.db.models import MerchantRule
from financial_dashboard.services.categorization.merchant_rules import (
    add_merchant_rule,
    get_merchant_rules,
    list_merchant_rules,
    load_merchant_rules,
)
from financial_dashboard.services.categorization.rules import (
    default_rule_config,
    match_rules,
)
from financial_dashboard.services.categorization.vocabulary import ensure_category

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _restore_mr_cache():
    """Snapshot/restore the merchant-rules module-level cache around each test."""
    snapshot = list(mr_mod._cache)
    try:
        yield
    finally:
        mr_mod._cache.clear()
        mr_mod._cache.extend(snapshot)


def _f(cp=None, raw=None, channel=None, direction="debit"):
    return {
        "counterparty": cp,
        "raw_description": raw,
        "channel": channel,
        "direction": direction,
    }


def test_cc_spend_guard_blocks_cc_payment_rule_only_on_spend_narration():
    """A 'spent on ... Credit Card' narration must NOT be labelled
    credit_card_payment even when a cc-payment merchant pattern matches."""
    cfg = default_rule_config()._replace(
        merchant_rules=(("cardpay", "credit_card_payment"), ("billco", "bill_payment"))
    )
    raw = "spent on your Credit Card cardpay"
    assert match_rules(_f(cp="CARDPAY", raw=raw), cfg) is None
    r = match_rules(_f(cp="BILLCO LIMITED", raw="spent on your Credit Card"), cfg)
    assert r is not None and r.slug == "bill_payment"
    r = match_rules(_f(cp="CARDPAY", raw="credit card payment"), cfg)
    assert r is not None and r.slug == "credit_card_payment"


async def test_load_merchant_rules_order_drives_first_match(session: AsyncSession):
    """Lower priority number wins first. Within a band, the longer pattern wins.
    Inactive rules are not loaded."""
    session.add(MerchantRule(pattern="acme", category="misc", priority=100))
    session.add(
        MerchantRule(
            pattern="acmepayments", category="credit_card_payment", priority=100
        )
    )
    session.add(MerchantRule(pattern="abc", category="investment", priority=50))
    session.add(MerchantRule(pattern="abc long name", category="expense"))
    session.add(MerchantRule(pattern="inactivepay", category="expense", active=False))
    await session.flush()

    await load_merchant_rules(_session=session)
    cfg = default_rule_config()._replace(merchant_rules=get_merchant_rules())

    cases = {
        "acmepayments.upi@okbank": "credit_card_payment",
        "acme.upi@okbank": "misc",
        "abc long name": "investment",
        "inactivepay": None,
    }
    for cp, slug in cases.items():
        r = match_rules(_f(cp=cp), cfg)
        assert (r and r.slug) == slug, cp


# ---------------------------------------------------------------------------
# add_merchant_rule
# ---------------------------------------------------------------------------


async def test_add_merchant_rule_normalizes_upserts_and_rejects_unknown_category(
    session: AsyncSession,
):
    await ensure_category(session, "expense")
    await ensure_category(session, "salary")
    assert await add_merchant_rule(session, "  Acme Corp  ", "expense") is True
    await session.flush()
    await add_merchant_rule(session, "acme corp", "salary", priority=50)
    await session.flush()

    rules = [r for r in await list_merchant_rules(session) if r.pattern == "acme corp"]
    assert len(rules) == 1
    assert rules[0].category == "salary" and rules[0].priority == 50

    # A valid slug that is not in the vocabulary is a typo.
    with pytest.raises(ValueError):
        await add_merchant_rule(session, "somemerchant", "dinng")


# ---------------------------------------------------------------------------
# built-in default merchant rules
# ---------------------------------------------------------------------------


def test_default_merchant_rules_are_valid():
    """Every shipped default must use a normalized pattern and a real category
    slug — init_db inserts these raw (no add_merchant_rule validation)."""
    from financial_dashboard.services.categorization.merchant_defaults import (
        DEFAULT_MERCHANT_RULES,
    )
    from financial_dashboard.services.categorization.normalize import normalize_text
    from financial_dashboard.services.categorization.vocabulary import (
        SEED_CATEGORIES,
        is_valid_slug,
    )

    vocab = set(SEED_CATEGORIES)
    seen: set[str] = set()
    for category, patterns in DEFAULT_MERCHANT_RULES.items():
        assert is_valid_slug(category), f"bad slug: {category!r}"
        assert category in vocab, f"category not in vocabulary: {category!r}"
        for pattern in patterns:
            assert pattern and pattern == normalize_text(pattern), (
                f"unnormalized: {pattern!r}"
            )
            assert pattern not in seen, f"duplicate pattern: {pattern!r}"
            seen.add(pattern)


@pytest.mark.parametrize(
    ("legacy_category", "p2p_slug"),
    [("cashback_rewards", None), ("reimbursement", "reimbursement")],
)
async def test_init_db_retires_bare_supermoney_rule_and_keeps_cashback(
    monkeypatch, legacy_category, p2p_slug
):
    """A friend who pays through SuperMoney is not cashback. A payout from
    SuperMoney's own handle is. The first boot deletes the old seeded rule but
    keeps a user rule on that pattern with another category. A later boot
    keeps a rule that the user adds again."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from financial_dashboard.db.init_db import init_db
    from financial_dashboard.services import settings as settings_mod
    from tests.conftest import new_test_engine

    async def _noop():
        return None

    monkeypatch.setattr(settings_mod, "load_all_settings", _noop)
    monkeypatch.setattr(mr_mod, "load_merchant_rules", _noop)

    engine, holder = new_test_engine()
    try:
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as s:
            await ensure_category(s, legacy_category)
            s.add(MerchantRule(pattern="supermoney", category=legacy_category))
            await s.commit()

        await init_db(engine)

        async with maker() as s:
            await load_merchant_rules(_session=s)
        cfg = default_rule_config()._replace(merchant_rules=get_merchant_rules())
        cases = {
            "UPI/123456789012/CR/ALEX/ICIC/alex@superyes/ Paidalex@superyes/"
            "ALEX DOE/Paid via SuperMoney": p2p_slug,
            "UPI Credit-ALEX DOE-alex@superyes-ICIC0000001-123456789012-"
            "Paid via SuperMoney": p2p_slug,
            "UPI/supermoney/supermoney1@ye/Supermoney/YES BANKL/123456789012": (
                "cashback_rewards"
            ),
        }
        for raw, slug in cases.items():
            r = match_rules(_f(raw=raw, channel="upi", direction="credit"), cfg)
            assert (r and r.slug) == slug, raw

        async with maker() as s:
            await add_merchant_rule(s, "supermoney", "cashback_rewards")
            await s.commit()
        await init_db(engine)
        async with maker() as s:
            kept = await s.scalar(
                select(MerchantRule.category).where(
                    MerchantRule.pattern == "supermoney"
                )
            )
        assert kept == "cashback_rewards"
    finally:
        await engine.dispose()
        holder.close()
