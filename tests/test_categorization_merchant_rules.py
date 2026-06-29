"""Tests for the DB-backed merchant-rules layer.

Tests cover: match_rules engine behaviour with merchant_rules in config,
the is_cc_spend guard, load_merchant_rules ordering, add_merchant_rule, and
list_merchant_rules.  All DB tests use the in-memory `session` fixture from
conftest.py so the real DB is never touched.
"""

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


# ---------------------------------------------------------------------------
# match_rules engine with merchant_rules in config
# ---------------------------------------------------------------------------


def test_match_rules_bill_payment():
    cfg = default_rule_config()._replace(
        merchant_rules=(
            ("billco", "bill_payment"),
            ("investco", "investment"),
        )
    )
    r = match_rules(_f(cp="BILLCO LIMITED"), cfg)
    assert r is not None and r.slug == "bill_payment"
    assert r.confidence == 0.9


def test_match_rules_investment():
    cfg = default_rule_config()._replace(merchant_rules=(("investco", "investment"),))
    r = match_rules(_f(cp="INVESTCO BROKING LIMITED"), cfg)
    assert r is not None and r.slug == "investment"


def test_match_rules_no_rule_returns_none():
    cfg = default_rule_config()._replace(merchant_rules=(("investco", "investment"),))
    r = match_rules(_f(cp="UNKNOWN MERCHANT XYZ"), cfg)
    assert r is None


# ---------------------------------------------------------------------------
# is_cc_spend guard
# ---------------------------------------------------------------------------


def test_is_cc_spend_guard_blocks_cc_payment_rule():
    """A 'spent on ... Credit Card' narration must NOT be labelled credit_card_payment
    even when a cc-payment merchant pattern matches."""
    cfg = default_rule_config()._replace(
        merchant_rules=(("cardpay", "credit_card_payment"),)
    )
    raw = "spent on your Credit Card cardpay"
    r = match_rules(_f(cp="CARDPAY", raw=raw), cfg)
    assert r is None, "cc-payment rule must be skipped on a spend narration"


def test_is_cc_spend_guard_allows_bill_payment_rule():
    """A bill_payment rule is never blocked by the cc-spend guard."""
    cfg = default_rule_config()._replace(
        merchant_rules=(
            ("billco", "bill_payment"),
            ("cardpay", "credit_card_payment"),
        )
    )
    raw = "spent on your Credit Card"
    r = match_rules(_f(cp="BILLCO LIMITED", raw=raw), cfg)
    assert r is not None and r.slug == "bill_payment"


def test_cc_spend_guard_does_not_fire_without_spent_on():
    """Without 'spent on' in the text the guard is inactive — normal cc payment."""
    cfg = default_rule_config()._replace(
        merchant_rules=(("cardpay", "credit_card_payment"),)
    )
    r = match_rules(_f(cp="CARDPAY", raw="credit card payment"), cfg)
    assert r is not None and r.slug == "credit_card_payment"


# ---------------------------------------------------------------------------
# load_merchant_rules ordering
# ---------------------------------------------------------------------------


async def test_load_merchant_rules_longer_pattern_wins(session: AsyncSession):
    """Within the same priority band, the longer (more-specific) pattern is
    tried first — 'acmepayments' wins over 'acme'."""
    session.add(MerchantRule(pattern="acme", category="misc", priority=100))
    session.add(
        MerchantRule(
            pattern="acmepayments", category="credit_card_payment", priority=100
        )
    )
    await session.flush()

    await load_merchant_rules(_session=session)
    rules = get_merchant_rules()

    patterns = [p for p, _ in rules]
    assert "acmepayments" in patterns and "acme" in patterns
    assert patterns.index("acmepayments") < patterns.index("acme"), (
        "acmepayments (longer) must precede acme (shorter) in the cache"
    )


async def test_load_merchant_rules_priority_wins_over_length(session: AsyncSession):
    """Lower priority number beats a longer pattern at a higher priority number."""
    # priority 50 but short
    session.add(MerchantRule(pattern="abc", category="salary", priority=50))
    # priority 100 but long
    session.add(
        MerchantRule(
            pattern="very long merchant name here", category="expense", priority=100
        )
    )
    await session.flush()

    await load_merchant_rules(_session=session)
    rules = get_merchant_rules()

    patterns = [p for p, _ in rules]
    assert patterns.index("abc") < patterns.index("very long merchant name here")


async def test_load_merchant_rules_ordering_drives_first_match(session: AsyncSession):
    """match_rules returns the FIRST matching rule in the ordered list."""
    session.add(MerchantRule(pattern="acme", category="misc", priority=100))
    session.add(
        MerchantRule(
            pattern="acmepayments", category="credit_card_payment", priority=100
        )
    )
    await session.flush()

    await load_merchant_rules(_session=session)
    cfg = default_rule_config()._replace(merchant_rules=get_merchant_rules())

    # 'acmepayments' is longer → checked first → matches → credit_card_payment
    r = match_rules(_f(cp="acmepayments.upi@okbank"), cfg)
    assert r is not None and r.slug == "credit_card_payment"

    # 'acme' matches but 'acmepayments' does not → misc
    r = match_rules(_f(cp="acme.upi@okbank"), cfg)
    assert r is not None and r.slug == "misc"


async def test_load_merchant_rules_only_active(session: AsyncSession):
    """Inactive rules must not appear in the cache."""
    session.add(MerchantRule(pattern="inactivepay", category="expense", active=False))
    session.add(MerchantRule(pattern="activepay", category="salary", active=True))
    await session.flush()

    await load_merchant_rules(_session=session)
    patterns = [p for p, _ in get_merchant_rules()]
    assert "activepay" in patterns
    assert "inactivepay" not in patterns


# ---------------------------------------------------------------------------
# add_merchant_rule
# ---------------------------------------------------------------------------


async def test_add_merchant_rule_inserts_and_strips(session: AsyncSession):
    await ensure_category(session, "expense")
    result = await add_merchant_rule(session, "  My Merchant  ", "expense")
    assert result is True
    await session.flush()

    rules = await list_merchant_rules(session)
    patterns = [r.pattern for r in rules]
    assert "my merchant" in patterns  # lowercased + stripped


async def test_add_merchant_rule_rejects_invalid_slug(session: AsyncSession):
    with pytest.raises(ValueError, match="Invalid category slug"):
        await add_merchant_rule(session, "somemerchant", "Not A Valid Slug!")


async def test_add_merchant_rule_rejects_unknown_category(session: AsyncSession):
    # valid slug format, but not in the categories vocabulary -> rejected (typo guard)
    with pytest.raises(ValueError, match="Unknown category"):
        await add_merchant_rule(session, "somemerchant", "dinng")


async def test_add_merchant_rule_upsert_updates_category(session: AsyncSession):
    await ensure_category(session, "expense")
    await ensure_category(session, "salary")
    await add_merchant_rule(session, "acme corp", "expense", priority=100)
    await session.flush()
    await add_merchant_rule(session, "acme corp", "salary", priority=50)
    await session.flush()

    rules = await list_merchant_rules(session)
    acme = next(r for r in rules if r.pattern == "acme corp")
    assert acme.category == "salary"
    assert acme.priority == 50


# ---------------------------------------------------------------------------
# list_merchant_rules
# ---------------------------------------------------------------------------


async def test_list_merchant_rules_returns_all(session: AsyncSession):
    session.add(MerchantRule(pattern="alpha", category="salary", priority=100))
    session.add(MerchantRule(pattern="beta", category="investment", priority=50))
    await session.flush()

    rules = await list_merchant_rules(session)
    assert len(rules) >= 2

    # beta has priority 50, must come before alpha (priority 100)
    patterns = [r.pattern for r in rules]
    assert patterns.index("beta") < patterns.index("alpha")


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
