"""Deterministic, high-confidence categorization rules (run before the LLM).

Structural rules live here in code; merchant→category mappings live in the
DB-backed merchant_rules table and are loaded into memory at startup.
"""

from typing import NamedTuple

from financial_dashboard.services.categorization.normalize import (
    normalize_counterparty,
    normalize_text,
)
from financial_dashboard.services.categorization.merchant_rules import (
    get_merchant_rules,
)
from financial_dashboard.services.categorization.polarity import (
    EXPENSE_SLUGS,
    INCOME_SLUGS,
)
from financial_dashboard.services.settings import get_self_identifier_tokens

RULESET_VERSION = "rules-v1"


class RuleConfig(NamedTuple):
    self_name_tokens: tuple[str, ...]
    merchant_rules: tuple[tuple[str, str], ...]


class RuleResult(NamedTuple):
    slug: str
    confidence: float


def default_rule_config() -> RuleConfig:
    return RuleConfig(
        self_name_tokens=("self",),
        merchant_rules=(),
    )


def load_rule_config() -> RuleConfig:
    # Drop tokens shorter than 3 chars: substring-matched against the normalized
    # counterparty, a 1-2 char token would false-positive on many merchants.
    extra = [t for t in get_self_identifier_tokens() if len(t) >= 3]
    base = default_rule_config()
    tokens = tuple(dict.fromkeys((*base.self_name_tokens, *(t.lower() for t in extra))))
    return base._replace(
        self_name_tokens=tokens,
        merchant_rules=get_merchant_rules(),
    )


def match_rules(fields: dict, config: RuleConfig) -> RuleResult | None:
    text_norm = normalize_text(
        " ".join(
            filter(None, (fields.get("counterparty"), fields.get("raw_description")))
        )
    )
    cp_norm = normalize_counterparty(fields.get("counterparty"))
    channel = (fields.get("channel") or "").lower()
    direction = (fields.get("direction") or "").lower()

    # A "spent on ... credit card" alert is a SPEND (purchase), never a payment
    # toward a card bill — guard so cc-payment merchant rules don't mislabel it.
    is_cc_spend = "spent on" in text_norm and "credit card" in text_norm

    # Interest EARNED is a credit. A debit on the interest channel is interest
    # PAID (a cost) — let it fall through to the spend side, not income.
    if channel == "interest" and direction != "debit":
        return RuleResult("interest", 1.0)

    # Structural email_type signals: parsers tag card payoffs/refunds even when
    # there's no counterparty/narration. Spend alerts lack these tokens (so are
    # excluded); generic cc_credit is ambiguous and left to the rules/LLM below.
    email_type = (fields.get("email_type") or "").lower()
    if "cc_refund" in email_type:
        return RuleResult("refund", 0.9)
    if (
        "cc_payment" in email_type
        or "cc_bill_paid" in email_type
        or "cc_smartpay" in email_type
    ):
        return RuleResult("credit_card_payment", 0.95)

    # Merchant/type rules run BEFORE the self-by-counterparty rule: a specific
    # narration signal (e.g. "CASHBACK FOR BILLPAY", a CRED payment) must win
    # over a weak "Self"/own-name counterparty label (common in bank statements).
    for pattern, category in config.merchant_rules:
        if is_cc_spend and category == "credit_card_payment":
            continue
        # Polarity guard: a spend-merchant rule must not fire on a CREDIT (a
        # refund at that merchant), nor an income rule on a DEBIT. Let those
        # fall through to the LLM/fallback (e.g. a credit at a dining merchant →
        # refund, not dining). Neutral categories (credit_card_payment,
        # investment) are unaffected and apply either direction.
        if direction == "credit" and category in EXPENSE_SLUGS:
            continue
        if direction == "debit" and category in INCOME_SLUGS:
            continue
        if pattern in text_norm:
            # Investing is an outflow (debit). A CREDIT at an investment
            # merchant is money coming back — a redemption/payout/dividend.
            if category == "investment" and direction == "credit":
                return RuleResult("investment_redemption", 0.9)
            return RuleResult(category, 0.9)

    # Self-transfer by counterparty token — only after no merchant/type rule hit.
    if any(
        tok and tok in cp_norm
        for tok in (normalize_counterparty(t) for t in config.self_name_tokens)
    ):
        return RuleResult("self_transfer", 0.9)

    return None
