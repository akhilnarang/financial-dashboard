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
from financial_dashboard.services.categorization.slugs import (
    CREDIT_CARD_ACCOUNT_TYPE,
    CREDIT_CARD_PAYMENT_SLUG,
    REFUND_SLUG,
)
from financial_dashboard.services.settings import get_self_identifier_tokens

RULESET_VERSION = "rules-v1"

# Narration evidence for the two things that can credit a credit card. Matched as
# whole tokens (not bare substrings) so e.g. "cc payment" cannot fire on an
# unrelated "...acc payment..." run of characters.
CARD_BILL_PAYMENT_MARKERS: tuple[str, ...] = (
    "cc payment",
    "credit card payment",
    "card payment",
    "bill payment",
    "bill repayment",
    "bppy",
    "bbps pmt",
    "payment received",
)
CARD_REFUND_MARKERS: tuple[str, ...] = (
    "refund",
    "refunded",
    "reversal",
    "reversed",
    "chargeback",
    "cancellation",
)

# Narration evidence for a credit that reverses a previously-debited fee. A
# real fee reversal nets against the fee it reversed, so it belongs on the
# ``fees_charges`` line as a contra-credit rather than on ``refund`` /
# ``repayment``. This is the one explicit, evidence-gated path that lets a
# credit ``fees_charges`` through: the polarity guard still flips a credit
# ``fees_charges`` (LLM output, no evidence) to ``repayment`` / a card bill,
# so ordinary contra-credit slugs stay guarded. Whole-token matched like the
# card markers, so a generic "...fees..." run of characters cannot fire it.
FEE_REVERSAL_MARKERS: tuple[str, ...] = (
    "fee reversal",
    "fee reversed",
    "fee refund",
    "fee waived",
    "fees reversed",
    "fees refund",
    "fees waived",
    "annual fee reversal",
    "annual fee reversed",
    "annual fee refund",
    "annual charges reversal",
    "annual charges reversed",
    "joining fee reversal",
    "joining fee reversed",
    "card fee reversal",
    "card fee reversed",
    "late fee reversal",
    "late fee reversed",
    "finance charges reversal",
    "finance charges reversed",
)


def _has_marker(text_norm: str, markers: tuple[str, ...]) -> bool:
    padded = f" {text_norm} "
    return any(f" {marker} " in padded for marker in markers)


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

    # Fee reversal: a credit reversing a previously-debited fee nets against the
    # fee it reversed, so it belongs on the fees_charges line as a contra-credit
    # (the cashflow report's expense bucket reads it as a negative contra). This
    # is the explicit, evidence-gated path the polarity guard deliberately does
    # NOT take: a credit fees_charges is directionally impossible for the LLM
    # (which has no evidence to offer), so the guard still flips a no-evidence
    # credit fees_charges to 'repayment' / a card bill. Direction-gated to
    # credit because the markers themselves describe a credit event.
    if direction == "credit" and _has_marker(text_norm, FEE_REVERSAL_MARKERS):
        return RuleResult("fees_charges", 0.9)

    # A credit on a credit card is money returning TO the card — the account
    # holds what you owe, so it cannot receive inbound income. Only two things
    # produce one: a merchant reversing a charge, or a payment against the bill.
    # Decide on narration evidence here, ahead of the merchant rules, so a bill
    # payment routed through a merchant-looking rail is still read as a payment.
    is_card_credit = (
        direction == "credit" and fields.get("account_type") == CREDIT_CARD_ACCOUNT_TYPE
    )
    if is_card_credit:
        if _has_marker(text_norm, CARD_REFUND_MARKERS):
            return RuleResult(REFUND_SLUG, 0.9)
        # A leading "Payment/..." is the bank's own label for a card bill payment.
        if _has_marker(text_norm, CARD_BILL_PAYMENT_MARKERS) or text_norm.startswith(
            "payment "
        ):
            return RuleResult(CREDIT_CARD_PAYMENT_SLUG, 0.95)

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

    # No narration or merchant evidence, but the account still constrains the
    # answer: a card credit with nothing to go on is overwhelmingly a bill
    # payment, and letting it through would hand it to the credit fallback
    # ('repayment'), which counts as inbound money on a card that cannot receive
    # any. Lower confidence than the narration-backed rules above: it is a
    # structural default, not a positive identification.
    if is_card_credit:
        return RuleResult(CREDIT_CARD_PAYMENT_SLUG, 0.7)

    return None
