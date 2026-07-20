"""Categorization-direction semantics, pinned at every layer they cross.

These are the invariants the cashflow report's bucket map, the polarity guard
and the rules layer are mutually answerable for, written down in one place so
a change to any one of the three that drifts from either of the others is
caught here rather than in a number that quietly stops reconciling.

The cases are deliberately small and pinned at the layer they belong to, so a
regression is named: a fee-reversal rule test that fails says "the fee-reversal
rule regressed", not "the cashflow report changed by 50".

Pinned semantics:

* Refund / cashback / **fee reversal** are the three contra-expense credits.
  Refund and cashback are income-guard slugs re-homed to expense by the bucket
  map; fee reversal is an expense-guard slug that the polarity guard would
  otherwise flip to ``repayment``. The fee-reversal rule is the one explicit,
  evidence-gated path that lets a credit ``fees_charges`` through.
* **Card-credit refund vs card-credit payment**: only two things credit a card
  (a refund/reversal or a bill payment); they are distinguished by narration,
  never by direction.
* **Investment vs redemption**: the slug is direction-dependent — a debit is a
  contribution, a credit at an investment merchant is a redemption.
* **Repayment**: a bank credit somebody-paying-you-back keeps its own line; it
  never inflates income.
* **Interest paid vs earned**: the interest *channel* on a credit is income; a
  debit on the same channel is interest paid and falls through to spend.
* **Neutral card-credit semantics**: a credit-card payment is direction-neutral
  (the bank-side debit and the card-side credit are the same event), and a
  self-transfer is direction-neutral on either side.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.services.cashflow.buckets import (
    BUCKET_BY_SLUG,
    CONTRA_EXPENSE_SLUGS,
    bucket_for_slug,
)
from financial_dashboard.services.categorization import engine as eng
from financial_dashboard.services.categorization import gemini as gem
from financial_dashboard.services.categorization.polarity import (
    EXPENSE_SLUGS,
    INCOME_SLUGS,
    resolve_direction,
)
from financial_dashboard.services.categorization.rules import (
    FEE_REVERSAL_MARKERS,
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

# Stand-in config used by every rule test below — self-name tokens + one
# merchant per category the cases need, so the engine under test does not
# depend on whatever happens to be in the merchant_rules seed.
CFG = default_rule_config()._replace(
    self_name_tokens=("alex", "doe"),
    merchant_rules=(
        ("dinerco", "dining"),
        ("investco", "investment"),
        ("cardpay", "credit_card_payment"),
        ("foodco", "groceries"),
    ),
)


def _f(
    cp=None,
    raw=None,
    direction="debit",
    account_type=None,
    channel=None,
    email_type=None,
):
    return {
        "counterparty": cp,
        "raw_description": raw,
        "channel": channel,
        "email_type": email_type,
        "direction": direction,
        "account_type": account_type,
    }


# ---------------------------------------------------------------------------
# Contra-expense polarity: refund / cashback / fee reversal
# ---------------------------------------------------------------------------


def test_contra_expense_slugs_are_pinned():
    """The cashflow bucket map's contra-expense set is named once: refund and
    cashback_rewards are income-guard slugs re-homed to expense, while a credit
    fees_charges is the explicit fee-reversal path. A change to either side of
    that union is a change to which credits net against spend."""
    assert CONTRA_EXPENSE_SLUGS == frozenset({"refund", "cashback_rewards"})
    # fees_charges is in EXPENSE_SLUGS, NOT in CONTRA_EXPENSE_SLUGS — the
    # contra path for it is the rules layer's fee-reversal evidence, not a
    # blanket bucket-map re-homing.
    assert "fees_charges" in EXPENSE_SLUGS
    assert "fees_charges" not in CONTRA_EXPENSE_SLUGS
    # And the contra pair stays in the expense bucket under every scope so the
    # credit leg of a contra nets against the spend it contra'd.
    for slug in CONTRA_EXPENSE_SLUGS | {"fees_charges"}:
        assert bucket_for_slug(slug, scope="bank") == "expense"
        assert bucket_for_slug(slug) == "expense"


def test_polarity_guard_keeps_contra_expense_credits_on_a_bank():
    # Refund and cashback_rewards are income-guard slugs that survive a credit
    # unchanged on a bank — they are contra-expense credits, not 'repayment'.
    for slug in CONTRA_EXPENSE_SLUGS:
        resolved, changed = resolve_direction(slug, "credit", "bank_account")
        assert (resolved, changed) == (slug, False), slug


def test_polarity_guard_flips_a_no_evidence_credit_fees_charges_to_repayment():
    """The whole point of the fee-reversal rule: a credit fees_charges with NO
    narration evidence is directionally impossible, so the polarity guard flips
    it. Only the explicit fee-reversal rule path can land a credit
    fees_charges, and the rule bypasses the polarity guard entirely."""
    # On a bank account: a credit fees_charges → repayment (somebody paying
    # you back is the credit default).
    resolved, changed = resolve_direction("fees_charges", "credit", "bank_account")
    assert (resolved, changed) == (REPAYMENT_SLUG, True)
    # On a card: a credit fees_charges → credit_card_payment (a card cannot
    # receive inbound income, so the unexplained credit is a bill payment).
    resolved, changed = resolve_direction(
        "fees_charges", "credit", CREDIT_CARD_ACCOUNT_TYPE
    )
    assert (resolved, changed) == (CREDIT_CARD_PAYMENT_SLUG, True)
    # No account_type known: still flipped, to the bank-side default.
    resolved, changed = resolve_direction("fees_charges", "credit", None)
    assert (resolved, changed) == (REPAYMENT_SLUG, True)


def test_polarity_guard_does_not_weaken_ordinary_expense_slugs_on_credit():
    """Every expense slug that is NOT in the contra-expense set, with no
    fee-reversal narration, is flipped on a credit. A debit stays unchanged.
    The fee-reversal path is the explicit exception, not a blanket allowance."""
    ordinary = sorted(EXPENSE_SLUGS - CONTRA_EXPENSE_SLUGS)
    assert ordinary, "EXPENSE_SLUGS - CONTRA_EXPENSE_SLUGS is non-empty"
    for slug in ordinary:
        resolved, changed = resolve_direction(slug, "credit", "bank_account")
        assert resolved == REPAYMENT_SLUG, slug
        assert changed is True, slug
        # And the debit reads unchanged — the guard is direction-gated.
        resolved, changed = resolve_direction(slug, "debit", "bank_account")
        assert (resolved, changed) == (slug, False), slug


@pytest.mark.parametrize("marker", FEE_REVERSAL_MARKERS)
def test_fee_reversal_rule_fires_for_every_marker(marker):
    """Each narration marker in FEE_REVERSAL_MARKERS, on a credit row, is
    categorized as fees_charges. The markers are whole-token matched, so a bare
    'fee' substring could never fire — these explicit markers are the only
    evidence that produces a contra-credit fees_charges."""
    r = match_rules(_f(raw=f"ACME BANK {marker}", direction="credit"), CFG)
    assert r is not None, marker
    assert r.slug == "fees_charges", marker
    assert r.confidence == 0.9, marker


def test_fee_reversal_rule_is_direction_gated():
    """A DEBIT row with a fee-reversal marker is not fees_charges — the markers
    describe a credit event, and a debit at a fee-charging merchant is the
    original fee. Direction-gated so the rule cannot produce a debit contra."""
    r = match_rules(_f(raw="ACME BANK annual fee reversal", direction="debit"), CFG)
    # No fee-reversal rule hit on a debit. No other rule fires either (no
    # merchant match, no self token), so the rule pass returns None and the
    # row falls through to the LLM/unknown path.
    assert r is None


def test_fee_reversal_rule_fires_on_bank_and_card_account_types():
    """A real fee reversal lands on either side: a bank-side fee reversal
    (account_type=bank_account) and a card-side reversal
    (account_type=credit_card) both produce fees_charges — the rule is
    direction-gated, not account-type-gated, because the contra semantics are
    the same on either side."""
    for account_type in ("bank_account", CREDIT_CARD_ACCOUNT_TYPE):
        r = match_rules(
            _f(
                raw="Annual fee reversal credited",
                direction="credit",
                account_type=account_type,
            ),
            CFG,
        )
        assert r is not None, account_type
        assert r.slug == "fees_charges", account_type


def test_fee_reversal_rule_beats_a_generic_refund_marker_on_a_card():
    """On a credit card, the card-credit refund block would catch the word
    'reversal' and call it a refund. The fee-reversal rule must fire first so a
    fee reversal nets against fees_charges, not against the generic refund line.
    Ordering the fee-reversal check ahead of the card-credit block is what
    makes that happen."""
    r = match_rules(
        _f(
            raw="Annual fee reversal",
            direction="credit",
            account_type=CREDIT_CARD_ACCOUNT_TYPE,
        ),
        CFG,
    )
    assert r is not None
    assert r.slug == "fees_charges"


def test_fee_reversal_rule_beats_a_card_payment_marker_on_a_card():
    # A narration carrying BOTH "fee reversal" and "payment received" must be
    # read as the fee reversal — the more-specific contra-expense signal wins
    # over the generic card-bill-payment marker.
    r = match_rules(
        _f(
            raw="Annual fee reversal payment received",
            direction="credit",
            account_type=CREDIT_CARD_ACCOUNT_TYPE,
        ),
        CFG,
    )
    assert r is not None
    assert r.slug == "fees_charges"


def test_fee_reversal_rule_does_not_fire_on_a_generic_fee_substring():
    """The markers are whole-token matched: a generic 'fee' or 'fees' substring
    cannot trip the rule. A narration like 'ACME BANK fees applied' is the
    original fee DEBIT, not a reversal credit, and the rule must leave it
    alone."""
    for raw in (
        "ACME BANK annual fee debited",
        "ACME BANK fees applied",
        "ACME BANK convenience fee",
        "ACME BANK late fee charged",
    ):
        # On a credit: no fee-reversal marker, no other rule fires — None.
        r = match_rules(_f(raw=raw, direction="credit"), CFG)
        assert r is None, raw
        # On a debit: same — no rule fires (no merchant match, no self token).
        r = match_rules(_f(raw=raw, direction="debit"), CFG)
        assert r is None, raw


# ---------------------------------------------------------------------------
# Card-credit refund vs card-credit payment
# ---------------------------------------------------------------------------


def test_card_credit_refund_vs_payment_distinguished_by_narration():
    # The two things that can credit a card: a refund/reversal, or a payment
    # against the bill. They are decided by narration, ahead of the merchant
    # rules, so a bill payment routed through a merchant rail is still a
    # payment and a refund at a merchant is still a refund.
    refund = match_rules(
        _f(
            cp="DINERCO",
            raw="Refund for order 991",
            direction="credit",
            account_type=CREDIT_CARD_ACCOUNT_TYPE,
        ),
        CFG,
    )
    assert refund is not None and refund.slug == "refund"

    payment = match_rules(
        _f(
            raw="BPPY CC PAYMENT 000000",
            direction="credit",
            account_type=CREDIT_CARD_ACCOUNT_TYPE,
        ),
        CFG,
    )
    assert payment is not None and payment.slug == CREDIT_CARD_PAYMENT_SLUG


def test_card_credit_unexplained_is_card_payment_never_repayment():
    # A card credit with no narration evidence at all defaults to a bill
    # payment, not 'repayment' — a card cannot receive inbound money.
    r = match_rules(
        _f(
            cp="SOMETHING ODD",
            raw="SOMETHING ODD REF 12",
            direction="credit",
            account_type=CREDIT_CARD_ACCOUNT_TYPE,
        ),
        CFG,
    )
    assert r is not None and r.slug == CREDIT_CARD_PAYMENT_SLUG


@pytest.mark.parametrize(
    "slug", ["repayment", "salary", "shopping", "dining", "unknown", "interest"]
)
def test_card_credit_polarity_guard_never_produces_repayment(slug):
    # The polarity guard mirrors the rule: a card credit can never land on
    # 'repayment' (income), and the income/expense/unknown slugs that would
    # produce it are all routed to a card-bill-payment instead.
    resolved, changed = resolve_direction(slug, "credit", CREDIT_CARD_ACCOUNT_TYPE)
    assert resolved == CREDIT_CARD_PAYMENT_SLUG
    assert changed is True


# ---------------------------------------------------------------------------
# Investment vs redemption
# ---------------------------------------------------------------------------


def test_investment_debit_is_contribution_credit_is_redemption():
    # The 'investment' merchant maps to a debit. On a credit at the same
    # merchant the rule remaps to investment_redemption — money coming back,
    # not going in.
    cfg = CFG._replace(merchant_rules=(("investco", "investment"),))
    debit = match_rules(_f(cp="INVESTCO BROKING LIMITED", direction="debit"), cfg)
    assert debit is not None and debit.slug == "investment"

    credit = match_rules(_f(cp="INVESTCO BROKING LIMITED", direction="credit"), cfg)
    assert credit is not None and credit.slug == "investment_redemption"


def test_investment_slugs_bucket_correctly():
    # The cashflow report splits the same slug by direction: investment is a
    # contribution (debit) and investment_redemption is a redemption (credit),
    # both in the investment bucket. A credit 'investment' does not appear in
    # the rule output (it's remapped above), but the bucket map still has to
    # accept the slug on either side so the report's direction-split works.
    for slug in ("investment", "investment_redemption"):
        assert bucket_for_slug(slug, scope="bank") == "investment"
        assert bucket_for_slug(slug) == "investment"


# ---------------------------------------------------------------------------
# Repayment
# ---------------------------------------------------------------------------


def test_repayment_is_transfers_in_not_income():
    # Repayment is somebody handing money back. The bucket map re-homes it out
    # of the income guard and onto its own transfers_in line, so it never
    # inflates income.
    assert REPAYMENT_SLUG in INCOME_SLUGS  # polarity guard's set
    assert BUCKET_BY_SLUG[REPAYMENT_SLUG] == "transfers_in"
    assert bucket_for_slug(REPAYMENT_SLUG, scope="bank") == "transfers_in"


def test_repayment_survives_a_bank_credit_unchanged():
    # A bank credit that the rules/LLM call 'repayment' is left alone by the
    # polarity guard — it's a valid credit-side slug.
    resolved, changed = resolve_direction(REPAYMENT_SLUG, "credit", "bank_account")
    assert (resolved, changed) == (REPAYMENT_SLUG, False)


# ---------------------------------------------------------------------------
# Interest paid vs earned
# ---------------------------------------------------------------------------


def test_interest_channel_credit_is_interest_income():
    # The interest channel on a credit is interest earned.
    r = match_rules(_f(channel="interest", direction="credit"), CFG)
    assert r is not None and r.slug == "interest"
    assert bucket_for_slug("interest", scope="bank") == "income"


def test_interest_channel_debit_falls_through_to_spend():
    # A debit on the interest channel is interest PAID (a cost). The interest
    # rule is gated on direction != debit, so it does not fire and the row
    # falls through. The test pins that fall-through: no interest rule hit on
    # a debit means the LLM/unknown path picks it up, not the income bucket.
    r = match_rules(_f(channel="interest", direction="debit"), CFG)
    assert r is None
    # And even if a confident LLM said 'interest' on a debit, the polarity
    # guard would flip it — an income slug on a debit is impossible.
    resolved, changed = resolve_direction("interest", "debit", "bank_account")
    assert (resolved, changed) == ("expense", True)


# ---------------------------------------------------------------------------
# Neutral card-credit semantics: card payment and self-transfer
# ---------------------------------------------------------------------------


def test_credit_card_payment_is_direction_neutral_in_the_bucket_map():
    # Over every account, a credit_card_payment is internal (counting both legs
    # would charge the same rupee twice). Over the bank alone it is the moment
    # cash leaves — the bill — and is expense. Direction-neutral in shape: the
    # bank-side debit and the card-side credit are the same event.
    assert bucket_for_slug(CREDIT_CARD_PAYMENT_SLUG) == "internal"
    assert bucket_for_slug(CREDIT_CARD_PAYMENT_SLUG, scope=None) == "internal"
    assert bucket_for_slug(CREDIT_CARD_PAYMENT_SLUG, scope="bank") == "expense"
    # The same slug survives a credit and a debit unchanged on either side —
    # the polarity guard does not touch the neutral slugs.
    for direction in ("debit", "credit"):
        for account_type in ("bank_account", CREDIT_CARD_ACCOUNT_TYPE, None):
            resolved, changed = resolve_direction(
                CREDIT_CARD_PAYMENT_SLUG, direction, account_type
            )
            assert (resolved, changed) == (CREDIT_CARD_PAYMENT_SLUG, False)


def test_self_transfer_is_direction_neutral_everywhere():
    # self_transfer is internal under every scope and survives either direction
    # unchanged — the slug itself describes a paired movement, not a spend or
    # an income.
    assert bucket_for_slug("self_transfer", scope="bank") == "internal"
    assert bucket_for_slug("self_transfer") == "internal"
    for direction in ("debit", "credit"):
        for account_type in ("bank_account", CREDIT_CARD_ACCOUNT_TYPE, None):
            resolved, changed = resolve_direction(
                "self_transfer", direction, account_type
            )
            assert (resolved, changed) == ("self_transfer", False)


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
        return gem.GeminiResult("fees_charges", 0.95, "looks like a fee credit")

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
