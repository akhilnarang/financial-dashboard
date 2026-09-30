from financial_dashboard.services import settings as settings_mod
from financial_dashboard.services.categorization.rules import (
    default_rule_config,
    load_rule_config,
    match_rules,
)

# Merchant rules are DATA (DB-backed), so these tests use synthetic stand-in
# patterns — one per category the behavioral cases need — rather than real
# merchant names. The engine logic under test is independent of which merchants
# happen to be seeded.
CFG = default_rule_config()._replace(
    self_name_tokens=("alex", "doe"),
    merchant_rules=(
        ("cardpay", "credit_card_payment"),
        ("payroll inc", "salary"),
        ("investco", "investment"),
    ),
)


def _f(cp=None, raw=None, channel=None, direction="debit", bank=None):
    return {
        "counterparty": cp,
        "raw_description": raw,
        "channel": channel,
        "direction": direction,
        "bank": bank,
    }


def test_self_transfer_by_name():
    r = match_rules(_f(cp="ALE X QUINN DOE", direction="credit"), CFG)
    assert r is not None and r.slug == "self_transfer"


def test_card_payment_merchant_matches_across_fields():
    # The pattern matches the counterparty, the narration, or a normalized
    # UPI handle ("cardpay.loans@upi" -> "cardpay loans upi").
    for cp, raw in (
        ("CARDPAY SETTLEMENT", None),
        ("UPI", "Paid Via CardPay"),
        ("cardpay.loans@upi", None),
    ):
        r = match_rules(_f(cp=cp, raw=raw), CFG)
        assert r is not None and r.slug == "credit_card_payment", cp


def test_interest_channel():
    r = match_rules(_f(channel="interest", direction="credit"), CFG)
    assert r is not None and r.slug == "interest"


def test_fd_maturity_beats_interest_channel():
    # A maturity row carries the "interest" channel, but the whole credit is a
    # redemption. The FD rule must win over the interest shortcut.
    r = match_rules(
        _f(cp="IDFC FD", channel="interest", direction="credit", bank="idfc"), CFG
    )
    assert r is not None and r.slug == "investment_redemption"


def test_fd_label_is_derived_from_the_row_bank():
    # The label must be the row's OWN bank plus " FD". An unrelated beneficiary
    # ending in " FD", or a label naming a different bank, must NOT match.
    assert (
        match_rules(_f(cp="ICICI FD", direction="debit", bank="icici"), CFG).slug
        == "investment"
    )
    assert match_rules(_f(cp="ACME FD", direction="debit", bank="idfc"), CFG) is None
    assert match_rules(_f(cp="IDFC FD", direction="debit", bank="slice"), CFG) is None
    assert match_rules(_f(cp="ACMEFD", direction="debit", bank="idfc"), CFG) is None


def test_email_type_card_alerts_with_blank_fields():
    # Card payoff/refund alerts have no counterparty or narration. The
    # email_type substring decides; the bank prefix is irrelevant.
    def f(et, direction="credit"):
        return {
            "counterparty": None,
            "raw_description": None,
            "email_type": et,
            "direction": direction,
        }

    assert match_rules(f("banka_cc_payment_alert"), CFG).slug == "credit_card_payment"
    assert (
        match_rules(f("bankb_cc_bill_paid", "debit"), CFG).slug == "credit_card_payment"
    )
    assert match_rules(f("bankc_cc_smartpay_bbps_alert", "debit"), CFG).slug == (
        "credit_card_payment"
    )
    assert match_rules(f("bankd_cc_refund_alert"), CFG).slug == "refund"
    # A card SPEND alert must NOT be treated as a payment.
    assert match_rules(f("banke_cc_transaction_alert", "debit"), CFG) is None


def test_investment_credit_is_redemption():
    assert match_rules(_f(cp="INVESTCO", direction="debit"), CFG).slug == "investment"
    assert (
        match_rules(_f(cp="INVESTCO", direction="credit"), CFG).slug
        == "investment_redemption"
    )


def test_merchant_rule_skips_refund_credit():
    # A spend-merchant rule (foodco->dining) must NOT fire on a credit (a refund
    # at that merchant); it falls through so the LLM/fallback can call it a refund.
    cfg = CFG._replace(merchant_rules=(("foodco", "dining"),))
    assert match_rules(_f(cp="WWW FOODCO", direction="debit"), cfg).slug == "dining"
    assert match_rules(_f(cp="WWW FOODCO", direction="credit"), cfg) is None


def test_merchant_type_beats_self_by_counterparty():
    # Precedence: a specific narration signal beats a weak "Self"/own-name
    # counterparty label (banks tag own-account credits as "Self", but the
    # narration — e.g. CASHBACK — is the truth).
    cfg = CFG._replace(
        self_name_tokens=("self",),
        merchant_rules=(("cashback", "cashback_rewards"),),
    )
    r = match_rules(_f(cp="Self", raw="CASHBACK FOR BILLPAY", direction="credit"), cfg)
    assert r is not None and r.slug == "cashback_rewards"


def test_load_rule_config_self_identifier_triggers_rule(monkeypatch):
    monkeypatch.setitem(
        settings_mod._cache, "categorization.self_identifiers", "alex, lee"
    )
    cfg = load_rule_config()
    for cp in ("ALEX SAVINGS ACCOUNT", "LEE SAVINGS ACCOUNT", "Self"):
        r = match_rules(_f(cp=cp, direction="credit"), cfg)
        assert r is not None and r.slug == "self_transfer", cp


def test_dividend_credit_is_other_income():
    for raw in ("ACME HOTELS COMPANY LTD FINAL DIV 25 26", "INTERIM DIVIDEND ACME LTD"):
        r = match_rules(_f(raw=raw, direction="credit", channel="bank_statement"), CFG)
        assert r is not None and r.slug == "other_income", raw


def test_dividend_marker_is_whole_token_credit_only_and_not_on_a_card():
    # "div" inside a word is not a dividend; a debit with the token is not one.
    assert match_rules(_f(raw="INDIVIDUAL STORE", direction="credit"), CFG) is None
    assert match_rules(_f(raw="ACME FINAL DIV 25 26", direction="debit"), CFG) is None
    fields = _f(raw="ACME FINAL DIV 25 26", direction="credit")
    fields["account_type"] = "credit_card"
    assert match_rules(fields, CFG).slug == "credit_card_payment"


def test_ach_credit_is_a_dividend():
    # Every ACH credit on this ledger is a share dividend. A listed bank as the
    # payer is the company paying the dividend, not a bank moving money. The
    # "ACH C-" narration matches when the channel is missing.
    for channel in ("ach_credit", None):
        r = match_rules(
            _f(
                cp="ACH C- SOME BANK LIMITED-9573935",
                direction="credit",
                channel=channel,
            ),
            CFG,
        )
        assert r is not None and r.slug == "other_income", channel


def test_ach_credit_rule_is_credit_only_and_yields_to_merchant_rules():
    assert match_rules(_f(cp="ACH D- ACME LTD", channel="ach_debit"), CFG) is None
    r = match_rules(
        _f(cp="ACH C- PAYROLL INC-123", direction="credit", channel="ach_credit"), CFG
    )
    assert r is not None and r.slug == "salary"
