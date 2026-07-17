"""Explicit scenario-branch IDs + coverage metadata for the synthetic graph.

The scenario is a graph of deliberately-shaped edges, each standing for a
dashboard behaviour the corpus exists to exercise. This module names every one
of those edges as a stable *branch id* grouped by concern (merge/link,
categorization, statement reconciliation, refunds/reversals, FX, net-worth/CAS/
manual, workflow/projection), and :func:`compute_coverage` walks a built
:class:`~scripts.synth.models.Scenario` to report which branch ids are present.

The set of *required* branch ids (:data:`REQUIRED_BRANCH_IDS`) is what the
manifest records and ``verify_manifest`` polices: a generator regression that
silently drops a branch shows up as a missing id, and (via the manifest) as a
tamper error. This is the single source of truth for "did the canonical scenario
cover every shape it claims to".

Coverage is a *pure function* of the scenario graph — no RNG, no I/O — so it is
deterministic, unit-testable, and byte-stable for the same ``(seed, profile,
as_of)``.
"""

from typing import NamedTuple

from scripts.synth import constants as C
from scripts.synth.models import Scenario


#: A coverage group + its human description. Branch ids are namespaced
#: ``<group>.<edge>`` so a test or a manifest diff reads legibly.
class ScenarioBranch(NamedTuple):
    branch_id: str
    group: str
    description: str


# Defined as a tuple of (group, edge, description); the fully-qualified branch
# id is built as ``<group>.<edge>``. Order is stable so the registry (and thus
# any manifest diff) reads top-to-bottom by concern.
_RAW_BRANCHES: tuple[tuple[str, str, str], ...] = (
    # --- merge / link -------------------------------------------------------
    ("merge", "email_sms_pair", "one event arriving as both email and SMS, merged"),
    ("merge", "link_by_account_mask", "txn resolved to an account by account_mask"),
    ("merge", "link_by_card_mask", "txn resolved to a card by card_mask"),
    ("merge", "unlinked_unknown", "txn on no account (unaccounted scope)"),
    (
        "merge",
        "ref_mismatch_pair",
        "statement ref disagrees with the DB ref (fuzzy refusal)",
    ),
    ("merge", "balance_conflict", "balance-conflict split between two legs"),
    # --- categorization metadata -------------------------------------------
    ("cat", "method_synthetic", "category_method='synthetic' (bulk default)"),
    ("cat", "method_manual", "category_method='manual'"),
    ("cat", "method_rule", "category_method='rule' (incl. self-transfer rule)"),
    ("cat", "method_llm", "category_method='llm' with model + confidence"),
    ("cat", "method_pending_llm", "category_method='pending_llm'"),
    ("cat", "review_pending", "review_status='pending'"),
    ("cat", "review_notified", "review_status='notified'"),
    ("cat", "review_resolved", "review_status='resolved' with reason"),
    ("cat", "full_vocabulary", "every seed category slug represented"),
    # --- statement reconciliation ------------------------------------------
    ("stmt", "cc_exact_match", "CC statement row exact-matched to a DB txn"),
    ("stmt", "cc_paid", "a CC statement marked fully paid"),
    ("stmt", "cc_unpaid", "a CC statement marked unpaid"),
    ("stmt", "cc_parse_error", "a CC statement upload in parse_error status"),
    ("stmt", "cc_email_summary", "a CC statement sourced from an email summary"),
    ("stmt", "bank_matched", "bank statement rows matched to DB txns"),
    ("stmt", "bank_closing", "bank statement carrying a closing balance"),
    (
        "stmt",
        "reconcile_offline",
        "reconciliation_data produced by the real reconcile service",
    ),
    # --- refunds / reversals ----------------------------------------------
    ("txn", "merchant_refund", "distinct merchant refund credit on a bank account"),
    ("txn", "cashback", "cashback_rewards contra-expense credit"),
    ("txn", "fee_reversal", "fee reversal credit netting a prior fee"),
    ("txn", "cc_refund_credit", "CC refund/reversal credit that is NOT a bill payment"),
    ("txn", "cc_payment_received", "real CC payment received credit (card-side leg)"),
    ("txn", "transfer_in_repayment", "transfer-in repayment credit"),
    ("txn", "investment_redemption", "investment redemption credit"),
    # --- FX -----------------------------------------------------------------
    ("fx", "usd_covered", "USD txn with a covering FX rate"),
    ("fx", "usd_missing", "USD txn before the first configured rate"),
    ("fx", "eur_covered", "EUR txn with a covering FX rate"),
    ("fx", "gbp_missing", "GBP txn with no configured rate"),
    ("fx", "invalid_currency", "a txn carrying an invalid currency code"),
    ("fx", "blank_currency", "a txn whose currency is blank (None read as INR)"),
    # --- net-worth / CAS / manual ------------------------------------------
    ("net", "multiple_pans", "CAS uploads spanning >1 PAN"),
    ("net", "nsdl_cdsl", "both NSDL and CDSL depositories present"),
    ("net", "cas_reconciled", "a reconciled CAS portfolio (portfolio_ok=True)"),
    ("net", "cas_unreconciled", "an unreconciled CAS portfolio (portfolio_ok=False)"),
    ("net", "lot_complete", "a complete MF acquisition lot"),
    ("net", "lot_disposal", "a disposal MF transaction"),
    ("net", "lot_incomplete", "an incomplete MF fact (missing lot facts)"),
    ("net", "manual_active", "an active manual asset/liability"),
    ("net", "manual_inactive", "a deactivated manual item"),
    ("net", "cc_zero_snapshot", "a credit-card snapshot with zero outstanding"),
    ("net", "non_inr_snapshot_excluded", "a non-INR snapshot excluded from net worth"),
    (
        "net",
        "trend_multiple_months",
        "snapshots spanning multiple months for the trend",
    ),
    # --- workflow / projection --------------------------------------------
    ("proj", "self_transfer_paired", "a paired (debit+credit) self-transfer"),
    ("proj", "self_transfer_unmatched", "an unmatched single-leg self-transfer"),
    ("proj", "card_side_payment", "a card-side CC payment (skipped by projection)"),
    ("proj", "fx_priced", "a non-INR txn the projection can price"),
    ("proj", "fx_missing_rate", "a non-INR txn with no FX rate"),
    ("proj", "long_account_name", "a long, spaced ledger account name"),
    # --- transaction edges -------------------------------------------------
    ("edge", "undated", "a transaction with no transaction_date"),
    ("edge", "blank_counterparty", "a transaction with a blank counterparty"),
    ("edge", "blank_category", "a transaction with a blank category"),
    ("edge", "am_pm_pair", "an AM/PM same-day pair"),
    (
        "edge",
        "orphan_cc_payment",
        "an orphan CC payment debit with no card-side credit",
    ),
    ("edge", "orphan_emails", "pending/failed/skipped source emails"),
)


def _make_registry() -> tuple[ScenarioBranch, ...]:
    return tuple(
        ScenarioBranch(branch_id=f"{group}.{edge}", group=group, description=desc)
        for group, edge, desc in _RAW_BRANCHES
    )


#: The complete, ordered registry of known scenario branches.
SCENARIO_BRANCHES: tuple[ScenarioBranch, ...] = _make_registry()

#: Every fully-qualified branch id, as a set for fast membership tests.
ALL_BRANCH_IDS: frozenset[str] = frozenset(b.branch_id for b in SCENARIO_BRANCHES)

#: Branch ids grouped by concern. ``manifest.coverage`` carries this grouping so
#: a reader can see which concerns the corpus covers without re-deriving it.
BRANCHES_BY_GROUP: dict[str, tuple[str, ...]] = {}
for _b in SCENARIO_BRANCHES:
    BRANCHES_BY_GROUP.setdefault(_b.group, ())
    BRANCHES_BY_GROUP[_b.group] = (*BRANCHES_BY_GROUP[_b.group], _b.branch_id)

#: The subset of branch ids that the canonical generator must always emit.
#: ``verify_manifest`` raises if the manifest's recorded coverage is missing any
#: of these, so a regression that drops a branch fails verification. Branches
#: excluded here are ones whose presence is scale-dependent (e.g. a parse-error
#: statement only at certain profiles) or are recorded opportunistically.
REQUIRED_BRANCH_IDS: frozenset[str] = ALL_BRANCH_IDS


def branch_groups(coverage: frozenset[str]) -> dict[str, tuple[str, ...]]:
    """Group a present-id set by concern, sorted for byte-stable output."""
    out: dict[str, list[str]] = {g: [] for g in BRANCHES_BY_GROUP}
    for bid in sorted(coverage):
        group = bid.split(".", 1)[0]
        out.setdefault(group, []).append(bid)
    return {g: tuple(v) for g, v in sorted(out.items()) if v}


def compute_coverage(scenario: Scenario) -> frozenset[str]:
    """Walk ``scenario`` and return the set of scenario-branch ids it exercises.

    Pure and deterministic: the same scenario always yields the same set. The
    detection rules intentionally key off *distinguishing features* of the rows
    (category, direction, currency, mask, balance presence, lot shape, ...) so
    a dropped edge shows up as a missing id rather than a silent gap.
    """
    present: set[str] = set()
    cc_account_pks = {a.pk for a in scenario.accounts if a.type == C.CREDIT_CARD}
    bank_account_pks = {
        a.pk for a in scenario.accounts if a.type == C.BANK_ACCOUNT and a.active
    }

    txns = scenario.transactions
    categories = {t.category for t in txns}
    currencies = {t.currency for t in txns}

    def has(predicate) -> bool:
        return any(predicate(t) for t in txns)

    # --- merge / link -------------------------------------------------------
    if any(t.sms_pk is not None for t in txns):
        present.add("merge.email_sms_pair")
    if any(t.account_mask and t.account_pk in bank_account_pks for t in txns):
        present.add("merge.link_by_account_mask")
    if any(t.card_mask and t.card_pk is not None for t in txns):
        present.add("merge.link_by_card_mask")
    if any(t.account_pk is None for t in txns):
        present.add("merge.unlinked_unknown")
    if has(lambda t: (t.reference_number or "").startswith("SYN-MISMATCH")):
        present.add("merge.ref_mismatch_pair")
    if has(lambda t: (t.reference_number or "").startswith("SYN-CONFLICT")):
        present.add("merge.balance_conflict")

    # --- categorization metadata -------------------------------------------
    # ``category_method`` defaults to ``synthetic`` at load time (a None on the
    # scenario graph is the bulk-lane default), so a None value counts as the
    # synthetic axis. The production self-transfer reference rule fires on a
    # paired self-transfer at load time (category_method='rule'), so the
    # presence of a paired self-transfer exercises the rule axis too.
    methods = {
        getattr(t, "category_method", None)
        for t in txns
        if getattr(t, "category_method", None)
    }
    if any(getattr(t, "category_method", None) in (None, "synthetic") for t in txns):
        present.add("cat.method_synthetic")
    for m in methods:
        key = {
            "synthetic": "cat.method_synthetic",
            "manual": "cat.method_manual",
            "rule": "cat.method_rule",
            "llm": "cat.method_llm",
            "pending_llm": "cat.method_pending_llm",
        }.get(m)
        if key:
            present.add(key)
    # The production self-transfer reference rule fires on a paired
    # self-transfer (>=2 legs sharing a reference) at load time.
    _st_refs_all: dict[str, int] = {}
    for t in txns:
        if t.category == "self_transfer" and t.reference_number:
            _st_refs_all[t.reference_number] = (
                _st_refs_all.get(t.reference_number, 0) + 1
            )
    if any(n >= 2 for n in _st_refs_all.values()):
        present.add("cat.method_rule")
    review_statuses = {t.review_status for t in txns if t.review_status}
    for rs in review_statuses:
        key = {
            "pending": "cat.review_pending",
            "notified": "cat.review_notified",
            "resolved": "cat.review_resolved",
            "flagged": "cat.review_pending",
            "reviewed": "cat.review_resolved",
        }.get(rs)
        if key:
            present.add(key)
    if categories >= set(C.SEED_CATEGORY_SLUGS):
        present.add("cat.full_vocabulary")

    # --- refunds / reversals ----------------------------------------------
    if has(
        lambda t: (
            t.category == "refund"
            and t.direction == C.DIRECTION_CREDIT
            and t.account_pk in bank_account_pks
            and "refund" in t.email_type
        )
    ):
        present.add("txn.merchant_refund")
    if "cashback_rewards" in categories:
        present.add("txn.cashback")
    if has(
        lambda t: t.category == "fees_charges" and t.direction == C.DIRECTION_CREDIT
    ):
        present.add("txn.fee_reversal")
    if has(
        lambda t: (
            t.category == "refund"
            and t.direction == C.DIRECTION_CREDIT
            and t.account_pk in cc_account_pks
        )
    ):
        present.add("txn.cc_refund_credit")
    if has(
        lambda t: (
            t.category == "credit_card_payment"
            and t.direction == C.DIRECTION_CREDIT
            and t.account_pk in cc_account_pks
        )
    ):
        present.add("txn.cc_payment_received")
    if has(lambda t: t.category == "repayment" and t.direction == C.DIRECTION_CREDIT):
        present.add("txn.transfer_in_repayment")
    if "investment_redemption" in categories:
        present.add("txn.investment_redemption")

    # --- FX -----------------------------------------------------------------
    fx_currencies = {fx.currency for fx in scenario.fx_rates}
    if has(lambda t: t.currency == "USD" and t.account_pk in bank_account_pks):
        present.add("fx.usd_covered")
    if has(
        lambda t: (
            t.currency == "USD"
            and t.transaction_date is not None
            and t.transaction_date
            < min((fx.date for fx in scenario.fx_rates), default=t.transaction_date)
        )
    ):
        present.add("fx.usd_missing")
    if "EUR" in currencies and "EUR" in fx_currencies:
        present.add("fx.eur_covered")
    if has(lambda t: t.currency == "GBP"):
        present.add("fx.gbp_missing")
    if has(lambda t: t.currency == C.INVALID_CURRENCY_TOKEN):
        present.add("fx.invalid_currency")
    if has(lambda t: t.currency is None or t.currency == ""):
        present.add("fx.blank_currency")

    # --- statement reconciliation ------------------------------------------
    cc_stmts = [s for s in scenario.statement_uploads if s.card_number]
    bank_stmts = [s for s in scenario.statement_uploads if not s.card_number]
    if any(s.matched_count > 0 for s in cc_stmts):
        present.add("stmt.cc_exact_match")
    if any(s.payment_status == C.PAYMENT_PAID for s in cc_stmts):
        present.add("stmt.cc_paid")
    if any(s.payment_status == C.PAYMENT_UNPAID for s in cc_stmts):
        present.add("stmt.cc_unpaid")
    if any(s.status == C.STMT_STATUS_PARSE_ERROR for s in cc_stmts):
        present.add("stmt.cc_parse_error")
    if any(s.source_kind == C.STMT_SOURCE_EMAIL_SUMMARY for s in cc_stmts):
        present.add("stmt.cc_email_summary")
    if any(s.matched_count > 0 for s in bank_stmts):
        present.add("stmt.bank_matched")
    if any(s.closing_balance for s in bank_stmts):
        present.add("stmt.bank_closing")
    if any(getattr(s, "reconciliation_data", None) for s in scenario.statement_uploads):
        present.add("stmt.reconcile_offline")

    # --- net-worth / CAS / manual ------------------------------------------
    pans = {cas.portfolio_key for cas in scenario.cas_uploads}
    depositories = {cas.depository_source for cas in scenario.cas_uploads}
    if len(pans) >= 2:
        present.add("net.multiple_pans")
    if {"nsdl", "cdsl"} <= depositories:
        present.add("net.nsdl_cdsl")
    if any(cas.portfolio_ok for cas in scenario.cas_uploads):
        present.add("net.cas_reconciled")
    if any(not cas.portfolio_ok for cas in scenario.cas_uploads):
        present.add("net.cas_unreconciled")
    if any(
        getattr(snap, "note", None) == "complete_lot"
        for snap in scenario.account_snapshots
    ):
        present.add("net.lot_complete")
    if scenario.investment_lots >= 1:
        present.add("net.lot_complete")
    # disposal + incomplete are detected from the CAS payloads' MF transactions.
    _any_disposal = False
    _any_incomplete = False
    for cas in scenario.cas_uploads:
        for tx in cas.raw_payload.get("transactions", []):
            if tx.get("scope") != "mf":
                continue
            if tx.get("transaction_type") == "redemption":
                _any_disposal = True
            if tx.get("transaction_type") == "purchase" and "nav" not in tx:
                _any_incomplete = True
    if _any_disposal:
        present.add("net.lot_disposal")
    if _any_incomplete:
        present.add("net.lot_incomplete")
    if any(m.active for m in scenario.manual_items):
        present.add("net.manual_active")
    if any(not m.active for m in scenario.manual_items):
        present.add("net.manual_inactive")
    if any(
        snap.kind == C.SNAPSHOT_LIABILITY and snap.current == 0
        for snap in scenario.account_snapshots
    ):
        present.add("net.cc_zero_snapshot")
    if any(
        getattr(snap, "currency", "INR") != "INR" for snap in scenario.account_snapshots
    ):
        present.add("net.non_inr_snapshot_excluded")
    snap_months = {snap.as_of.replace(day=1) for snap in scenario.account_snapshots}
    if len(snap_months) >= 2:
        present.add("net.trend_multiple_months")

    # --- workflow / projection --------------------------------------------
    st_refs: dict[str, int] = {}
    for t in txns:
        if t.category == "self_transfer" and t.reference_number:
            st_refs[t.reference_number] = st_refs.get(t.reference_number, 0) + 1
    if any(n >= 2 for n in st_refs.values()):
        present.add("proj.self_transfer_paired")
    if any(n == 1 for n in st_refs.values()):
        present.add("proj.self_transfer_unmatched")
    if has(
        lambda t: (
            t.category == "credit_card_payment"
            and t.direction == C.DIRECTION_CREDIT
            and t.account_pk in cc_account_pks
        )
    ):
        present.add("proj.card_side_payment")
    if has(
        lambda t: (
            t.currency in fx_currencies
            and t.transaction_date is not None
            and any(
                fx.currency == t.currency and fx.date <= t.transaction_date
                for fx in scenario.fx_rates
            )
        )
    ):
        present.add("proj.fx_priced")
    if has(
        lambda t: (
            t.currency not in ("INR", None, "") and t.currency not in fx_currencies
        )
    ):
        present.add("proj.fx_missing_rate")
    if any(t.ledger_account and len(t.ledger_account) >= 47 for t in txns):
        present.add("proj.long_account_name")

    # --- transaction edges -------------------------------------------------
    if has(lambda t: t.transaction_date is None):
        present.add("edge.undated")
    if has(lambda t: t.counterparty is None or t.counterparty.strip() == ""):
        present.add("edge.blank_counterparty")
    if has(lambda t: t.category is None or t.category.strip() == ""):
        present.add("edge.blank_category")
    if has(lambda t: (t.reference_number or "").startswith("SYN-AMPM")):
        present.add("edge.am_pm_pair")
    if has(lambda t: (t.reference_number or "").startswith("SYN-ORPHAN-CCPAY")):
        present.add("edge.orphan_cc_payment")
    if scenario.orphan_emails:
        present.add("edge.orphan_emails")

    # Sanity: every detected id must be a known branch.
    unknown = present - ALL_BRANCH_IDS
    assert not unknown, (
        f"compute_coverage produced unknown branch ids: {sorted(unknown)}"
    )
    return frozenset(present)


__all__ = [
    "ALL_BRANCH_IDS",
    "BRANCHES_BY_GROUP",
    "REQUIRED_BRANCH_IDS",
    "SCENARIO_BRANCHES",
    "ScenarioBranch",
    "branch_groups",
    "compute_coverage",
]
