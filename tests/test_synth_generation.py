"""Determinism, profile sizing, Decimal and stable-id tests for the generator.

These run without a database — :mod:`scripts.synth.scenario` and
:mod:`scripts.synth.paisa` are pure, so generation is checked directly.
"""

import datetime
from decimal import Decimal

import pytest

from scripts.synth import build_corpus, build_scenario
from scripts.synth.ids import stable_id, txn_reference
from scripts.synth.scenario import DEFAULT_AS_OF, PROFILES


def test_profiles_exist_and_cover_required_size_bands():
    assert set(PROFILES) >= {"smoke", "ci", "stress"}
    assert PROFILES["smoke"].expected_transaction_floor >= 100  # hundreds
    assert PROFILES["ci"].expected_transaction_floor >= 1000  # several thousand
    assert PROFILES["stress"].expected_transaction_floor >= 200_000  # task floor


def test_smoke_is_hundreds_ci_is_several_thousand():
    smoke = build_scenario(profile="smoke")
    ci = build_scenario(profile="ci")
    assert 100 <= len(smoke.transactions) <= 999
    assert 1000 <= len(ci.transactions) <= 9999


def test_stress_generation_without_loading_hits_floor():
    # Generation only — no DB. This is the "generate stress on demand" path.
    stress = build_scenario(profile="stress")
    assert len(stress.transactions) >= 200_000


def test_same_inputs_produce_identical_scenarios():
    a = build_scenario(seed=4242, as_of=DEFAULT_AS_OF, profile="smoke")
    b = build_scenario(seed=4242, as_of=DEFAULT_AS_OF, profile="smoke")
    assert a.transactions == b.transactions
    assert a.accounts == b.accounts
    assert a.emails == b.emails
    assert a.counts() == b.counts()


def test_different_seed_produces_different_scenarios():
    a = build_scenario(seed=4242, profile="smoke")
    b = build_scenario(seed=9999, profile="smoke")
    assert a.transactions != b.transactions
    # Structure (accounts/cards/categories) is seed-independent in shape...
    assert len(a.accounts) == len(b.accounts)
    # ...but the money/dates differ.
    assert {t.amount for t in a.transactions} != {t.amount for t in b.transactions}


def test_different_profiles_produce_independent_graphs():
    smoke = build_scenario(seed=4242, profile="smoke")
    ci = build_scenario(seed=4242, profile="ci")
    # Same seed, different profile -> different transaction streams.
    assert smoke.transactions[0].stable_id != ci.transactions[0].stable_id


def test_all_amounts_are_decimal_not_float():
    s = build_scenario(profile="smoke")
    for t in s.transactions:
        assert isinstance(t.amount, Decimal)
        if t.balance is not None:
            assert isinstance(t.balance, Decimal)
    for m in s.manual_items:
        assert isinstance(m.value, Decimal)
    for cas in s.cas_uploads:
        assert isinstance(cas.grand_total, Decimal)


def test_amounts_quantized_to_two_decimals():
    s = build_scenario(profile="smoke")
    for t in s.transactions:
        assert t.amount == t.amount.quantize(Decimal("0.01"))


def test_stable_ids_are_deterministic_and_unique_per_kind():
    s = build_scenario(profile="smoke")
    account_ids = {a.stable_id for a in s.accounts}
    txn_ids = {t.stable_id for t in s.transactions}
    # No collisions within a kind...
    assert len(account_ids) == len(s.accounts)
    assert len(txn_ids) == len(s.transactions)
    # ...and UUIDv5-derived hex strings (32 chars).
    assert all(len(sid) == 32 for sid in account_ids)
    assert all(len(sid) == 32 for sid in txn_ids)
    # The same stable_id inputs reproduce the same id.
    assert stable_id("account", "1") == stable_id("account", "1")


def test_reference_numbers_are_stable_and_prefixed():
    sid = stable_id("txn", "x")
    ref = txn_reference(sid)
    assert ref.startswith("SYN-")
    assert txn_reference(sid) == txn_reference(sid)


def test_scenario_covers_every_required_transaction_kind():
    s = build_scenario(profile="smoke")
    email_types = {t.email_type for t in s.transactions}
    channels = {t.channel for t in s.transactions}
    categories = {t.category for t in s.transactions}
    currencies = {t.currency for t in s.transactions}

    # salaries, expenses, refunds/reversals, fees, withdrawals, self-transfers,
    # credit-card purchases and payments, missing refs, backdated, non-INR,
    # unknowns.
    assert any("salary" in et for et in email_types)
    assert "cash_withdrawal" in categories
    assert "self_transfer" in categories
    assert "credit_card_payment" in categories
    assert "fees_charges" in categories
    assert "shopping" in categories  # cc purchases / generic expenses
    # refund (credit) and reversal (credit) families.
    assert "refund" in categories
    assert any("reversal" in et for et in email_types)
    # missing reference (NACH rent + cc payment)
    assert any(t.reference_number is None for t in s.transactions)
    # backdated row (~400 days before as_of)
    assert any(
        t.transaction_date < DEFAULT_AS_OF - datetime.timedelta(days=300)
        for t in s.transactions
    )
    # non-INR
    assert currencies != {"INR"}
    # unknown category + unlinked
    assert "unknown" in categories
    assert any(t.account_pk is None for t in s.transactions)
    # channels the dashboard recognizes
    assert {"upi", "neft"} <= channels


def test_scenario_covers_unmatched_self_transfer_and_card_side_payment():
    # An unmatched self-transfer is a single debit leg whose reference has no
    # matching credit; the projection reports it as unmatched_self_transfer.
    s = build_scenario(profile="smoke")
    st_refs: dict[str, list] = {}
    for t in s.transactions:
        if t.category == "self_transfer" and t.reference_number:
            st_refs.setdefault(t.reference_number, []).append(t)
    assert any(len(legs) == 1 for legs in st_refs.values()), (
        "expected an unmatched (single-leg) self-transfer"
    )
    # The card-side CC payment is a credit on a credit-card account.
    cc_pks = {a.pk for a in s.accounts if a.type == "credit_card"}
    card_side = [
        t
        for t in s.transactions
        if t.category == "credit_card_payment"
        and t.direction == "credit"
        and t.account_pk in cc_pks
    ]
    assert card_side, "expected a card-side credit_card_payment leg"


def test_scenario_exercises_every_seed_category():
    # The full seed category vocabulary must be represented at every scale so
    # the cashflow buckets, the projection contra-accounts and the coverage
    # matrix all see every slug.
    from scripts.synth import constants as C

    for profile in ("golden", "smoke", "ci"):
        s = build_scenario(profile=profile)
        present = {t.category for t in s.transactions}
        missing = sorted(set(C.SEED_CATEGORY_SLUGS) - present)
        assert not missing, f"{profile}: categories never seeded: {missing}"


def test_statement_overlap_dedup_pairs_exist():
    s = build_scenario(profile="smoke")
    paired = [t for t in s.transactions if t.sms_pk is not None]
    assert paired, "expected at least one email+SMS paired (dedup) event"
    # Each paired event shares a dedup_group with itself.
    assert all(t.dedup_group is not None for t in paired)


def test_stale_and_deactivated_entities_present():
    s = build_scenario(profile="smoke")
    assert any(not a.active for a in s.accounts), "expected a deactivated account"
    assert any(not c.active for c in s.cards), "expected a deactivated card"
    assert any(not src.active for src in s.email_sources)
    assert any(not m.active for m in s.manual_items)


def test_cas_and_manual_and_statement_entities_present():
    s = build_scenario(profile="smoke")
    assert len(s.cas_uploads) >= 2
    depositories = {cas.depository_source for cas in s.cas_uploads}
    assert {"nsdl", "cdsl"} <= depositories
    assert len(s.manual_items) >= 5
    assert any(m.kind == "asset" for m in s.manual_items)
    assert any(m.kind == "liability" for m in s.manual_items)
    assert len(s.statement_uploads) >= 2


def test_paisa_corpus_artefacts_are_present_and_deterministic():
    s = build_scenario(profile="smoke")
    corpus_a = build_corpus(s)
    corpus_b = build_corpus(s)
    assert set(corpus_a.artefacts) == {
        "main.ledger",
        "user-authored.ledger",
        "dashboard-generated.ledger",
        "paisa.yaml",
        "prices.ledger",
    }
    for name in corpus_a.artefacts:
        assert corpus_a.artefacts[name] == corpus_b.artefacts[name]


def test_paisa_corpus_caps_ledger_size_for_reviewability():
    s = build_scenario(profile="stress")
    corpus = build_corpus(s)
    # The journal stays hand-reviewable even for a 200k+ txn scenario.
    assert corpus.entries <= 500
    generated = corpus.artefacts["dashboard-generated.ledger"].decode("utf-8")
    # No single artefact blows up into an unreviewable file.
    assert len(generated) < 130_000


@pytest.mark.parametrize("profile", ["golden", "smoke", "ci"])
def test_emails_have_unique_message_ids(profile):
    s = build_scenario(profile=profile)
    ids = [e.message_id for e in s.emails]
    assert len(set(ids)) == len(ids)
