"""Tests for ``reconcile_bank_statement``.

The reconciler matches statement rows against existing DB transactions in
two passes: deterministic ``(reference_number, direction)`` first, then a
reference-aware ``(date ±1, amount, direction)`` fallback. These tests
pin down the matching rules so the greedy-fallback regression — where an
earlier statement row consumed a DB row that a later row owned by ref —
cannot come back.

Reconciliation is a pure function over Pydantic stmt rows + a list of
DB-Transaction-like objects, so we use a lightweight stub instead of a
real DB.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from bank_statement_parser.models import BankTransaction, ParsedBankStatement

from financial_dashboard.services.statements.bank import reconcile_bank_statement


@dataclass
class StubDbTxn:
    """Minimal stand-in for a DB ``Transaction`` row."""

    id: int
    transaction_date: datetime.date
    amount: Decimal
    direction: str
    reference_number: str | None = None
    counterparty: str | None = None
    raw_description: str | None = None
    channel: str | None = None


def _stmt(
    transactions: list[BankTransaction],
    *,
    opening_balance: str | None = None,
    closing_balance: str | None = None,
    account_holder_name: str | None = None,
) -> ParsedBankStatement:
    return ParsedBankStatement(
        file="test.pdf",
        bank="kotak",
        transactions=transactions,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        account_holder_name=account_holder_name,
    )


def _txn(
    *,
    date: str,
    amount: str,
    direction: Literal["debit", "credit"],
    ref: str | None = None,
    narration: str = "",
    channel: str | None = None,
) -> BankTransaction:
    return BankTransaction(
        date=date,
        narration=narration,
        amount=amount,
        transaction_type=direction,
        reference_number=ref,
        channel=channel,
    )


def test_ref_match_takes_priority_over_date_fallback():
    """Regression: a statement row whose ref doesn't appear in the DB must
    not greedily consume (via date+amount fallback) a DB row that a
    later statement row owns by reference.

    Setup mirrors the production failure: db_txn 5257 (14 Apr, ₹2000,
    debit, ref X) plus a statement with two debit ₹2000 rows — first on
    13 Apr with a different ref Y not in the DB, second on 14 Apr with
    ref X. Old reconciler matched the 13 Apr row to db_txn via date
    offset, leaving the 14 Apr row unmatched and triggering a duplicate
    insert.
    """
    db_txn = StubDbTxn(
        id=5257,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("2000.00"),
        direction="debit",
        reference_number="REF-X",
    )

    parsed = _stmt(
        [
            _txn(date="13/04/2026", amount="2,000.00", direction="debit", ref="REF-Y"),
            _txn(date="14/04/2026", amount="2,000.00", direction="debit", ref="REF-X"),
        ]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    matched_refs = {m["reference_number"] for m in recon["matched"]}
    missing_refs = {m["reference_number"] for m in recon["missing"]}

    # The ref-X row matches db_txn 5257 by reference.
    assert matched_refs == {"REF-X"}
    matched_x = next(m for m in recon["matched"] if m["reference_number"] == "REF-X")
    assert matched_x["db_txn_id"] == 5257

    # The ref-Y row stays missing (DB has nothing with that ref, and the
    # date-fallback candidate has a different non-null ref).
    assert missing_refs == {"REF-Y"}


def test_fallback_refuses_match_when_refs_disagree():
    """A statement row with ref A must not fuzzy-match a DB row with ref
    B even when date/amount/direction line up. They cannot be the same
    logical transaction."""
    db_txn = StubDbTxn(
        id=10,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("500.00"),
        direction="debit",
        reference_number="REF-A",
    )

    parsed = _stmt(
        [_txn(date="14/04/2026", amount="500.00", direction="debit", ref="REF-B")]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    assert recon["matched"] == []
    assert len(recon["missing"]) == 1
    assert recon["missing"][0]["reference_number"] == "REF-B"


def test_fallback_matches_when_db_row_has_no_ref():
    """Email-derived DB rows often lack a parsed reference number, but
    the statement row has one. Date+amount+direction fallback should
    still allow this to match."""
    db_txn = StubDbTxn(
        id=11,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("750.00"),
        direction="debit",
        reference_number=None,
    )

    parsed = _stmt(
        [_txn(date="14/04/2026", amount="750.00", direction="debit", ref="REF-Z")]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    assert len(recon["matched"]) == 1
    assert recon["matched"][0]["db_txn_id"] == 11


def test_fallback_matches_when_stmt_row_has_no_ref():
    """If the statement row has no ref but date+amount+direction line
    up with an unconsumed DB row, that's a valid fallback match."""
    db_txn = StubDbTxn(
        id=12,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("250.00"),
        direction="credit",
        reference_number="REF-Q",
    )

    parsed = _stmt(
        [_txn(date="14/04/2026", amount="250.00", direction="credit", ref=None)]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    assert len(recon["matched"]) == 1
    assert recon["matched"][0]["db_txn_id"] == 12


def test_ref_with_opposite_direction_does_not_collide():
    """UPI refunds reuse the same ref with the opposite direction. The
    ref pool keys on ``(ref, direction)``, so a debit and credit sharing
    a ref do not match each other."""
    db_debit = StubDbTxn(
        id=20,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("100.00"),
        direction="debit",
        reference_number="UPI-1",
    )
    parsed = _stmt(
        [_txn(date="14/04/2026", amount="100.00", direction="credit", ref="UPI-1")]
    )

    recon = reconcile_bank_statement(parsed, [db_debit], account_id=1)

    # The credit stmt row cannot match the debit DB row even though they
    # share a ref. No date-fallback either: directions differ.
    assert recon["matched"] == []
    assert len(recon["missing"]) == 1


def test_unparseable_amount_lands_in_missing():
    """Rows whose amount/date can't be parsed go straight to missing
    without consuming any DB candidate."""
    db_txn = StubDbTxn(
        id=30,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("100.00"),
        direction="debit",
        reference_number=None,
    )
    parsed = _stmt(
        [
            _txn(date="not-a-date", amount="100.00", direction="debit", ref=None),
            _txn(date="14/04/2026", amount="100.00", direction="debit", ref=None),
        ]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    # The valid second row should still match the DB candidate; the
    # garbage row stays in missing.
    assert len(recon["matched"]) == 1
    assert len(recon["missing"]) == 1
    assert recon["matched"][0]["db_txn_id"] == 30


def test_db_ref_inside_stmt_narration_matches_despite_ref_disagreement():
    """Some banks (e.g. slice) embed the UPI/IMPS UTR inside the
    statement narration but report a different bank-internal ref in the
    ref column. The email-derived DB row carries the UTR as
    ``reference_number``. When the DB ref appears verbatim inside the
    statement narration we treat it as the same logical transaction
    even though both rows have refs that disagree.

    Example shape: db row with ref=100200300400 (the UTR) vs statement
    row with ref=20990428180878701 (bank-internal txn id) and narration
    ``UPI-Credit-100200300400-Sample Payer-...``. Old reconciler
    refused; new reconciler accepts on UTR-substring evidence.
    """
    db_txn = StubDbTxn(
        id=6710,
        transaction_date=datetime.date(2026, 4, 28),
        amount=Decimal("650.00"),
        direction="credit",
        reference_number="100200300400",
        counterparty="Sample Payer",
        raw_description="You have received Rs.650 via UPI in your savings account xx1234",
        channel="upi",
    )

    parsed = _stmt(
        [
            _txn(
                date="28/04/2026",
                amount="650.00",
                direction="credit",
                ref="20990428180878701",
                narration="UPI-Credit-100200300400-Sample Payer-KARB-sample.payer-2@okaxis-amazon prime",
                channel="upi",
            )
        ]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    assert len(recon["matched"]) == 1, recon
    assert recon["matched"][0]["db_txn_id"] == 6710


def test_stmt_ref_inside_db_raw_description_matches_despite_ref_disagreement():
    """Symmetric direction: if the statement's reference number appears
    inside the DB row's raw narration, that's evidence too. (Less common
    in practice, but the rule is symmetric so we cover it.)"""
    db_txn = StubDbTxn(
        id=42,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("100.00"),
        direction="credit",
        reference_number="UTR-FROM-EMAIL",
        raw_description="Credited via UPI ref STMT-INTERNAL-XYZ from John Doe",
        channel="upi",
    )

    parsed = _stmt(
        [
            _txn(
                date="14/04/2026",
                amount="100.00",
                direction="credit",
                ref="STMT-INTERNAL-XYZ",
                narration="UPI Credit-John Doe-...",
                channel="upi",
            )
        ]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    assert len(recon["matched"]) == 1, recon
    assert recon["matched"][0]["db_txn_id"] == 42


def test_imps_with_only_account_holder_name_overlap_is_refused():
    """Two IMPS self-transfers on the same day with same amount and
    differing refs (because email and statement use different identifier
    namespaces) must NOT auto-merge on token overlap alone — the only
    overlapping tokens would be the account holder's own name, which is
    not distinctive evidence. Better to false-split (manual cleanup) than
    false-merge (silent data loss)."""
    db_txn = StubDbTxn(
        id=5277,
        transaction_date=datetime.date(2026, 4, 15),
        amount=Decimal("200000.00"),
        direction="credit",
        reference_number="700200300400",
        counterparty="Mr Sample Holder",
        raw_description="You have received Rs.2,00,000 via IMPS in your savings a/c xx1234",
        channel="imps",
    )

    parsed = _stmt(
        [
            _txn(
                date="15/04/2026",
                amount="200000.00",
                direction="credit",
                ref="89901234567890",
                narration="IMPS Credit-Mr Sample Holder-ABCD0009 999-XX5678-999900000001-Selftransfer",
                channel="imps",
            )
        ],
        account_holder_name="Sample Holder Account",
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    # Refused — the only token overlap is the account holder's own name,
    # which is excluded from the distinctive-token set. Falls into missing
    # so the human can resolve it manually via cleanup.
    assert recon["matched"] == [], recon
    assert len(recon["missing"]) == 1


def test_upi_token_overlap_with_distinctive_counterparty_matches():
    """When both refs differ, channel is UPI on both sides, and the
    counterparties share at least one distinctive token (not the account
    holder, not a banking stopword), accept the match. Covers UPI cases
    where the UTR happens to be missing from the narration but the
    counterparty name is unambiguous."""
    db_txn = StubDbTxn(
        id=99,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("500.00"),
        direction="credit",
        reference_number="UTR-EMAIL",
        counterparty="SAMPLE MERCHANT",
        raw_description="UPI credit from SAMPLE MERCHANT",
        channel="upi",
    )

    parsed = _stmt(
        [
            _txn(
                date="14/04/2026",
                amount="500.00",
                direction="credit",
                ref="STMT-INTERNAL",
                narration="UPI Credit-SAMPLE MERCHANT-sample.merchant014@okaxis-Payment via wallet",
                channel="upi",
            )
        ],
        account_holder_name="Sample Holder Account",
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    assert len(recon["matched"]) == 1, recon
    assert recon["matched"][0]["db_txn_id"] == 99


def test_disagreeing_refs_require_exact_date_no_offset():
    """When both rows have refs and they differ, ±1 day fuzzy is too
    permissive — it doubles the collision window after we've already
    overridden the strong negative signal of mismatched refs. Require
    exact date in that path. (Pure ref-less rows still get ±1.)"""
    db_txn = StubDbTxn(
        id=200,
        transaction_date=datetime.date(2026, 4, 13),
        amount=Decimal("1000.00"),
        direction="debit",
        reference_number="DB-REF",
        raw_description="UPI debit ref STMT-REF to MERCHANT",
        channel="upi",
    )

    # Statement row dated 14 Apr (1 day off) with a different ref. Even
    # though ref-substring evidence exists (STMT-REF in db raw_desc), we
    # require exact date when refs disagree.
    parsed = _stmt(
        [
            _txn(
                date="14/04/2026",
                amount="1000.00",
                direction="debit",
                ref="STMT-REF",
                narration="UPI Debit MERCHANT",
                channel="upi",
            )
        ]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    assert recon["matched"] == [], recon
    assert len(recon["missing"]) == 1


def test_ambiguous_multiple_compatible_candidates_refused():
    """If two unmatched DB rows on the same date+amount+direction both
    pass the compatibility filter (e.g. same channel, both could plausibly
    be the stmt row's counterpart), we must not silently pick the first
    one. Refuse and leave the stmt row in missing so the human/operator
    can resolve."""
    db_a = StubDbTxn(
        id=301,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("1000.00"),
        direction="debit",
        reference_number=None,
        counterparty="MERCHANT A",
        raw_description="UPI to MERCHANT A",
        channel="upi",
    )
    db_b = StubDbTxn(
        id=302,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("1000.00"),
        direction="debit",
        reference_number=None,
        counterparty="MERCHANT B",
        raw_description="UPI to MERCHANT B",
        channel="upi",
    )

    # Stmt row has no ref either; date+amount+direction matches both.
    # Without distinguishing evidence, we must refuse.
    parsed = _stmt(
        [
            _txn(
                date="14/04/2026",
                amount="1000.00",
                direction="debit",
                ref=None,
                narration="UPI Debit to unknown",
                channel="upi",
            )
        ]
    )

    recon = reconcile_bank_statement(parsed, [db_a, db_b], account_id=1)

    assert recon["matched"] == [], recon
    assert len(recon["missing"]) == 1


def test_ambiguous_disambiguates_by_distinctive_narration():
    """When two DB rows are candidates but only one has its UTR
    appearing in the stmt narration, we should pick that unique one
    rather than refusing — narration evidence breaks the tie."""
    db_a = StubDbTxn(
        id=401,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("500.00"),
        direction="credit",
        reference_number="UTR-A",
        counterparty="ALICE",
        raw_description="UPI from ALICE",
        channel="upi",
    )
    db_b = StubDbTxn(
        id=402,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("500.00"),
        direction="credit",
        reference_number="UTR-B",
        counterparty="BOB",
        raw_description="UPI from BOB",
        channel="upi",
    )

    parsed = _stmt(
        [
            _txn(
                date="14/04/2026",
                amount="500.00",
                direction="credit",
                ref="STMT-XYZ",
                narration="UPI-Credit-UTR-A-ALICE-something",
                channel="upi",
            )
        ]
    )

    recon = reconcile_bank_statement(parsed, [db_a, db_b], account_id=1)

    assert len(recon["matched"]) == 1, recon
    assert recon["matched"][0]["db_txn_id"] == 401


def test_stmt_ref_db_noref_fuzzy_date_still_matches():
    """Regression: when stmt has a ref but the DB candidate has no ref,
    there is no ref *disagreement* — so the ±1 day timezone fuzz that
    we keep for ref-less rows should still apply. Earlier draft of the
    fix gated on ``stmt_ref`` alone and accidentally locked this case
    to exact-date, breaking matches that the old code (and the
    real-world dataset) relied on."""
    db_txn = StubDbTxn(
        id=12,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("250.00"),
        direction="credit",
        reference_number=None,  # email parser missed the ref
        counterparty="ALICE",
    )

    parsed = _stmt(
        [
            _txn(
                date="15/04/2026",  # +1 day off (timezone slop)
                amount="250.00",
                direction="credit",
                ref="STMT-REF",
                narration="UPI Credit-ALICE-ref STMT-REF",
                channel="upi",
            )
        ]
    )

    recon = reconcile_bank_statement(parsed, [db_txn], account_id=1)

    assert len(recon["matched"]) == 1, recon
    assert recon["matched"][0]["db_txn_id"] == 12


def test_disagreeing_refs_with_fuzzy_date_refused_even_with_substring_evidence():
    """Once both refs differ AND the date is off, substring/token evidence
    is no longer enough — refuse. (Already covered by an earlier test
    for offset 1; this case checks that the per-candidate enforcement
    still kicks in even when the call site allowed ±1 because some
    candidate in the date pool has no ref.)"""
    db_with_ref = StubDbTxn(
        id=200,
        transaction_date=datetime.date(2026, 4, 13),
        amount=Decimal("1000.00"),
        direction="debit",
        reference_number="DB-REF",
        raw_description="UPI debit ref STMT-REF to MERCHANT",
        channel="upi",
    )
    db_without_ref = StubDbTxn(
        id=201,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("1000.00"),
        direction="debit",
        reference_number=None,
        counterparty="OTHER MERCHANT",
        raw_description="UPI debit OTHER MERCHANT",
        channel="upi",
    )

    parsed = _stmt(
        [
            _txn(
                date="14/04/2026",
                amount="1000.00",
                direction="debit",
                ref="STMT-REF",
                narration="UPI Debit MERCHANT",
                channel="upi",
            )
        ]
    )

    # The exact-date candidate (db 201) has no ref → compatible. The
    # ±1 day candidate (db 200) has a different ref + substring evidence,
    # but ±1 with both-refs-disagree should be refused. Result: db 201
    # matches; db 200 does not.
    recon = reconcile_bank_statement(parsed, [db_with_ref, db_without_ref], account_id=1)

    assert len(recon["matched"]) == 1, recon
    assert recon["matched"][0]["db_txn_id"] == 201


def test_substring_match_requires_word_boundary_and_min_length():
    """``stmt_ref in cand_raw`` is too permissive: a short or non-bounded
    ref can be an accidental substring of unrelated content (a date, a
    longer txn id, a phone number). The substring rule must require
    word-boundary alignment and a minimum length to count as evidence."""

    # Case A: stmt_ref is a 4-digit number that happens to appear inside
    # an unrelated longer numeric string in the DB raw_description.
    # Should NOT count as a match.
    db_short = StubDbTxn(
        id=500,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("100.00"),
        direction="debit",
        reference_number="OTHER-REF",
        raw_description="UPI debit txn id 12345678901234 something",
        channel="upi",
    )
    parsed_short = _stmt(
        [
            _txn(
                date="14/04/2026",
                amount="100.00",
                direction="debit",
                ref="3456",  # appears in 12345678901234 but not at word boundary
                narration="some narration",
                channel="upi",
            )
        ]
    )
    recon = reconcile_bank_statement(parsed_short, [db_short], account_id=1)
    assert recon["matched"] == [], (
        "short ref appearing as embedded substring must not count as evidence"
    )

    # Case B: stmt_ref is a 12-digit id that appears at a word boundary
    # in the DB raw_description. SHOULD count as a match.
    db_bounded = StubDbTxn(
        id=501,
        transaction_date=datetime.date(2026, 4, 14),
        amount=Decimal("100.00"),
        direction="debit",
        reference_number="OTHER-REF",
        raw_description="UPI debit ref 123456789012 to MERCHANT",
        channel="upi",
    )
    parsed_bounded = _stmt(
        [
            _txn(
                date="14/04/2026",
                amount="100.00",
                direction="debit",
                ref="123456789012",
                narration="some narration",
                channel="upi",
            )
        ]
    )
    recon = reconcile_bank_statement(parsed_bounded, [db_bounded], account_id=1)
    assert len(recon["matched"]) == 1, recon


def test_balance_verification_runs_when_balances_present():
    parsed = _stmt(
        [_txn(date="14/04/2026", amount="100.00", direction="debit", ref=None)],
        opening_balance="1,000.00",
        closing_balance="900.00",
    )
    parsed.debit_total = "100.00"
    parsed.credit_total = "0.00"

    recon = reconcile_bank_statement(parsed, [], account_id=1)

    bv = recon["balance_verification"]
    assert bv is not None
    assert bv["is_balanced"] is True
    assert bv["delta"] == "0.00"
