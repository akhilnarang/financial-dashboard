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

from bank_email_fetcher.services.statements.bank import reconcile_bank_statement


@dataclass
class StubDbTxn:
    """Minimal stand-in for a DB ``Transaction`` row."""

    id: int
    transaction_date: datetime.date
    amount: Decimal
    direction: str
    reference_number: str | None = None
    counterparty: str | None = None


def _stmt(
    transactions: list[BankTransaction],
    *,
    opening_balance: str | None = None,
    closing_balance: str | None = None,
) -> ParsedBankStatement:
    return ParsedBankStatement(
        file="test.pdf",
        bank="kotak",
        transactions=transactions,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
    )


def _txn(
    *,
    date: str,
    amount: str,
    direction: Literal["debit", "credit"],
    ref: str | None = None,
    narration: str = "",
) -> BankTransaction:
    return BankTransaction(
        date=date,
        narration=narration,
        amount=amount,
        transaction_type=direction,
        reference_number=ref,
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
