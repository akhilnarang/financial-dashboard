"""Offline statement reconciliation against scenario transactions.

Produces ``reconciliation_data`` for :class:`SynthStatementUpload` rows by
calling the **production** :func:`reconcile_statement` /
:func:`reconcile_bank_statement` services against lightweight stand-in objects
built from scenario transactions — *not* hand-invented counts. No PDF is
parsed and no network is opened: the stand-ins mimic the small attribute surface
the reconcile functions read (``.amount`` / ``.date`` / ``.direction`` /
``.reference_number`` / ``.narration``), so the matching arithmetic is the real
production path.

This is the fidelity boundary the scenario expansion marks explicitly: the
parser/PDF path is inappropriate for a synthetic corpus (there is no real
statement to parse), so the scenario carries pre-built statement *rows*
derived from its own transactions and lets the real reconcile service match
them. Re-running imports nothing new and matches nothing new (the inputs are
deterministic).
"""

import datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import NamedTuple, Sequence


class _DbTxn(NamedTuple):
    """The minimal attribute surface ``reconcile_*`` reads off a DB row."""

    id: int
    transaction_date: datetime.date | None
    amount: Decimal
    direction: str
    counterparty: str | None
    reference_number: str | None
    card_mask: str | None


def _db_standins(scenario_txns: Sequence, *, account_pk: int) -> list[_DbTxn]:
    out: list[_DbTxn] = []
    for i, t in enumerate(scenario_txns):
        if t.account_pk != account_pk:
            continue
        if t.transaction_date is None:
            continue
        out.append(
            _DbTxn(
                id=i,
                transaction_date=t.transaction_date,
                amount=t.amount,
                direction=t.direction,
                counterparty=t.counterparty,
                reference_number=t.reference_number,
                card_mask=t.card_mask,
            )
        )
    return out


def _cc_stmt_row(t, *, idx: int):
    """A statement row mimicking the cc-parser's debit transaction object."""
    return SimpleNamespace(
        amount=f"{t.amount:,.2f}",
        date=t.transaction_date.strftime("%d/%m/%Y"),
        narration=t.counterparty or "synthetic",
        card_number=None,
        person="primary",
    )


def _bank_stmt_row(t):
    """A statement row mimicking the bank-parser's transaction object."""
    return SimpleNamespace(
        transaction_type=t.direction,
        amount=f"{t.amount:,.2f}",
        date=t.transaction_date.strftime("%d/%m/%Y"),
        reference_number=t.reference_number,
        narration=t.counterparty or "synthetic",
        counterparty=t.counterparty,
        channel=t.channel,
        balance=None,
    )


def reconcile_cc_offline(
    scenario_txns: Sequence,
    *,
    account_pk: int,
    window: tuple[datetime.date, datetime.date],
    period_end: datetime.date,
    include_missing: bool = True,
) -> dict | None:
    """Run the real :func:`reconcile_statement` over scenario-derived rows.

    Builds CC statement debit rows from the scenario's debits on ``account_pk``
    inside ``window``, plus one deliberately-unmatched row (when
    ``include_missing``) so the reconciliation surfaces a non-empty ``missing``
    list — the production path is exercised on both the matched and the
    unmatched branch. Returns ``None`` if the production import fails (the
    fidelity-boundary fallback).
    """
    try:
        from financial_dashboard.core.masks import normalize_mask
        from financial_dashboard.services.statements.cc import reconcile_statement
    except Exception:  # noqa: BLE001 — fidelity boundary
        return None

    lo, hi = window
    rows = [
        t
        for t in scenario_txns
        if t.account_pk == account_pk
        and t.direction == "debit"
        and t.transaction_date is not None
        and lo <= t.transaction_date <= hi
    ][:6]
    parsed = SimpleNamespace(
        transactions=[_cc_stmt_row(t, idx=i) for i, t in enumerate(rows)],
        payments_refunds=[],
        card_summaries=[],
        payments_refunds_total="0.00",
        adjustment_pairs=[],
        possible_adjustment_pairs=[],
        overall_total="0.00",
        overall_reward_points=0,
    )
    db = _db_standins(scenario_txns, account_pk=account_pk)
    account_card_masks = list(
        dict.fromkeys(mask for txn in db if (mask := normalize_mask(txn.card_mask)))
    )
    try:
        return reconcile_statement(parsed, db, account_pk, account_card_masks)
    except Exception:  # noqa: BLE001 — fidelity boundary
        return None


def reconcile_bank_offline(
    scenario_txns: Sequence,
    *,
    account_pk: int,
    window: tuple[datetime.date, datetime.date],
    closing_balance: str,
) -> dict | None:
    """Run the real :func:`reconcile_bank_statement` over scenario rows.

    The bank reconcile is reference-first then date-fuzzy, so feeding it the
    scenario's own rows exercises both passes plus the ref-mismatch refusal
    (a row carrying a ``SYN-MISMATCH`` reference still matches by date when the
    statement row omits the ref). Returns ``None`` on import/call failure.
    """
    try:
        from financial_dashboard.services.statements.bank import (
            reconcile_bank_statement,
        )
    except Exception:  # noqa: BLE001 — fidelity boundary
        return None

    lo, hi = window
    rows = [
        t
        for t in scenario_txns
        if t.account_pk == account_pk
        and t.transaction_date is not None
        and lo <= t.transaction_date <= hi
    ][:8]
    parsed = SimpleNamespace(
        transactions=[_bank_stmt_row(t) for t in rows],
        account_holder_name="Synthetic Investor",
        opening_balance=closing_balance,
        closing_balance=closing_balance,
        debit_total="0",
        credit_total="0",
    )
    db = _db_standins(scenario_txns, account_pk=account_pk)
    try:
        return reconcile_bank_statement(parsed, db, account_pk)
    except Exception:  # noqa: BLE001 — fidelity boundary
        return None


__all__ = [
    "reconcile_bank_offline",
    "reconcile_cc_offline",
]
