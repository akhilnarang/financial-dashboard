"""Tests for amount-based CC payment disambiguation.

Covers the public API of financial_dashboard.services.cc_disambiguation.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import (
    Account,
    PaymentStatus,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.cc_disambiguation import (
    find_cc_account_by_total_due,
    is_cc_payment_received_email,
    resolve_cc_payment_account,
    should_auto_reconcile_statement,
)


def _stmt(
    *,
    account_id: int,
    total: str,
    due_date: str = "20/05/2026",
    status: PaymentStatus | None = PaymentStatus.UNPAID,
    paid: Decimal | None = None,
) -> StatementUpload:
    return StatementUpload(
        account_id=account_id,
        bank="indusind",
        filename="x.pdf",
        file_path="/tmp/x.pdf",
        status="imported",
        due_date=due_date,
        total_amount_due=total,
        payment_status=status,
        payment_paid_amount=paid,
    )


async def _seed_three_indusind_ccs(session: AsyncSession) -> tuple[int, int, int]:
    a1 = Account(bank="indusind", type="credit_card", label="IndusInd A", active=True)
    a2 = Account(bank="indusind", type="credit_card", label="IndusInd B", active=True)
    a3 = Account(bank="indusind", type="credit_card", label="IndusInd C", active=True)
    session.add_all([a1, a2, a3])
    await session.flush()
    return a1.id, a2.id, a3.id


# ---------- payment gates ----------


def _bare_txn(**kwargs) -> Transaction:
    return Transaction(
        bank="indusind",
        email_type=kwargs.pop("email_type", "indusind_cc_payment_alert"),
        direction=kwargs.pop("direction", "credit"),
        amount=kwargs.pop("amount", Decimal("133")),
        currency="INR",
        account_id=kwargs.pop("account_id", 7),
    )


def test_payment_gates_accept_linked_bill_payment_credits_only():
    """Each suffix names a real parser shape. A refund or reversal is a
    credit but not a bill payment: it must not mark a statement paid."""
    for email_type in (
        "indusind_cc_payment_alert",
        "icici_cc_upi_payment_alert",
        "hsbc_cc_credit_alert",
        "kotak_cc_payment",
        "kotak_cc_bill_paid",
        "axis_cc_payment_received_alert",
        "slice_cc_bill_paid_alert",
        "slice_cc_repayment_received_alert",
        "sbi_payment_ack",
    ):
        assert is_cc_payment_received_email(email_type) is True, email_type
    for email_type in (
        "icici_cc_reversal",
        "hdfc_cc_refund_alert",
        "icici_cc_transaction_alert",
        None,
        "",
    ):
        assert is_cc_payment_received_email(email_type) is False, email_type
    assert should_auto_reconcile_statement(_bare_txn()) is True
    assert should_auto_reconcile_statement(_bare_txn(direction="debit")) is False
    assert should_auto_reconcile_statement(_bare_txn(account_id=None)) is False


# ---------- find_cc_account_by_total_due ----------


@pytest.mark.anyio
async def test_multiple_matches_returns_none(session):
    """Two open statements share the same total — refuse to guess."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="133.00"),
            _stmt(account_id=b, total="133.00"),
            _stmt(account_id=c, total="500.00"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out is None


@pytest.mark.anyio
async def test_paid_and_undated_statements_are_excluded(session):
    """A PAID statement, one with no due_date and one on another bank are
    not candidates, even when their totals match the payment."""
    a, b, c = await _seed_three_indusind_ccs(session)
    other = Account(bank="hdfc", type="credit_card", label="HDFC", active=True)
    session.add(other)
    await session.flush()
    session.add_all(
        [
            _stmt(account_id=a, total="133.00", status=PaymentStatus.PAID),
            _stmt(account_id=b, total="133.00", due_date=None),
            _stmt(account_id=c, total="133.00"),
            _stmt(account_id=other.id, total="133.00"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out == c


@pytest.mark.anyio
async def test_only_latest_cycle_per_account_is_considered(session):
    """An older cycle whose total matches must NOT win when a newer
    cycle on the same account is already on file with a different
    total — matches the rule in check_payment_received."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            # Account a: older cycle matches, but newer cycle is different.
            _stmt(account_id=a, total="133.00", due_date="20/04/2026"),
            _stmt(account_id=a, total="999.00", due_date="20/05/2026"),
            # Account b: latest cycle matches.
            _stmt(account_id=b, total="133.00", due_date="20/05/2026"),
            # Account c: an unparseable total is skipped, not an error.
            _stmt(account_id=c, total="₹133.00"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out == b


# ---------- resolve_cc_payment_account ----------


async def _txn(
    session: AsyncSession,
    *,
    bank: str = "indusind",
    email_type: str = "indusind_cc_payment_alert",
    direction: str = "credit",
    amount: Decimal | None = Decimal("133"),
    account_id: int | None = None,
) -> Transaction:
    t = Transaction(
        bank=bank,
        email_type=email_type,
        direction=direction,
        amount=amount,
        currency="INR",
        account_id=account_id,
    )
    session.add(t)
    await session.flush()
    return t


@pytest.mark.anyio
async def test_resolver_ignores_rows_that_are_not_maskless_payments(session):
    """Linked, debit, non-payment and zero-amount rows are left untouched.
    Three CCs are on file, so a row that passes the gate gets a prompt."""
    await _seed_three_indusind_ccs(session)
    rows = [
        (_bare_txn(account_id=42), 42),
        (_bare_txn(account_id=None, direction="debit"), None),
        (_bare_txn(account_id=None, email_type="icici_cc_transaction_alert"), None),
        (_bare_txn(account_id=None, amount=Decimal("0")), None),
    ]
    for t, expected_account in rows:
        assert await resolve_cc_payment_account(session, t) is None
        assert t.account_id == expected_account


@pytest.mark.anyio
async def test_resolver_auto_resolves_when_single_candidate(session):
    """No CC for the bank → no-op. A single CC → auto-resolve account_id."""
    t = await _txn(session)
    assert await resolve_cc_payment_account(session, t) is None
    assert t.account_id is None

    only = Account(bank="indusind", type="credit_card", label="solo", active=True)
    session.add(only)
    await session.flush()
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id == only.id


@pytest.mark.anyio
async def test_resolver_amount_match_auto_resolves(session):
    """Multi-CC bank, one matching open statement → auto-resolve to it."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="1,616.00"),
            _stmt(account_id=b, total="133.00"),  # match
            _stmt(account_id=c, total="4,661.00"),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=Decimal("133"))
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id == b


@pytest.mark.anyio
async def test_resolver_returns_prompt_payload_when_no_amount_match(session):
    """Multi-CC, no amount match → prompt payload, account_id stays None."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="1,616.00"),
            _stmt(account_id=b, total="500.00"),
            _stmt(account_id=c, total="4,661.00"),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=Decimal("133"))
    out = await resolve_cc_payment_account(session, t)
    assert t.account_id is None
    assert out is not None
    assert out["txn_id"] == t.id
    assert out["bank"] == "indusind"
    assert out["amount"] == Decimal("133")
    assert set(out["candidate_account_ids"]) == {a, b, c}
    assert set(out["candidate_labels"].keys()) == {a, b, c}


# ---------- resolve: outstanding-match fallback ----------


@pytest.mark.anyio
async def test_resolver_outstanding_match_both_outstanding_same(session):
    """Two cards have identical outstanding and neither matches total_due
    exactly. The resolver must fall through to the prompt."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(
                account_id=a,
                total="2,300.00",
                status=PaymentStatus.PARTIALLY_PAID,
                paid=Decimal("600"),
            ),
            _stmt(
                account_id=b,
                total="3,400.00",
                status=PaymentStatus.PARTIALLY_PAID,
                paid=Decimal("1700"),
            ),
            _stmt(account_id=c, total="0.00", status=PaymentStatus.PAID),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=Decimal("1700"))
    out = await resolve_cc_payment_account(session, t)
    assert t.account_id is None
    assert out is not None


# ---------- resolve: sole-outstanding fallback ----------


@pytest.mark.anyio
async def test_resolver_sole_outstanding(session):
    """Only one card has remaining balance > 0. Payment is less than
    that balance."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(
                account_id=a,
                total="8,347.00",
                status=PaymentStatus.PARTIALLY_PAID,
                paid=Decimal("1347"),
            ),
            _stmt(
                account_id=b,
                total="512.00",
                status=PaymentStatus.PAID,
                paid=Decimal("512"),
            ),
            _stmt(
                account_id=c, total="0.00", status=PaymentStatus.PAID, paid=Decimal("0")
            ),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=Decimal("4000"))
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id == a


@pytest.mark.anyio
async def test_resolver_sole_outstanding_rejects_overpayment(session):
    """Payment is more than the sole outstanding balance. The resolver
    must fall through to the prompt."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(
                account_id=a,
                total="6,200.00",
                status=PaymentStatus.PARTIALLY_PAID,
                paid=Decimal("2200"),
            ),
            _stmt(
                account_id=b,
                total="512.00",
                status=PaymentStatus.PAID,
                paid=Decimal("512"),
            ),
            _stmt(
                account_id=c, total="0.00", status=PaymentStatus.PAID, paid=Decimal("0")
            ),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=Decimal("5000"))
    out = await resolve_cc_payment_account(session, t)
    assert t.account_id is None
    assert out is not None


@pytest.mark.anyio
@pytest.mark.parametrize("amount", [Decimal("2500"), Decimal("4000")])
async def test_resolver_sole_outstanding_skips_untracked(session, amount):
    """An untracked candidate (payment_status=None) blocks the
    sole-outstanding and outstanding-match tiers. The resolver must fall
    through to the prompt."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(
                account_id=a,
                total="6,200.00",
                status=PaymentStatus.PARTIALLY_PAID,
                paid=Decimal("2200"),
            ),
            _stmt(account_id=b, total="819.00", status=None),
            _stmt(
                account_id=c, total="0.00", status=PaymentStatus.PAID, paid=Decimal("0")
            ),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=amount)
    out = await resolve_cc_payment_account(session, t)
    assert t.account_id is None
    assert out is not None


# ---------- resolve: missing-statement guard ----------


@pytest.mark.anyio
async def test_resolver_outstanding_skips_when_candidate_has_no_statement(session):
    """A candidate with no statement makes outstanding-based tiers
    unreliable. The resolver must fall through to the prompt."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(
                account_id=a,
                total="9,400.00",
                status=PaymentStatus.PARTIALLY_PAID,
                paid=Decimal("4400"),
            ),
            _stmt(
                account_id=b,
                total="512.00",
                status=PaymentStatus.PAID,
                paid=Decimal("512"),
            ),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=Decimal("5000"))
    out = await resolve_cc_payment_account(session, t)
    assert t.account_id is None
    assert out is not None
