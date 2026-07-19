"""Projection: selection scope, opening balances, account/category mappings,
non-INR skipping, self-transfer/card-payment handling, idempotency, ledger
safety, and the read-only guarantee (no core writes).

Uses the shared in-memory ``session`` fixture so each test gets a fresh DB.
Non-INR is always skipped — there is no ``include`` path.
"""

import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db.enums import (
    SnapshotCategory,
    SnapshotSource,
)
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    Transaction,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.projection import (
    ProjectionError,
    project,
)
from financial_dashboard.services.paisa.renderer import (
    CARD_PAYMENT_CLEARING,
    EQUITY_OPENING,
)

pytestmark = pytest.mark.anyio

CUTOVER = dt.date(2026, 1, 1)


def _config(**overrides) -> PaisaProjectionConfig:
    base = dict(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=(1,),
        cutover_date=CUTOVER,
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
        ledger_cli="ledger",
        fx_rates={},
    )
    base.update(overrides)
    return PaisaProjectionConfig(**base)


async def _bank(session, *, id=1, bank="hdfc", label="Savings", active=True):
    acct = Account(id=id, bank=bank, label=label, type="bank_account", active=active)
    session.add(acct)
    await session.flush()
    return acct


async def _card(session, *, id=2, bank="icici", label="Platinum", active=True):
    acct = Account(id=id, bank=bank, label=label, type="credit_card", active=active)
    session.add(acct)
    await session.flush()
    return acct


async def _txn(
    session,
    *,
    account_id,
    direction,
    amount,
    date,
    category=None,
    currency="INR",
    id=None,
    bank="hdfc",
    counterparty=None,
    reference_number=None,
    balance=None,
):
    kwargs = dict(
        account_id=account_id,
        bank=bank,
        email_type="test_account_transaction",
        direction=direction,
        amount=Decimal(amount),
        currency=currency,
        transaction_date=date,
        category=category,
        counterparty=counterparty,
        reference_number=reference_number,
        balance=Decimal(balance) if balance is not None else None,
    )
    if id is not None:
        kwargs["id"] = id
    session.add(Transaction(**kwargs))
    await session.flush()


async def _snapshot(session, account_id, date, value, *, kind="asset", currency="INR"):
    session.add(
        BalanceSnapshot(
            account_id=account_id,
            kind=kind,
            category=(
                SnapshotCategory.bank_balance.value
                if kind == "asset"
                else SnapshotCategory.cc_outstanding.value
            ),
            as_of_date=date,
            value=Decimal(value),
            source=SnapshotSource.bank_statement.value
            if kind == "asset"
            else SnapshotSource.cc_statement.value,
            currency=currency,
        )
    )
    await session.flush()


# ---------------------------------------------------------------------------
# Selection scope
# ---------------------------------------------------------------------------


async def test_only_selected_accounts_projected(session):
    bank = await _bank(session, id=1)
    other = await _bank(session, id=99, bank="axis", label="Other")
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Store",
    )
    await _txn(
        session,
        account_id=other.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 2),
        category="shopping",
        counterparty="BigSpend",
    )
    report = await project(session, _config(selected_account_ids=(1,)))
    assert "BigSpend" not in report.journal
    assert "Store" in report.journal
    assert report.emitted_count == 1


async def test_only_post_cutover_transactions_emitted(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="50.00",
        date=dt.date(2025, 12, 30),
        category="groceries",
        counterparty="Old",
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="75.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="New",
    )
    report = await project(session, _config())
    assert "Old" not in report.journal
    assert "New" in report.journal
    assert report.emitted_count == 1


async def test_cutover_required(session):
    cfg = _config(cutover_date=None)
    with pytest.raises(ProjectionError):
        await project(session, cfg)


async def test_empty_selection_yields_empty_journal(session):
    report = await project(session, _config(selected_account_ids=()))
    assert report.emitted_count == 0
    assert report.journal == ""


# ---------------------------------------------------------------------------
# Opening balances
# ---------------------------------------------------------------------------


async def test_opening_from_snapshot(session):
    bank = await _bank(session)
    await _snapshot(session, bank.id, dt.date(2025, 12, 15), "100000.00")
    report = await project(session, _config())
    assert "Opening Balances" in report.journal
    assert EQUITY_OPENING in report.journal
    assert "100000.00 INR" in report.journal
    assert report.openings[0].source == "snapshot"


async def test_opening_falls_back_to_transaction_balance(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="5000.00",
        date=dt.date(2025, 12, 20),
        category="salary",
        balance="90000.00",
        counterparty="Employer",
    )
    report = await project(session, _config())
    assert report.openings
    assert report.openings[0].source == "transaction_balance"
    assert report.openings[0].amount == Decimal("90000.00")


async def test_opening_ignores_newer_foreign_snapshot(session):
    """An explicit foreign snapshot can never be relabelled as an INR opening."""
    bank = await _bank(session)
    await _snapshot(session, bank.id, dt.date(2025, 11, 30), "90000.00", currency="INR")
    await _snapshot(session, bank.id, dt.date(2025, 12, 31), "1000.00", currency="USD")

    report = await project(session, _config())

    assert len(report.openings) == 1
    assert report.openings[0].amount == Decimal("90000.00")
    assert report.openings[0].as_of == dt.date(2025, 11, 30)


async def test_foreign_snapshot_uses_only_inr_running_balance_fallback(session):
    bank = await _bank(session)
    await _snapshot(session, bank.id, dt.date(2025, 12, 31), "1000.00", currency="EUR")
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="1",
        date=dt.date(2025, 12, 30),
        category="salary",
        currency="USD",
        balance="1111.00",
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="1",
        date=dt.date(2025, 12, 29),
        category="salary",
        currency="INR",
        balance="88000.00",
    )

    report = await project(session, _config())

    assert report.openings[0].source == "transaction_balance"
    assert report.openings[0].amount == Decimal("88000.00")


async def test_no_inr_snapshot_or_running_balance_emits_no_opening(session):
    bank = await _bank(session)
    await _snapshot(session, bank.id, dt.date(2025, 12, 31), "1000.00", currency="USD")
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="1",
        date=dt.date(2025, 12, 30),
        category="salary",
        currency="USD",
        balance="1111.00",
    )

    report = await project(session, _config())

    assert report.openings == ()


async def test_liability_opening_negated(session):
    card = await _card(session)
    await _snapshot(
        session, card.id, dt.date(2025, 12, 15), "12000.00", kind="liability"
    )
    report = await project(session, _config(selected_account_ids=(2,)))
    # Liability opening lands negative in ledger.
    assert "-12000.00 INR" in report.journal


# ---------------------------------------------------------------------------
# Account & category mappings
# ---------------------------------------------------------------------------


async def test_account_mapping_override(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="X",
    )
    report = await project(
        session,
        _config(account_mappings={"1": "Assets:Bank:Custom:Salary"}),
    )
    assert "Assets:Bank:Custom:Salary" in report.journal


async def test_invalid_account_mapping_raises(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="X",
    )
    # No hierarchy + comment char -> invalid; projection fails loudly.
    with pytest.raises(ProjectionError):
        await project(session, _config(account_mappings={"1": "NoColon; oops"}))


async def test_invalid_category_mapping_raises(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="X",
    )
    with pytest.raises(ProjectionError):
        await project(
            session, _config(category_mappings={"groceries": "Expenses\n:Evil"})
        )


async def test_category_mapping_override(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="X",
    )
    report = await project(
        session,
        _config(category_mappings={"groceries": "Expenses:Food:Groceries"}),
    )
    assert "Expenses:Food:Groceries" in report.journal
    assert "Expenses:Groceries" not in report.journal  # default not used


async def test_default_contra_for_debit_is_expense(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="dining",
        counterparty="Cafe",
    )
    report = await project(session, _config())
    assert "Expenses:Dining" in report.journal


async def test_default_contra_for_credit_is_income(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="50000.00",
        date=dt.date(2026, 2, 1),
        category="salary",
        counterparty="Employer",
    )
    report = await project(session, _config())
    assert "Income:Salary" in report.journal


async def test_bank_asset_card_liability_mapping(session):
    bank = await _bank(session, id=1)
    card = await _card(session, id=2)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="A",
    )
    await _txn(
        session,
        account_id=card.id,
        direction="debit",
        amount="20.00",
        date=dt.date(2026, 2, 2),
        category="groceries",
        counterparty="B",
    )
    report = await project(session, _config(selected_account_ids=(1, 2)))
    assert "Assets:Bank:Hdfc:Savings" in report.journal
    assert "Liabilities:Card:Icici:Platinum" in report.journal


# ---------------------------------------------------------------------------
# Non-INR: ALWAYS skipped (no include path)
# ---------------------------------------------------------------------------


async def test_non_inr_always_skipped(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="USD Txn",
        currency="USD",
    )
    report = await project(session, _config())
    assert report.non_inr_count == 1
    assert report.emitted_count == 0
    assert report.skipped[0].reason == "non_inr"


async def test_non_inr_policy_include_still_skips(session):
    # The policy is skip-only at the projection layer; even if a future config
    # carried "include", a non-INR amount is never emitted labelled INR.
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="USD Txn",
        currency="USD",
    )
    report = await project(session, _config(non_inr_policy="include"))  # type: ignore[arg-type]
    assert report.non_inr_count == 1
    assert report.emitted_count == 0


async def test_non_inr_amount_never_labelled_inr(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="USD Txn",
        currency="USD",
    )
    report = await project(session, _config())
    # The USD amount (10.00) must not appear as "10.00 INR".
    assert "10.00 INR" not in report.journal
    assert "USD" not in report.journal


async def test_null_currency_treated_as_inr(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Legacy",
        currency=None,
    )
    report = await project(session, _config())
    assert report.non_inr_count == 0
    assert report.emitted_count == 1


# ---------------------------------------------------------------------------
# Self-transfer / card payment handling
# ---------------------------------------------------------------------------


async def test_self_transfer_pair_emitted_once(session):
    a = await _bank(session, id=1, bank="hdfc", label="A")
    b = await _bank(session, id=3, bank="axis", label="B")
    ref = "IMPS12345"
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="hdfc",
    )
    await _txn(
        session,
        account_id=b.id,
        direction="credit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="axis",
    )
    report = await project(session, _config(selected_account_ids=(1, 3)))
    assert report.self_transfer_pairs == 1
    assert report.emitted_count == 1
    posting_lines = [
        line
        for line in report.journal.splitlines()
        if line.startswith("    Assets:Bank:Hdfc:A")
        or line.startswith("    Assets:Bank:Axis:B")
    ]
    assert len(posting_lines) == 2
    assert "; txn:1, 2" in report.journal


async def test_unmatched_self_transfer_skipped(session):
    a = await _bank(session, id=1)
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number="LONE",
        bank="hdfc",
    )
    report = await project(session, _config())
    assert report.self_transfer_pairs == 0
    assert report.unmatched_count == 1
    assert report.emitted_count == 0
    assert report.skipped[0].reason == "unmatched_self_transfer"


async def test_self_transfer_non_inr_leg_skipped_before_pairing(session):
    # A non-INR self-transfer leg must obey the skip policy BEFORE pairing, so
    # it cannot drag a foreign amount into an INR-labelled transfer.
    a = await _bank(session, id=1, bank="hdfc", label="A")
    b = await _bank(session, id=3, bank="axis", label="B")
    ref = "WISE999"
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="hdfc",
        currency="USD",
    )
    await _txn(
        session,
        account_id=b.id,
        direction="credit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="axis",
        currency="USD",
    )
    report = await project(session, _config(selected_account_ids=(1, 3)))
    assert report.self_transfer_pairs == 0
    assert report.non_inr_count == 2
    assert report.emitted_count == 0
    # No INR-labelled posting for the USD amount.
    assert "1000.00 INR" not in report.journal


async def test_self_transfer_amount_mismatch_refused(session):
    # A debit/credit pair sharing a reference but differing in magnitude is not
    # a clean transfer — the reference is shared by distinct events. Both legs
    # are refused (never collapsed into one entry) and reported verbatim.
    a = await _bank(session, id=1, bank="hdfc", label="A")
    b = await _bank(session, id=3, bank="axis", label="B")
    ref = "MIS123"
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="hdfc",
        id=1,
    )
    await _txn(
        session,
        account_id=b.id,
        direction="credit",
        amount="999.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="axis",
        id=2,
    )
    report = await project(session, _config(selected_account_ids=(1, 3)))
    assert report.self_transfer_pairs == 0
    assert report.unmatched_count == 2
    assert report.emitted_count == 0
    mismatched = [
        s
        for s in report.skipped
        if s.reason == "unmatched_self_transfer" and "amount mismatch" in s.detail
    ]
    assert len(mismatched) == 2
    assert {s.txn_id for s in mismatched} == {1, 2}
    assert all(f"reference {ref}" in s.detail for s in mismatched)
    # Neither leg's account appears as a posting line (4-space indent);
    # declarations may still name the accounts.
    assert not any(
        line.startswith("    Assets:Bank:Hdfc:A")
        or line.startswith("    Assets:Bank:Axis:B")
        for line in report.journal.splitlines()
    )


async def test_self_transfer_non_clean_multileg_refused(session):
    # A reference shared by 2 debits and 1 credit is not a clean 1+1 pair; the
    # projection refuses to guess which legs go together and reports all three.
    # The two debit legs use distinct banks so they do not collide on the
    # (bank, reference_number, direction) unique index.
    a = await _bank(session, id=1, bank="hdfc", label="A")
    b = await _bank(session, id=3, bank="axis", label="B")
    c = await _bank(session, id=4, bank="kotak", label="C")
    ref = "MULTI123"
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="hdfc",
        id=1,
    )
    await _txn(
        session,
        account_id=b.id,
        direction="debit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="axis",
        id=2,
    )
    await _txn(
        session,
        account_id=c.id,
        direction="credit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="kotak",
        id=3,
    )
    report = await project(session, _config(selected_account_ids=(1, 3, 4)))
    assert report.self_transfer_pairs == 0
    assert report.unmatched_count == 3
    assert report.emitted_count == 0
    multileg = [
        s
        for s in report.skipped
        if s.reason == "unmatched_self_transfer" and "debit(s)" in s.detail
    ]
    assert len(multileg) == 3
    assert {s.txn_id for s in multileg} == {1, 2, 3}
    assert all("2 debit(s), 1 credit(s)" in s.detail for s in multileg)
    assert all(f"reference {ref}" in s.detail for s in multileg)
    # No transfer posting for any of the three legs (4-space indent = posting).
    assert not any(
        line.startswith("    Assets:Bank:Hdfc:A")
        or line.startswith("    Assets:Bank:Axis:B")
        or line.startswith("    Assets:Bank:Kotak:C")
        for line in report.journal.splitlines()
    )


async def test_self_transfer_account_out_of_scope_refused(session):
    # Both legs reference selected accounts, but account 3 has no Account row
    # (metadata missing / out of resolvable scope). _build_self_transfer_entry
    # cannot resolve the credit leg, so the pair is refused and BOTH legs are
    # reported "account not in scope" rather than partially emitted.
    a = await _bank(session, id=1, bank="hdfc", label="A")
    # NOTE: no Account with id=3 is created. account_id=3 IS in the selected
    # set, so its leg is loaded by the projection query, but its metadata
    # cannot be resolved into accounts_by_id.
    ref = "ORPHAN123"
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="hdfc",
        id=1,
    )
    await _txn(
        session,
        account_id=3,
        direction="credit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        bank="axis",
        id=2,
    )
    report = await project(session, _config(selected_account_ids=(1, 3)))
    assert report.self_transfer_pairs == 0
    assert report.unmatched_count == 2
    assert report.emitted_count == 0
    oos = [
        s
        for s in report.skipped
        if s.reason == "unmatched_self_transfer" and "account not in scope" in s.detail
    ]
    assert len(oos) == 2
    assert {s.txn_id for s in oos} == {1, 2}
    assert all(f"reference {ref}" in s.detail for s in oos)
    # The missing account 3 is also surfaced as unknown_account (it is in the
    # selected set but has no row).
    assert any(
        s.reason == "unknown_account" and "3" in s.detail for s in report.skipped
    )
    # The resolvable debit leg was NOT emitted as a posting.
    assert not any(
        line.startswith("    Assets:Bank:Hdfc:A")
        for line in report.journal.splitlines()
    )


async def test_card_payment_on_bank_emitted_as_liability_transfer(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Card Bill",
        bank="hdfc",
    )
    report = await project(session, _config())
    assert report.card_payments == 1
    assert CARD_PAYMENT_CLEARING in report.journal
    # Must NOT be treated as an expense contra.
    assert "Expenses:" not in report.journal


async def test_card_side_payment_not_misposted(session):
    # A credit_card_payment on the CARD account (liability) is the card being
    # paid down. The bank-side leg is the authoritative event; the card-side
    # leg must be skipped, not emitted as a bank payment.
    card = await _card(session, id=2)
    await _txn(
        session,
        account_id=card.id,
        direction="credit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Payment",
        bank="icici",
    )
    report = await project(session, _config(selected_account_ids=(2,)))
    assert report.card_payments == 0
    assert report.card_side_payments == 1
    assert report.emitted_count == 0
    assert any(s.reason == "card_side_payment" for s in report.skipped)
    # Not posted to the generic clearing liability from the card side.
    assert CARD_PAYMENT_CLEARING not in report.journal


async def test_card_spend_and_bank_payment_not_double_counted(session):
    card = await _card(session, id=2)
    bank = await _bank(session, id=1)
    await _txn(
        session,
        account_id=card.id,
        direction="debit",
        amount="500.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Store",
        bank="icici",
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="500.00",
        date=dt.date(2026, 2, 5),
        category="credit_card_payment",
        counterparty="Payment",
        bank="hdfc",
    )
    report = await project(session, _config(selected_account_ids=(1, 2)))
    assert report.emitted_count == 2
    assert report.journal.count("Expenses:Groceries") == 1


# ---------------------------------------------------------------------------
# Idempotency & deterministic ordering
# ---------------------------------------------------------------------------


async def test_projection_is_deterministic(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 5),
        category="groceries",
        counterparty="B",
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="20.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="A",
    )
    r1 = await project(session, _config())
    r2 = await project(session, _config())
    assert r1.journal == r2.journal
    a_idx = r1.journal.index("2026-02-01")
    b_idx = r1.journal.index("2026-02-05")
    assert a_idx < b_idx


async def test_txn_id_comments_present(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="X",
        id=42,
    )
    report = await project(session, _config())
    assert "; txn:42" in report.journal


async def test_unknown_category_counted(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="unknown",
        counterparty="X",
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="20.00",
        date=dt.date(2026, 2, 2),
        category=None,
        counterparty="Y",
    )
    report = await project(session, _config())
    assert report.unknown_count == 2
    assert report.emitted_count == 2  # still emitted, into a suspense contra


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


async def test_projection_writes_nothing_to_core_tables(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="X",
    )
    txn_before = (await session.execute(select(Transaction))).scalars().all()
    acct_before = (await session.execute(select(Account))).scalars().all()
    snap_before = (await session.execute(select(BalanceSnapshot))).scalars().all()

    await project(session, _config())

    txn_after = (await session.execute(select(Transaction))).scalars().all()
    acct_after = (await session.execute(select(Account))).scalars().all()
    snap_after = (await session.execute(select(BalanceSnapshot))).scalars().all()

    assert [t.id for t in txn_before] == [t.id for t in txn_after]
    assert [a.id for a in acct_before] == [a.id for a in acct_after]
    assert snap_before == snap_after
    assert list(session.sync_session.new) == []
    assert list(session.sync_session.dirty) == []


async def test_unknown_selected_account_reported(session):
    report = await project(session, _config(selected_account_ids=(1, 777)))
    assert any(
        s.reason == "unknown_account" and "777" in s.detail for s in report.skipped
    )


# ---------------------------------------------------------------------------
# Multi-currency: priced policy emits foreign commodity + price directives
# ---------------------------------------------------------------------------

from financial_dashboard.services.paisa.config import FxRate  # noqa: E402


def _fx(usd_rates, eur_rates=()):
    """Build an fx_rates dict from ``(date_str, rate_str)`` tuples."""
    out: dict[str, tuple[FxRate, ...]] = {
        "USD": tuple(
            FxRate(date=dt.date.fromisoformat(d), rate=Decimal(r)) for d, r in usd_rates
        )
    }
    if eur_rates:
        out["EUR"] = tuple(
            FxRate(date=dt.date.fromisoformat(d), rate=Decimal(r)) for d, r in eur_rates
        )
    return out


async def test_priced_foreign_txn_emitted_in_own_commodity(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Cafe",
        currency="USD",
    )
    report = await project(
        session,
        _config(
            non_inr_policy="priced",
            fx_rates=_fx((("2026-01-01", "83.00"),)),
        ),
    )
    # Emitted, in USD, never relabelled to INR.
    assert report.emitted_count == 1
    assert report.projected_foreign_count == 1
    assert report.missing_fx_rate_count == 0
    assert "10.00 USD" in report.journal
    assert "-10.00 USD" in report.journal
    # The USD amount is NOT silently labelled INR.
    assert "10.00 INR" not in report.journal
    # A price directive was emitted for the currency on the txn date.
    assert "P 2026-02-01 USD 83.0000 INR" in report.journal
    assert "USD" in report.source_currencies


async def test_priced_foreign_txn_without_rate_skipped_missing_fx(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Cafe",
        currency="USD",
    )
    # A rate exists but only AFTER the txn date → not eligible → missing_fx_rate.
    report = await project(
        session,
        _config(non_inr_policy="priced", fx_rates=_fx((("2026-03-01", "83.00"),))),
    )
    assert report.emitted_count == 0
    assert report.projected_foreign_count == 0
    assert report.missing_fx_rate_count == 1
    assert any(s.reason == "missing_fx_rate" for s in report.skipped)
    # Nothing foreign leaked into the journal.
    assert "USD" not in report.journal


async def test_priced_chooses_latest_rate_on_or_before(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 15),
        category="groceries",
        counterparty="Cafe",
        currency="USD",
    )
    report = await project(
        session,
        _config(
            non_inr_policy="priced",
            fx_rates=_fx((("2026-01-01", "82.00"), ("2026-02-10", "84.5000"))),
        ),
    )
    # Latest on/before Feb 15 is the Feb 10 rate (84.5000), not the Jan one.
    assert "P 2026-02-15 USD 84.5000 INR" in report.journal
    assert "82.0000" not in report.journal


async def test_priced_price_directives_deduped_per_currency_date(session):
    bank = await _bank(session)
    # Two USD txns on the same date → one price directive.
    for i, cp in enumerate(("Cafe", "Store")):
        await _txn(
            session,
            account_id=bank.id,
            direction="debit",
            amount="10.00",
            date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty=cp,
            currency="USD",
            id=100 + i,
        )
    report = await project(
        session,
        _config(non_inr_policy="priced", fx_rates=_fx((("2026-01-01", "83.00"),))),
    )
    assert report.projected_foreign_count == 2
    # Exactly one P directive for USD on 2026-02-01.
    assert report.journal.count("P 2026-02-01 USD 83.0000 INR") == 1


async def test_priced_emits_one_directive_per_distinct_currency(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="USD Cafe",
        currency="USD",
        id=1,
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="EUR Store",
        currency="EUR",
        id=2,
    )
    report = await project(
        session,
        _config(
            non_inr_policy="priced",
            fx_rates=_fx((("2026-01-01", "83.00"),), (("2026-01-01", "90.0000"),)),
        ),
    )
    assert report.projected_foreign_count == 2
    assert set(report.source_currencies) == {"USD", "EUR"}
    assert "P 2026-02-01 USD 83.0000 INR" in report.journal
    assert "P 2026-02-01 EUR 90.0000 INR" in report.journal


async def test_priced_foreign_card_payment_emitted_in_commodity(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Card Bill",
        currency="USD",
    )
    report = await project(
        session,
        _config(non_inr_policy="priced", fx_rates=_fx((("2026-01-01", "83.00"),))),
    )
    assert report.card_payments == 1
    assert report.projected_foreign_count == 1
    assert "5000.00 USD" in report.journal
    assert "P 2026-02-01 USD 83.0000 INR" in report.journal


async def test_priced_self_transfer_same_currency_pair_emitted(session):
    a = await _bank(session, id=1, bank="hdfc", label="A")
    b = await _bank(session, id=3, bank="axis", label="B")
    ref = "WISE999"
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        currency="USD",
        id=1,
    )
    await _txn(
        session,
        account_id=b.id,
        direction="credit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        currency="USD",
        id=2,
    )
    report = await project(
        session,
        _config(
            selected_account_ids=(1, 3),
            non_inr_policy="priced",
            fx_rates=_fx((("2026-01-01", "83.00"),)),
        ),
    )
    assert report.self_transfer_pairs == 1
    assert report.projected_foreign_count == 1
    assert "100.00 USD" in report.journal
    assert "P 2026-02-01 USD 83.0000 INR" in report.journal


async def test_priced_self_transfer_cross_currency_pair_rejected(session):
    # A debit in USD and a credit in EUR sharing a reference is NOT a clean
    # pair — collapsing it would need an FX conversion we do not perform.
    a = await _bank(session, id=1, bank="hdfc", label="A")
    b = await _bank(session, id=3, bank="axis", label="B")
    ref = "CROSS1"
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        currency="USD",
        id=1,
    )
    await _txn(
        session,
        account_id=b.id,
        direction="credit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        counterparty="Me",
        reference_number=ref,
        currency="EUR",
        id=2,
    )
    report = await project(
        session,
        _config(
            selected_account_ids=(1, 3),
            non_inr_policy="priced",
            fx_rates=_fx((("2026-01-01", "83.00"),), (("2026-01-01", "90.0000"),)),
        ),
    )
    assert report.self_transfer_pairs == 0
    assert report.unmatched_count == 2
    assert any(
        s.reason == "unmatched_self_transfer" and "currency mismatch" in s.detail
        for s in report.skipped
    )


async def test_skip_policy_still_skips_non_inr_under_priced_disabled(session):
    # When policy is the default skip, non-INR is skipped as non_inr even if a
    # rate is configured — priced only takes effect under policy=priced.
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Cafe",
        currency="USD",
    )
    report = await project(
        session,
        _config(non_inr_policy="skip", fx_rates=_fx((("2026-01-01", "83.00"),))),
    )
    assert report.non_inr_count == 1
    assert report.projected_foreign_count == 0
    assert report.missing_fx_rate_count == 0
    assert report.emitted_count == 0
    assert "USD" not in report.journal


# ---------------------------------------------------------------------------
# FX currency sanitization: a malformed/control-laced currency is normalized
# to a legal uppercase symbol or skipped with a clear invalid_currency
# diagnostic — never reaching a posting amount or price directive in any
# backend (beancount emits the commodity BARE, so this guard is mandatory).
# ---------------------------------------------------------------------------


async def test_whitespace_laced_currency_normalized_to_legal_symbol(session):
    # Surrounding/internal whitespace and control characters are noise around a
    # valid ISO code; they normalize to the clean uppercase symbol and the row
    # is emitted in its own commodity (priced), never relabelled INR.
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Cafe",
        currency=" \tus\nd ",
    )
    report = await project(
        session,
        _config(non_inr_policy="priced", fx_rates=_fx((("2026-01-01", "83.00"),))),
    )
    assert report.emitted_count == 1
    assert report.projected_foreign_count == 1
    assert "10.00 USD" in report.journal
    assert "P 2026-02-01 USD 83.0000 INR" in report.journal
    assert "USD" in report.source_currencies


async def test_punctuation_laced_currency_normalized_safely(session):
    # A stray punctuation mark is stripped (normalize safely) to the letters,
    # yielding a legal symbol — never an invalid directive.
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Cafe",
        currency="US-D",
    )
    report = await project(
        session,
        _config(non_inr_policy="priced", fx_rates=_fx((("2026-01-01", "83.00"),))),
    )
    assert report.emitted_count == 1
    assert "10.00 USD" in report.journal


@pytest.mark.parametrize("bad", ["123", "1USD", ";", "{}", "€", "   -"])
async def test_unrepresentable_currency_skipped_with_diagnostic(session, bad):
    # A digit-led, empty-after-normalization, or non-ASCII currency cannot be a
    # legal backend commodity; it is skipped as invalid_currency (regardless of
    # policy) and never reaches the journal.
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Cafe",
        currency=bad,
    )
    report = await project(
        session,
        _config(non_inr_policy="priced", fx_rates=_fx((("2026-01-01", "83.00"),))),
    )
    assert report.emitted_count == 0
    assert report.projected_foreign_count == 0
    skipped = [s for s in report.skipped if s.reason == "invalid_currency"]
    assert skipped, f"expected invalid_currency skip for {bad!r}"
    assert repr(bad) in skipped[0].detail
    # No entry and no price directive is generated for the bad currency.
    assert "P " not in report.journal


async def test_control_laced_currency_never_reaches_price_directive(session):
    # The headline safety property: a currency carrying control chars / a
    # semicolon that could corrupt a directive is normalized BEFORE any backend
    # render, so beancount's bare ``price``/posting commodity stays legal and
    # the payload never appears verbatim in the journal.
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Cafe",
        currency="U\nS;D",
    )
    report = await project(
        session,
        _config(non_inr_policy="priced", fx_rates=_fx((("2026-01-01", "83.00"),))),
    )
    # Normalized to the legal ISO code "USD" and emitted in its own commodity.
    assert report.emitted_count == 1
    assert report.projected_foreign_count == 1
    assert "10.00 USD" in report.journal
    assert "P 2026-02-01 USD 83.0000 INR" in report.journal
    # The control/semicolon payload never appears verbatim anywhere.
    assert ";\n" not in report.journal
    assert "\n" not in " ".join(
        ln for ln in report.journal.splitlines() if ln.startswith("    ")
    )
