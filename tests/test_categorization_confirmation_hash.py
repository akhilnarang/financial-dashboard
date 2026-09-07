from decimal import Decimal

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.hashing import (
    compute_transaction_confirmation_hash,
)


def _transaction() -> Transaction:
    return Transaction(
        bank="test",
        email_type="transaction",
        direction="debit",
        amount=Decimal("10.00"),
        currency="INR",
        counterparty="Merchant",
        raw_description="Merchant purchase",
        note=None,
        exclude_from_cashflow=False,
        category="dining",
        category_input_hash="classifier-input-v1",
    )


def test_confirmation_hash_changes_for_each_mutable_field():
    txn = _transaction()
    original = compute_transaction_confirmation_hash(txn, "bank_account")

    txn.category = "groceries"
    category_hash = compute_transaction_confirmation_hash(txn, "bank_account")
    txn.category = "dining"
    txn.note = "work lunch"
    note_hash = compute_transaction_confirmation_hash(txn, "bank_account")
    txn.note = None
    txn.exclude_from_cashflow = True
    excluded_hash = compute_transaction_confirmation_hash(txn, "bank_account")

    assert len({original, category_hash, note_hash, excluded_hash}) == 4
