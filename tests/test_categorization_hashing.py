from decimal import Decimal

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.hashing import (
    build_input_payload,
    compute_input_hash,
)


def _txn(**kw):
    base = dict(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("10"),
        currency="INR",
        counterparty="ACME STORE",
        channel="upi",
        raw_description="ACME STORE MUMBAI",
    )
    base.update(kw)
    return Transaction(**base)


def test_hash_is_stable_and_sensitive():
    h1 = compute_input_hash(build_input_payload(_txn(), "bank_account"))
    h2 = compute_input_hash(build_input_payload(_txn(), "bank_account"))
    h3 = compute_input_hash(
        build_input_payload(_txn(direction="credit"), "bank_account")
    )
    # Outputs are not inputs: a stored category must not change the hash.
    h4 = compute_input_hash(build_input_payload(_txn(category="x"), "bank_account"))
    assert h1 == h2 == h4
    assert h1 != h3
    assert len(h1) == 64
