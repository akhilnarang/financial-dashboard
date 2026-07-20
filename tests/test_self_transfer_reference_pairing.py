from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization import engine
from financial_dashboard.services.categorization.self_transfer import (
    REFERENCE_PAIR_RULESET_VERSION,
    apply_reference_self_transfer_rule,
)
from financial_dashboard.services.txn_merge import merge_transaction

pytestmark = pytest.mark.anyio


def _transaction(
    *,
    bank: str,
    direction: str,
    reference_number: str | None,
    account_id: int | None = None,
    account_mask: str | None = None,
):
    return Transaction(
        account_id=account_id,
        bank=bank,
        email_type=f"{bank}_imps_alert",
        direction=direction,
        amount=Decimal("135541.88"),
        currency="INR",
        reference_number=reference_number,
        account_mask=account_mask,
        channel="imps",
    )


def _assert_reference_rule(txn: Transaction) -> None:
    assert txn.category == "self_transfer"
    assert txn.category_method == "rule"
    assert txn.category_confidence == 1.0
    assert txn.category_model == REFERENCE_PAIR_RULESET_VERSION
    assert txn.category_input_hash is not None
    assert txn.categorized_at is not None
    assert txn.review_status is None
    assert txn.review_reason is None


async def test_ingest_matching_opposite_reference_marks_both_legs(
    session: AsyncSession,
):
    debit = _transaction(
        bank="hdfc",
        direction="debit",
        reference_number="619445758035",
        account_mask="XX7702",
    )
    debit.category = "self_transfer"
    debit.category_method = "rule"
    debit.review_status = "pending"
    debit.review_reason = "old review"
    session.add(debit)
    await session.flush()

    outcome, credit, _ = await merge_transaction(
        session,
        "sms",
        {
            "bank": "icici",
            "email_type": "icici_imps_credit_alert",
            "direction": "credit",
            "amount": Decimal("135541.88"),
            "currency": "INR",
            "reference_number": "619445758035",
            "account_mask": "XX214",
            "channel": "imps",
        },
    )

    assert outcome == "created"
    assert credit is not None
    _assert_reference_rule(debit)
    _assert_reference_rule(credit)


async def test_same_direction_reference_does_not_pair(session: AsyncSession):
    first = _transaction(
        bank="hdfc", direction="debit", reference_number="619445758035"
    )
    second = _transaction(
        bank="icici", direction="debit", reference_number="619445758035"
    )
    session.add_all([first, second])
    await session.flush()

    method = await engine.categorize_one(session, second, use_llm=False)

    assert method == "skip"
    assert first.category is None
    assert second.category is None
    assert second.category_method == "pending_llm"


async def test_categorization_pass_pairs_directly_inserted_legs(
    session: AsyncSession,
):
    debit = _transaction(
        bank="hdfc",
        direction="debit",
        reference_number="619445758035",
        account_id=35,
    )
    credit = _transaction(
        bank="icici",
        direction="credit",
        reference_number="619445758035",
        account_id=4,
    )
    credit.category = "repayment"
    credit.category_method = "llm"
    credit.review_status = "pending"
    session.add_all([debit, credit])
    await session.flush()

    method = await engine.categorize_one(session, debit, use_llm=True)

    assert method == "rule"
    _assert_reference_rule(debit)
    _assert_reference_rule(credit)


async def test_same_account_uber_auth_reversal_is_not_self_transfer(
    session: AsyncSession,
):
    charge = Transaction(
        account_id=4,
        bank="icici",
        email_type="icici_cc_transaction_alert",
        direction="debit",
        amount=Decimal("1.00"),
        currency="INR",
        counterparty="UBER",
        card_mask="XX1234",
        account_mask="xxxxxxxxxx8214",
        reference_number="UBER-AUTH-6276-6280",
        channel="credit_card",
        category="expense",
        category_method="rule",
    )
    session.add(charge)
    await session.flush()

    outcome, reversal, _ = await merge_transaction(
        session,
        "sms",
        {
            "bank": "icici",
            "email_type": "icici_cc_refund_alert",
            "direction": "credit",
            "amount": Decimal("1.00"),
            "currency": "INR",
            "counterparty": "UBER",
            "card_mask": "1234",
            "account_mask": "xxxxxxxxxx8214",
            "reference_number": "UBER-AUTH-6276-6280",
            "channel": "credit_card",
        },
    )

    assert outcome == "created"
    assert reversal is not None
    assert charge.category == "expense"
    assert charge.category_method == "rule"
    assert reversal.category != "self_transfer"
    assert reversal.category_method is None


async def test_apply_reference_self_transfer_rule_idempotent_second_call(
    session: AsyncSession,
):
    """Calling apply_reference_self_transfer_rule twice on the same pair is a
    no-op the second time: both legs stay categorized and the second call
    does not churn category timestamps or re-fire."""
    debit = _transaction(
        bank="hdfc",
        direction="debit",
        reference_number="619445758035",
        account_mask="XX7702",
    )
    credit = _transaction(
        bank="icici",
        direction="credit",
        reference_number="619445758035",
        account_mask="XX214",
    )
    session.add_all([debit, credit])
    await session.flush()

    first = await apply_reference_self_transfer_rule(session, credit)
    assert first is True
    _assert_reference_rule(debit)
    _assert_reference_rule(credit)
    debit_categorized_at = debit.categorized_at
    credit_categorized_at = credit.categorized_at

    # Second call: opposite leg still exists → still returns True, but the
    # already-current guard must leave the categorization untouched (no
    # timestamp churn on either leg).
    second = await apply_reference_self_transfer_rule(session, credit)
    assert second is True
    _assert_reference_rule(debit)
    _assert_reference_rule(credit)
    assert credit.categorized_at == credit_categorized_at
    assert debit.categorized_at == debit_categorized_at


async def test_short_account_masks_are_not_enough_to_prove_different_accounts(
    session: AsyncSession,
):
    debit = _transaction(
        bank="icici",
        direction="debit",
        reference_number="SHORT-MASK-REF",
        account_mask="XX1",
    )
    credit = _transaction(
        bank="icici",
        direction="credit",
        reference_number="SHORT-MASK-REF",
        account_mask="XX2",
    )
    session.add_all([debit, credit])
    await session.flush()

    paired = await apply_reference_self_transfer_rule(session, credit)

    assert paired is False
    assert debit.category is None
    assert credit.category is None


async def test_same_linked_account_id_does_not_pair(session: AsyncSession):
    charge = _transaction(
        bank="icici",
        direction="debit",
        reference_number="SAME-ACCOUNT-REF",
        account_id=4,
    )
    refund = _transaction(
        bank="icici",
        direction="credit",
        reference_number="SAME-ACCOUNT-REF",
        account_id=4,
    )
    session.add_all([charge, refund])
    await session.flush()

    paired = await apply_reference_self_transfer_rule(session, refund)

    assert paired is False
    assert charge.category is None
    assert refund.category is None


def test_transaction_schema_has_reference_lookup_index():
    index_names = {index.name for index in Transaction.__table__.indexes}
    assert "ix_transactions_reference_number" in index_names
