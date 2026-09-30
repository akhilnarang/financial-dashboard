from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    CategoryReviewDecision,
    TelegramOutboundDelivery,
    Transaction,
)
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


async def test_reference_pair_supersedes_active_decisions_for_both_legs(
    session: AsyncSession,
):
    debit = _transaction(
        bank="hdfc",
        direction="debit",
        reference_number="PAIR-WITH-REVIEWS",
        account_mask="XX7702",
    )
    credit = _transaction(
        bank="icici",
        direction="credit",
        reference_number="PAIR-WITH-REVIEWS",
        account_mask="XX214",
    )
    session.add_all([debit, credit])
    await session.flush()
    decisions = []
    deliveries = []
    for ordinal, txn in enumerate((debit, credit)):
        decision = CategoryReviewDecision(
            transaction_id=txn.id,
            category_input_hash=f"stale-{ordinal}",
            candidates_json='[{"category":"groceries"}]',
            gate_reason="old review",
        )
        session.add(decision)
        await session.flush()
        delivery = TelegramOutboundDelivery(
            category_review_decision_id=decision.id,
            recipient_chat_id=7,
            ordinal=0,
            transaction_id=txn.id,
            text="choose",
            delivery_token=f"pair-stale-{ordinal}",
            status="pending",
        )
        session.add(delivery)
        decisions.append(decision)
        deliveries.append(delivery)
    await session.flush()

    assert await apply_reference_self_transfer_rule(session, credit)

    assert all(decision.status == "superseded" for decision in decisions)
    assert all(delivery.status == "cancelled" for delivery in deliveries)
    _assert_reference_rule(debit)
    _assert_reference_rule(credit)
    categorized_at = (debit.categorized_at, credit.categorized_at)

    # A second call finds the pair again but must not churn either leg.
    assert await apply_reference_self_transfer_rule(session, credit)
    _assert_reference_rule(debit)
    _assert_reference_rule(credit)
    assert (debit.categorized_at, credit.categorized_at) == categorized_at


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


async def test_fd_counterparty_is_never_self_transfer(session: AsyncSession):
    # A slice FD booking carries a statement reference and could match an
    # opposite-direction row in another account, but an FD is an investment, not
    # a self-transfer. The authoritative rule must skip an FD-labeled row.
    credit = _transaction(
        bank="idfc",
        direction="credit",
        reference_number="900000000001",
        account_mask="XX1234",
    )
    session.add(credit)
    fd = _transaction(
        bank="slice",
        direction="debit",
        reference_number="900000000001",
        account_mask="XX5678",
    )
    fd.counterparty = "Slice FD"
    session.add(fd)
    await session.flush()

    # The FD leg triggers the rule.
    assert await apply_reference_self_transfer_rule(session, fd) is False
    # The other leg triggers the rule. The FD match must be excluded.
    fd.category = "investment"
    fd.category_method = "manual"
    await session.flush()
    assert await apply_reference_self_transfer_rule(session, credit) is False

    assert fd.category == "investment"
    assert fd.category_method == "manual"
    assert credit.category != "self_transfer"


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


async def test_pair_needs_proof_of_different_accounts(session: AsyncSession):
    """Masks with fewer than three digits prove nothing. One linked account
    on both legs is a charge and its refund."""
    short_debit = _transaction(
        bank="icici",
        direction="debit",
        reference_number="SHORT-MASK-REF",
        account_mask="XX1",
    )
    short_credit = _transaction(
        bank="icici",
        direction="credit",
        reference_number="SHORT-MASK-REF",
        account_mask="XX2",
    )
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
    legs = [short_debit, short_credit, charge, refund]
    session.add_all(legs)
    await session.flush()

    assert await apply_reference_self_transfer_rule(session, short_credit) is False
    assert await apply_reference_self_transfer_rule(session, refund) is False
    assert all(leg.category is None for leg in legs)
