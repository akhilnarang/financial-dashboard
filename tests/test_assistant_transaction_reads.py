import datetime
from decimal import Decimal

import pytest

from financial_dashboard.db.models import CategoryReviewDecision, Transaction
from financial_dashboard.services import settings as settings_service
from financial_dashboard.services.assistant.transaction_reads import (
    MAX_CONTEXT_ROWS,
    get_transaction,
    list_transactions,
)


def _transaction(**overrides: object) -> Transaction:
    values: dict[str, object] = {
        "bank": "HDFC",
        "email_type": "debit",
        "direction": "debit",
        "amount": Decimal("960.00"),
        "transaction_date": datetime.date(2026, 9, 1),
        "counterparty": "PUREBERRYSMUMBAI",
        "category": "groceries",
        "note": "fruit and snacks",
    }
    values.update(overrides)
    return Transaction(**values)


@pytest.mark.anyio
async def test_get_transaction_uses_compact_allowlisted_projection(session):
    settings_service._cache["categorization.hidden_identifiers"] = "Alice"
    transaction = _transaction(
        raw_description="merchant metadata",
        reference_number="123456789",
        note="lunch with Alice",
    )
    session.add(transaction)
    await session.flush()

    result = await get_transaction(session, transaction.id)

    assert result is not None
    assert result.id == transaction.id
    assert result.counterparty == "PUREBERRYSMUMBAI"
    assert result.reference_number == "[redacted-num]"
    assert result.note == "lunch with [redacted-name]"
    assert not hasattr(result, "raw_description")
    assert not hasattr(result, "balance")


@pytest.mark.anyio
async def test_get_transaction_includes_saved_review_gate(session):
    transaction = _transaction(review_status="pending")
    session.add(transaction)
    await session.flush()
    session.add(
        CategoryReviewDecision(
            transaction_id=transaction.id,
            category_input_hash="hash",
            candidates_json="[]",
            gate_reason="confidence 0.42 below 0.60",
        )
    )
    await session.flush()

    result = await get_transaction(session, transaction.id)

    assert result is not None
    assert result.review_gate_reason == "confidence 0.42 below 0.60"


@pytest.mark.anyio
async def test_list_matches_filters_and_exclusion_semantics(session):
    session.add_all(
        [
            _transaction(id=1),
            _transaction(
                id=2,
                account_id=7,
                transaction_date=datetime.date(2026, 8, 1),
                direction="credit",
                category="salary",
                counterparty="ACME",
            ),
            _transaction(id=3, exclude_from_cashflow=True),
        ]
    )
    await session.flush()

    included = await list_transactions(
        session,
        account_id=None,
        date_from=datetime.date(2026, 9, 1),
        direction="debit",
        category="groceries",
        search="berry",
        excluded=False,
    )
    excluded = await list_transactions(session, excluded=True)

    assert [row.id for row in included] == [1]
    assert [row.id for row in excluded] == [3]


@pytest.mark.anyio
async def test_list_caps_model_context_rows(session):
    session.add_all(_transaction(id=index) for index in range(1, 25))
    await session.flush()

    result = await list_transactions(session, limit=MAX_CONTEXT_ROWS + 100)

    assert len(result) == MAX_CONTEXT_ROWS


@pytest.mark.anyio
async def test_missing_transaction_is_none(session):
    assert await get_transaction(session, 404) is None
