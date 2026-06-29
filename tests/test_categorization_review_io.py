# tests/test_categorization_review_io.py
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.review_io import (
    apply_reviewed_rows,
    build_export_row,
)

pytestmark = pytest.mark.anyio


def test_build_export_row():
    """build_export_row maps a categorized Transaction to the correct dict."""
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("1500.50"),
        currency="INR",
        channel="upi",
        counterparty="BigMart",
        raw_description="Payment to BigMart",
        category="groceries",
        category_method="rule",
        category_confidence=0.9,
        review_reason=None,
    )
    row = build_export_row(txn)

    # suggested and final both pre-filled from txn.category
    assert row["suggested_category"] == "groceries"
    assert row["final_category"] == "groceries"
    # confidence is stringified
    assert row["confidence"] == "0.9"
    assert row["suggested_method"] == "rule"
    assert row["review_reason"] == ""
    assert row["direction"] == "debit"
    assert row["currency"] == "INR"
    assert row["channel"] == "upi"
    assert row["counterparty"] == "BigMart"
    assert row["amount"] == "1500.50"


async def test_apply_reviewed_rows_basic(session: AsyncSession):
    """applied=1 for a valid slug, skipped=1 for blank final_category."""
    txn1 = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("100")
    )
    txn2 = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("200")
    )
    session.add(txn1)
    session.add(txn2)
    await session.flush()

    rows = [
        {
            "id": str(txn1.id),
            "final_category": "Groceries",
            "amount": str(txn1.amount),
            "date": str(txn1.transaction_date or ""),
            "direction": txn1.direction or "",
        },
        {
            "id": str(txn2.id),
            "final_category": "",
            "amount": str(txn2.amount),
            "date": str(txn2.transaction_date or ""),
            "direction": txn2.direction or "",
        },
    ]
    result = await apply_reviewed_rows(session, rows)

    assert result.applied == 1
    assert result.skipped == 1
    assert result.invalid == []
    assert txn1.category == "groceries"
    assert txn1.category_method == "manual"


async def test_apply_reviewed_rows_invalid_slug(session: AsyncSession):
    """Rows with invalid slugs are recorded in invalid, not applied."""
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("100")
    )
    session.add(txn)
    await session.flush()

    rows = [
        {
            "id": str(txn.id),
            "final_category": "123",
            "amount": str(txn.amount),
            "date": str(txn.transaction_date or ""),
            "direction": txn.direction or "",
        }
    ]
    result = await apply_reviewed_rows(session, rows)

    assert result.applied == 0
    assert result.skipped == 0
    assert len(result.invalid) == 1
    assert result.invalid[0] == f"{txn.id}:123"
    # txn must not have been modified
    assert txn.category_method != "manual"


async def test_apply_reviewed_rows_malformed_id(session: AsyncSession):
    """A non-integer id lands in invalid without crashing; other valid rows still apply."""
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("50")
    )
    session.add(txn)
    await session.flush()

    rows = [
        {"id": "not-an-int", "final_category": "groceries"},
        {
            "id": str(txn.id),
            "final_category": "groceries",
            "amount": str(txn.amount),
            "date": str(txn.transaction_date or ""),
            "direction": txn.direction or "",
        },
    ]
    result = await apply_reviewed_rows(session, rows)

    assert result.applied == 1
    assert result.skipped == 0
    assert len(result.invalid) == 1
    assert result.invalid[0] == "not-an-int:groceries"
    assert txn.category == "groceries"


async def test_apply_reviewed_rows_mismatch(session: AsyncSession):
    """A row whose amount/date don't match the DB txn is recorded as mismatch; txn unchanged."""
    txn = Transaction(
        bank="testbank", email_type="x", direction="debit", amount=Decimal("300")
    )
    session.add(txn)
    await session.flush()

    rows = [
        {
            "id": str(txn.id),
            "final_category": "groceries",
            # Wrong amount — does not match txn.amount=300
            "amount": "999.99",
            "date": str(txn.transaction_date or ""),
            "direction": txn.direction or "",
        }
    ]
    result = await apply_reviewed_rows(session, rows)

    assert result.applied == 0
    assert result.skipped == 0
    assert len(result.invalid) == 1
    assert result.invalid[0] == f"{txn.id}:mismatch"
    # txn must not have been modified
    assert txn.category_method != "manual"
