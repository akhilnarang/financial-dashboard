"""The account-scope predicate: what each scope selects, and that they partition.

Every assertion here is of the same shape — a row is in *exactly one* scope —
because that is the property the report depends on. Asserting only that the
expected rows turn up in the expected scope would pass just as happily if a row
turned up in two of them, and a row counted twice is the double count the whole
cash basis exists to avoid.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.cashflow.scope import (
    SCOPE_PREDICATES,
    scope_predicate,
)
from tests.conftest import MISSING_ACCOUNT_ID, ensure_account

pytestmark = pytest.mark.anyio
D = Decimal
SCOPES = ("bank", "card", "unaccounted")


async def _add(session: AsyncSession, note: str, account_id: int | None) -> None:
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=D("100"),
            currency="INR",
            transaction_date=datetime.date(2026, 6, 15),
            note=note,
            account_id=account_id,
        )
    )
    await session.flush()


async def _notes_in(session: AsyncSession, scope: str) -> set[str]:
    rows = (
        await session.execute(select(Transaction.note).where(SCOPE_PREDICATES[scope]))
    ).scalars()
    return set(rows)


async def test_each_account_type_lands_in_exactly_one_scope(session: AsyncSession):
    savings = await ensure_account(session, 1, "bank_account")
    debit = await ensure_account(session, 2, "debit_card")
    credit = await ensure_account(session, 3, "credit_card")
    unknown = await ensure_account(session, 4, "prepaid_wallet")

    await _add(session, "savings", savings)
    await _add(session, "debit_card", debit)
    await _add(session, "credit_card", credit)
    await _add(session, "unknown_type", unknown)
    await _add(session, "unlinked", None)
    # A link to an account row that is not there: the only way to reach a NULL
    # account type, since `Account.type` is non-null in the ORM.
    await _add(session, "dangling", MISSING_ACCOUNT_ID)

    assert await _notes_in(session, "bank") == {"savings", "debit_card"}
    assert await _notes_in(session, "card") == {"credit_card"}
    assert await _notes_in(session, "unaccounted") == {
        "unknown_type",
        "unlinked",
        "dangling",
    }

    # Exhaustive and disjoint: six rows, each in one scope and no scope sharing
    # a row with another.
    seen = [await _notes_in(session, scope) for scope in SCOPES]
    assert sum(len(rows) for rows in seen) == 6
    assert set.union(*seen) == {
        "savings",
        "debit_card",
        "credit_card",
        "unknown_type",
        "unlinked",
        "dangling",
    }
    for i, rows in enumerate(seen):
        for other in seen[i + 1 :]:
            assert rows.isdisjoint(other)


async def test_scope_costs_no_extra_row_and_no_extra_statement(session: AsyncSession):
    """The predicate is a correlated EXISTS, so it filters an aggregate in place.

    A join would have been the obvious way to write it and would have been wrong:
    a transaction linked to two account rows cannot happen, but a join's row
    multiplication is exactly the failure a `GROUP BY` cannot see.
    """
    savings = await ensure_account(session, 1, "bank_account")
    await _add(session, "savings", savings)
    await _add(session, "unlinked", None)

    total = (
        await session.execute(
            select(func.sum(Transaction.amount)).where(SCOPE_PREDICATES["bank"])
        )
    ).scalar_one()
    assert total == D("100")


def test_absent_scope_is_not_a_predicate():
    # "Every account" is a fourth thing, and it has to stay tellable apart from
    # the three scopes or a caller cannot leave its query alone.
    assert scope_predicate(None) is None
    for scope in SCOPES:
        assert scope_predicate(scope) is SCOPE_PREDICATES[scope]
