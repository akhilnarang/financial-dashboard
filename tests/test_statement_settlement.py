"""Folding a recorded statement row into the stored alert for one purchase.

A pump authorises 2,500.00 and the statement bills 2,529.00 for one fill. The
statement row is recorded, and a person folds it into the alert.
"""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Account, Base, StatementUpload, Transaction
from financial_dashboard.services.statement_settlement import (
    SettlementError,
    answer,
    row_digest,
)

pytestmark = pytest.mark.anyio

ACCOUNT_ID = 1
AUTHORISED = "2500.00"
SETTLED = "2,529.00"
NARRATION = "MW SAMPLE FUEL STATION Pune"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


def _row(*, stmt_idx=0, amount=SETTLED, candidates=(1,), recorded=2) -> dict:
    """A held row, as it stands after the statement was imported."""
    return {
        "stmt_idx": stmt_idx,
        "date": "07/04/2026",
        "amount": amount,
        "direction": "debit",
        "narration": NARRATION,
        "card_number": None,
        "imported": True,
        "imported_txn_id": recorded,
        "ambiguous": True,
        "candidate_transaction_ids": list(candidates),
    }


def _recon(rows=None, matched=()) -> dict:
    return {"matched": list(matched), "missing": list(rows or [_row()])}


async def _seed(maker, *, stored=(AUTHORISED,), recon=None) -> int:
    """One account, one imported statement, and one stored row per amount.

    The statement row is in the ledger, because the reconciler imports a held
    row like any other. The stored rows are the card alerts it may state.
    """
    payload = recon or _recon()
    async with maker() as session:
        session.add(Account(id=ACCOUNT_ID, bank="hdfc", label="CC", type="credit_card"))
        for amount in stored:
            session.add(
                Transaction(
                    account_id=ACCOUNT_ID,
                    bank="hdfc",
                    email_type="transaction",
                    direction="debit",
                    amount=Decimal(amount),
                    currency="INR",
                    transaction_date=datetime.date(2026, 4, 7),
                    counterparty="SAMPLE FUEL STATION",
                    channel="card",
                    card_mask="9012",
                )
            )
        upload = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="imported",
            due_date="20/04/2026",
            reconciliation_data=json.dumps(payload),
        )
        session.add(upload)
        await session.flush()

        for entry in payload["missing"]:
            session.add(
                Transaction(
                    id=entry["imported_txn_id"],
                    statement_upload_id=upload.id,
                    account_id=ACCOUNT_ID,
                    bank="hdfc",
                    email_type="cc_statement",
                    direction=entry["direction"],
                    amount=Decimal(entry["amount"].replace(",", "")),
                    currency="INR",
                    transaction_date=datetime.date(2026, 4, 7),
                    counterparty=entry["narration"],
                    channel="cc_statement",
                )
            )
        await session.commit()
        return upload.id


async def _rows(maker) -> list[Transaction]:
    async with maker() as session:
        return list(
            (await session.scalars(select(Transaction).order_by(Transaction.id))).all()
        )


async def test_a_fold_states_what_the_statement_states(maker):
    """One row stays, at the billed amount, date and merchant.

    The alert can be a day earlier and name the merchant more briefly.
    """
    upload_id = await _seed(maker)
    async with maker() as session:
        alert = await session.get(Transaction, 1)
        alert.transaction_date = datetime.date(2026, 4, 6)
        await session.commit()

    async with maker() as session:
        result = await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert result.outcome == "merged"
    [row] = await _rows(maker)
    assert (row.id, row.amount, row.transaction_date, row.counterparty) == (
        1,
        Decimal("2529.00"),
        datetime.date(2026, 4, 7),
        NARRATION,
    )


async def test_only_the_row_the_prompt_showed_is_folded_once(maker):
    """A changed row is refused, and a second tap changes nothing."""
    upload_id = await _seed(maker)

    async with maker() as session:
        changed = await answer(
            session, upload_id, 0, row_digest(_row(amount="9,999.00")), 1
        )
    async with maker() as session:
        first = await answer(session, upload_id, 0, row_digest(_row()), 1)
    async with maker() as session:
        second = await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert (changed.outcome, first.outcome, second.outcome) == (
        "stale",
        "merged",
        "stale",
    )
    assert [row.amount for row in await _rows(maker)] == [Decimal("2529.00")]


async def test_a_row_another_statement_row_holds_is_refused(maker):
    """Folding onto it would put two purchases on one row."""
    claimed = _recon(matched=[{"stmt_idx": 1, "db_txn_id": 1}])
    upload_id = await _seed(maker, recon=claimed)

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2529.00"),
    ]


async def test_a_row_the_prompt_did_not_offer_is_refused(maker):
    """The prompt named its candidates. Any other id is a forged callback."""
    upload_id = await _seed(
        maker, stored=(AUTHORISED, "2512.00"), recon=_recon([_row(recorded=3)])
    )

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row(recorded=3)), 2)

    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2512.00"),
        Decimal("2529.00"),
    ]
