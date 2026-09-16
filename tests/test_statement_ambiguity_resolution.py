"""Resolving a statement row the reconciler held back.

A card authorises one amount and settles another, so the same purchase
reaches the DB and the statement as two amounts. The reconciler cannot tell
that from two separate purchases of a similar size, so it holds the statement
row back. These tests drive the real resolver and count rows, because both
wrong answers cost a row: a merge that should have created leaves one
purchase unrecorded, and a create that should have merged stores one twice.
"""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Account, Base, StatementUpload, Transaction
from financial_dashboard.services.statement_ambiguity_resolution import (
    StatementAmbiguityError,
    resolve_statement_ambiguity,
)

pytestmark = pytest.mark.anyio

ACCOUNT_ID = 1
CARD_NUMBER = "4111XXXXXXXX9012"


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


def _entry(*, stmt_idx=0, amount="1,012.00", candidates=(1,), **overrides) -> dict:
    entry = {
        "stmt_idx": stmt_idx,
        "stmt_list": "transactions",
        "date": "07/04/2026",
        "amount": amount,
        "direction": "debit",
        "narration": "MW MERCHANT Pune",
        "card_number": CARD_NUMBER,
        "person": "A PERSON",
        "imported": False,
        "imported_txn_id": None,
        "ambiguous": True,
        "candidate_transaction_ids": list(candidates),
        "candidate_count": len(candidates),
    }
    entry.update(overrides)
    return entry


async def _seed(maker, *, entries, stored=("1000.00",)) -> int:
    """One account, one held-back statement, and one DB row per stored amount."""
    async with maker() as session:
        session.add(
            Account(
                id=ACCOUNT_ID,
                bank="hdfc",
                label="HDFC Credit Card",
                type="credit_card",
                account_number=CARD_NUMBER,
            )
        )
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
                    counterparty="MERCHANT",
                    channel="card",
                    card_mask="9012",
                )
            )
        upload = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="partial_import",
            reconciliation_data=json.dumps({"matched": [], "missing": entries}),
        )
        session.add(upload)
        await session.commit()
        return upload.id


async def _rows(maker) -> list[Transaction]:
    async with maker() as session:
        return list(
            (await session.scalars(select(Transaction).order_by(Transaction.id))).all()
        )


async def test_merge_writes_the_settled_amount_onto_the_stored_row(maker):
    """The statement states what the card really billed, so its amount wins."""
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        result = await resolve_statement_ambiguity(
            session, upload_id, 0, "merge", transaction_id=1
        )

    assert result.status == "merged"
    rows = await _rows(maker)
    assert [(row.id, row.amount) for row in rows] == [(1, Decimal("1012.00"))]


async def test_create_new_imports_the_row_as_its_own_transaction(maker):
    """The two really were different purchases, so both must be stored."""
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        result = await resolve_statement_ambiguity(session, upload_id, 0, "create_new")

    assert result.status == "created"
    rows = await _rows(maker)
    assert [row.amount for row in rows] == [Decimal("1000.00"), Decimal("1012.00")]
    assert rows[1].statement_upload_id == upload_id


async def test_a_second_tap_does_not_write_twice(maker):
    """Telegram redelivers, and a person taps again. Neither may write."""
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        first = await resolve_statement_ambiguity(
            session, upload_id, 0, "merge", transaction_id=1
        )
    async with maker() as session:
        second = await resolve_statement_ambiguity(
            session, upload_id, 0, "merge", transaction_id=1
        )

    assert first.status == "merged"
    assert second.status == "already_resolved"
    assert second.transaction_id == first.transaction_id
    assert [(row.id, row.amount) for row in await _rows(maker)] == [
        (1, Decimal("1012.00"))
    ]


async def test_create_new_after_a_merge_does_not_add_a_row(maker):
    """The other button, tapped on an already-resolved row."""
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        await resolve_statement_ambiguity(
            session, upload_id, 0, "merge", transaction_id=1
        )
    async with maker() as session:
        again = await resolve_statement_ambiguity(session, upload_id, 0, "create_new")

    assert again.status == "already_resolved"
    assert len(await _rows(maker)) == 1


async def test_a_target_outside_the_candidates_is_refused(maker):
    """The button names a candidate. Anything else is a forged callback."""
    upload_id = await _seed(
        maker, entries=[_entry(candidates=(1,))], stored=("1000.00", "1005.00")
    )

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 0, "merge", transaction_id=2
            )

    assert [row.amount for row in await _rows(maker)] == [
        Decimal("1000.00"),
        Decimal("1005.00"),
    ]


async def test_a_target_that_drifted_out_of_the_band_is_refused(maker):
    """The row moved after the prompt was sent, so it is not what was shown.

    Re-reading the band at the write is what makes a stale button safe.
    """
    upload_id = await _seed(maker, entries=[_entry()], stored=("1000.00",))
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.amount = Decimal("500.00")
        await session.commit()

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 0, "merge", transaction_id=1
            )

    assert [row.amount for row in await _rows(maker)] == [Decimal("500.00")]


async def test_a_row_that_was_never_held_back_is_refused(maker):
    """Only a held-back row has a question to answer."""
    upload_id = await _seed(maker, entries=[_entry(ambiguous=False)])

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 0, "merge", transaction_id=1
            )


async def test_an_unknown_statement_row_is_refused(maker):
    """A callback naming an index the statement does not have."""
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 7, "merge", transaction_id=1
            )


async def test_one_answer_does_not_resolve_a_second_held_row(maker):
    """Two held rows, one answered. The other must still await its own."""
    upload_id = await _seed(
        maker,
        entries=[_entry(stmt_idx=0), _entry(stmt_idx=1, amount="1,008.00")],
        stored=("1000.00",),
    )

    async with maker() as session:
        await resolve_statement_ambiguity(session, upload_id, 0, "create_new")

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        recon = json.loads(upload.reconciliation_data)
    held = [entry for entry in recon["missing"] if entry.get("ambiguous")]
    assert [entry["stmt_idx"] for entry in held] == [1]
    assert upload.missing_count == 1
