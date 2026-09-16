"""Resolution of a statement row that the import held back.

The example is a fuel purchase. Fuel shows the two amounts every time. The
pump authorises the amount on its display. It adds a surcharge of about 1% at
settlement, some days later. The card alert therefore states 2,500.00 and the
statement states 2,529.00 for one fill.

The import cannot tell this from two purchases of a similar size. It holds
the statement row back and asks.

These tests call the real resolver and count the rows. Both wrong answers
cost a row. A merge in place of a create loses one purchase. A create in
place of a merge stores one purchase twice.
"""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Account, Base, StatementUpload, Transaction
from financial_dashboard.services.statements.cc import carry_resolved_answers
from financial_dashboard.services.statement_ambiguity_resolution import (
    StatementAmbiguityError,
    resolve_statement_ambiguity,
)

pytestmark = pytest.mark.anyio

ACCOUNT_ID = 1
CARD_NUMBER = "4111XXXXXXXX9012"

# One fill. The pump authorised the first amount. The statement settled the
# second amount.
AUTHORISED = "2500.00"
SETTLED = "2,529.00"
FUEL_NARRATION = "MW SAMPLE FUEL STATION Pune"


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


def _entry(*, stmt_idx=0, amount=SETTLED, candidates=(1,), **overrides) -> dict:
    entry = {
        "stmt_idx": stmt_idx,
        "stmt_list": "transactions",
        "date": "07/04/2026",
        "amount": amount,
        "direction": "debit",
        "narration": FUEL_NARRATION,
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


async def _seed(maker, *, entries, stored=(AUTHORISED,)) -> int:
    """One account, one held-back statement, and one row for each amount."""
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
    """The statement states what the bank billed. Its amount wins."""
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        result = await resolve_statement_ambiguity(
            session, upload_id, 0, "merge", transaction_id=1
        )

    assert result.status == "merged"
    rows = await _rows(maker)
    assert [(row.id, row.amount) for row in rows] == [(1, Decimal("2529.00"))]


async def test_create_new_imports_the_row_as_its_own_transaction(maker):
    """The two rows are different purchases. The code must store both."""
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        result = await resolve_statement_ambiguity(session, upload_id, 0, "create_new")

    assert result.status == "created"
    rows = await _rows(maker)
    assert [row.amount for row in rows] == [Decimal("2500.00"), Decimal("2529.00")]
    assert rows[1].statement_upload_id == upload_id


async def test_a_second_tap_does_not_write_twice(maker):
    """Telegram sends the prompt again, and a person taps again.

    Neither tap writes a second time.
    """
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
        (1, Decimal("2529.00"))
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
    """The button gives a candidate. A different id is a forged callback."""
    upload_id = await _seed(
        maker, entries=[_entry(candidates=(1,))], stored=(AUTHORISED, "2512.00")
    )

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 0, "merge", transaction_id=2
            )

    assert [row.amount for row in await _rows(maker)] == [
        Decimal("2500.00"),
        Decimal("2512.00"),
    ]


async def test_a_target_that_drifted_out_of_the_band_is_refused(maker):
    """The row changed after the prompt. It is not the row that the person saw.

    The resolver reads the band again at the write. An old button is
    therefore safe.
    """
    upload_id = await _seed(maker, entries=[_entry()], stored=(AUTHORISED,))
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.amount = Decimal("900.00")
        await session.commit()

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 0, "merge", transaction_id=1
            )

    assert [row.amount for row in await _rows(maker)] == [Decimal("900.00")]


async def test_a_row_that_was_never_held_back_is_refused(maker):
    """A held-back row has a question. No other row has one."""
    upload_id = await _seed(maker, entries=[_entry(ambiguous=False)])

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 0, "merge", transaction_id=1
            )


async def test_an_unknown_statement_row_is_refused(maker):
    """The callback gives an index that the statement does not have."""
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 7, "merge", transaction_id=1
            )


async def test_one_answer_does_not_resolve_a_second_held_row(maker):
    """Two held rows. A person answers one. The other still waits."""
    upload_id = await _seed(
        maker,
        entries=[_entry(stmt_idx=0), _entry(stmt_idx=1, amount="2,521.00")],
        stored=(AUTHORISED,),
    )

    async with maker() as session:
        await resolve_statement_ambiguity(session, upload_id, 0, "create_new")

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        recon = json.loads(upload.reconciliation_data)
    held = [entry for entry in recon["missing"] if entry.get("ambiguous")]
    assert [entry["stmt_idx"] for entry in held] == [1]
    assert upload.missing_count == 1


async def test_a_row_with_no_candidate_still_creates(maker):
    """No row can be this purchase. Create must still work."""
    upload_id = await _seed(
        maker, entries=[_entry(candidates=())], stored=(AUTHORISED,)
    )

    async with maker() as session:
        result = await resolve_statement_ambiguity(session, upload_id, 0, "create_new")

    assert result.status == "created"
    assert [row.amount for row in await _rows(maker)] == [
        Decimal("2500.00"),
        Decimal("2529.00"),
    ]


async def test_the_chosen_candidate_is_the_one_written(maker):
    """Two candidates. Only the row that the person chose can change."""
    upload_id = await _seed(
        maker,
        entries=[_entry(candidates=(1, 2))],
        stored=(AUTHORISED, "2512.00"),
    )

    async with maker() as session:
        await resolve_statement_ambiguity(
            session, upload_id, 0, "merge", transaction_id=2
        )

    assert [(row.id, row.amount) for row in await _rows(maker)] == [
        (1, Decimal("2500.00")),
        (2, Decimal("2529.00")),
    ]


async def test_a_merge_keeps_the_amount_the_card_authorised(maker):
    """An alert for this purchase can arrive after the statement.

    That alert states the authorised amount. Duplicate detection uses the
    amount. Without the authorised amount on the row, the alert finds no row
    and stores the purchase a second time.
    """
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        await resolve_statement_ambiguity(
            session, upload_id, 0, "merge", transaction_id=1
        )

    row = (await _rows(maker))[0]
    assert row.amount == Decimal("2529.00")
    assert row.authorised_amount == Decimal("2500.00")


async def test_a_target_that_moved_out_of_the_window_is_refused(maker):
    """The prompt gave a row of the statement day. That row moved after it."""
    upload_id = await _seed(maker, entries=[_entry()])
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.transaction_date = datetime.date(2025, 1, 1)
        await session.commit()

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 0, "merge", transaction_id=1
            )

    assert [row.amount for row in await _rows(maker)] == [Decimal("2500.00")]


async def test_a_merge_refuses_a_row_in_another_currency(maker):
    """A statement states rupees.

    A row in another currency holds a different unit, so the two amounts are
    not comparable. Writing the statement amount would state rupees in that
    unit and lose the real amount.
    """
    upload_id = await _seed(maker, entries=[_entry()], stored=())
    async with maker() as session:
        session.add(
            Transaction(
                account_id=ACCOUNT_ID,
                bank="hdfc",
                email_type="transaction",
                direction="debit",
                amount=Decimal(AUTHORISED),
                currency="USD",
                transaction_date=datetime.date(2026, 4, 7),
                counterparty="FOREIGN SHOP",
                channel="card",
                card_mask="9012",
            )
        )
        await session.commit()

    async with maker() as session:
        with pytest.raises(StatementAmbiguityError):
            await resolve_statement_ambiguity(
                session, upload_id, 0, "merge", transaction_id=1
            )

    row = (await _rows(maker))[0]
    assert row.amount == Decimal(AUTHORISED)
    assert row.currency == "USD"


async def test_a_reparse_keeps_an_answer_given_while_it_ran(maker):
    """A person can answer a held row while a reparse computes.

    The reparse holds a reconciliation that predates the answer. Writing it
    would revive the row: it asks again, and a second answer imports a second
    transaction.
    """
    upload_id = await _seed(maker, entries=[_entry()])

    async with maker() as session:
        await resolve_statement_ambiguity(session, upload_id, 0, "create_new")

    # What the reparse computed before the answer landed.
    stale = {"matched": [], "missing": [_entry()]}
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        carry_resolved_answers(upload.reconciliation_data, stale)

    entry = stale["missing"][0]
    assert entry["imported"] is True
    assert entry["ambiguous"] is False
    assert entry["imported_txn_id"] is not None
