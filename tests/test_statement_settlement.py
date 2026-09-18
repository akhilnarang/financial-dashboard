"""Answering a statement row that settled near a stored amount.

The example is a fuel purchase, because fuel shows the two amounts every time.
The pump authorises the amount on its display and adds a surcharge of about 1%
at settlement, so the card alert states 2,500.00 and the statement 2,529.00 for
one fill.

Nothing in the two rows says whether that is one purchase or two, so a person
answers. Both wrong answers cost a row: a merge that should have been a skip
rewrites a real purchase, and a skip that should have been a merge leaves the
ledger holding one purchase twice.

The question carries itself. The callback names the statement, the row and a
digest of what the row stated, so nothing is looked up from a stored question.
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
    held_rows,
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


def _row(*, stmt_idx=0, amount=SETTLED, candidates=(1,)) -> dict:
    return {
        "stmt_idx": stmt_idx,
        "date": "07/04/2026",
        "amount": amount,
        "direction": "debit",
        "narration": NARRATION,
        "card_number": None,
        "imported": False,
        "ambiguous": True,
        "candidate_transaction_ids": list(candidates),
    }


def _recon(rows=None, matched=()) -> dict:
    return {"matched": list(matched), "missing": list(rows or [_row()])}


async def _seed(maker, *, stored=(AUTHORISED,), recon=None, currency="INR") -> int:
    """One account, one held statement, and one stored row per amount."""
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
                    currency=currency,
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
            reconciliation_data=json.dumps(payload),
        )
        session.add(upload)
        await session.commit()
        return upload.id


async def _rows(maker) -> list[Transaction]:
    async with maker() as session:
        return list(
            (await session.scalars(select(Transaction).order_by(Transaction.id))).all()
        )


async def _recon_of(maker, upload_id: int) -> dict:
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        return json.loads(upload.reconciliation_data or "{}")


async def test_merge_writes_the_settled_amount(maker):
    """The statement states what the bank billed, so its amount wins."""
    upload_id = await _seed(maker)
    digest = row_digest(_row())

    async with maker() as session:
        result = await answer(session, upload_id, 0, digest, "merge", 1)

    assert result.outcome == "merged"
    assert [(row.id, row.amount) for row in await _rows(maker)] == [
        (1, Decimal("2529.00"))
    ]


async def test_skip_records_the_second_purchase(maker):
    """Two purchases of a similar size, so the statement row is a purchase.

    The stored row keeps the amount it was authorised at, and the statement
    row enters the ledger at the amount it was billed at. Leaving it out would
    lose a real purchase.
    """
    upload_id = await _seed(maker)
    digest = row_digest(_row())

    async with maker() as session:
        result = await answer(session, upload_id, 0, digest, "skip")

    assert result.outcome == "skipped"
    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2529.00"),
    ]

    recon = await _recon_of(maker, upload_id)
    assert held_rows(recon) == []
    assert recon["missing"][0]["imported"] is True
    assert recon["missing"][0]["imported_txn_id"] == result.transaction_id


async def test_a_second_tap_changes_nothing(maker):
    """Telegram resends a prompt, and a person taps again."""
    upload_id = await _seed(maker)
    digest = row_digest(_row())

    async with maker() as session:
        first = await answer(session, upload_id, 0, digest, "merge", 1)
    async with maker() as session:
        second = await answer(session, upload_id, 0, digest, "merge", 1)

    assert first.outcome == "merged"
    assert second.outcome == "stale"
    assert [(row.id, row.amount) for row in await _rows(maker)] == [
        (1, Decimal("2529.00"))
    ]


async def test_a_row_that_changed_since_the_prompt_is_refused(maker):
    """A reparse can put another row at this position.

    The digest names what the prompt showed, so the answer applies to that row
    or to none.
    """
    upload_id = await _seed(maker)
    stale_digest = row_digest(_row(amount="9,999.00"))

    async with maker() as session:
        result = await answer(session, upload_id, 0, stale_digest, "merge", 1)

    assert result.outcome == "stale"
    assert [row.amount for row in await _rows(maker)] == [Decimal(AUTHORISED)]


async def test_a_transaction_another_row_holds_is_refused(maker):
    """A transaction answers one statement row.

    A matched row holds the one it won, so merging onto it would put two
    purchases on one ledger row and lose one of them.
    """
    claimed = _recon(matched=[{"stmt_idx": 1, "db_txn_id": 1}])
    upload_id = await _seed(maker, recon=claimed)
    digest = row_digest(_row())

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 1)

    assert [row.amount for row in await _rows(maker)] == [Decimal(AUTHORISED)]


async def test_a_transaction_that_was_not_offered_is_refused(maker):
    """The prompt named its candidates. Any other id is a forged callback."""
    upload_id = await _seed(maker, stored=(AUTHORISED, "2512.00"))
    digest = row_digest(_row())

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 2)

    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2512.00"),
    ]


async def test_a_transaction_that_moved_out_of_the_band_is_refused(maker):
    """The stored row changed after the prompt, so it is not what was shown."""
    upload_id = await _seed(maker)
    digest = row_digest(_row())
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.amount = Decimal("900.00")
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 1)

    assert [row.amount for row in await _rows(maker)] == [Decimal("900.00")]


async def test_a_transaction_in_another_currency_is_refused(maker):
    """A statement states rupees, so another unit is not comparable."""
    upload_id = await _seed(maker, currency="USD")
    digest = row_digest(_row())

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 1)

    row = (await _rows(maker))[0]
    assert row.amount == Decimal(AUTHORISED)
    assert row.currency == "USD"


async def test_a_statement_row_is_not_a_merge_target(maker):
    """A row that came from a statement already states a settled amount.

    A near amount beside it is a second purchase, so merging would overwrite a
    billed amount.
    """
    upload_id = await _seed(maker)
    digest = row_digest(_row())
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.statement_upload_id = upload_id
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 1)

    assert (await _rows(maker))[0].amount == Decimal(AUTHORISED)


async def test_two_identical_rows_are_two_questions(maker):
    """A statement can print one line twice, and mean it.

    The rows differ only in position, so each carries its own answer and one
    answer cannot stand for both.
    """
    two = _recon([_row(stmt_idx=0), _row(stmt_idx=1)])
    upload_id = await _seed(maker, recon=two)
    digest = row_digest(_row())

    async with maker() as session:
        first = await answer(session, upload_id, 0, digest, "skip")
    async with maker() as session:
        second = await answer(session, upload_id, 1, digest, "skip")

    assert first.outcome == "skipped"
    assert second.outcome == "skipped"
    assert held_rows(await _recon_of(maker, upload_id)) == []


async def test_an_answered_row_keeps_the_transaction_it_took(maker):
    """A merge claims its target, so a later row cannot take the same one.

    Two identical rows share a digest, because a digest states what a row
    states and these rows state the same thing. The claim, not the digest,
    keeps the second row off the transaction the first one took.
    """
    two = _recon([_row(stmt_idx=0), _row(stmt_idx=1)])
    upload_id = await _seed(maker, recon=two)
    digest = row_digest(_row())

    async with maker() as session:
        first = await answer(session, upload_id, 0, digest, "merge", 1)
    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 1, digest, "merge", 1)

    assert first.outcome == "merged"
    assert [(row.id, row.amount) for row in await _rows(maker)] == [
        (1, Decimal("2529.00"))
    ]


async def test_a_merge_links_the_row_to_its_statement(maker):
    """A merged row states a billed amount, so it names the statement.

    The link sits on the row, so a reparse cannot drop it. A replayed prompt
    then finds a row that already came from a statement.
    """
    upload_id = await _seed(maker)
    digest = row_digest(_row())

    async with maker() as session:
        await answer(session, upload_id, 0, digest, "merge", 1)
    assert (await _rows(maker))[0].statement_upload_id == upload_id

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        upload.reconciliation_data = json.dumps(_recon())
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 1)

    assert [row.amount for row in await _rows(maker)] == [Decimal("2529.00")]


async def test_one_printed_row_is_answered_once_across_uploads(maker):
    """The same statement can be uploaded twice.

    Each upload asks its own question about the same printed purchase. Two
    answers would write that one purchase onto two stored rows.
    """
    upload_id = await _seed(
        maker,
        stored=(AUTHORISED, "2512.00"),
        recon=_recon([_row(candidates=(1, 2))]),
    )
    async with maker() as session:
        second = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="partial_import",
            reconciliation_data=json.dumps(_recon([_row(candidates=(1, 2))])),
        )
        session.add(second)
        await session.commit()
        second_id = second.id

    digest = row_digest(_row(candidates=(1, 2)))

    async with maker() as session:
        await answer(session, upload_id, 0, digest, "merge", 1)
    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, second_id, 0, digest, "merge", 2)

    assert [row.amount for row in await _rows(maker)] == [
        Decimal("2529.00"),
        Decimal("2512.00"),
    ]


async def test_a_transaction_that_runs_the_other_way_is_refused(maker):
    """A statement debit cannot settle a stored credit."""
    upload_id = await _seed(maker)
    digest = row_digest(_row())
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.direction = "credit"
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 1)

    assert (await _rows(maker))[0].amount == Decimal(AUTHORISED)


async def test_a_transaction_that_moved_out_of_the_window_is_refused(maker):
    """A settlement follows its purchase by about a day, not by weeks."""
    upload_id = await _seed(maker)
    digest = row_digest(_row())
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.transaction_date = datetime.date(2026, 3, 1)
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 1)

    assert (await _rows(maker))[0].amount == Decimal(AUTHORISED)


async def test_a_transaction_on_another_account_is_refused(maker):
    """A statement speaks for one card only."""
    upload_id = await _seed(maker)
    digest = row_digest(_row())
    async with maker() as session:
        session.add(Account(id=2, bank="hdfc", label="Other", type="credit_card"))
        row = await session.get(Transaction, 1)
        row.account_id = 2
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, digest, "merge", 1)

    assert (await _rows(maker))[0].amount == Decimal(AUTHORISED)


async def test_one_printed_row_is_recorded_once_across_uploads(maker):
    """A statement uploaded twice states its purchases once.

    The second upload asks the same question about the same printed row, so
    answering both would record one purchase twice.
    """
    first_id = await _seed(maker)
    async with maker() as session:
        second = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="partial_import",
            reconciliation_data=json.dumps(_recon()),
        )
        session.add(second)
        await session.commit()
        second_id = second.id

    digest = row_digest(_row())
    async with maker() as session:
        await answer(session, first_id, 0, digest, "skip")
    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, second_id, 0, digest, "skip")

    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2529.00"),
    ]


async def test_answering_a_credit_re_derives_the_paid_state(maker, monkeypatch):
    """A credit answer changes what the paid state was derived from."""
    seen = []

    async def fake_resync(session, upload):
        seen.append(upload.id)
        return True

    monkeypatch.setattr(
        "financial_dashboard.services.reminders.resync_tracked_cc_payment_state",
        fake_resync,
    )
    credit = _row()
    credit["direction"] = "credit"
    upload_id = await _seed(maker, recon=_recon([credit]))

    async with maker() as session:
        await answer(session, upload_id, 0, row_digest(credit), "skip")

    assert seen == [upload_id]


async def test_the_digest_follows_what_the_row_states():
    """Equal rows give one digest; a changed row gives another."""
    assert row_digest(_row()) == row_digest(_row())
    assert row_digest(_row()) != row_digest(_row(amount="1,000.00"))
