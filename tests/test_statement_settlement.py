"""A person answers a statement row that settled near a stored amount.

The example is a fuel purchase, because fuel shows the two amounts every time.
The pump authorises the amount on its display and adds a surcharge of about 1%
at settlement, so the card alert states 2,500.00 and the statement states
2,529.00 for one fill.

Nothing in the two rows says whether that is one purchase or two, so the code
asks. These tests drive the real resolver and count rows: a merge that should
have created loses one purchase, and a create that should have merged stores
one twice.
"""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import (
    Account,
    Base,
    StatementRowDecision,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.statement_settlement import (
    SettlementError,
    _carry_answers,
    parse_revision,
    record_pending_decisions,
    resolve_settlement,
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


def _recon(*, candidates=(1,), amount=SETTLED, held=True) -> dict:
    entry = {
        "stmt_idx": 0,
        "date": "07/04/2026",
        "amount": amount,
        "direction": "debit",
        "narration": NARRATION,
        "imported": False,
        "ambiguous": held,
    }
    return {
        "matched": [],
        "missing": [entry],
        "settlement_groups": (
            [{"stmt_idxs": [0], "txn_ids": list(candidates), "held_idxs": [0]}]
            if held
            else []
        ),
    }


async def _seed(maker, *, stored=(AUTHORISED,), currency="INR", recon=None) -> int:
    """One account, one statement, and one stored row per amount."""
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
        payload = recon or _recon()
        upload = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="partial_import",
            # The importer always stores this, and the resolver reads it back
            # under its lock to check the question is still live.
            reconciliation_data=json.dumps(payload),
        )
        session.add(upload)
        await session.flush()
        await record_pending_decisions(session, upload, payload)
        await session.commit()
        return upload.id


async def _decision(maker, upload_id: int) -> StatementRowDecision:
    async with maker() as session:
        return (
            await session.scalars(
                select(StatementRowDecision).where(
                    StatementRowDecision.statement_upload_id == upload_id
                )
            )
        ).one()


async def _rows(maker) -> list[Transaction]:
    async with maker() as session:
        return list(
            (await session.scalars(select(Transaction).order_by(Transaction.id))).all()
        )


async def test_a_held_row_becomes_a_pending_question(maker):
    """The question is recorded with the row copied in, so it outlives a reparse."""
    upload_id = await _seed(maker)

    decision = await _decision(maker, upload_id)

    assert decision.status == "pending"
    assert decision.row_amount == SETTLED
    assert json.loads(decision.candidate_txn_ids or "[]") == [1]


async def test_create_new_stores_the_row_as_its_own_transaction(maker):
    """The two rows are different purchases, so both must be stored."""
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)

    async with maker() as session:
        result = await resolve_settlement(session, decision.id)

    assert result.status == "created"
    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2529.00"),
    ]


async def test_a_reparse_that_moves_the_row_supersedes_the_old_question(maker):
    """``stmt_idx`` is a position within one parse, so a new parse replaces it.

    An answer that named the old ordinal must not be applied to a different
    row, so the pending question of the old parse is retired.
    """
    upload_id = await _seed(maker)

    moved = _recon()
    moved["missing"][0]["amount"] = "9,999.00"

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await record_pending_decisions(session, upload, moved)
        await session.commit()

    async with maker() as session:
        decisions = list(
            (
                await session.scalars(
                    select(StatementRowDecision).order_by(StatementRowDecision.id)
                )
            ).all()
        )

    assert [d.status for d in decisions] == ["superseded", "pending"]
    assert decisions[1].row_amount == "9,999.00"


async def test_an_unchanged_reparse_keeps_the_pending_question(maker):
    """The same rows in the same order are the same parse, so the answer stands."""
    upload_id = await _seed(maker)

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await record_pending_decisions(session, upload, _recon())
        await session.commit()

    async with maker() as session:
        decisions = list((await session.scalars(select(StatementRowDecision))).all())

    assert [d.status for d in decisions] == ["pending"]


async def test_the_revision_token_follows_the_rows():
    """Equal rows give one token; a changed row gives another."""
    assert parse_revision(_recon()) == parse_revision(_recon())
    assert parse_revision(_recon()) != parse_revision(_recon(amount="1,000.00"))


async def test_the_revision_token_survives_a_restart():
    """The token is stored and read back, so it must not change per process.

    ``hash()`` is salted per process. A token built from it would differ after
    a restart, so an unchanged reparse would retire every pending question and
    ask it again, and a tap on the older prompt would be dropped.
    """
    import subprocess
    import sys

    script = (
        "from financial_dashboard.services.statement_settlement import "
        "parse_revision;"
        "print(parse_revision({'matched': [], 'missing': ["
        "{'stmt_idx': 0, 'date': '07/04/2026', 'amount': '2,529.00',"
        " 'direction': 'debit', 'narration': 'X'}]}))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }

    assert len(runs) == 1


async def test_a_created_row_takes_the_card_of_its_statement_row(maker):
    """A statement bills several cards, so the header card is not the row's."""
    with_card = _recon()
    with_card["missing"][0]["card_number"] = "4111XXXXXXXX7777"
    upload_id = await _seed(maker, recon=with_card)
    decision = await _decision(maker, upload_id)

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        upload.card_number = "4111XXXXXXXX9012"
        await session.commit()

    async with maker() as session:
        await resolve_settlement(session, decision.id)

    created = (await _rows(maker))[1]
    assert created.card_mask == "7777"


async def test_a_tap_revalidates_against_the_stored_reconciliation(maker):
    """A reparse can resolve a row without the decision being retired first.

    The importer retires questions it can see, but a tap can arrive while that
    runs. So the resolver reads the reconciliation again under its lock, and a
    row that no longer asks is not answered.
    """
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)

    # Only the stored reconciliation changes: the decision stays pending, the
    # way it would between the importer's write and its retire.
    resolved = _recon()
    resolved["missing"][0]["imported"] = True
    resolved["missing"][0]["ambiguous"] = False
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        upload.reconciliation_data = json.dumps(resolved)
        await session.commit()

    async with maker() as session:
        before = await session.get(StatementRowDecision, decision.id)
    assert before is not None and before.status == "pending"

    async with maker() as session:
        result = await resolve_settlement(session, decision.id)

    assert result.status == "superseded"
    assert len(await _rows(maker)) == 1


async def test_a_reused_question_describes_the_row_of_this_parse(maker):
    """A reparse can change a row while its question waits.

    The question is reused when the parse returns to a shape it held before, so
    it must take the values of the current row, not the ones it was written
    from.
    """
    upload_id = await _seed(maker)

    moved = _recon()
    moved["missing"][0]["card_number"] = "4111XXXXXXXX7777"
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await record_pending_decisions(session, upload, moved)
        await session.commit()

    async with maker() as session:
        live = (
            await session.scalars(
                select(StatementRowDecision).where(
                    StatementRowDecision.status == "pending"
                )
            )
        ).one()

    assert live.row_card_number == "4111XXXXXXXX7777"


async def test_a_carried_answer_comes_from_the_decision(maker):
    """The answers carried across a reparse come from the decision table.

    That table is scoped to the parse, so an ordinal means the same row. The
    stored reconciliation is not the source: it cannot say which answer a
    person gave.
    """
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)
    async with maker() as session:
        await resolve_settlement(session, decision.id)

    fresh = _recon()
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await record_pending_decisions(session, upload, fresh)
        await session.commit()

    entry = fresh["missing"][0]
    assert entry["imported"] is True
    assert entry["imported_txn_id"] is not None


async def test_the_importer_retires_a_question_whose_row_stopped_asking(maker):
    """The importer sees the row settle, so it retires the question there.

    The resolver checks again under its lock, but that is a second line: a
    question nobody can answer should not be left pending in the first place.
    """
    upload_id = await _seed(maker)

    resolved = _recon()
    resolved["missing"][0]["imported"] = True
    resolved["missing"][0]["ambiguous"] = False
    resolved["settlement_groups"] = []

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await record_pending_decisions(session, upload, resolved)
        await session.commit()

    async with maker() as session:
        after = (await session.scalars(select(StatementRowDecision))).one()

    assert after.status == "superseded"


async def test_a_card_change_retires_the_question_of_the_old_card(maker):
    """A row that moved to another card is another row.

    The token covers each row's card, so the parse that moved it is a different
    parse and its questions retire. An old prompt then answers nothing.
    """
    upload_id = await _seed(maker)

    moved = _recon()
    moved["missing"][0]["card_number"] = "4111XXXXXXXX7777"

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await record_pending_decisions(session, upload, moved)
        await session.commit()

    async with maker() as session:
        statuses = sorted(
            row.status
            for row in (await session.scalars(select(StatementRowDecision))).all()
        )

    assert statuses == ["pending", "superseded"]


async def test_the_count_follows_the_carried_answers(maker):
    """A writer counts before the carry runs, so the count is recomputed.

    Otherwise a statement reports an unresolved row it has no question for.
    """
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)
    async with maker() as session:
        await resolve_settlement(session, decision.id)

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        upload.missing_count = 1
        await record_pending_decisions(session, upload, _recon())
        await session.commit()

    async with maker() as session:
        after = await session.get(StatementUpload, upload_id)

    assert after is not None
    assert after.missing_count == 0


async def test_an_answer_naming_a_gone_transaction_is_not_carried(maker):
    """A transaction can vanish without the decision being updated.

    A delete outside this session, or a row removed by hand, leaves the
    decision naming an id that is not there. Carrying it would mark the row
    resolved to nothing, so the row must ask again.
    """
    upload_id = await _seed(maker)
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        decision = (await session.scalars(select(StatementRowDecision))).one()
        decision.status = "created"
        decision.transaction_id = 9999  # never existed
        await session.commit()

        fresh = _recon()
        await record_pending_decisions(session, upload, fresh)
        await session.commit()

    assert fresh["missing"][0].get("imported") is False


async def test_a_second_tap_returns_the_first_answer(maker):
    """Telegram sends the prompt again, and a person taps again."""
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)

    async with maker() as session:
        first = await resolve_settlement(session, decision.id)
    async with maker() as session:
        second = await resolve_settlement(session, decision.id)

    assert first.status == "created"
    assert second.transaction_id == first.transaction_id
    assert len(await _rows(maker)) == 2


async def test_an_answer_shows_on_the_statement(maker):
    """The statement page reads the reconciliation, so an answer must land there."""
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)

    async with maker() as session:
        await resolve_settlement(session, decision.id)

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        recon = json.loads(upload.reconciliation_data or "{}")

    assert recon["missing"][0]["imported"] is True
    assert upload is not None and upload.missing_count == 0


async def test_an_answer_outlives_the_statement_it_came_from(maker):
    """Deleting a statement must not erase what a person decided."""
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)
    async with maker() as session:
        await resolve_settlement(session, decision.id)

    async with maker() as session:
        await session.delete(await session.get(StatementUpload, upload_id))
        await session.commit()

    async with maker() as session:
        kept = await session.get(StatementRowDecision, decision.id)

    assert kept is not None
    assert kept.status == "created"
    assert kept.row_narration == NARRATION


async def test_a_deleted_answer_lets_its_row_ask_again(maker):
    """Deleting the created transaction is how a wrong answer is undone.

    The row must then have a live question, not merely stop being marked
    imported: a retired decision asks nobody.
    """
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)
    async with maker() as session:
        result = await resolve_settlement(session, decision.id)

    async with maker() as session:
        await session.delete(await session.get(Transaction, result.transaction_id))
        await session.commit()

    fresh = _recon()
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await record_pending_decisions(session, upload, fresh)
        await session.commit()

    async with maker() as session:
        again = await session.get(StatementRowDecision, decision.id)

    assert fresh["missing"][0].get("imported") is False
    assert again is not None
    assert again.status == "pending"
    assert again.transaction_id is None


async def test_an_unreadable_row_is_refused(maker):
    """A decision copies what the row stated, and that can be unparseable.

    Importing it would store a row with no amount or no date.
    """
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)
    async with maker() as session:
        stored = await session.get(StatementRowDecision, decision.id)
        stored.row_amount = "not an amount"
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await resolve_settlement(session, decision.id)

    assert len(await _rows(maker)) == 1


async def test_an_unknown_decision_is_refused(maker):
    """A callback can name a decision that is not there."""
    await _seed(maker)

    async with maker() as session:
        with pytest.raises(SettlementError):
            await resolve_settlement(session, 9999)


async def test_the_same_statement_uploaded_twice_imports_once(maker):
    """One statement can be uploaded twice, so one row can ask twice.

    The first answer imported the purchase. The second must report that
    answer, not import the purchase again.
    """
    first_id = await _seed(maker)
    async with maker() as session:
        second = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="again.pdf",
            file_path="/nonexistent/again.pdf",
            status="partial_import",
            reconciliation_data=json.dumps(_recon()),
        )
        session.add(second)
        await session.flush()
        await record_pending_decisions(session, second, _recon())
        await session.commit()

    async with maker() as session:
        decisions = list(
            (
                await session.scalars(
                    select(StatementRowDecision).order_by(StatementRowDecision.id)
                )
            ).all()
        )

    async with maker() as session:
        first = await resolve_settlement(session, decisions[0].id)
    async with maker() as session:
        again = await resolve_settlement(session, decisions[1].id)

    assert first.status == "created"
    assert again.status == "superseded"
    assert again.transaction_id == first.transaction_id
    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2529.00"),
    ]
    assert first_id != decisions[1].statement_upload_id


async def test_an_answer_is_not_carried_onto_a_row_that_matched_it(maker):
    """A reparse can match the created transaction to another statement row.

    That row holds it now, so the answered row must not also claim it: two
    statement rows would report one transaction and the statement would look
    fully reconciled with one row missing.
    """
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)
    async with maker() as session:
        result = await resolve_settlement(session, decision.id)

    # The reparse gives the created transaction to a different row. The rows
    # themselves are unchanged, so the revision is the one the answer was
    # recorded under.
    fresh = _recon()
    revision = parse_revision(fresh)
    fresh["matched"] = [{"stmt_idx": 1, "db_txn_id": result.transaction_id}]
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await _carry_answers(session, upload, fresh, revision)

    assert fresh["missing"][0].get("imported") is False


async def test_a_reused_row_id_is_not_taken_for_the_answer(maker):
    """SQLite gives the id of a deleted row to the next insert.

    An answer naming that id must not be carried onto whatever now holds it:
    the row would be resolved to an unrelated purchase.
    """
    upload_id = await _seed(maker)
    decision = await _decision(maker, upload_id)
    async with maker() as session:
        result = await resolve_settlement(session, decision.id)

    async with maker() as session:
        await session.delete(await session.get(Transaction, result.transaction_id))
        await session.commit()

    # A separate write, so SQLite gives the freed id to the new row.
    async with maker() as session:
        session.add(
            Transaction(
                account_id=ACCOUNT_ID,
                bank="hdfc",
                email_type="transaction",
                direction="debit",
                amount=Decimal("55.00"),
                currency="INR",
                transaction_date=datetime.date(2026, 9, 9),
                counterparty="SOMETHING ELSE",
                channel="card",
            )
        )
        await session.commit()

    fresh = _recon()
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        await _carry_answers(session, upload, fresh, parse_revision(fresh))

    assert fresh["missing"][0].get("imported") is False
