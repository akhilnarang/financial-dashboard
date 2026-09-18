"""Folding a statement row into the stored alert that states the same purchase.

The example is a fuel purchase, because fuel shows the two amounts every time.
The pump authorises the amount on its display and adds a surcharge of about 1%
at settlement, so the card alert states 2,500.00 and the statement 2,529.00 for
one fill.

A statement states a purchase, so the row is in the ledger before anyone is
asked. The question is only whether a stored row states the same purchase. A
person answers by folding the two together. A row nobody folds is a purchase of
its own, and it already stands in the ledger as one.

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


async def _seed(maker, *, stored=(AUTHORISED,), recon=None, currency="INR") -> int:
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


async def _recon_of(maker, upload_id: int) -> dict:
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        return json.loads(upload.reconciliation_data or "{}")


async def test_a_fold_leaves_one_row_at_the_billed_amount(maker):
    """Two rows stating one purchase become one row stating it once.

    The stored row keeps its history and takes the billed amount. The statement
    row goes, because the purchase is already in the ledger under the row that
    survives.
    """
    upload_id = await _seed(maker)

    async with maker() as session:
        result = await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert result.outcome == "merged"
    assert [(row.id, row.amount) for row in await _rows(maker)] == [
        (1, Decimal("2529.00"))
    ]
    assert held_rows(await _recon_of(maker, upload_id)) == []


async def test_an_unanswered_row_stands_as_its_own_purchase(maker):
    """Nobody folds the row, so the purchase stays in the ledger.

    This is the case a person never answers. The statement stated a purchase,
    so the ledger holds it, and no answer is needed to keep it.
    """
    upload_id = await _seed(maker)

    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2529.00"),
    ]
    assert len(held_rows(await _recon_of(maker, upload_id))) == 1


async def test_a_second_tap_changes_nothing(maker):
    """Telegram resends a prompt, and a person taps again."""
    upload_id = await _seed(maker)
    digest = row_digest(_row())

    async with maker() as session:
        first = await answer(session, upload_id, 0, digest, 1)
    async with maker() as session:
        second = await answer(session, upload_id, 0, digest, 1)

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
        result = await answer(session, upload_id, 0, stale_digest, 1)

    assert result.outcome == "stale"
    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2529.00"),
    ]


async def test_a_transaction_another_row_holds_is_refused(maker):
    """A transaction states one purchase.

    A matched row holds the one it won, so folding onto it would put two
    purchases on one ledger row and lose one of them.
    """
    claimed = _recon(matched=[{"stmt_idx": 1, "db_txn_id": 1}])
    upload_id = await _seed(maker, recon=claimed)

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert [row.amount for row in await _rows(maker)] == [
        Decimal(AUTHORISED),
        Decimal("2529.00"),
    ]


async def test_a_transaction_that_was_not_offered_is_refused(maker):
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


async def test_the_statement_row_is_not_its_own_target(maker):
    """Folding a row into itself would delete the purchase."""
    upload_id = await _seed(maker, recon=_recon([_row(candidates=(1, 2))]))

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row(candidates=(1, 2))), 2)

    assert len(await _rows(maker)) == 2


async def test_a_transaction_in_another_currency_is_refused(maker):
    """A statement states rupees, so another unit is not comparable."""
    upload_id = await _seed(maker, currency="USD")

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert (await _rows(maker))[0].amount == Decimal(AUTHORISED)


async def test_a_transaction_that_runs_the_other_way_is_refused(maker):
    """A statement debit cannot state the same purchase as a stored credit."""
    upload_id = await _seed(maker)
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.direction = "credit"
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert (await _rows(maker))[0].amount == Decimal(AUTHORISED)


async def test_a_transaction_that_moved_out_of_the_window_is_refused(maker):
    """A settlement follows its purchase by about a day, not by weeks."""
    upload_id = await _seed(maker)
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.transaction_date = datetime.date(2026, 3, 1)
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert (await _rows(maker))[0].amount == Decimal(AUTHORISED)


async def test_a_fold_keeps_a_one_day_offset_row(maker):
    """A settlement follows its purchase by a day, and that is still one."""
    upload_id = await _seed(maker)
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.transaction_date = datetime.date(2026, 4, 6)
        await session.commit()

    async with maker() as session:
        result = await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert result.outcome == "merged"
    assert [(row.id, row.amount) for row in await _rows(maker)] == [
        (1, Decimal("2529.00"))
    ]


async def test_a_transaction_that_moved_out_of_the_band_is_refused(maker):
    """The stored row changed after the prompt, so it is not what was shown."""
    upload_id = await _seed(maker)
    async with maker() as session:
        row = await session.get(Transaction, 1)
        row.amount = Decimal("900.00")
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert (await _rows(maker))[0].amount == Decimal("900.00")


async def test_a_transaction_on_another_account_is_refused(maker):
    """A statement speaks for one card only."""
    upload_id = await _seed(maker)
    async with maker() as session:
        session.add(Account(id=2, bank="hdfc", label="Other", type="credit_card"))
        row = await session.get(Transaction, 1)
        row.account_id = 2
        await session.commit()

    async with maker() as session:
        with pytest.raises(SettlementError):
            await answer(session, upload_id, 0, row_digest(_row()), 1)

    assert (await _rows(maker))[0].amount == Decimal(AUTHORISED)


async def test_a_statement_that_prints_one_line_twice_folds_each(maker):
    """Two coffees of one price on one day are two purchases.

    Each row is asked about on its own, and folding one must leave the other
    standing.
    """
    two = _recon([_row(stmt_idx=0, recorded=2), _row(stmt_idx=1, recorded=3)])
    upload_id = await _seed(maker, recon=two)

    async with maker() as session:
        result = await answer(session, upload_id, 0, row_digest(two["missing"][0]), 1)

    assert result.outcome == "merged"
    assert [(row.id, row.amount) for row in await _rows(maker)] == [
        (1, Decimal("2529.00")),
        (3, Decimal("2529.00")),
    ]
    assert len(held_rows(await _recon_of(maker, upload_id))) == 1


async def test_folding_a_credit_re_derives_the_paid_state(maker, monkeypatch):
    """A fold rewrites a payment credit, so the paid state must follow."""
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
        row = await session.get(Transaction, 1)
        row.direction = "credit"
        await session.commit()

    async with maker() as session:
        result = await answer(session, upload_id, 0, row_digest(credit), 1)

    assert result.outcome == "merged"
    assert seen == [upload_id]


async def test_a_claimed_transaction_is_not_offered(maker, monkeypatch):
    """A button that always refuses is not worth sending."""
    sent = []

    async def fake_send(payload, chat_id):
        sent.append(payload)

    claimed = _recon(matched=[{"stmt_idx": 1, "db_txn_id": 1}])
    upload_id = await _seed(maker, recon=claimed)

    monkeypatch.setattr(
        "financial_dashboard.services.telegram.send_settlement_prompt", fake_send
    )
    monkeypatch.setattr(
        "financial_dashboard.services.settings.get_telegram_chat_id", lambda: 1
    )
    monkeypatch.setattr("financial_dashboard.db.async_session", maker)

    from financial_dashboard.services.statement_settlement import _send_prompts

    await _send_prompts(upload_id)

    assert sent and sent[0]["candidates"] == []


async def test_the_digest_survives_a_restart():
    """The digest travels to Telegram and comes back, possibly after a restart.

    ``hash()`` is salted per process, so a digest built from it would differ
    and every prompt would be refused.
    """
    import subprocess
    import sys

    script = (
        "from financial_dashboard.services.statement_settlement import row_digest;"
        "print(row_digest({'date': '07/04/2026', 'amount': '2,529.00',"
        " 'direction': 'debit', 'narration': 'X', 'card_number': None}))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }

    assert len(runs) == 1


async def test_the_digest_follows_what_the_row_states():
    """Equal rows give one digest; a changed row gives another."""
    assert row_digest(_row()) == row_digest(_row())
    assert row_digest(_row()) != row_digest(_row(amount="1,000.00"))
