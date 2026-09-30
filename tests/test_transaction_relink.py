"""Tests for the manual relink endpoint POST /api/transactions/{id}/relink.

Use case: a Transaction was originally orphaned (account_id=None) by
the ingestion pipeline — typically a maskless CC bill-payment where
the user has multiple CCs at the same bank and the auto-resolver
gave up. The operator picks the right account/card from the UI; the
endpoint sets the FKs and fires check_payment_received when the
transaction is a CC bill-payment credit, so the matching statement
auto-marks paid in one click.
"""

import datetime
from decimal import Decimal

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.core.deps as core_deps
import financial_dashboard.services.reminders as reminders_module
from financial_dashboard.api import get_router as get_api_router
from financial_dashboard.core.deps import get_session
from financial_dashboard.db import (
    Account,
    Card,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus
from tests.conftest import new_test_engine


@pytest.fixture
async def session_maker(monkeypatch):
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(reminders_module, "async_session", maker)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()
    holder.close()


def _build_test_app(maker):
    app = FastAPI()
    app.include_router(get_api_router(paisa_enabled=True))

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    return app


async def _seed(maker, *, orphan: bool = True):
    """Seed a single IndusInd CC, a card, a UNPAID ₹500 statement, and an
    orphan (or already-linked) ₹500 CC-payment txn. Return ids."""
    async with maker() as session:
        account = Account(
            bank="indusind",
            type="credit_card",
            label="IndusInd Rupay",
            active=True,
        )
        session.add(account)
        await session.flush()
        card = Card(
            account_id=account.id,
            card_mask="5785",
            label="Primary",
            is_primary=True,
            active=True,
        )
        session.add(card)
        await session.flush()  # so card.id is populated before the txn refs it

        statement = StatementUpload(
            account_id=account.id,
            bank="indusind",
            filename="x.pdf",
            file_path="/tmp/x.pdf",
            status="imported",
            due_date="20/05/2026",
            total_amount_due="500.00",
            payment_status=PaymentStatus.UNPAID,
            payment_paid_amount=Decimal("0"),
            created_at=datetime.datetime(2026, 4, 30, tzinfo=datetime.UTC),
        )
        session.add(statement)

        txn = Transaction(
            bank="indusind",
            email_type="indusind_cc_payment_alert",
            direction="credit",
            amount=Decimal("500"),
            currency="INR",
            transaction_date=datetime.date(2026, 5, 17),
            counterparty="Payment received",
            channel="card",
            account_id=None if orphan else account.id,
            card_id=None if orphan else card.id,
        )
        session.add(txn)
        await session.commit()
        return account.id, card.id, statement.id, txn.id


async def _relink(maker, txn_id: int, payload: dict):
    app = _build_test_app(maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(f"/api/transactions/{txn_id}/relink", json=payload)


@pytest.mark.anyio
async def test_relink_orphan_by_card_derives_account_and_marks_statement_paid(
    session_maker,
):
    """A maskless CC-payment orphan gets the operator's card. The service derives
    the account from the card, and the statement auto-pays in the same request."""
    account_id, card_id, stmt_id, txn_id = await _seed(session_maker)

    r = await _relink(session_maker, txn_id, {"card_id": card_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["account_id"] == account_id
    assert body["card_id"] == card_id
    assert body["statement_marked_paid"] is True

    async with session_maker() as s:
        txn = await s.get(Transaction, txn_id)
        assert txn.account_id == account_id
        assert txn.card_id == card_id
        stmt = await s.get(StatementUpload, stmt_id)
        assert stmt.payment_status == PaymentStatus.PAID
        assert stmt.payment_paid_amount == Decimal("500")


@pytest.mark.anyio
async def test_relink_already_linked_does_not_double_credit_and_null_clears(
    session_maker,
):
    """Relink of an already-linked CC payment must not re-fire
    check_payment_received. A null means 'clear': a null card keeps the account,
    and null/null clears both, which is the undo."""
    account_id, card_id, stmt_id, txn_id = await _seed(session_maker, orphan=False)
    # The original ingestion already credited the statement.
    async with session_maker() as s:
        stmt = await s.get(StatementUpload, stmt_id)
        stmt.payment_paid_amount = Decimal("500")
        stmt.payment_status = PaymentStatus.PAID
        await s.commit()

    r = await _relink(
        session_maker, txn_id, {"account_id": account_id, "card_id": card_id}
    )
    assert r.status_code == 200, r.text
    assert r.json()["statement_marked_paid"] is False

    r = await _relink(
        session_maker, txn_id, {"account_id": account_id, "card_id": None}
    )
    assert r.status_code == 200, r.text
    assert (r.json()["account_id"], r.json()["card_id"]) == (account_id, None)

    r = await _relink(session_maker, txn_id, {"account_id": None, "card_id": None})
    assert r.status_code == 200, r.text

    async with session_maker() as s:
        stmt = await s.get(StatementUpload, stmt_id)
        assert stmt.payment_paid_amount == Decimal("500")
        txn = await s.get(Transaction, txn_id)
        assert txn.account_id is None
        assert txn.card_id is None


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["other_bank", "card_elsewhere", "missing"])
async def test_relink_to_a_wrong_account_is_rejected(session_maker, target):
    """A misclick must not move a CC payment onto the wrong account."""
    _account_id, card_id, stmt_id, txn_id = await _seed(session_maker)
    async with session_maker() as s:
        other = Account(
            bank="hdfc" if target == "other_bank" else "indusind",
            type="credit_card",
            label="Other card",
            active=True,
        )
        s.add(other)
        await s.commit()
        other_id = other.id

    payload = {
        "other_bank": {"account_id": other_id},
        # The card belongs to the first account, not this one.
        "card_elsewhere": {"account_id": other_id, "card_id": card_id},
        "missing": {"account_id": 999999},
    }[target]
    r = await _relink(session_maker, txn_id, payload)
    assert r.status_code == 400

    async with session_maker() as s:
        txn = await s.get(Transaction, txn_id)
        assert (txn.account_id, txn.card_id) == (None, None)
        stmt = await s.get(StatementUpload, stmt_id)
        assert stmt.payment_paid_amount == Decimal("0")
