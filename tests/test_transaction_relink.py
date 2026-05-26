"""Tests for the manual relink endpoint POST /api/transactions/{id}/relink.

Use case: a Transaction was originally orphaned (account_id=None) by
the ingestion pipeline — typically a maskless CC bill-payment where
the user has multiple CCs at the same bank and the auto-resolver
gave up. The operator picks the right account/card from the UI; the
endpoint sets the FKs and fires check_payment_received when the
transaction is a CC bill-payment credit, so the matching statement
auto-marks paid in one click.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.core.deps as core_deps
import financial_dashboard.services.reminders as reminders_module
from financial_dashboard.api import router as api_router
from financial_dashboard.core.deps import get_session
from financial_dashboard.db import (
    Account,
    Base,
    Card,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_maker(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(reminders_module, "async_session", maker)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()


def _build_test_app(maker):
    app = FastAPI()
    app.include_router(api_router)

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
            card_mask="0000",
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


@pytest.mark.anyio
async def test_relink_orphan_sets_account_and_marks_statement_paid(session_maker):
    """The headline scenario: a maskless CC-payment orphan gets the
    operator's chosen account via the relink endpoint; statement
    auto-pays in the same request."""
    account_id, card_id, stmt_id, txn_id = await _seed(session_maker)

    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/api/transactions/{txn_id}/relink",
            json={"account_id": account_id, "card_id": card_id},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
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
async def test_relink_already_linked_does_not_double_credit(session_maker):
    """Relink of an already-linked CC-payment txn must not re-fire
    check_payment_received — payment_paid_amount stays untouched."""
    account_id, card_id, stmt_id, txn_id = await _seed(session_maker, orphan=False)

    # Pretend the original ingestion already credited the statement.
    async with session_maker() as s:
        stmt = await s.get(StatementUpload, stmt_id)
        stmt.payment_paid_amount = Decimal("500")
        stmt.payment_status = PaymentStatus.PAID
        await s.commit()

    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/api/transactions/{txn_id}/relink",
            json={"account_id": account_id, "card_id": card_id},
        )
        assert r.status_code == 200, r.text
        assert r.json()["statement_marked_paid"] is False

    async with session_maker() as s:
        stmt = await s.get(StatementUpload, stmt_id)
        # Same as before — no double credit.
        assert stmt.payment_paid_amount == Decimal("500")


@pytest.mark.anyio
async def test_relink_card_only_derives_account(session_maker):
    """If only card_id is given, the service derives account_id from
    the card's owning account."""
    account_id, card_id, _stmt_id, txn_id = await _seed(session_maker)

    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/api/transactions/{txn_id}/relink",
            json={"card_id": card_id},  # no account_id
        )
        assert r.status_code == 200, r.text
        assert r.json()["account_id"] == account_id


@pytest.mark.anyio
async def test_relink_bank_mismatch_rejected(session_maker):
    """An account in a different bank than the transaction must be
    refused — protects against operator misclicks."""
    _idsind_acct, _card, _stmt, txn_id = await _seed(session_maker)

    # Create an HDFC account.
    async with session_maker() as s:
        hdfc = Account(
            bank="hdfc",
            type="credit_card",
            label="HDFC Diners",
            active=True,
        )
        s.add(hdfc)
        await s.commit()
        hdfc_id = hdfc.id

    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/api/transactions/{txn_id}/relink",
            json={"account_id": hdfc_id},
        )
        assert r.status_code == 400
        assert "bank" in r.json()["detail"].lower()


@pytest.mark.anyio
async def test_relink_card_account_mismatch_rejected(session_maker):
    """If both account_id and card_id are given and the card's
    account_id != account_id, refuse the relink."""
    _acct1, card_id, _stmt, txn_id = await _seed(session_maker)

    # Make a second IndusInd account WITHOUT the given card.
    async with session_maker() as s:
        acct2 = Account(
            bank="indusind",
            type="credit_card",
            label="IndusInd Pinnacle",
            active=True,
        )
        s.add(acct2)
        await s.commit()
        acct2_id = acct2.id

    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/api/transactions/{txn_id}/relink",
            json={"account_id": acct2_id, "card_id": card_id},
        )
        assert r.status_code == 400
        assert (
            "mismatch" in r.json()["detail"].lower()
            or "account" in r.json()["detail"].lower()
        )


@pytest.mark.anyio
async def test_relink_clears_when_both_null(session_maker):
    """Passing null/null clears the existing link — useful for undo."""
    account_id, card_id, _stmt, txn_id = await _seed(session_maker, orphan=False)

    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/api/transactions/{txn_id}/relink",
            json={"account_id": None, "card_id": None},
        )
        assert r.status_code == 200
        assert r.json()["account_id"] is None
        assert r.json()["card_id"] is None

    async with session_maker() as s:
        txn = await s.get(Transaction, txn_id)
        assert txn.account_id is None
        assert txn.card_id is None


@pytest.mark.anyio
async def test_relink_clears_card_while_keeping_account(session_maker):
    """Hybrid: operator sets account_id but leaves card_id null. The
    null is interpreted as 'clear' (not 'leave alone'), matching the
    null/null clear semantic. Useful when the operator knows the
    account but not which specific card."""
    account_id, card_id, _stmt, txn_id = await _seed(session_maker, orphan=False)
    # Sanity: prior state has both linked.
    async with session_maker() as s:
        txn = await s.get(Transaction, txn_id)
        assert txn.account_id == account_id
        assert txn.card_id == card_id

    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/api/transactions/{txn_id}/relink",
            json={"account_id": account_id, "card_id": None},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["account_id"] == account_id
        assert body["card_id"] is None

    async with session_maker() as s:
        txn = await s.get(Transaction, txn_id)
        assert txn.account_id == account_id
        assert txn.card_id is None


@pytest.mark.anyio
async def test_relink_nonexistent_txn_404(session_maker):
    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/api/transactions/999999/relink",
            json={"account_id": None, "card_id": None},
        )
        assert r.status_code == 404


@pytest.mark.anyio
async def test_relink_nonexistent_account_400(session_maker):
    _acct, _card, _stmt, txn_id = await _seed(session_maker)
    app = _build_test_app(session_maker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            f"/api/transactions/{txn_id}/relink",
            json={"account_id": 999999},
        )
        assert r.status_code == 400
        assert "account" in r.json()["detail"].lower()
