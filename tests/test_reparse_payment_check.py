"""Verifies that the web reparse routes invoke ``check_payment_received``
for credit transactions, mirroring the polling pipeline.

The bug this guards against: ``services/emails.py:handle_polled_email``
collects credit-direction transactions whose ``account_id`` was set by the
linker and after commit calls ``services.reminders.check_payment_received``
to bump the matching active StatementUpload's ``payment_paid_amount`` /
``payment_status``. Both web reparse routes (``reparse_email`` and
``reparse_all_failed``) used to skip that call, leaving statements
``unpaid`` even when a credit txn that should have satisfied them had been
created via reparse.
"""

import datetime
from decimal import Decimal
from email.message import EmailMessage
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.core.deps as core_deps
import financial_dashboard.services.reminders as reminders_module
from financial_dashboard.core.deps import get_session
from financial_dashboard.web import get_router as get_web_router
from financial_dashboard.db import (
    Account,
    Card,
    Email,
    FetchRule,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus
from financial_dashboard.integrations.email.body import RawEmailResult
from tests.conftest import new_test_engine


@pytest.fixture
async def session_maker(monkeypatch):
    """In-memory aiosqlite session-maker, also installed as the global
    ``async_session`` used by ``check_payment_received``."""
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(reminders_module, "async_session", maker)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()
    holder.close()


def _build_test_app(maker):
    app = FastAPI()
    app.include_router(get_web_router())

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    return app


async def _post(maker, url: str):
    raw = _equitas_payment_eml("12,345.00", "9999")
    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            new=AsyncMock(return_value=RawEmailResult(raw, None, "provider")),
        ),
        patch(
            "financial_dashboard.web.emails.should_notify_transactions",
            return_value=False,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=_build_test_app(maker)),
            base_url="http://test",
        ) as client:
            return await client.post(url)


def _equitas_payment_eml(amount: str, card_last4: str) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Payment received !"
    msg["From"] = "cc-alerts@equitas.bank.in"
    msg["Date"] = "Wed, 6 May 2026 00:28:00 +0530"
    msg.set_content(
        "Dear Mr. Test Customer,\n\n"
        f"We inform you that INR {amount} was received on 06/05/2026 and was "
        f"credited to your Equitas Credit Card XX{card_last4}.\n"
    )
    return msg.as_bytes()


async def _seed(
    maker,
    *,
    due_amount: str = "100,000.00",
    email_count: int = 1,
) -> list[int]:
    """Create a credit_card account, matching card, active statement upload,
    and ``email_count`` failed Email rows tied to a fetch rule. Returns the
    list of email ids in insertion order."""
    async with maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="cc-alerts@equitas.bank.in",
            bank="equitas",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        await session.flush()

        account = Account(
            bank="equitas",
            label="Equitas Test CC",
            type="credit_card",
            account_number="6530XXXXXXXX9999",
            active=True,
        )
        session.add(account)
        await session.flush()

        card = Card(
            account_id=account.id,
            card_mask="6530XXXXXXXX9999",
            label="self",
            is_primary=True,
            active=True,
        )
        session.add(card)

        upload = StatementUpload(
            account_id=account.id,
            bank="equitas",
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            status="parsed",
            due_date="10/05/2026",
            total_amount_due=due_amount,
            payment_status=PaymentStatus.UNPAID,
            payment_paid_amount=Decimal("0"),
            # Statement generated before the payment arrives, so the 06/05
            # payment credit falls inside this cycle's recompute scope.
            created_at=datetime.datetime(2026, 4, 20, tzinfo=datetime.UTC),
        )
        session.add(upload)

        email_ids: list[int] = []
        for i in range(email_count):
            email_row = Email(
                provider="gmail",
                message_id=f"test-msg-id-{i + 1}",
                sender="cc-alerts@equitas.bank.in",
                subject="Payment received !",
                received_at=datetime.datetime(
                    2026, 5, 6, 0, 28 + i, tzinfo=datetime.UTC
                ),
                status="failed",
                error="Previous parse failed",
                rule_id=rule.id,
            )
            session.add(email_row)
            await session.flush()
            email_ids.append(email_row.id)
        await session.commit()
        return email_ids


@pytest.mark.anyio
class TestReparseEmailInvokesPaymentCheck:
    async def test_credit_txn_fully_pays_active_statement(self, session_maker):
        [email_id] = await _seed(session_maker, due_amount="12,345.00")

        r = await _post(session_maker, f"/emails/{email_id}/reparse")
        assert r.status_code == 200, r.text

        async with session_maker() as s:
            txn = (await s.execute(select(Transaction))).scalars().one()
            assert txn.direction == "credit"
            assert txn.amount == Decimal("12345.00")
            assert txn.account_id is not None

            upload = (await s.execute(select(StatementUpload))).scalars().one()
            assert upload.payment_paid_amount == Decimal("12345.00")
            assert upload.payment_status == PaymentStatus.PAID
            assert upload.payment_paid_at is not None


async def _seed_dup_defer(session_maker, *, candidates: int) -> int:
    """A [dup-defer] email plus balance-less rows of the same event."""
    [email_id] = await _seed(session_maker, due_amount="100,000.00")
    async with session_maker() as s:
        em = await s.get(Email, email_id)
        em.status = "skipped"
        em.error = "[dup-defer] possible duplicate"
        for t in range(candidates):
            s.add(
                Transaction(
                    bank="equitas",
                    email_type="equitas_cc_payment_received_alert",
                    direction="credit",
                    amount=Decimal("12345.00"),
                    currency="INR",
                    transaction_date=datetime.date(2026, 5, 6),
                    transaction_time=datetime.time(0, 28 + t),
                    counterparty="Payment received",
                    card_mask="XX9999",
                    balance=None,
                    source="sms",
                )
            )
        await s.commit()
    return email_id


@pytest.mark.anyio
async def test_dup_defer_reparse_stays_skipped_even_when_now_matchable(session_maker):
    """Without force_new a [dup-defer] email stays skipped, even when
    find_match would now return one clean match. Otherwise a deferred row
    silently flips to enriched on reparse."""
    email_id = await _seed_dup_defer(session_maker, candidates=1)

    r = await _post(session_maker, f"/emails/{email_id}/reparse")
    assert r.status_code == 200, r.text
    assert r.json()["new_status"] == "skipped"
    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        assert len(rows) == 1
        assert rows[0].email_id is None
        assert (await s.get(Email, email_id)).status == "skipped"


@pytest.mark.anyio
async def test_dup_defer_reparse_force_new_creates_row(session_maker):
    """A plain reparse inserts no row. force_new creates a real transaction."""
    email_id = await _seed_dup_defer(session_maker, candidates=2)

    r = await _post(session_maker, f"/emails/{email_id}/reparse")
    assert r.json()["new_status"] == "skipped"
    async with session_maker() as s:
        assert len((await s.execute(select(Transaction))).scalars().all()) == 2

    r = await _post(session_maker, f"/emails/{email_id}/reparse?force_new=true")
    assert r.status_code == 200, r.text
    assert r.json()["new_status"] == "parsed"
    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        assert len(rows) == 3
        assert (await s.get(Email, email_id)).status == "parsed"
        assert any(t.email_id == email_id for t in rows)


@pytest.mark.anyio
class TestReparseAllFailedBulkRoute:
    """Regression coverage for /emails/reparse-all-failed.

    Without ``session.expunge_all()`` before the post-select rollback, the
    loop's first access of ``email_row.provider`` triggered an async
    lazy-load with no greenlet attached and raised MissingGreenlet.
    """

    async def test_bulk_reparse_processes_each_email_and_bumps_statement(
        self, session_maker
    ):
        await _seed(session_maker, due_amount="100,000.00", email_count=2)

        r = await _post(session_maker, "/emails/reparse-all-failed")
        assert r.status_code == 200, r.text
        assert r.json()["succeeded"] == 2
        assert r.json()["failed"] == 0

        async with session_maker() as s:
            txns = (await s.execute(select(Transaction))).scalars().all()
            assert len(txns) == 2
            assert all(t.direction == "credit" for t in txns)
            assert all(t.amount == Decimal("12345.00") for t in txns)

            upload = (await s.execute(select(StatementUpload))).scalars().one()
            # Both credit transactions should have bumped paid_amount.
            assert upload.payment_paid_amount == Decimal("24690.00")
            assert upload.payment_status == PaymentStatus.PARTIALLY_PAID
