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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import bank_email_fetcher.core.deps as core_deps
import bank_email_fetcher.services.reminders as reminders_module
from bank_email_fetcher.core.deps import get_session
from bank_email_fetcher.web import get_router as get_web_router
from bank_email_fetcher.db import (
    Account,
    Base,
    Card,
    Email,
    FetchRule,
    StatementUpload,
    Transaction,
)
from bank_email_fetcher.db.enums import PaymentStatus


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_maker(monkeypatch):
    """In-memory aiosqlite session-maker, also installed as the global
    ``async_session`` used by ``check_payment_received``."""
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
    app.include_router(get_web_router())

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    return app


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
    async def test_credit_txn_partially_pays_active_statement(self, session_maker):
        [email_id] = await _seed(session_maker, due_amount="100,000.00")

        raw = _equitas_payment_eml("12,345.00", "9999")
        with (
            patch(
                "bank_email_fetcher.web.emails.load_or_fetch_raw_email",
                new=AsyncMock(return_value=(raw, None)),
            ),
            patch(
                "bank_email_fetcher.web.emails.should_notify_transactions",
                return_value=False,
            ),
        ):
            app = _build_test_app(session_maker)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post(f"/emails/{email_id}/reparse")
                assert r.status_code == 200, r.text

        async with session_maker() as s:
            txn = (await s.execute(select(Transaction))).scalars().one()
            assert txn.direction == "credit"
            assert txn.amount == Decimal("12345.00")
            assert txn.account_id is not None

            upload = (await s.execute(select(StatementUpload))).scalars().one()
            assert upload.payment_paid_amount == Decimal("12345.00")
            assert upload.payment_status == PaymentStatus.PARTIALLY_PAID

    async def test_credit_txn_fully_pays_active_statement(self, session_maker):
        [email_id] = await _seed(session_maker, due_amount="12,345.00")

        raw = _equitas_payment_eml("12,345.00", "9999")
        with (
            patch(
                "bank_email_fetcher.web.emails.load_or_fetch_raw_email",
                new=AsyncMock(return_value=(raw, None)),
            ),
            patch(
                "bank_email_fetcher.web.emails.should_notify_transactions",
                return_value=False,
            ),
        ):
            app = _build_test_app(session_maker)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post(f"/emails/{email_id}/reparse")
                assert r.status_code == 200, r.text

        async with session_maker() as s:
            upload = (await s.execute(select(StatementUpload))).scalars().one()
            assert upload.payment_paid_amount == Decimal("12345.00")
            assert upload.payment_status == PaymentStatus.PAID
            assert upload.payment_paid_at is not None


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
        email_ids = await _seed(
            session_maker, due_amount="100,000.00", email_count=2
        )
        assert len(email_ids) == 2

        raw = _equitas_payment_eml("12,345.00", "9999")
        with (
            patch(
                "bank_email_fetcher.web.emails.load_or_fetch_raw_email",
                new=AsyncMock(return_value=(raw, None)),
            ),
            patch(
                "bank_email_fetcher.web.emails.should_notify_transactions",
                return_value=False,
            ),
        ):
            app = _build_test_app(session_maker)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post("/emails/reparse-all-failed")
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["succeeded"] == 2
                assert body["failed"] == 0

        async with session_maker() as s:
            txns = (await s.execute(select(Transaction))).scalars().all()
            assert len(txns) == 2
            assert all(t.direction == "credit" for t in txns)
            assert all(t.amount == Decimal("12345.00") for t in txns)

            upload = (await s.execute(select(StatementUpload))).scalars().one()
            # Both credit transactions should have bumped paid_amount.
            assert upload.payment_paid_amount == Decimal("24690.00")
            assert upload.payment_status == PaymentStatus.PARTIALLY_PAID
