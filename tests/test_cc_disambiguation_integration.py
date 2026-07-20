"""End-to-end integration test for the maskless CC bill-payment
disambiguation path through ``/emails/{id}/reparse``.

Mirrors the production scenario that motivated the feature: an IndusInd
CC payment-confirmation email body carries no card mask. The user has
three IndusInd credit cards. Exactly one has an open statement whose
``total_amount_due`` matches the incoming payment. The reparse endpoint
must route the txn to that account and bump the statement's
``payment_paid_amount`` / ``payment_status``.
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

import financial_dashboard.core.deps as core_deps
import financial_dashboard.services.reminders as reminders_module
from financial_dashboard.core.deps import get_session
from financial_dashboard.db import (
    Account,
    Base,
    Card,
    Email,
    FetchRule,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus
from financial_dashboard.integrations.email.body import RawEmailResult
from financial_dashboard.web import get_router as get_web_router


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
    app.include_router(get_web_router())

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    return app


def _indusind_payment_eml(amount: str) -> bytes:
    """The body shape that originally produced the orphan: amount only,
    no card mask. ``_date_pattern`` requires the 'credited to your
    Credit Card account on DD/MM/YYYY' suffix."""
    msg = EmailMessage()
    msg["Subject"] = "Payment Confirmation on your IndusInd Bank Credit Card"
    msg["From"] = "transactionalert@indusind.com"
    msg["Date"] = "Sun, 17 May 2026 18:42:35 +0530"
    msg.set_content(
        f"Thank you for your Payment of INR {amount} towards your "
        f"IndusInd Bank Credit Card. Your payment is credited to your "
        f"Credit Card account on 17/05/2026.\n"
    )
    return msg.as_bytes()


async def _seed_three_indusind_ccs_with_statements(
    maker, *, target_total: str
) -> tuple[int, int, int, int]:
    """Three IndusInd CC accounts, each with an open statement. Only
    the middle account's statement total matches ``target_total``. A
    single failed Email row is created and returned alongside the
    three account ids."""
    async with maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="transactionalert@indusind.com",
            bank="indusind",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        await session.flush()

        a = Account(bank="indusind", type="credit_card", label="A", active=True)
        b = Account(bank="indusind", type="credit_card", label="B", active=True)
        c = Account(bank="indusind", type="credit_card", label="C", active=True)
        session.add_all([a, b, c])
        await session.flush()

        session.add_all(
            [
                StatementUpload(
                    account_id=a.id,
                    bank="indusind",
                    filename="a.pdf",
                    file_path="/tmp/a.pdf",
                    status="imported",
                    due_date="20/05/2026",
                    total_amount_due="1,616.00",
                    payment_status=PaymentStatus.UNPAID,
                    payment_paid_amount=Decimal("0"),
                    created_at=datetime.datetime(2026, 4, 30, tzinfo=datetime.UTC),
                ),
                StatementUpload(
                    account_id=b.id,
                    bank="indusind",
                    filename="b.pdf",
                    file_path="/tmp/b.pdf",
                    status="imported",
                    due_date="20/05/2026",
                    total_amount_due=target_total,
                    payment_status=PaymentStatus.UNPAID,
                    payment_paid_amount=Decimal("0"),
                    created_at=datetime.datetime(2026, 4, 30, tzinfo=datetime.UTC),
                ),
                StatementUpload(
                    account_id=c.id,
                    bank="indusind",
                    filename="c.pdf",
                    file_path="/tmp/c.pdf",
                    status="imported",
                    due_date="20/05/2026",
                    total_amount_due="4,661.00",
                    payment_status=PaymentStatus.UNPAID,
                    payment_paid_amount=Decimal("0"),
                    created_at=datetime.datetime(2026, 4, 30, tzinfo=datetime.UTC),
                ),
            ]
        )

        email_row = Email(
            provider="gmail",
            message_id="test-indusind-1",
            sender="transactionalert@indusind.com",
            subject="Payment Confirmation on your IndusInd Bank Credit Card",
            received_at=datetime.datetime(2026, 5, 17, 18, 42, 35, tzinfo=datetime.UTC),
            status="failed",
            error="Previous parse failed",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()
        return a.id, b.id, c.id, email_row.id


@pytest.mark.anyio
async def test_maskless_indusind_cc_payment_resolves_via_amount(session_maker):
    """The matching-total CC account (b) receives the txn link, and
    its statement gets marked fully paid by check_payment_received."""
    a_id, b_id, c_id, email_id = await _seed_three_indusind_ccs_with_statements(
        session_maker, target_total="133.00"
    )

    raw = _indusind_payment_eml("133.00")
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
        app = _build_test_app(session_maker)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(f"/emails/{email_id}/reparse")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        txn = (await s.execute(select(Transaction))).scalars().one()
        assert txn.direction == "credit"
        assert txn.amount == Decimal("133")
        assert txn.account_id == b_id  # ← the matching-total CC

        uploads = {
            u.account_id: u
            for u in (await s.execute(select(StatementUpload))).scalars().all()
        }
        # Account b's statement was fully paid by check_payment_received.
        assert uploads[b_id].payment_status == PaymentStatus.PAID
        assert uploads[b_id].payment_paid_amount == Decimal("133")
        # Other accounts' statements untouched — both a and c.
        for other_id in (a_id, c_id):
            assert uploads[other_id].payment_status == PaymentStatus.UNPAID
            assert uploads[other_id].payment_paid_amount == Decimal("0")


@pytest.mark.anyio
async def test_maskless_indusind_cc_payment_with_no_match_stays_unlinked(
    session_maker,
):
    """When no open statement total matches the payment, the txn is
    inserted but stays unlinked and no statement gets marked paid.
    (The pipeline would dispatch a Telegram prompt — we patch
    send_disambiguation_prompt to a no-op since Telegram isn't configured
    in tests.)"""
    a_id, b_id, c_id, email_id = await _seed_three_indusind_ccs_with_statements(
        session_maker,
        target_total="999.00",  # ← doesn't match payment
    )

    raw = _indusind_payment_eml("133.00")
    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            new=AsyncMock(return_value=RawEmailResult(raw, None, "provider")),
        ),
        patch(
            "financial_dashboard.web.emails.should_notify_transactions",
            return_value=False,
        ),
        patch(
            "financial_dashboard.web.emails.send_disambiguation_prompt",
            new=AsyncMock(return_value=None),
        ) as prompt,
    ):
        app = _build_test_app(session_maker)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(f"/emails/{email_id}/reparse")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        txn = (await s.execute(select(Transaction))).scalars().one()
        assert txn.amount == Decimal("133")
        assert txn.account_id is None  # ← still orphaned, by design

        # No statement was touched.
        uploads = {
            u.account_id: u
            for u in (await s.execute(select(StatementUpload))).scalars().all()
        }
        for acct_id in (a_id, b_id, c_id):
            assert uploads[acct_id].payment_status == PaymentStatus.UNPAID
            assert uploads[acct_id].payment_paid_amount == Decimal("0")

    # Prompt dispatched once with the full payload shape consumed by
    # send_disambiguation_prompt — txn_id, candidates, amount, bank.
    prompt.assert_called_once()
    payload = prompt.call_args.args[0]
    assert set(payload["candidate_account_ids"]) == {a_id, b_id, c_id}
    assert payload["txn_id"] == txn.id
    assert payload["amount"] == Decimal("133")
    assert payload["bank"] == "indusind"
    assert set(payload["candidate_labels"].keys()) == {a_id, b_id, c_id}


@pytest.mark.anyio
async def test_bulk_reparse_disambiguates_maskless_cc_payment(session_maker):
    """Regression: the /emails/reparse-all-failed bulk endpoint used to
    skip the disambiguation block, so a failed-then-bulk-reparsed
    maskless multi-CC payment would re-link to None. Now the helper
    runs on the bulk path too and the txn lands on the matching CC."""
    a_id, b_id, c_id, email_id = await _seed_three_indusind_ccs_with_statements(
        session_maker, target_total="133.00"
    )

    raw = _indusind_payment_eml("133.00")
    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            new=AsyncMock(return_value=RawEmailResult(raw, None, "provider")),
        ),
    ):
        app = _build_test_app(session_maker)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/emails/reparse-all-failed")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        txn = (await s.execute(select(Transaction))).scalars().one()
        assert txn.account_id == b_id
        uploads = {
            u.account_id: u
            for u in (await s.execute(select(StatementUpload))).scalars().all()
        }
        assert uploads[b_id].payment_status == PaymentStatus.PAID
        for other_id in (a_id, c_id):
            assert uploads[other_id].payment_status == PaymentStatus.UNPAID


@pytest.mark.anyio
async def test_reparse_upserts_existing_attached_transaction(session_maker):
    """Reparsing an email that already has an attached transaction must
    update the existing row (and relink), not create a duplicate. The
    user's intended workflow for fixing a historical orphan."""
    a_id, b_id, _c_id, email_id = await _seed_three_indusind_ccs_with_statements(
        session_maker, target_total="133.00"
    )

    # Seed an orphan transaction already attached to the email (the
    # state #7078 was in production: parsed, but unlinked).
    async with session_maker() as s:
        orphan = Transaction(
            email_id=email_id,
            bank="indusind",
            email_type="indusind_cc_payment_alert",
            direction="credit",
            amount=Decimal("133"),
            currency="INR",
            transaction_date=datetime.date(2026, 5, 17),
            counterparty="Payment received",
            channel="card",
            account_id=None,
        )
        s.add(orphan)
        await s.commit()
        orphan_id = orphan.id

    raw = _indusind_payment_eml("133.00")
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
        app = _build_test_app(session_maker)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(f"/emails/{email_id}/reparse")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        # Exactly one transaction — the orphan, now linked.
        rows = (await s.execute(select(Transaction))).scalars().all()
        assert len(rows) == 1
        assert rows[0].id == orphan_id
        assert rows[0].account_id == b_id

        # Statement #b got marked paid.
        ups = {
            u.account_id: u
            for u in (await s.execute(select(StatementUpload))).scalars().all()
        }
        assert ups[b_id].payment_status == PaymentStatus.PAID
        assert ups[a_id].payment_status == PaymentStatus.UNPAID


def _icici_reversal_eml(amount: str, card_last4: str) -> bytes:
    """Real ICICI merchant-refund email shape (icici_cc_reversal). This
    is a credit on the CC but NOT a bill payment — it must not satisfy
    an open statement."""
    msg = EmailMessage()
    msg["Subject"] = "ICICI Bank Credit Card Reversal"
    msg["From"] = "credit_cards@icici.bank.in"
    msg["Date"] = "Sun, 17 May 2026 12:00:00 +0530"
    msg.set_content(
        f"We have received merchant credit refund on your "
        f"ICICI Bank Credit Card XX{card_last4} for INR {amount} on "
        f"May 17, 2026 from SOME MERCHANT.\n"
    )
    return msg.as_bytes()


@pytest.mark.anyio
async def test_cc_reversal_credit_does_not_mark_statement_paid(session_maker):
    """An icici_cc_reversal email is direction=credit but represents a
    merchant refund, not a bill payment. The email-path gate must skip
    check_payment_received so the open statement stays UNPAID. Before
    the gate was added, this credit silently bumped payment_paid_amount
    and could flip a statement to PARTIALLY_PAID against a refund."""
    async with session_maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="credit_cards@icici.bank.in",
            bank="icici",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        await session.flush()

        account = Account(
            bank="icici",
            type="credit_card",
            label="ICICI CC",
            active=True,
        )
        session.add(account)
        await session.flush()

        card = Card(
            account_id=account.id,
            card_mask="XX2308",
            label="self",
            is_primary=True,
            active=True,
        )
        session.add(card)

        upload = StatementUpload(
            account_id=account.id,
            bank="icici",
            filename="x.pdf",
            file_path="/tmp/x.pdf",
            status="imported",
            due_date="20/05/2026",
            total_amount_due="500.00",
            payment_status=PaymentStatus.UNPAID,
            payment_paid_amount=Decimal("0"),
        )
        session.add(upload)

        email_row = Email(
            provider="gmail",
            message_id="test-icici-reversal-1",
            sender="credit_cards@icici.bank.in",
            subject="ICICI Bank Credit Card Reversal",
            received_at=datetime.datetime(2026, 5, 17, 12, 0, 0, tzinfo=datetime.UTC),
            status="failed",
            error="Previous parse failed",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()
        email_id = email_row.id

    raw = _icici_reversal_eml("250.00", "2308")
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
        app = _build_test_app(session_maker)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(f"/emails/{email_id}/reparse")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        txn = (await s.execute(select(Transaction))).scalars().one()
        assert txn.direction == "credit"
        assert txn.amount == Decimal("250")
        # The reversal is correctly linked to the card's account by the
        # linker (it carries a card_mask, so this is the normal path).
        assert txn.account_id is not None

        upload = (await s.execute(select(StatementUpload))).scalars().one()
        # CRITICAL: the open statement must be untouched. Before the
        # email-path gate was added, payment_paid_amount would have
        # bumped to 250 and payment_status would have flipped to
        # PARTIALLY_PAID against a merchant refund.
        assert upload.payment_status == PaymentStatus.UNPAID
        assert upload.payment_paid_amount == Decimal("0")


@pytest.mark.anyio
async def test_reparse_does_not_re_credit_already_linked_txn(session_maker):
    """Regression: reparse of an email whose original parse already
    linked the txn to a CC account must NOT re-fire
    check_payment_received and double-count payment_paid_amount. The
    upsert path is meant for fixing a historical orphan (account_id
    was None on the original parse), not for re-applying a credit
    that already landed.

    Setup: a CC bill-payment email of ₹133 was originally parsed,
    linked to account b, and the statement is UNPAID with
    payment_paid_amount=0 (simulating that for whatever reason the
    initial check_payment_received result was missed or rolled back).
    The user reparses. With the WAS_ORPHANED guard at web/emails.py:
    reparse must NOT call check_payment_received because the prior
    txn was already linked — operator-induced state mismatches like
    this don't get silently "fixed" by reparse, which would in the
    common case double-count.

    Without the guard, the reparse would credit the statement and
    bump payment_paid_amount to 133, which is exactly the corruption
    we're guarding against in the partially-paid common case."""
    # Statement total matches the payment so the resolver re-links
    # cleanly on reparse — keeps the test focused on the double-fire
    # behavior, not on re-linking.
    a_id, b_id, _c_id, email_id = await _seed_three_indusind_ccs_with_statements(
        session_maker, target_total="133.00"
    )
    # Seed: a txn already attached AND linked to b.
    async with session_maker() as s:
        prior = Transaction(
            email_id=email_id,
            bank="indusind",
            email_type="indusind_cc_payment_alert",
            direction="credit",
            amount=Decimal("133"),
            currency="INR",
            account_id=b_id,
            transaction_date=datetime.date(2026, 5, 17),
            counterparty="Payment received",
            channel="card",
        )
        s.add(prior)
        await s.commit()

    raw = _indusind_payment_eml("133.00")
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
        app = _build_test_app(session_maker)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(f"/emails/{email_id}/reparse")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        # Exactly one Transaction, still attached to email and (re-)linked.
        rows = (await s.execute(select(Transaction))).scalars().all()
        assert len(rows) == 1
        assert rows[0].account_id == b_id

        ups = {
            u.account_id: u
            for u in (await s.execute(select(StatementUpload))).scalars().all()
        }
        # The CRITICAL assertion: payment_paid_amount stayed at 0.
        # Without the WAS_ORPHANED guard, reparse would have queued
        # check_payment_received and bumped it to 133, silently
        # crediting the statement a second time.
        assert ups[b_id].payment_paid_amount == Decimal("0"), (
            f"Reparse re-credited an already-linked txn: "
            f"payment_paid_amount={ups[b_id].payment_paid_amount}"
        )
        assert ups[b_id].payment_status == PaymentStatus.UNPAID
        # Other accounts untouched.
        assert ups[a_id].payment_paid_amount == Decimal("0")
