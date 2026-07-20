"""Email reparse must dedup against an existing SMS-sourced transaction.

Regression test for the production double-count seen on the HDFC
savings-to-PPF transfer: the same event arrives as an SMS (creates a
``source='sms'`` row) and an email. When the email is *reparsed* (as
opposed to ingested live), the reparse handler must not blind-insert a
second row — it must find the existing SMS row via the cross-channel
matcher and enrich it (``source='sms+email'``), exactly as the live
ingest path does.

The reparse handler keeps its bespoke ``email_id``-keyed upsert for the
"fix a historical orphan attached to *this* email" workflow; the dedup
only fires when no transaction is yet attached to the email.
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
from financial_dashboard.db import Account, Base, Email, FetchRule, Transaction
from financial_dashboard.integrations.email.body import RawEmailResult
from financial_dashboard.services.emails import _process_email_full
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


def _hdfc_ppf_transfer_eml() -> bytes:
    """HDFC savings-to-PPF transfer debit email body (no time, has date)."""
    msg = EmailMessage()
    msg["Subject"] = "View: Account update for your HDFC Bank A/c"
    msg["From"] = "HDFC Bank InstaAlerts <alerts@hdfcbank.bank.in>"
    msg["Date"] = "Fri, 05 Jun 2026 09:45:11 +0000"
    msg.set_content(
        "Dear Customer,\n"
        "You have transferred Rs. 1,00,000.00 to your PPF/Sukanya Samriddhi "
        "Yojana Account No. ending with XX0000 from your A/c No. XX1111, "
        "through Online Banking on 05-06-2026.\n"
        "Not you? Call 18002586161\n"
    )
    return msg.as_bytes()


async def _seed_sms_row_and_failed_email(maker) -> tuple[int, int]:
    """Seed an existing SMS-sourced HDFC transfer Transaction (no email
    attached) plus a matching failed Email row. Returns (sms_txn_id,
    email_id)."""
    async with maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="alerts@hdfcbank.bank.in",
            bank="hdfc",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)

        # Source account so the email's account_mask can link.
        account = Account(
            bank="hdfc",
            type="bank_account",
            label="HDFC Savings",
            account_number="000000001111",
            active=True,
        )
        session.add(account)
        await session.flush()

        sms_txn = Transaction(
            bank="hdfc",
            email_type="hdfc_account_transfer_debit_alert",
            direction="debit",
            amount=Decimal("100000"),
            currency="INR",
            transaction_date=datetime.date(2026, 6, 5),
            transaction_time=datetime.time(15, 15, 12),
            counterparty="PPF/SSY A/c XX0000",
            channel="online",
            source="sms",
            notified_channel="sms",
            sms_message_id=None,
        )
        session.add(sms_txn)

        email_row = Email(
            provider="gmail",
            message_id="test-hdfc-ppf-1",
            sender="alerts@hdfcbank.bank.in",
            subject="View: Account update for your HDFC Bank A/c",
            received_at=datetime.datetime(2026, 6, 5, 9, 45, 11, tzinfo=datetime.UTC),
            status="failed",
            error="Previous parse failed",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()
        return sms_txn.id, email_row.id


@pytest.mark.anyio
async def test_reparse_email_enriches_existing_sms_row(session_maker):
    """Reparsing the email for an event already captured by SMS must
    enrich the SMS row, not create a second transaction."""
    sms_txn_id, email_id = await _seed_sms_row_and_failed_email(session_maker)

    raw = _hdfc_ppf_transfer_eml()
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
        rows = (await s.execute(select(Transaction))).scalars().all()
        # Exactly one row — the SMS row, now enriched with the email.
        assert len(rows) == 1, (
            f"expected 1 row, got {len(rows)}: {[r.id for r in rows]}"
        )
        row = rows[0]
        assert row.id == sms_txn_id
        assert row.source == "sms+email"
        assert row.email_id == email_id
        assert row.sms_message_id is None  # SMS row had none; preserved
        # Email carried the source mask → account link gets filled.
        assert row.account_mask == "XX1111"
        assert row.account_id is not None
        # Downgrade-safe enrichment: the SMS row's in-body time must NOT be
        # clobbered by the email's missing time.
        assert row.transaction_time == datetime.time(15, 15, 12)


@pytest.mark.anyio
async def test_reparse_email_no_match_still_creates_row(session_maker):
    """With no pre-existing cross-channel row, reparse must still insert a
    fresh transaction — the dedup probe must not suppress the normal path."""
    # Seed only the failed email + account; NO SMS transaction.
    async with session_maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="alerts@hdfcbank.bank.in",
            bank="hdfc",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        session.add(
            Account(
                bank="hdfc",
                type="bank_account",
                label="HDFC Savings",
                account_number="000000001111",
                active=True,
            )
        )
        await session.flush()
        email_row = Email(
            provider="gmail",
            message_id="test-hdfc-ppf-nomatch",
            sender="alerts@hdfcbank.bank.in",
            subject="View: Account update for your HDFC Bank A/c",
            received_at=datetime.datetime(2026, 6, 5, 9, 45, 11, tzinfo=datetime.UTC),
            status="failed",
            error="Previous parse failed",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()
        email_id = email_row.id

    raw = _hdfc_ppf_transfer_eml()
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
        rows = (await s.execute(select(Transaction))).scalars().all()
        # The dedup probe found nothing, so the normal insert path runs.
        # (Reparse-created rows leave source unset, as they always have.)
        assert len(rows) == 1
        assert rows[0].email_id == email_id
        assert rows[0].sms_message_id is None


@pytest.mark.anyio
async def test_reparse_email_does_not_match_different_amount(session_maker):
    """A pre-existing SMS row for a *different* amount must NOT be merged
    into — the email gets its own row (no false cross-channel dedup)."""
    # Seed an SMS row for a different amount (₹2,00,000) than the email
    # (₹1,00,000), same bank/date.
    async with session_maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="alerts@hdfcbank.bank.in",
            bank="hdfc",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        await session.flush()
        session.add(
            Transaction(
                bank="hdfc",
                email_type="hdfc_account_transfer_debit_alert",
                direction="debit",
                amount=Decimal("200000"),
                currency="INR",
                transaction_date=datetime.date(2026, 6, 5),
                transaction_time=datetime.time(15, 15, 12),
                counterparty="PPF/SSY A/c XX0000",
                channel="online",
                source="sms",
                notified_channel="sms",
            )
        )
        email_row = Email(
            provider="gmail",
            message_id="test-hdfc-ppf-diffamt",
            sender="alerts@hdfcbank.bank.in",
            subject="View: Account update for your HDFC Bank A/c",
            received_at=datetime.datetime(2026, 6, 5, 9, 45, 11, tzinfo=datetime.UTC),
            status="failed",
            error="Previous parse failed",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()
        email_id = email_row.id

    raw = _hdfc_ppf_transfer_eml()  # ₹1,00,000
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
        rows = (await s.execute(select(Transaction))).scalars().all()
        # Two distinct events → two rows (no false cross-channel merge).
        assert len(rows) == 2
        by_amount = {r.amount: r for r in rows}
        assert by_amount[Decimal("200000")].source == "sms"
        assert by_amount[Decimal("200000")].email_id is None
        assert by_amount[Decimal("100000")].email_id == email_id


def _kotak_digital_eml(amount: str, txn_id: str) -> bytes:
    """Kotak "Transaction Successful" digital debit email. Carries a
    Transaction ID (reference_number) in a labeled grid, no transaction
    date."""
    msg = EmailMessage()
    msg["Subject"] = "Transaction Successful"
    msg["From"] = "Kotak Mahindra Bank <no-reply@kotak.com>"
    msg["Date"] = "Tue, 09 Jun 2026 12:14:07 +0000"
    msg.add_alternative(
        f"""<html><body>
        <table width="100%"><tbody><tr><td>
          <p>Hello CUSTOMER,</p>
          <p>Your transaction of &#8377; {amount} has been processed successfully.</p>
          <table width="600"><tbody>
            <tr><th>Transaction ID</th><th>Amount in &#8377;</th><th>Status</th></tr>
            <tr><td rowspan="2">{txn_id}</td><td rowspan="2">{amount}</td></tr>
            <tr><td>SUCCESS</td></tr>
          </tbody></table>
        </td></tr></tbody></table>
        </body></html>""",
        subtype="html",
    )
    return msg.as_bytes()


@pytest.mark.anyio
async def test_reparse_same_ref_different_amount_defers_not_409(session_maker):
    """Repro of the Kotak digital-transaction ref-collision class on the
    reparse path. A pre-existing
    row carries a reference_number that a *different-amount* email reparse
    also produces. The exact-ref match path now defers (amount mismatch), so
    the reparse handler must route that to the [dup-defer] manual-review
    state — NOT fall through to a blind insert that violates the
    (bank, reference_number, direction) unique index and 500s/409s.

    The shared reference_number is the literal the *currently pinned* parser
    extracts from this email ("Transaction ID"); the test exercises the
    dashboard merge/reparse layer independently of the parser deploy chain,
    so it stays valid before and after the parser ref-scrape fix lands."""
    shared_ref = _process_email_full(
        "kotak", _kotak_digital_eml(amount="7777.00", txn_id="999000111222")
    ).txn_data["reference_number"]
    async with session_maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="no-reply@kotak.com",
            bank="kotak",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        await session.flush()
        # Pre-existing ₹5,555 row sharing the parsed reference_number. The Kotak
        # digital "Transaction Successful" mail is a credit, so the row must be a
        # credit too — direction is part of the dedup match key.
        session.add(
            Transaction(
                bank="kotak",
                email_type="kotak_digital_transaction",
                direction="credit",
                amount=Decimal("5555"),
                currency="INR",
                transaction_date=datetime.date(2026, 5, 3),
                reference_number=shared_ref,
                source="email",
            )
        )
        email_row = Email(
            provider="gmail",
            message_id="test-kotak-digital-diffamt",
            sender="no-reply@kotak.com",
            subject="Transaction Successful",
            received_at=datetime.datetime(2026, 6, 9, 12, 14, 7, tzinfo=datetime.UTC),
            status="failed",
            error="Previous parse failed",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()
        email_id = email_row.id

    # Reparse a ₹7,777 email sharing the SAME Transaction ID.
    raw = _kotak_digital_eml(amount="7777.00", txn_id="999000111222")
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
        rows = (await s.execute(select(Transaction))).scalars().all()
        # The ₹5,555 row is untouched; no second row inserted.
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
        assert rows[0].amount == Decimal("5555")
        assert rows[0].transaction_date == datetime.date(2026, 5, 3)
        # The email is parked for manual review, not silently lost.
        email = await s.get(Email, email_id)
        assert email.status == "skipped"
        assert "dup-defer" in (email.error or "")


@pytest.mark.anyio
async def test_bulk_reparse_enriches_existing_sms_row(session_maker):
    """Bulk reparse-all-failed must dedup against an existing SMS-sourced
    transaction exactly like the single-email reparse: a failed email whose
    event already exists from an SMS row (no ref) must NOT create a second
    transaction — it must match/enrich the existing row instead of blind
    inserting."""
    sms_txn_id, email_id = await _seed_sms_row_and_failed_email(session_maker)

    raw = _hdfc_ppf_transfer_eml()
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
            r = await client.post("/emails/reparse-all-failed")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        # Exactly one row — the SMS row, now enriched with the email.
        assert len(rows) == 1, (
            f"expected 1 row, got {len(rows)}: {[r.id for r in rows]}"
        )
        row = rows[0]
        assert row.id == sms_txn_id
        assert row.source == "sms+email"
        assert row.email_id == email_id
        # Downgrade-safe enrichment: the SMS row's in-body time must NOT be
        # clobbered by the email's missing time.
        assert row.transaction_time == datetime.time(15, 15, 12)


@pytest.mark.anyio
async def test_bulk_reparse_retains_sms_enrichment_on_none(session_maker):
    """Bulk reparse must not NULL an SMS-provided value when the email's
    txn_data carries that field as None (fill-don't-clobber)."""
    # Seed an SMS row with a balance + counterparty the email lacks.
    async with session_maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="alerts@hdfcbank.bank.in",
            bank="hdfc",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        await session.flush()
        sms_txn = Transaction(
            bank="hdfc",
            email_type="hdfc_account_transfer_debit_alert",
            direction="debit",
            amount=Decimal("100000"),
            currency="INR",
            transaction_date=datetime.date(2026, 6, 5),
            transaction_time=datetime.time(15, 15, 12),
            counterparty="PPF/SSY A/c XX0000",
            channel="online",
            balance=Decimal("4242.00"),
            source="sms",
            notified_channel="sms",
        )
        session.add(sms_txn)
        email_row = Email(
            provider="gmail",
            message_id="test-hdfc-ppf-bulk-enrich",
            sender="alerts@hdfcbank.bank.in",
            subject="View: Account update for your HDFC Bank A/c",
            received_at=datetime.datetime(2026, 6, 5, 9, 45, 11, tzinfo=datetime.UTC),
            status="failed",
            error="Previous parse failed",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()
        sms_txn_id = sms_txn.id

    raw = _hdfc_ppf_transfer_eml()  # carries no balance
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
            r = await client.post("/emails/reparse-all-failed")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        row = await s.get(Transaction, sms_txn_id)
        # SMS-only balance + counterparty must survive the email reparse.
        assert row.balance == Decimal("4242.00")
        assert row.counterparty == "PPF/SSY A/c XX0000"


@pytest.mark.anyio
async def test_single_reparse_retains_sms_enrichment_on_none(session_maker):
    """Single-email reparse upsert of an existing SMS-enriched row whose
    email txn_data has balance/counterparty as None must RETAIN the SMS
    values rather than clobber them with None."""
    # Seed an SMS row already attached to THIS email (the in-place upsert
    # path), with a balance + counterparty the email lacks.
    async with session_maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="alerts@hdfcbank.bank.in",
            bank="hdfc",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        await session.flush()
        email_row = Email(
            provider="gmail",
            message_id="test-hdfc-ppf-single-enrich",
            sender="alerts@hdfcbank.bank.in",
            subject="View: Account update for your HDFC Bank A/c",
            received_at=datetime.datetime(2026, 6, 5, 9, 45, 11, tzinfo=datetime.UTC),
            status="parsed",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.flush()
        txn = Transaction(
            bank="hdfc",
            email_type="hdfc_account_transfer_debit_alert",
            direction="debit",
            amount=Decimal("100000"),
            currency="INR",
            transaction_date=datetime.date(2026, 6, 5),
            transaction_time=datetime.time(15, 15, 12),
            counterparty="PPF/SSY A/c XX0000",
            channel="online",
            balance=Decimal("4242.00"),
            source="sms+email",
            notified_channel="sms",
            email_id=email_row.id,
        )
        session.add(txn)
        await session.commit()
        txn_id = txn.id
        email_id = email_row.id

    raw = _hdfc_ppf_transfer_eml()  # carries no balance
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
        row = await s.get(Transaction, txn_id)
        # SMS-only balance + counterparty must survive the same-source reparse.
        assert row.balance == Decimal("4242.00")
        assert row.counterparty == "PPF/SSY A/c XX0000"


@pytest.mark.anyio
async def test_reparse_does_not_steal_row_claimed_by_another_email(session_maker):
    """If the matched cross-channel row is already attached to a DIFFERENT
    email, reparsing a second email must NOT steal that link — it inserts
    its own row. (Guards against orphaning the first email.)"""
    # Seed: one SMS row, ALREADY claimed by email A; plus failed email B
    # whose parse matches the same event.
    async with session_maker() as session:
        rule = FetchRule(
            provider="gmail",
            sender="alerts@hdfcbank.bank.in",
            bank="hdfc",
            enabled=True,
            email_kind="transaction",
        )
        session.add(rule)
        await session.flush()

        email_a = Email(
            provider="gmail",
            message_id="test-hdfc-ppf-A",
            sender="alerts@hdfcbank.bank.in",
            subject="View: Account update for your HDFC Bank A/c",
            received_at=datetime.datetime(2026, 6, 5, 9, 45, 11, tzinfo=datetime.UTC),
            status="parsed",
            rule_id=rule.id,
        )
        email_b = Email(
            provider="gmail",
            message_id="test-hdfc-ppf-B",
            sender="alerts@hdfcbank.bank.in",
            subject="View: Account update for your HDFC Bank A/c",
            received_at=datetime.datetime(2026, 6, 5, 9, 46, 0, tzinfo=datetime.UTC),
            status="failed",
            error="Previous parse failed",
            rule_id=rule.id,
        )
        session.add_all([email_a, email_b])
        await session.flush()

        claimed = Transaction(
            bank="hdfc",
            email_type="hdfc_account_transfer_debit_alert",
            direction="debit",
            amount=Decimal("100000"),
            currency="INR",
            transaction_date=datetime.date(2026, 6, 5),
            counterparty="PPF/SSY A/c XX0000",
            channel="online",
            source="email",
            email_id=email_a.id,
        )
        session.add(claimed)
        await session.commit()
        email_a_id, email_b_id = email_a.id, email_b.id

    raw = _hdfc_ppf_transfer_eml()
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
            r = await client.post(f"/emails/{email_b_id}/reparse")
            assert r.status_code == 200, r.text

    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        # Email A keeps its row; email B gets its own. Neither orphaned.
        assert len(rows) == 2
        by_email = {r.email_id: r for r in rows}
        assert by_email[email_a_id].email_id == email_a_id
        assert by_email[email_b_id].email_id == email_b_id
