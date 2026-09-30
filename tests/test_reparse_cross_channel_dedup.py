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
from contextlib import contextmanager
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
from financial_dashboard.db import Account, Email, FetchRule, Transaction
from financial_dashboard.integrations.email.body import RawEmailResult
from financial_dashboard.services.emails import _process_email_full
from financial_dashboard.web import get_router as get_web_router
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


@contextmanager
def _reparse_patches(raw: bytes, *, notify: bool = False):
    """Serve ``raw`` as the stored email. Yield the two message mocks."""
    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            new=AsyncMock(return_value=RawEmailResult(raw, None, "provider")),
        ),
        patch(
            "financial_dashboard.web.emails.should_notify_transactions",
            return_value=notify,
        ),
        patch("financial_dashboard.web.emails.get_telegram_chat_id", return_value=1),
        patch(
            "financial_dashboard.web.emails.send_enrichment_notification",
            new=AsyncMock(),
        ) as enrich_msg,
        patch(
            "financial_dashboard.web.emails.send_transaction_notification",
            new=AsyncMock(),
        ) as txn_msg,
    ):
        yield enrich_msg, txn_msg


async def _post(maker, url: str) -> None:
    app = FastAPI()
    app.include_router(get_web_router())

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(url)
        assert r.status_code == 200, r.text


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


def _hdfc_neft_eml() -> bytes:
    """HDFC outward-NEFT email. Its event time comes from message arrival."""
    msg = EmailMessage()
    msg["Subject"] = "View: Account update for your HDFC Bank A/c"
    msg["From"] = "HDFC Bank InstaAlerts <alerts@hdfcbank.bank.in>"
    msg["Date"] = "Sun, 26 Jul 2026 19:33:51 +0000"
    msg.set_content(
        "Dear Customer, Rs. 1,234.56 has been deducted from your HDFC Bank "
        "account ending in XX0000 for a transfer to payee Sample Payee via "
        "NEFT using HDFC Bank Online Banking. Not you? Call 00000000000 from "
        "your registered mobile number."
    )
    return msg.as_bytes()


def _hdfc_rule() -> FetchRule:
    return FetchRule(
        provider="gmail",
        sender="alerts@hdfcbank.bank.in",
        bank="hdfc",
        enabled=True,
        email_kind="transaction",
    )


def _hdfc_email(rule_id: int, message_id: str, **fields) -> Email:
    values = {
        "status": "failed",
        "error": "Previous parse failed",
        "received_at": datetime.datetime(2026, 6, 5, 9, 45, 11, tzinfo=datetime.UTC),
    }
    values.update(fields)
    return Email(
        provider="gmail",
        message_id=message_id,
        sender="alerts@hdfcbank.bank.in",
        subject="View: Account update for your HDFC Bank A/c",
        rule_id=rule_id,
        **values,
    )


def _ppf_sms_row(**fields) -> Transaction:
    values = {
        "amount": Decimal("100000"),
        "transaction_time": datetime.time(15, 15, 12),
        "source": "sms",
        "notified_channel": "sms",
    }
    values.update(fields)
    return Transaction(
        bank="hdfc",
        email_type="hdfc_account_transfer_debit_alert",
        direction="debit",
        currency="INR",
        transaction_date=datetime.date(2026, 6, 5),
        counterparty="PPF/SSY A/c XX0000",
        channel="online",
        **values,
    )


async def _seed_sms_row_and_failed_email(maker) -> tuple[int, int]:
    """Seed an existing SMS-sourced HDFC transfer Transaction (no email
    attached) plus a matching failed Email row. Returns (sms_txn_id,
    email_id)."""
    async with maker() as session:
        rule = _hdfc_rule()
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

        sms_txn = _ppf_sms_row()
        session.add(sms_txn)
        email_row = _hdfc_email(rule.id, "test-hdfc-ppf-1")
        session.add(email_row)
        await session.commit()
        return sms_txn.id, email_row.id


async def _assert_sms_row_enriched(maker, sms_txn_id: int, email_id: int) -> None:
    async with maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        # Exactly one row — the SMS row, now enriched with the email.
        assert [r.id for r in rows] == [sms_txn_id]
        row = rows[0]
        assert row.source == "sms+email"
        assert row.email_id == email_id
        # Email carried the source mask → account link gets filled.
        assert row.account_mask == "XX1111"
        assert row.account_id is not None
        # Downgrade-safe enrichment: the SMS row's in-body time must NOT be
        # clobbered by the email's missing time.
        assert row.transaction_time == datetime.time(15, 15, 12)


@pytest.mark.anyio
async def test_reparse_twice_enriches_the_sms_row_once(session_maker):
    """Reparse the same email twice. The first run adds the email data to the
    row that the SMS made and sends one enrichment message. The second run
    finds the same row and changes no field.

    Send nothing. The row is not new, so a message about a new transaction is
    wrong. No field changed, so an enrichment message says nothing.
    """
    sms_txn_id, email_id = await _seed_sms_row_and_failed_email(session_maker)

    with _reparse_patches(_hdfc_ppf_transfer_eml(), notify=True) as (
        enrich_msg,
        txn_msg,
    ):
        await _post(session_maker, f"/emails/{email_id}/reparse")
        assert enrich_msg.await_count == 1, "the SMS row changed"
        assert txn_msg.await_count == 0, "there is only one payment"
        enrich_msg.reset_mock()
        txn_msg.reset_mock()

        await _post(session_maker, f"/emails/{email_id}/reparse")

    assert txn_msg.await_count == 0, "the row is not new"
    assert enrich_msg.await_count == 0, "no field changed"
    await _assert_sms_row_enriched(session_maker, sms_txn_id, email_id)


@pytest.mark.anyio
async def test_bulk_reparse_enriches_existing_sms_row(session_maker):
    """Bulk reparse-all-failed must dedup against an existing SMS-sourced
    transaction exactly like the single-email reparse."""
    sms_txn_id, email_id = await _seed_sms_row_and_failed_email(session_maker)

    with _reparse_patches(_hdfc_ppf_transfer_eml()):
        await _post(session_maker, "/emails/reparse-all-failed")

    await _assert_sms_row_enriched(session_maker, sms_txn_id, email_id)


@pytest.mark.anyio
async def test_reparse_email_does_not_match_different_amount(session_maker):
    """A pre-existing SMS row for a *different* amount must NOT be merged
    into — the email gets its own row (no false cross-channel dedup), and the
    user gets a message about a new transaction."""
    async with session_maker() as session:
        rule = _hdfc_rule()
        session.add(rule)
        await session.flush()
        session.add(_ppf_sms_row(amount=Decimal("200000")))
        email_row = _hdfc_email(rule.id, "test-hdfc-ppf-diffamt")
        session.add(email_row)
        await session.commit()
        email_id = email_row.id

    with _reparse_patches(_hdfc_ppf_transfer_eml(), notify=True) as (
        enrich_msg,
        txn_msg,
    ):
        await _post(session_maker, f"/emails/{email_id}/reparse")

    assert txn_msg.await_count == 1
    assert enrich_msg.await_count == 0
    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
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
    """A pre-existing row carries a reference_number that a *different-amount*
    email reparse also produces. The reparse must park the email in the
    [dup-defer] state. It must not blind-insert, which violates the
    (bank, reference_number, direction) unique index.

    The shared reference_number is the literal the pinned parser extracts
    from this email, so the test does not depend on the parser deploy chain."""
    raw = _kotak_digital_eml(amount="7777.00", txn_id="999000111222")
    shared_ref = _process_email_full("kotak", raw).txn_data["reference_number"]
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
        # The Kotak digital mail is a credit. Direction is part of the match key.
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

    with _reparse_patches(raw):
        await _post(session_maker, f"/emails/{email_id}/reparse")

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
async def test_single_reparse_retains_sms_enrichment_on_none(session_maker):
    """Single-email reparse upsert of an existing SMS-enriched row whose
    email txn_data has balance/counterparty as None must RETAIN the SMS
    values rather than clobber them with None."""
    # Seed an SMS row already attached to THIS email (the in-place upsert
    # path), with a balance + counterparty the email lacks.
    async with session_maker() as session:
        rule = _hdfc_rule()
        session.add(rule)
        await session.flush()
        email_row = _hdfc_email(
            rule.id, "test-hdfc-ppf-single-enrich", status="parsed", error=None
        )
        session.add(email_row)
        await session.flush()
        txn = _ppf_sms_row(
            balance=Decimal("4242.00"), source="sms+email", email_id=email_row.id
        )
        session.add(txn)
        await session.commit()
        txn_id = txn.id
        email_id = email_row.id

    with _reparse_patches(_hdfc_ppf_transfer_eml()):
        await _post(session_maker, f"/emails/{email_id}/reparse")

    async with session_maker() as s:
        row = await s.get(Transaction, txn_id)
        # SMS-only balance + counterparty must survive the same-source reparse.
        assert row.balance == Decimal("4242.00")
        assert row.counterparty == "PPF/SSY A/c XX0000"


@pytest.mark.anyio
async def test_reparse_does_not_steal_row_claimed_by_another_email(session_maker):
    """A fuzzy candidate that another email owns is not a match. The second
    email inserts its own row. Thus neither email is orphaned."""
    async with session_maker() as session:
        rule = _hdfc_rule()
        session.add(rule)
        await session.flush()

        email_a = _hdfc_email(rule.id, "test-hdfc-ppf-A", status="parsed", error=None)
        email_b = _hdfc_email(
            rule.id,
            "test-hdfc-ppf-B",
            received_at=datetime.datetime(2026, 6, 5, 9, 46, 0, tzinfo=datetime.UTC),
        )
        session.add_all([email_a, email_b])
        await session.flush()

        session.add(
            _ppf_sms_row(transaction_time=None, source="email", email_id=email_a.id)
        )
        await session.commit()
        email_a_id, email_b_id = email_a.id, email_b.id

    with _reparse_patches(_hdfc_ppf_transfer_eml()):
        await _post(session_maker, f"/emails/{email_b_id}/reparse")

    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        # Email A keeps its row; email B gets its own. Neither orphaned.
        assert sorted(r.email_id for r in rows) == sorted([email_a_id, email_b_id])


@pytest.mark.anyio
async def test_reparse_ref_hit_does_not_steal_row_of_another_email(session_maker):
    """An exact reference hit returns a match even when another email owns the
    row. The reparse must not move that row to the second email."""
    raw = _kotak_digital_eml(amount="7777.00", txn_id="999000111222")
    txn_data = _process_email_full("kotak", raw).txn_data
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
        emails = [
            Email(
                provider="gmail",
                message_id=f"test-kotak-digital-{name}",
                sender="no-reply@kotak.com",
                subject="Transaction Successful",
                received_at=datetime.datetime(
                    2026, 6, 9, 12, 14, 7, tzinfo=datetime.UTC
                ),
                status=status,
                rule_id=rule.id,
            )
            for name, status in (("A", "parsed"), ("B", "failed"))
        ]
        session.add_all(emails)
        await session.flush()
        session.add(
            Transaction(
                bank="kotak",
                email_type="kotak_digital_transaction",
                direction=txn_data["direction"],
                amount=txn_data["amount"],
                currency="INR",
                transaction_date=txn_data["transaction_date"],
                reference_number=txn_data["reference_number"],
                source="email",
                email_id=emails[0].id,
            )
        )
        await session.commit()
        email_a_id, email_b_id = emails[0].id, emails[1].id

    app = FastAPI()
    app.include_router(get_web_router())

    async def _override():
        async with session_maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    with _reparse_patches(raw):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(f"/emails/{email_b_id}/reparse")

    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        assert [r.email_id for r in rows] == [email_a_id]
        assert (await s.get(Email, email_b_id)).status == "skipped"


@pytest.mark.anyio
async def test_reparse_moves_received_time_provenance_with_a_changed_time(
    session_maker,
):
    """The time-source flag describes the stored transaction time.

    A parser fix can make a reparse change the time. If the new time comes
    from message arrival, the flag must change to true. If the flag stays
    false, the matcher uses the unsafe 10-minute window.
    """
    async with session_maker() as session:
        rule = _hdfc_rule()
        session.add(rule)
        await session.flush()
        email_row = _hdfc_email(
            rule.id,
            "test-hdfc-neft-time-provenance",
            received_at=datetime.datetime(2026, 7, 26, 19, 33, 51, tzinfo=datetime.UTC),
            error="Previous parser supplied a different time",
        )
        session.add(email_row)
        await session.flush()
        transaction = Transaction(
            email_id=email_row.id,
            bank="hdfc",
            email_type="hdfc_account_neft_debit_alert",
            direction="debit",
            amount=Decimal("1234.56"),
            currency="INR",
            transaction_date=datetime.date(2026, 7, 26),
            transaction_time=datetime.time(19, 33, 0),
            transaction_time_is_received_time=False,
            source="email",
        )
        session.add(transaction)
        await session.commit()
        email_id = email_row.id
        txn_id = transaction.id

    with _reparse_patches(_hdfc_neft_eml(), notify=True) as (enrich_msg, txn_msg):
        await _post(session_maker, f"/emails/{email_id}/reparse")

    assert enrich_msg.await_count == 1, "the existing row changed"
    assert txn_msg.await_count == 0, "the row is not new"
    diff = enrich_msg.await_args.args[1]
    assert "transaction_time" in diff.overwritten

    async with session_maker() as session:
        stored = await session.get(Transaction, txn_id)
        assert stored is not None
        assert stored.transaction_date == datetime.date(2026, 7, 27)
        assert stored.transaction_time == datetime.time(1, 3, 51)
        assert stored.transaction_time_is_received_time is True
