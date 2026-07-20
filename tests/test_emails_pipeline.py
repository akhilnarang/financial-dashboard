"""Focused integration tests for the live email pipeline
(``services.emails.handle_polled_email``) and the email reparse route.

These cover the branches ``handle_polled_email`` owns that no other test
exercises directly:

- SBI declined transaction alerts are notify-only (no Transaction row).
- The ``telegram.bulk_threshold`` boundary swaps per-row notifications for a
  single bulk summary.
- A created CC-payment credit reconciles its statement exactly once; a second
  source that enriches the same row must NOT re-fire ``check_payment_received``
  (``payment_paid_amount`` is cumulative).
- An enriched row re-links when the second source fills a mask the linker can
  now resolve.
- Reparsing an email with more than one attached transaction surfaces a 409
  (``MultipleResultsFound``) instead of silently picking one.

``handle_polled_email`` opens its own ``async_session()``; the tests install the
in-memory maker onto ``services.emails.async_session`` (the established pattern
used by ``test_reparse_*`` for ``core.deps``/``reminders``) and stub the parser
via ``_process_email_full`` so the scenarios don't depend on real bank email
corpora.
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
import financial_dashboard.services.emails as emails_mod
import financial_dashboard.services.reminders as reminders_module
from financial_dashboard.core.deps import get_session
from financial_dashboard.db import (
    Account,
    Base,
    Card,
    Email,
    FetchRule,
    Transaction,
)
from financial_dashboard.services.emails import (
    EmailDispatchResult,
    ProcessedEmailParse,
    handle_polled_email,
)
from financial_dashboard.services.linker import build_link_context
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
    # handle_polled_email / check_payment_received open their own async_session();
    # install the in-memory maker so they see (and commit to) the test DB.
    monkeypatch.setattr(emails_mod, "async_session", maker)
    monkeypatch.setattr(reminders_module, "async_session", maker)
    monkeypatch.setattr(core_deps, "async_session", maker)
    yield maker
    await engine.dispose()


def _build_web_app(maker):
    app = FastAPI()
    app.include_router(get_web_router())

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    return app


def _raw_email(subject: str = "Test", date_header: str | None = None) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "alerts@example.bank.in"
    msg["Date"] = date_header or "Tue, 02 Jun 2026 09:45:11 +0000"
    msg.set_content("body")
    return msg.as_bytes()


def _txn_data(**overrides) -> dict:
    base = {
        "bank": "axis",
        "email_type": "axis_cc_transaction_alert",
        "direction": "debit",
        "amount": Decimal("500"),
        "currency": "INR",
        "transaction_date": datetime.date(2026, 6, 2),
        "transaction_time": datetime.time(10, 0, 0),
        "counterparty": "Zomato",
        "card_mask": None,
        "account_mask": None,
        "reference_number": None,
        "channel": None,
        "balance": None,
        "raw_description": None,
    }
    base.update(overrides)
    return base


async def _seed_rule(maker, *, bank="axis", email_kind="transaction") -> int:
    async with maker() as s:
        rule = FetchRule(
            provider="gmail",
            sender="alerts@example.bank.in",
            bank=bank,
            enabled=True,
            email_kind=email_kind,
            initial_backfill_done_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        )
        s.add(rule)
        await s.commit()
        return rule.id


def _stub_parser(monkeypatch, txn_data):
    """Make services.emails._process_email_full return *txn_data*."""
    monkeypatch.setattr(
        emails_mod,
        "_process_email_full",
        lambda bank, raw: ProcessedEmailParse(None, txn_data, None, None),
    )


async def _run_handle_polled_email(
    maker,
    monkeypatch,
    *,
    rule_id,
    txn_data,
    msg_id,
    raw_bytes=None,
    should_notify=True,
):
    """Drive handle_polled_email with the parser stubbed to return *txn_data*.

    Only the parser is stubbed here; each test patches the Telegram senders /
    ``check_payment_received`` it needs to assert on (or passes
    ``should_notify=False`` to skip dispatch entirely).
    """
    _stub_parser(monkeypatch, txn_data)

    async with maker() as s:
        rule = await s.get(FetchRule, rule_id)
        link_ctx = await build_link_context(s)

    stats = {"parsed": 0, "skipped": 0, "failed": 0, "fetched": 0}
    await handle_polled_email(
        rule=rule,
        provider="gmail",
        source_id=1,
        msg_id=msg_id,
        # remote_id is unique per (source_id, remote_id); derive from msg_id so
        # two calls in one test don't collide on that constraint.
        remote_id=f"remote-{msg_id}",
        raw_bytes=raw_bytes or _raw_email(),
        should_notify=should_notify,
        link_context=link_ctx,
        stats=stats,
    )
    return stats


# ---------------------------------------------------------------------------
# SBI declined alerts are notify-only.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sbi_declined_is_notify_only(session_maker, monkeypatch):
    """An ``sbi_cc_transaction_declined`` txn is notify-only: the email is
    marked parsed, NO transaction row is created, and the primary notification
    carries the ``_declined`` flag."""
    rule_id = await _seed_rule(session_maker, bank="sbi")
    txn_data = _txn_data(
        bank="sbi",
        email_type="sbi_cc_transaction_declined",
        direction="debit",
        reference_number=None,
    )

    captured = []

    async def _capture_notification(txn_id, txn_info, chat_id, **kwargs):
        captured.append((txn_id, txn_info))

    _stub_parser(monkeypatch, txn_data)
    monkeypatch.setattr(
        emails_mod, "send_transaction_notification", _capture_notification
    )
    monkeypatch.setattr(emails_mod, "send_bulk_summary", AsyncMock())
    monkeypatch.setattr(emails_mod, "send_enrichment_notification", AsyncMock())
    monkeypatch.setattr(emails_mod, "send_disambiguation_prompt", AsyncMock())

    async with session_maker() as s:
        rule = await s.get(FetchRule, rule_id)
        link_ctx = await build_link_context(s)

    stats = {"parsed": 0, "skipped": 0, "failed": 0, "fetched": 0}
    await handle_polled_email(
        rule=rule,
        provider="gmail",
        source_id=1,
        msg_id="sbi-declined-1",
        remote_id="remote-1",
        raw_bytes=_raw_email(subject="Transaction declined"),
        should_notify=True,
        link_context=link_ctx,
        stats=stats,
    )

    assert stats["parsed"] == 1
    # No transaction row created.
    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        assert rows == []
        em = (await s.execute(select(Email))).scalar_one()
        assert em.status == "parsed"
        assert em.error is None
    # Declined notification fired with the _declined marker.
    assert len(captured) == 1
    _, info = captured[0]
    assert info.get("_declined") is True


# ---------------------------------------------------------------------------
# Bulk-threshold boundary: above the threshold, a single bulk summary replaces
# per-row notifications.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bulk_threshold_swaps_to_summary(session_maker, monkeypatch):
    """With ``telegram.bulk_threshold`` set below the number of primaries, the
    poll sends one bulk summary instead of per-row notifications."""
    from financial_dashboard.services import settings as settings_mod

    rule_id = await _seed_rule(session_maker, bank="axis")
    txn_data = _txn_data(reference_number="BULK-PRIMARY-1")

    settings_mod._cache["telegram.bulk_threshold"] = "0"

    primary = AsyncMock()
    bulk = AsyncMock()
    monkeypatch.setattr(emails_mod, "send_transaction_notification", primary)
    monkeypatch.setattr(emails_mod, "send_bulk_summary", bulk)
    monkeypatch.setattr(emails_mod, "send_enrichment_notification", AsyncMock())
    monkeypatch.setattr(emails_mod, "send_disambiguation_prompt", AsyncMock())

    await _run_handle_polled_email(
        session_maker,
        monkeypatch,
        rule_id=rule_id,
        txn_data=txn_data,
        msg_id="bulk-threshold-1",
        should_notify=True,
    )

    bulk.assert_awaited_once()
    primary.assert_not_awaited()


# ---------------------------------------------------------------------------
# CC-payment reconciliation fires on create only (never on enrich).
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_payment_created_only_reconciles_enriched_does_not(
    session_maker, monkeypatch
):
    """A created CC bill-payment credit fires ``check_payment_received`` exactly
    once; a second source that enriches the same row must NOT re-fire it
    (payment_paid_amount is cumulative and would double-count)."""
    rule_id = await _seed_rule(session_maker, bank="axis")

    # One CC account so resolve_cc_payment_account auto-resolves (1 candidate).
    async with session_maker() as s:
        s.add(
            Account(
                bank="axis",
                type="credit_card",
                label="Axis CC",
                account_number="1234",
                active=True,
            )
        )
        await s.commit()

    check_calls = []

    async def _fake_check(txn_id, account_id, amount):
        check_calls.append((txn_id, account_id, amount))

    monkeypatch.setattr(emails_mod, "check_payment_received", _fake_check)

    ref = "CCPAY-CREATED-ONLY-1"
    txn_data = _txn_data(
        bank="axis",
        email_type="axis_cc_payment_received_alert",
        direction="credit",
        amount=Decimal("500"),
        reference_number=ref,
        card_mask=None,
    )

    # First email → created → reconciliation fires once.
    await _run_handle_polled_email(
        session_maker,
        monkeypatch,
        rule_id=rule_id,
        txn_data=txn_data,
        msg_id="ccpay-1",
        should_notify=False,
    )
    assert len(check_calls) == 1

    # Second email (same event, different email row) → enriched → must NOT
    # re-fire. The matcher pairs them by the shared reference_number.
    await _run_handle_polled_email(
        session_maker,
        monkeypatch,
        rule_id=rule_id,
        txn_data=txn_data,
        msg_id="ccpay-2",
        should_notify=False,
    )
    assert len(check_calls) == 1  # still exactly one

    # Exactly one transaction row exists, linked to the CC account.
    async with session_maker() as s:
        rows = (await s.execute(select(Transaction))).scalars().all()
        assert len(rows) == 1
        assert rows[0].account_id is not None


# ---------------------------------------------------------------------------
# Enriched row re-links when the second source fills a mask.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_email_enriched_relinks_when_mask_filled(session_maker, monkeypatch):
    """An existing row whose card the first source couldn't see gets re-linked
    when the email fills card_mask and the linker can now resolve it."""
    rule_id = await _seed_rule(session_maker, bank="hdfc")

    async with session_maker() as s:
        acct = Account(
            bank="hdfc",
            type="bank_account",
            label="HDFC Savings",
            account_number="000000001111",
        )
        s.add(acct)
        await s.flush()
        card = Card(account_id=acct.id, card_mask="1111", label="HDFC Debit")
        s.add(card)
        # Pre-existing row from SMS: ref present but card_mask NULL → unlinked.
        s.add(
            Transaction(
                bank="hdfc",
                email_type="hdfc_dc_transaction_alert",
                direction="debit",
                amount=Decimal("500"),
                currency="INR",
                transaction_date=datetime.date(2026, 6, 2),
                reference_number="RELINK-EMAIL-1",
                source="sms",
                card_mask=None,
                account_id=None,
            )
        )
        await s.commit()
        txn_id = (await s.execute(select(Transaction))).scalar_one().id

    txn_data = _txn_data(
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        reference_number="RELINK-EMAIL-1",
        card_mask="1111",  # the mask the SMS row lacked
    )

    await _run_handle_polled_email(
        session_maker,
        monkeypatch,
        rule_id=rule_id,
        txn_data=txn_data,
        msg_id="relink-email-1",
        should_notify=False,
    )

    async with session_maker() as s:
        row = await s.get(Transaction, txn_id)
        assert row.card_mask == "1111"
        assert row.account_id is not None
        assert row.card_id is not None
        # Still a single row, now cross-channel.
        rows = (await s.execute(select(Transaction))).scalars().all()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Reparse with >1 attached transaction → 409 (MultipleResultsFound).
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reparse_multiple_transactions_returns_409(session_maker, monkeypatch):
    """A historical quirk that left two transactions attached to one email must
    surface loudly as 409 on reparse — the operator resolves the duplicates
    before the reparse can proceed, rather than the handler silently picking
    one."""
    rule_id = await _seed_rule(session_maker, bank="axis")
    async with session_maker() as s:
        em = Email(
            provider="gmail",
            message_id="multi-1",
            sender="alerts@example.bank.in",
            subject="Test",
            received_at=datetime.datetime(2026, 6, 2, 9, 45, 11, tzinfo=datetime.UTC),
            status="failed",
            error="prior",
            rule_id=rule_id,
        )
        s.add(em)
        await s.flush()
        for _ in range(2):
            s.add(
                Transaction(
                    bank="axis",
                    email_type="axis_cc_transaction_alert",
                    direction="debit",
                    amount=Decimal("500"),
                    currency="INR",
                    email_id=em.id,
                )
            )
        await s.commit()
        email_id = em.id

    txn_data = _txn_data(reference_number="MULTI-REPARSE-1")
    # Stub the parser so reparse produces a txn_data without needing a real
    # bank-email corpus (and without writing to data/failed/).
    monkeypatch.setattr(
        "financial_dashboard.web.emails.parse_email_by_kind",
        AsyncMock(
            return_value=EmailDispatchResult(
                error=None, txn_data=txn_data, password_hint=None, stmt_result=None
            )
        ),
    )

    with (
        patch(
            "financial_dashboard.web.emails.load_or_fetch_raw_email",
            new=AsyncMock(return_value=(_raw_email(), None)),
        ),
        patch(
            "financial_dashboard.web.emails.should_notify_transactions",
            return_value=False,
        ),
    ):
        app = _build_web_app(session_maker)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(f"/emails/{email_id}/reparse")

    assert r.status_code == 409
    assert "more than one" in r.json()["detail"].lower()
