"""Tests for Telegram source badge and enrichment notification."""

from datetime import date, time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db import TelegramMessageContext
from financial_dashboard.services.txn_merge import EnrichmentDiff


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_send_transaction_notification_with_sms_source_includes_badge():
    """Source badge 'via SMS' appears in the message text."""
    from financial_dashboard.services.telegram import send_transaction_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_transaction_notification(
                42,
                {
                    "bank": "hdfc",
                    "direction": "debit",
                    "amount": Decimal("500"),
                    "counterparty": "Zomato",
                    "transaction_date": date(2026, 5, 2),
                    "transaction_time": time(14, 23),
                    "card_mask": "x1234",
                    "account_label": "HDFC ····1234",
                    "channel": None,
                },
                chat_id=12345,
                source="sms",
            )
    assert "via SMS" in captured["text"]


@pytest.mark.anyio
async def test_send_transaction_notification_with_email_source_includes_badge():
    from financial_dashboard.services.telegram import send_transaction_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_transaction_notification(
                42,
                {
                    "bank": "hdfc",
                    "direction": "debit",
                    "amount": Decimal("500"),
                    "counterparty": "Zomato",
                    "card_mask": "x1234",
                },
                chat_id=12345,
                source="email",
            )
    assert "via Email" in captured["text"]


@pytest.mark.anyio
async def test_send_transaction_notification_no_source_no_badge():
    from financial_dashboard.services.telegram import send_transaction_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_transaction_notification(
                42,
                {"bank": "hdfc", "direction": "debit", "amount": Decimal("500")},
                chat_id=12345,
            )
    assert "via" not in captured["text"].lower()


async def _render_ledger_role_notification(role: str) -> str:
    """Render a notify-only SMS notification for a given ledger_role."""
    from financial_dashboard.services.telegram import send_transaction_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_transaction_notification(
                0,  # no txn id
                {
                    "_ledger_role": role,
                    "direction": "credit",
                    "amount": Decimal("50000"),
                    "bank": "hdfc",
                    "card_mask": "9710",
                },
                chat_id=12345,
                source="sms",
            )
    return captured["text"]


@pytest.mark.anyio
async def test_provisional_notification_renders_not_yet_settled():
    """A provisional ledger_role renders the 'not yet settled' copy and omits
    the #txn_id suffix (no transaction row exists for a provisional ping)."""
    text = await _render_ledger_role_notification("provisional")
    assert "not yet settled" in text.lower()
    assert "#" not in text  # no #txn_id suffix
    assert "50,000.00" in text


@pytest.mark.anyio
async def test_restatement_notification_renders_already_recorded():
    """A restatement ledger_role renders the 'already recorded' copy — the
    other notify-only label, distinct from provisional."""
    text = await _render_ledger_role_notification("restatement")
    assert "already recorded" in text.lower()
    assert "#" not in text
    assert "50,000.00" in text


@pytest.mark.anyio
async def test_send_enrichment_notification_renders_diff():
    from financial_dashboard.services.telegram import send_enrichment_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    diff = EnrichmentDiff(
        filled={"channel": "upi"},
        overwritten={"counterparty": ("PZCREDIT0000000", "Phone Pe")},
    )

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_enrichment_notification(42, diff, 12345, source="email")
    assert "#42" in captured["text"]
    assert "via Email" in captured["text"]
    assert "channel=upi" in captured["text"]
    assert "counterparty" in captured["text"]
    assert "PZCREDIT0000000" in captured["text"]
    assert "Phone Pe" in captured["text"]


@pytest.mark.anyio
async def test_send_enrichment_notification_inline_format_with_txn_info():
    """When txn_info is passed, the enrichment renders as a single
    inline line with bank/amount/counterparty context."""
    from financial_dashboard.services.telegram import send_enrichment_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    diff = EnrichmentDiff(filled={"channel": "upi"})

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_enrichment_notification(
                42,
                diff,
                12345,
                source="sms",
                txn_info={
                    "bank": "hdfc",
                    "direction": "debit",
                    "amount": Decimal("500"),
                    "counterparty": "Zomato",
                },
            )
    text = captured["text"]
    # Single-line inline form: txn id, bank, signed amount, counterparty,
    # diff fragment, source badge — all on one line.
    assert "\n" not in text
    assert "#42" in text
    assert "HDFC" in text
    assert "-₹500.00" in text
    assert "Zomato" in text
    assert "filled channel=upi" in text
    assert "via SMS" in text


@pytest.mark.anyio
async def test_enrichment_notification_records_reply_context(session, monkeypatch):
    from financial_dashboard.services import telegram

    maker = async_sessionmaker(
        session.bind, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(telegram, "async_session", maker)
    monkeypatch.setattr(telegram, "tg_app", object())

    async def fake_send(app, *, chat_id, text):
        assert text.splitlines()[0].endswith("#42")
        return SimpleNamespace(message_id=700)

    monkeypatch.setattr(telegram, "_send_with_retry", fake_send)

    await telegram.send_enrichment_notification(
        42,
        EnrichmentDiff(filled={"channel": "upi"}),
        12345,
        source="email",
    )

    async with maker() as verification:
        mapping = await verification.scalar(
            select(TelegramMessageContext).where(
                TelegramMessageContext.chat_id == 12345,
                TelegramMessageContext.message_id == 700,
            )
        )
    assert mapping.transaction_id == 42
    assert mapping.context_kind == "enrichment"


@pytest.mark.anyio
async def test_send_enrichment_notification_trims_time_microseconds():
    """transaction_time values with microsecond precision get rendered
    as HH:MM:SS — seconds are kept so seconds-level diffs (e.g. a
    received_at-derived guess vs a body-parsed exact time) are visible
    instead of looking like 'HH:MM→HH:MM' no-ops."""
    from financial_dashboard.services.telegram import send_enrichment_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    diff = EnrichmentDiff(filled={"transaction_time": "22:30:50.583000"})

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_enrichment_notification(99, diff, 12345, source="sms")
    text = captured["text"]
    assert "transaction_time=22:30:50" in text
    assert "583000" not in text  # microseconds dropped


@pytest.mark.anyio
async def test_send_enrichment_notification_shows_seconds_for_overwritten_time():
    """When an existing transaction_time gets overwritten, the diff
    renders both old and new with HH:MM:SS so a seconds-level diff
    (e.g. SMS-fallback 12:55:35 vs email-body 12:55:20) doesn't look
    like a no-op '12:55→12:55'."""
    from financial_dashboard.services.telegram import send_enrichment_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    diff = EnrichmentDiff(
        overwritten={"transaction_time": ("12:55:35", "12:55:20")},
    )

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_enrichment_notification(123, diff, 12345, source="email")
    text = captured["text"]
    assert "12:55:35→12:55:20" in text


@pytest.mark.anyio
async def test_foreign_currency_amount_is_not_shown_in_rupees():
    """A USD charge must carry its currency code, never the rupee sign."""
    from financial_dashboard.services.telegram import send_transaction_notification

    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    with patch("financial_dashboard.services.telegram.tg_app", new=object()):
        with patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ):
            await send_transaction_notification(
                7,
                {
                    "bank": "onecard",
                    "direction": "debit",
                    "amount": Decimal("12.34"),
                    "currency": "USD",
                    "counterparty": "CLOUDFLARE",
                    "transaction_date": date(2026, 9, 3),
                    "transaction_time": time(18, 41),
                    "card_mask": "x1234",
                    "account_label": "OneCard CC",
                    "channel": "card",
                },
                chat_id=12345,
                source="email",
            )
    assert "-USD 12.34" in captured["text"]
    assert "₹" not in captured["text"]


def test_format_money_defaults_to_rupees():
    from financial_dashboard.services.telegram import format_money

    assert format_money(Decimal("1234.5"), None) == "₹1,234.50"
    assert format_money(Decimal("1234.5"), "INR") == "₹1,234.50"
    assert format_money(Decimal("12.34"), "usd") == "USD 12.34"
