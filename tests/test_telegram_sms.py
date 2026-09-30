"""Tests for Telegram source badge and enrichment notification."""

from contextlib import contextmanager
from datetime import date, time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db import TelegramMessageContext
from financial_dashboard.services.telegram import (
    send_bulk_summary,
    send_enrichment_notification,
    send_transaction_notification,
)
from financial_dashboard.services.txn_merge import EnrichmentDiff


@pytest.fixture
def anyio_backend():
    return "asyncio"


@contextmanager
def _capture_text():
    """Capture the text that each Telegram send receives."""
    captured = {}

    async def fake_send(app, *, chat_id, text):
        captured["text"] = text

    with (
        patch("financial_dashboard.services.telegram.tg_app", new=object()),
        patch(
            "financial_dashboard.services.telegram._send_with_retry",
            new=AsyncMock(side_effect=fake_send),
        ),
    ):
        yield captured


@pytest.mark.anyio
async def test_send_transaction_notification_with_sms_source_includes_badge():
    with _capture_text() as captured:
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
@pytest.mark.parametrize(
    ("role", "label"),
    [("provisional", "not yet settled"), ("restatement", "already recorded")],
)
async def test_notify_only_notification_renders_role_label(role, label):
    """A notify-only role has no transaction row, so the text has no #txn_id."""
    with _capture_text() as captured:
        await send_transaction_notification(
            0,
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
    text = captured["text"]
    assert label in text.lower()
    assert "#" not in text
    assert "50,000.00" in text


@pytest.mark.anyio
async def test_send_enrichment_notification_inline_format_with_txn_info():
    """With txn_info, the enrichment renders as one line with context."""
    diff = EnrichmentDiff(
        filled={"channel": "upi"},
        overwritten={"counterparty": ("PZCREDIT0000000", "Phone Pe")},
    )

    with _capture_text() as captured:
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
    assert "\n" not in text
    assert "#42" in text
    assert "HDFC" in text
    assert "-₹500.00" in text
    assert "Zomato" in text
    assert "filled channel=upi" in text
    assert "PZCREDIT0000000" in text
    assert "Phone Pe" in text
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
async def test_send_enrichment_notification_renders_time_to_the_second():
    """Times render as HH:MM:SS. Microseconds drop, but seconds stay, so a
    seconds-level overwrite does not look like a no-op."""
    diff = EnrichmentDiff(
        filled={"transaction_time": "22:30:50.583000"},
        overwritten={"transaction_time": ("12:55:35", "12:55:20")},
    )

    with _capture_text() as captured:
        await send_enrichment_notification(99, diff, 12345, source="sms")
    text = captured["text"]
    assert "transaction_time=22:30:50" in text
    assert "583000" not in text
    assert "12:55:35→12:55:20" in text


@pytest.mark.anyio
async def test_foreign_currency_amount_is_not_shown_in_rupees():
    """A USD charge must carry its currency code, never the rupee sign."""
    with _capture_text() as captured:
        await send_transaction_notification(
            7,
            {
                "bank": "onecard",
                "direction": "debit",
                "amount": Decimal("12.34"),
                "currency": "usd",
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
    assert "via Email" in captured["text"]


@pytest.mark.anyio
async def test_bulk_summary_totals_each_currency_apart():
    txns = [
        (1, {"direction": "debit", "amount": Decimal("100.00"), "currency": "INR"}),
        (2, {"direction": "debit", "amount": Decimal("50.00"), "currency": None}),
        (3, {"direction": "debit", "amount": Decimal("12.34"), "currency": "USD"}),
    ]
    with _capture_text() as captured:
        await send_bulk_summary(3, 12345, source="email", txns=txns)
    assert "3 debits (₹150.00 + USD 12.34)" in captured["text"]
