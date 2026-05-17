"""Tests for Telegram source badge and enrichment notification."""

from datetime import date, time
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

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
                    "bank": "hdfc", "direction": "debit",
                    "amount": Decimal("500"), "counterparty": "Zomato",
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
