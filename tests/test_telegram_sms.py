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
