"""Tests for the Telegram CC-payment disambiguation prompt + callback."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_send_disambiguation_prompt_builds_keyboard():
    from financial_dashboard.services.telegram import send_disambiguation_prompt

    sent = {}

    async def fake_bot_send(chat_id, text, reply_markup, parse_mode=None):
        sent["text"] = text
        sent["markup"] = reply_markup

    fake_app = MagicMock()
    fake_app.bot.send_message = AsyncMock(side_effect=fake_bot_send)

    with patch("financial_dashboard.services.telegram.tg_app", new=fake_app):
        await send_disambiguation_prompt(
            {
                "txn_id": 42,
                "candidate_account_ids": [10, 20],
                "candidate_labels": {10: "Card-1234", 20: "Card-5678"},
                "amount": Decimal("2500"),
                "bank": "slice",
            },
            chat_id=12345,
        )
    assert (
        "couldn't auto-match" in sent["text"].lower()
        or "could not" in sent["text"].lower()
    )
    assert "₹2,500.00" in sent["text"] or "2500" in sent["text"]
    # The keyboard should have one button per candidate + a Skip.
    buttons = sent["markup"].inline_keyboard
    flat = [b for row in buttons for b in row]
    cb_data = [b.callback_data for b in flat]
    assert any(d.startswith("cc_pay_pick:42:10") for d in cb_data)
    assert any(d.startswith("cc_pay_pick:42:20") for d in cb_data)
    assert any(d.startswith("cc_pay_pick:42:skip") for d in cb_data)
