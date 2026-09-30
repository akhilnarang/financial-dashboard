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


@pytest.mark.anyio
async def test_send_sms_duplicate_prompt_single_candidate_has_two_safe_actions():
    from financial_dashboard.services.telegram import (
        send_sms_duplicate_disambiguation_prompt,
    )

    fake_app = MagicMock()
    fake_app.bot.send_message = AsyncMock()
    payload = {
        "sms_id": 17,
        "reason": "balance_ambiguous",
        "resolution_candidate_ids": [29],
        "amount": Decimal("246.80"),
        "bank": "<sample&bank>",
        "direction": "debit",
        "counterparty": "<synthetic shop>",
        "transaction_date": "2026-08-12",
    }

    with patch("financial_dashboard.services.telegram.tg_app", new=fake_app):
        await send_sms_duplicate_disambiguation_prompt(payload, chat_id=12345)

    kwargs = fake_app.bot.send_message.await_args.kwargs
    callbacks = [
        button.callback_data
        for row in kwargs["reply_markup"].inline_keyboard
        for button in row
    ]
    assert callbacks == ["smsdup:v1:m:17:29", "smsdup:v1:n:17"]
    assert "&lt;SAMPLE&amp;BANK&gt;" in kwargs["text"]
    assert "&lt;synthetic shop&gt;" in kwargs["text"]
    assert "<synthetic shop>" not in kwargs["text"]


@pytest.mark.anyio
async def test_send_sms_duplicate_prompt_never_offers_create_for_reference_mismatch():
    from financial_dashboard.services.telegram import (
        send_sms_duplicate_disambiguation_prompt,
    )

    fake_app = MagicMock()
    fake_app.bot.send_message = AsyncMock()
    payload = {
        "sms_id": 17,
        "reason": "reference_balance_mismatch",
        "resolution_candidate_ids": [],
        "amount": Decimal("246.80"),
        "bank": "samplebank",
        "direction": "debit",
    }

    with patch("financial_dashboard.services.telegram.tg_app", new=fake_app):
        await send_sms_duplicate_disambiguation_prompt(payload, chat_id=12345)

    kwargs = fake_app.bot.send_message.await_args.kwargs
    assert kwargs["reply_markup"] is None


@pytest.mark.parametrize(
    "data",
    [
        "smsdup:v1:m:0:2",
        "smsdup:v1:m:1:-2",
        "smsdup:v1:m:1",
        "smsdup:v1:n:1:2",
        "smsdup:v1:x:1",
        "smsdup:v1:n:not-an-int",
        "smsdup:v1:n:" + "1" * 65,
    ],
)
def test_sms_duplicate_callback_parser_rejects_invalid_contract(data):
    from financial_dashboard.services.telegram import _parse_sms_duplicate_callback

    assert _parse_sms_duplicate_callback(data) is None


def test_sms_duplicate_callback_parser_accepts_contract():
    from financial_dashboard.services.telegram import _parse_sms_duplicate_callback

    assert _parse_sms_duplicate_callback("smsdup:v1:m:17:29") == (
        "merge",
        17,
        29,
    )
    assert _parse_sms_duplicate_callback("smsdup:v1:n:17") == (
        "create_new",
        17,
        None,
    )


@pytest.mark.anyio
async def test_sms_duplicate_callback_rejects_wrong_chat(monkeypatch):
    from financial_dashboard.services.telegram import _handle_callback

    query = MagicMock()
    query.data = "smsdup:v1:m:17:29"
    query.message.chat.id = 999
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    monkeypatch.setattr(
        "financial_dashboard.services.telegram.get_telegram_chat_id", lambda: 12345
    )

    await _handle_callback(update, MagicMock())

    query.answer.assert_awaited_once_with("Unauthorized")
    query.edit_message_text.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("late_tap", [False, True])
async def test_sms_duplicate_double_tap_loser_sees_already_resolved(
    monkeypatch, late_tap
):
    from telegram.error import BadRequest

    from financial_dashboard.services.sms_duplicate_resolution import (
        SmsDuplicateResolutionResult,
    )
    from financial_dashboard.services.telegram import _handle_callback

    query = MagicMock()
    query.data = "smsdup:v1:n:17"
    query.message.chat.id = 12345
    # Telegram rejects the answer to a tap replayed after a restart. The
    # committed result must still reach the chat.
    query.answer = AsyncMock(
        side_effect=BadRequest("Query is too old") if late_tap else None
    )
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    class SessionContext:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    resolver = AsyncMock(
        return_value=SmsDuplicateResolutionResult("already_resolved", 29)
    )
    monkeypatch.setattr(
        "financial_dashboard.services.telegram.get_telegram_chat_id", lambda: 12345
    )
    monkeypatch.setattr(
        "financial_dashboard.services.telegram.async_session",
        lambda: SessionContext(),
    )
    monkeypatch.setattr(
        "financial_dashboard.services.sms_duplicate_resolution.resolve_sms_duplicate",
        resolver,
    )

    await _handle_callback(update, MagicMock())

    resolver.assert_awaited_once()
    query.edit_message_text.assert_awaited_once_with("Already resolved as #29")
