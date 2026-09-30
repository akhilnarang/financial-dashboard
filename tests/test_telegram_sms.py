"""Tests for Telegram notifications, prompts, callbacks, and send retries."""

from contextlib import contextmanager
from datetime import date, time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from financial_dashboard.db import TelegramMessageContext
from financial_dashboard.services.sms_duplicate_resolution import (
    SmsDuplicateResolutionResult,
)
from financial_dashboard.services.telegram import (
    _handle_callback,
    _parse_sms_duplicate_callback,
    _send_with_retry,
    send_bulk_summary,
    send_disambiguation_prompt,
    send_sms_duplicate_disambiguation_prompt,
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
async def test_notify_only_notification_names_no_transaction():
    """A notify-only role has no transaction row, so the text has no #txn_id."""
    with _capture_text() as captured:
        await send_transaction_notification(
            0,
            {
                "_ledger_role": "provisional",
                "direction": "credit",
                "amount": Decimal("50000"),
                "bank": "hdfc",
                "card_mask": "9710",
            },
            chat_id=12345,
            source="sms",
        )
    text = captured["text"]
    assert "not yet settled" in text.lower()
    assert "#" not in text
    assert "50,000.00" in text


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


def _mock_app(send_side_effect):
    app = MagicMock()
    app.bot.send_message = AsyncMock(side_effect=send_side_effect)
    return app


@pytest.mark.anyio
async def test_send_retry_after_does_not_consume_attempts(monkeypatch):
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)

    # Two RetryAfters do not count. One TimedOut uses one attempt.
    app = _mock_app([RetryAfter(2), RetryAfter(1), TimedOut(), None])
    await _send_with_retry(app, chat_id=1, text="hi")

    assert app.bot.send_message.await_count == 4
    assert sleep_mock.await_args_list == [((2.5,),), ((1.5,),), ((1,),)]


@pytest.mark.anyio
async def test_send_exhausts_retries_and_reraises(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    app = _mock_app([TimedOut(), TimedOut(), TimedOut()])
    with pytest.raises(NetworkError):
        await _send_with_retry(app, chat_id=1, text="hi")

    assert app.bot.send_message.await_count == 3


@pytest.mark.anyio
async def test_send_disambiguation_prompt_builds_keyboard():
    fake_app = MagicMock()
    fake_app.bot.send_message = AsyncMock()

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

    markup = fake_app.bot.send_message.await_args.kwargs["reply_markup"]
    cb_data = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert any(d.startswith("cc_pay_pick:42:10") for d in cb_data)
    assert any(d.startswith("cc_pay_pick:42:20") for d in cb_data)
    assert any(d.startswith("cc_pay_pick:42:skip") for d in cb_data)


@pytest.mark.anyio
async def test_sms_duplicate_prompt_offers_only_safe_actions():
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
        assert "<synthetic shop>" not in kwargs["text"]

        # A reference mismatch never offers "Create new".
        await send_sms_duplicate_disambiguation_prompt(
            {
                **payload,
                "reason": "reference_balance_mismatch",
                "resolution_candidate_ids": [],
            },
            chat_id=12345,
        )
        assert fake_app.bot.send_message.await_args.kwargs["reply_markup"] is None


def test_sms_duplicate_callback_parser_contract():
    assert _parse_sms_duplicate_callback("smsdup:v1:m:17:29") == ("merge", 17, 29)
    assert _parse_sms_duplicate_callback("smsdup:v1:n:17") == ("create_new", 17, None)
    for data in (
        "smsdup:v1:m:0:2",
        "smsdup:v1:n:1:2",
        "smsdup:v1:n:not-an-int",
        "smsdup:v1:n:" + "1" * 65,
    ):
        assert _parse_sms_duplicate_callback(data) is None


@pytest.mark.anyio
async def test_sms_duplicate_callback_rejects_wrong_chat(monkeypatch):
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
