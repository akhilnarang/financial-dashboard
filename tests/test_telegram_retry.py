"""Tests for _send_with_retry network-error retry logic."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import NetworkError, RetryAfter, TimedOut

from financial_dashboard.services.telegram import _send_with_retry


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _mock_app(send_side_effect):
    """Build a mock with `app.bot.send_message` that runs `send_side_effect`."""
    app = MagicMock()
    app.bot = MagicMock()
    app.bot.send_message = AsyncMock(side_effect=send_side_effect)
    return app


@pytest.mark.anyio
class TestSendWithRetry:
    async def test_retries_then_succeeds(self, monkeypatch):
        sleep_mock = AsyncMock()
        monkeypatch.setattr("asyncio.sleep", sleep_mock)

        app = _mock_app([TimedOut(), TimedOut(), None])
        await _send_with_retry(app, chat_id=42, text="hello", parse_mode="HTML")

        assert app.bot.send_message.await_count == 3
        app.bot.send_message.assert_awaited_with(
            chat_id=42, text="hello", parse_mode="HTML"
        )
        # 1s after first failure, 2s after second
        assert sleep_mock.await_args_list == [((1,),), ((2,),)]

    async def test_exhausts_retries_and_reraises(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", AsyncMock())

        app = _mock_app([TimedOut(), TimedOut(), TimedOut()])
        with pytest.raises(NetworkError):
            await _send_with_retry(app, chat_id=1, text="hi")

        assert app.bot.send_message.await_count == 3

    async def test_retry_after_does_not_consume_attempts(self, monkeypatch):
        sleep_mock = AsyncMock()
        monkeypatch.setattr("asyncio.sleep", sleep_mock)

        # 2 RetryAfters (don't count) + 1 TimedOut (1 attempt used) + success
        app = _mock_app(
            [
                RetryAfter(2),
                RetryAfter(1),
                TimedOut(),
                None,
            ]
        )
        await _send_with_retry(app, chat_id=1, text="hi")

        assert app.bot.send_message.await_count == 4
        # 2.5s + 1.5s for RetryAfter, then 1s for TimedOut backoff
        assert sleep_mock.await_args_list == [((2.5,),), ((1.5,),), ((1,),)]
