"""Fetch orchestration service stored on app.state."""

import asyncio
import logging

from financial_dashboard.integrations.email import orchestrator as fetch_orchestrator
from financial_dashboard.services.reminders import check_and_send_reminders
from financial_dashboard.services.settings import get_setting_int

logger = logging.getLogger(__name__)


def make_poll_status() -> dict:
    return {
        "state": "idle",
        "started_at": None,
        "finished_at": None,
        "last_stats": None,
        "last_error": None,
        "progress": None,
    }


class FetchService:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.status = make_poll_status()
        self._poll_loop_task: asyncio.Task | None = None
        self._active_poll_task: asyncio.Task | None = None

    def get_poll_status(self) -> dict:
        return fetch_orchestrator.get_poll_status(self.status)

    async def poll_all(self) -> dict:
        return await fetch_orchestrator.poll_all(
            poll_lock=self._lock,
            poll_status=self.status,
        )

    async def trigger_poll(self) -> bool:
        status = self.get_poll_status()
        if status["state"] == "polling" or (
            self._active_poll_task and not self._active_poll_task.done()
        ):
            return False
        self._active_poll_task = asyncio.create_task(self.poll_all())
        self._active_poll_task.add_done_callback(self._track_poll_task)
        return True

    async def start_poll_loop(self) -> None:
        if self._poll_loop_task and not self._poll_loop_task.done():
            return
        self._poll_loop_task = asyncio.create_task(self._poll_loop())

    async def stop_poll_loop(self) -> None:
        if self._poll_loop_task is not None:
            self._poll_loop_task.cancel()
            try:
                await self._poll_loop_task
            except asyncio.CancelledError:
                pass
            self._poll_loop_task = None

    def _track_poll_task(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception:
            logger.exception("Manual poll failed")
        finally:
            if self._active_poll_task is task:
                self._active_poll_task = None

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self.poll_all()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Background poll failed")

            try:
                if sent := await check_and_send_reminders():
                    logger.info("Sent %d payment reminder(s)", sent)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Reminder check failed")

            interval = max(1, get_setting_int("poll_interval_minutes", 15)) * 60
            await asyncio.sleep(interval)
