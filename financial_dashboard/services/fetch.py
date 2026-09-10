"""Fetch orchestration service stored on app.state."""

import asyncio
import logging
from typing import TYPE_CHECKING

from financial_dashboard.integrations.email import orchestrator as fetch_orchestrator
from financial_dashboard.db import async_session
from financial_dashboard.services.categorization.merchant_rules import (
    load_merchant_rules,
)
from financial_dashboard.services.categorization.sweep import (
    run_llm_sweep,
    run_review_notify,
    run_rule_sweep,
)
from financial_dashboard.services.categorization.vocabulary import refresh_vocab_cache
from financial_dashboard.services.reminders import check_and_send_reminders
from financial_dashboard.services.settings import get_setting_int
from financial_dashboard.services.assistant.delivery import recover_assistant_work

if TYPE_CHECKING:
    from financial_dashboard.services.extensions import ExtensionManager

logger = logging.getLogger(__name__)


async def run_categorization_cycle() -> None:
    # Reuse the existing fetch-cycle lifecycle for assistant lease recovery;
    # this avoids introducing a second process-wide polling loop.
    try:
        async with async_session() as session:
            await recover_assistant_work(session)
            await session.commit()
    except Exception:
        # Categorization remains useful during first-boot/test lifecycles where
        # the assistant tables have not been created yet.
        logger.exception("Assistant interaction recovery failed")
    # Refresh merchant-rule cache first so CLI edits land without a restart.
    await load_merchant_rules()
    await run_rule_sweep()
    await run_llm_sweep()
    await run_review_notify()
    # Category creation can happen through web/API/manual paths in another
    # worker. Refresh only after those services have committed their writes so
    # a rolled-back transaction never leaks a vocabulary version into cache.
    async with async_session() as session:
        await refresh_vocab_cache(session)
    try:
        from financial_dashboard.services.assistant.orchestrator import (
            resume_claimed_interactions,
        )
        from financial_dashboard.services import telegram
        from financial_dashboard.services.settings import (
            is_telegram_assistant_enabled,
        )

        if is_telegram_assistant_enabled():
            await resume_claimed_interactions(
                bot=telegram.tg_app.bot if telegram.tg_app is not None else None
            )
            await telegram.dispatch_pending_deliveries()
    except Exception:
        logger.exception("Assistant delivery recovery failed")


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
    def __init__(self, extension_manager: ExtensionManager | None = None) -> None:
        self._lock = asyncio.Lock()
        self.status = make_poll_status()
        self._poll_loop_task: asyncio.Task | None = None
        self._active_poll_task: asyncio.Task | None = None
        # Optional extension manager: when present, after-fetch-cycle hooks run
        # once per cycle (e.g. automatic Paisa sync). Absent (None) keeps the
        # legacy behavior so existing constructions stay compatible.
        self._extension_manager = extension_manager

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

            try:
                await run_categorization_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Categorization sweep failed")

            # Extension after-fetch-cycle hooks run ONCE per cycle, after native
            # polling/reminders/categorization, then the loop sleeps. The manager
            # isolates per-extension failures, so this can never break polling.
            if self._extension_manager is not None:
                try:
                    await self._extension_manager.after_fetch_cycle_all()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Extension after-fetch-cycle hooks failed")

            interval = max(1, get_setting_int("poll_interval_minutes", 15)) * 60
            await asyncio.sleep(interval)
