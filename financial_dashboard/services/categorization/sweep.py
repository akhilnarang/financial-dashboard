"""Query-driven categorization sweeps + durable review notifications."""

import asyncio
import logging

from sqlalchemy import select

from financial_dashboard.db import async_session
from financial_dashboard.db.models import Transaction, utc_now
from financial_dashboard.services.categorization.engine import (
    categorize_one,
    select_needs_work_stmt,
)
from financial_dashboard.services.settings import (
    get_active_llm_key,
    get_setting_bool,
    get_telegram_chat_id,
    is_telegram_configured,
)

logger = logging.getLogger(__name__)


async def run_rule_sweep(*, batch_limit: int = 500) -> int:
    """Run the rule pass over never-evaluated rows. Returns the number of rows
    PROCESSED (each becomes 'rule' or 'pending_llm'), so a backfill loop can
    terminate when the never-touched set is empty (0 processed)."""
    async with async_session() as session:
        stmt = select_needs_work_stmt(llm=False, limit=batch_limit)
        rows = (await session.execute(stmt)).scalars().all()
        for txn in rows:
            await categorize_one(session, txn, use_llm=False)
        await session.commit()
    return len(rows)


async def run_llm_sweep(*, batch_limit: int = 100) -> int:
    """Run the LLM fallback over rows the rule pass left as 'pending_llm'.

    Returns the number categorized this batch, so a backfill loop stops at 0.
    No-op (0) when LLM categorization is disabled or no provider is configured.
    """
    if not get_setting_bool("categorization.enabled", False) or not get_active_llm_key():
        return 0
    # Fetch IDs first, then process each in its own fresh session. A failed row
    # must not expire/contaminate the others (a rollback expires preloaded ORM
    # objects even with expire_on_commit=False), so we isolate per row.
    async with async_session() as session:
        stmt = select_needs_work_stmt(llm=True, limit=batch_limit)
        txn_ids = [t.id for t in (await session.execute(stmt)).scalars().all()]
    count = 0
    for txn_id in txn_ids:
        try:
            async with async_session() as session:
                txn = await session.get(Transaction, txn_id)
                if txn is None:
                    continue
                await categorize_one(session, txn, use_llm=True)
                await session.commit()
            count += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LLM categorization failed for txn %s", txn_id)
            await asyncio.sleep(1.0)
    return count


async def run_review_notify(*, max_attempts: int = 5) -> int:
    """Push rows flagged review_status='pending' to the Telegram review queue.

    Sends each pending transaction, marks it 'notified', and bumps
    notify_attempts; a row is retried until it succeeds or hits max_attempts,
    so a transient send failure never strands it. Returns the number sent;
    no-op (0) when Telegram isn't configured.
    """
    from financial_dashboard.services.telegram import _send_with_retry, tg_app

    # is_telegram_configured() already covers chat_id != 0; tg_app is a separate
    # concern — the bot Application may be uninitialized in this process.
    if not is_telegram_configured() or tg_app is None:
        return 0
    chat_id = get_telegram_chat_id()
    sent = 0
    async with async_session() as session:
        stmt = (
            select(Transaction)
            .where(
                Transaction.review_status == "pending",
                (Transaction.notify_attempts.is_(None))
                | (Transaction.notify_attempts < max_attempts),
            )
            .limit(50)
        )
        rows = (await session.execute(stmt)).scalars().all()
        for txn in rows:
            text = (
                f"\U0001f50d Needs a category: #{txn.id}\n"
                f"{txn.direction} {txn.amount} {txn.currency or 'INR'}\n"
                f"{txn.counterparty or txn.raw_description or ''}\n"
                f"Reason: {txn.review_reason or 'low confidence'}\n"
                f"Reply with: <note>\\n<category>"
            )
            try:
                # Plain text (parse_mode=None): counterparty/reason can contain
                # & or < which would break Telegram's HTML parser and strand the row.
                await _send_with_retry(
                    tg_app, chat_id=chat_id, text=text, parse_mode=None
                )
                txn.review_status = "notified"
                txn.last_notified_at = utc_now()
                txn.notify_attempts = (txn.notify_attempts or 0) + 1
                sent += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Review notify failed for txn %s", txn.id)
                txn.notify_attempts = (txn.notify_attempts or 0) + 1
        await session.commit()
    return sent
