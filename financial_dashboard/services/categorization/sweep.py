"""Query-driven categorization sweeps + durable review notifications."""

import asyncio
import html
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
    get_app_base_url,
    get_setting_bool,
    get_telegram_chat_id,
    is_telegram_configured,
)

logger = logging.getLogger(__name__)

# Abort an LLM sweep after this many back-to-back failures — a run of them is a
# systematic fault (bad key/model/quota), not one unlucky row.
_MAX_CONSECUTIVE_FAILURES = 5


def _needs_llm(txn: Transaction) -> bool:
    """Whether a row is still eligible for the LLM pass at write time.

    Mirrors select_needs_work_stmt(llm=True) minus the vocab-version filter: safe
    to re-run on never-evaluated, pending_llm, or prior 'unknown' rows, but never
    on a 'manual'/'rule'/finalised-'llm' row (guards the select→process window).
    """
    return txn.category_method in (None, "pending_llm") or (
        txn.category == "unknown" and txn.category_method == "llm"
    )


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
    consecutive_failures = 0
    for txn_id in txn_ids:
        try:
            async with async_session() as session:
                txn = await session.get(Transaction, txn_id)
                # Re-check eligibility: the row was selected earlier, but a manual
                # assignment (Telegram/API) may have landed during the in-flight
                # LLM call of a prior row. SQLite has no row locking / FOR UPDATE,
                # so guard in-app — never overwrite a manual (or already-final) row.
                if txn is None or not _needs_llm(txn):
                    continue
                await categorize_one(session, txn, use_llm=True)
                await session.commit()
            count += 1
            consecutive_failures = 0
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LLM categorization failed for txn %s", txn_id)
            consecutive_failures += 1
            # A run of failures means a systematic problem (bad key, model id,
            # quota) rather than one bad row — stop instead of burning the whole
            # batch of API calls (and poll-loop time) every cycle, forever.
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "LLM sweep aborting after %d consecutive failures",
                    consecutive_failures,
                )
                break
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
        base_url = get_app_base_url()
        for txn in rows:
            # HTML parse mode: link the id to its transaction page when a base URL
            # is configured. EVERY interpolated field is html-escaped — counterparty,
            # reason, direction, currency and the base_url are free text (parser/user
            # output) that could otherwise contain & or " or < and break Telegram's
            # HTML parser, stranding the row — as is the literal <note>/<category>
            # hint. txn.amount is a Decimal, so it's safe as-is.
            if base_url:
                href = html.escape(f"{base_url}/transactions/{txn.id}", quote=True)
                id_label = f'<a href="{href}">#{txn.id}</a>'
            else:
                id_label = f"#{txn.id}"
            detail = html.escape(txn.counterparty or txn.raw_description or "")
            reason = html.escape(txn.review_reason or "low confidence")
            direction = html.escape(txn.direction or "")
            currency = html.escape(txn.currency or "INR")
            text = (
                f"\U0001f50d Needs a category: {id_label}\n"
                f"{direction} {txn.amount} {currency}\n"
                f"{detail}\n"
                f"Reason: {reason}\n"
                f"Reply with: &lt;note&gt;\n&lt;category&gt;"
            )
            try:
                await _send_with_retry(
                    tg_app, chat_id=chat_id, text=text, parse_mode="HTML"
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
