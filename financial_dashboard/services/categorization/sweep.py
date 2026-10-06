"""Query-driven categorization sweeps + durable review notifications."""

import asyncio
import html
import json
import logging
from collections.abc import Sequence
from typing import NamedTuple
from urllib.parse import urlencode

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import async_session
from financial_dashboard.db.models import CategoryReviewDecision, Transaction, utc_now
from financial_dashboard.services.categorization.engine import (
    categorize_one,
    select_needs_work_stmt,
)
from financial_dashboard.services.categorization.review_decisions import (
    ensure_decision_delivery,
    ensure_legacy_decision,
)
from financial_dashboard.services.assistant.rendering import split_plain_text
from financial_dashboard.services.settings import (
    get_active_llm_key,
    get_app_base_url,
    get_setting_int,
    get_setting_bool,
    get_telegram_chat_id,
    is_telegram_configured,
    is_telegram_assistant_enabled,
)

logger = logging.getLogger(__name__)

# Abort an LLM sweep after this many back-to-back failures — a run of them is a
# systematic fault (bad key/model/quota), not one unlucky row.
_MAX_CONSECUTIVE_FAILURES = 5

# Stop retrying a review notification after this many failed sends.
_MAX_NOTIFY_ATTEMPTS = 5


def _needs_llm(txn: Transaction) -> bool:
    """Whether a row is still eligible for the LLM pass at write time.

    Mirrors select_needs_work_stmt(llm=True) minus the vocab-version filter: safe
    to re-run on never-evaluated, pending_llm, unknown, or unresolved review rows,
    but never
    on a 'manual'/'rule'/finalised-'llm' row (guards the select→process window).
    """
    return txn.category_method in (None, "pending_llm") or (
        txn.category_method == "llm"
        and (txn.category == "unknown" or txn.review_status in ("pending", "notified"))
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
    if (
        not get_setting_bool("categorization.enabled", False)
        or not get_active_llm_key()
    ):
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


class ImportKey(NamedTuple):
    """The statement upload that created a row, as a transactions API filter."""

    param: str
    upload_id: int


class PromptBatch(NamedTuple):
    """Rows to prompt one by one, and the overflow rows of each import."""

    prompt: list[Transaction]
    overflow: dict[ImportKey, list[Transaction]]


def _import_key(txn: Transaction) -> ImportKey | None:
    """Return the statement import of a row, or None for a single alert."""
    if (upload_id := txn.statement_upload_id) is not None:
        return ImportKey("statement_upload_id", upload_id)
    if (upload_id := txn.bank_statement_upload_id) is not None:
        return ImportKey("bank_statement_upload_id", upload_id)
    return None


async def _owned_by_conversation(session: AsyncSession, txn: Transaction) -> bool:
    """Whether an assistant conversation alone owns the review of a row."""
    active = (
        CategoryReviewDecision.transaction_id == txn.id,
        CategoryReviewDecision.status == "active",
    )
    background = select(CategoryReviewDecision.id).where(
        *active, CategoryReviewDecision.source_interaction_id.is_(None)
    )
    conversational = select(CategoryReviewDecision.id).where(
        *active, CategoryReviewDecision.source_interaction_id.is_not(None)
    )
    return bool(
        await session.scalar(select(conversational.exists() & ~background.exists()))
    )


async def _reviewable(
    session: AsyncSession, rows: Sequence[Transaction], assistant_enabled: bool
) -> list[Transaction]:
    """Drop rows whose review an assistant conversation already owns."""
    if not assistant_enabled:
        return list(rows)
    return [txn for txn in rows if not await _owned_by_conversation(session, txn)]


async def _notified_count(session: AsyncSession, key: ImportKey) -> int:
    """Count the rows of an import that already had a prompt or a summary."""
    count = await session.scalar(
        select(func.count()).where(
            getattr(Transaction, key.param) == key.upload_id,
            Transaction.last_notified_at.is_not(None),
        )
    )
    return count or 0


async def _pending_import_rows(
    session: AsyncSession, key: ImportKey, skip: set[int]
) -> Sequence[Transaction]:
    """Return every pending row of an import, except the ids in ``skip``."""
    stmt = (
        select(Transaction)
        .where(
            getattr(Transaction, key.param) == key.upload_id,
            Transaction.review_status == "pending",
            (Transaction.notify_attempts.is_(None))
            | (Transaction.notify_attempts < _MAX_NOTIFY_ATTEMPTS),
            Transaction.id.not_in(skip),
        )
        .order_by(Transaction.id)
    )
    return (await session.execute(stmt)).scalars().all()


async def _cap_per_import(
    session: AsyncSession, rows: Sequence[Transaction], assistant_enabled: bool
) -> PromptBatch:
    """Split pending rows so one statement import sends a bounded prompt count.

    Each import gets at most ``telegram.bulk_threshold`` prompts. Rows of the
    import that an earlier sweep notified use up the same budget. A row that
    had a prompt or a summary before goes to the summary. The overflow
    of an import holds all its other pending rows, also those past the sweep
    batch limit, so one summary covers them.

    Args:
        session: Open session that owns ``rows``.
        rows: Pending rows in id order.
        assistant_enabled: Whether the conversational assistant is on.

    Returns:
        The rows to prompt, and the overflow rows of each import.
    """
    cap = get_setting_int("telegram.bulk_threshold", 5)
    used: dict[ImportKey, int] = {}
    batch = PromptBatch([], {})
    for txn in await _reviewable(session, rows, assistant_enabled):
        if (key := _import_key(txn)) is None:
            batch.prompt.append(txn)
            continue
        if key not in used:
            used[key] = await _notified_count(session, key)
        if used[key] < cap and txn.last_notified_at is None:
            used[key] += 1
            batch.prompt.append(txn)
        else:
            batch.overflow[key] = []
    prompted = {txn.id for txn in batch.prompt}
    for key in batch.overflow:
        pending = await _pending_import_rows(session, key, prompted)
        batch.overflow[key] = await _reviewable(session, pending, assistant_enabled)
    return batch


async def _send_overflow_summary(
    chat_id: int, key: ImportKey, rows: list[Transaction], base_url: str
) -> None:
    """Send one line for the rows of an import above the prompt cap.

    The rows become 'notified' as if each one had its own prompt. A failed send
    leaves them 'pending' for the next sweep.

    Args:
        chat_id: Telegram chat to send to.
        key: The import that the rows came from.
        rows: Overflow rows of the import.
        base_url: Dashboard base URL. Empty means no link.
    """
    from financial_dashboard.services.telegram import _send_with_retry, tg_app

    text = f"\U0001f50d {len(rows)} more rows need a category"
    if base_url:
        query = urlencode({key.param: key.upload_id, "review_status": "notified"})
        text += f"\n{base_url}/api/transactions?{query}"
    try:
        await _send_with_retry(tg_app, chat_id=chat_id, text=text, parse_mode=None)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Review summary failed for %s=%s", key.param, key.upload_id)
        for txn in rows:
            txn.notify_attempts = (txn.notify_attempts or 0) + 1
        return
    now = utc_now()
    for txn in rows:
        txn.review_status = "notified"
        txn.last_notified_at = now
        txn.notify_attempts = (txn.notify_attempts or 0) + 1


async def _review_decision(
    session: AsyncSession, txn: Transaction, assistant_enabled: bool
) -> CategoryReviewDecision | None:
    """Return the background decision of a row, made on demand for the assistant."""
    decision = await session.scalar(
        select(CategoryReviewDecision)
        .where(
            CategoryReviewDecision.transaction_id == txn.id,
            CategoryReviewDecision.status == "active",
            CategoryReviewDecision.source_interaction_id.is_(None),
        )
        .order_by(CategoryReviewDecision.id.desc())
        .limit(1)
    )
    if decision is None and assistant_enabled:
        # Rows created before durable review decisions existed still need a
        # decision record so a later callback can fail closed. This path is
        # deliberately zero-candidate and never invokes the LLM. A row that a
        # conversational proposal owns never reaches here.
        decision = await ensure_legacy_decision(session, txn)
    return decision


async def _notify_assistant(
    session: AsyncSession,
    txn: Transaction,
    decision: CategoryReviewDecision,
    chat_id: int,
    base_url: str,
) -> bool:
    """Queue and send the assistant prompt of one row. Returns True when sent."""
    from financial_dashboard.services.telegram import dispatch_saved_delivery

    candidates = json.loads(decision.candidates_json)
    gate = decision.gate_reason or "manual review requested"
    next_step = (
        "Reply with context or choose a category below."
        if 2 <= len(candidates) <= 3
        else "Reply with context so I can categorize it."
    )
    proposed = decision.proposed_slug or (
        str(candidates[0].get("category", "")) if candidates else ""
    )
    plain_id = f"#{txn.id}"
    if base_url:
        plain_id += f" ({base_url}/transactions/{txn.id})"
    assistant_text = (
        f"\U0001f50d Needs a category: {plain_id}\n"
        f"{txn.direction or ''} {txn.amount} {txn.currency or 'INR'}\n"
        f"{txn.counterparty or txn.raw_description or ''}\n"
        f"Likely category: {proposed or 'uncertain'}\n"
        f"Why I asked: {gate}\n"
        f"Reasoning: {txn.review_reason or 'low confidence'}\n"
        f"{next_step}"
    )
    deliveries = []
    for ordinal, chunk in enumerate(split_plain_text(assistant_text, limit=4000)):
        delivery = await ensure_decision_delivery(
            session,
            decision,
            recipient_chat_id=chat_id,
            transaction_id=txn.id,
            text=chunk,
            reply_markup_json=None,
            ordinal=ordinal,
            parse_mode=None,
        )
        if "\nRef: " not in delivery.text:
            delivery.text = f"{delivery.text}\n\nRef: {delivery.delivery_token}"
        deliveries.append(delivery)
    first_delivery = deliveries[0]
    if 2 <= len(candidates) <= 3 and first_delivery.reply_markup_json is None:
        first_delivery.reply_markup_json = json.dumps(
            [
                [
                    {
                        "text": str(candidate.get("category", "")),
                        "callback_data": (
                            f"cat:v1:{decision.id}:{first_delivery.id}:{index}"
                        ),
                    }
                ]
                for index, candidate in enumerate(candidates)
            ],
            separators=(",", ":"),
        )
    await session.commit()
    delivered = all(
        [await dispatch_saved_delivery(delivery.id) for delivery in deliveries]
    )
    if not delivered:
        txn.notify_attempts = (txn.notify_attempts or 0) + 1
    return delivered


async def _notify_plain(txn: Transaction, chat_id: int, base_url: str) -> bool:
    """Send the HTML prompt of one row. Returns True when sent."""
    from financial_dashboard.services.telegram import _send_with_retry, tg_app

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
        await _send_with_retry(tg_app, chat_id=chat_id, text=text, parse_mode="HTML")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Review notify failed for txn %s", txn.id)
        txn.notify_attempts = (txn.notify_attempts or 0) + 1
        return False
    txn.review_status = "notified"
    txn.last_notified_at = utc_now()
    txn.notify_attempts = (txn.notify_attempts or 0) + 1
    return True


async def _notify_row(
    session: AsyncSession,
    txn: Transaction,
    chat_id: int,
    base_url: str,
    assistant_enabled: bool,
) -> bool:
    """Send the review prompt of one row. Returns True when sent."""
    decision = await _review_decision(session, txn, assistant_enabled)
    if not assistant_enabled:
        return await _notify_plain(txn, chat_id, base_url)
    return decision is not None and await _notify_assistant(
        session, txn, decision, chat_id, base_url
    )


async def run_review_notify() -> int:
    """Push rows flagged review_status='pending' to the Telegram review queue.

    Sends each pending transaction, marks it 'notified', and bumps
    notify_attempts; a row is retried until it succeeds or hits _MAX_NOTIFY_ATTEMPTS,
    so a transient send failure never strands it. One statement import sends at
    most ``telegram.bulk_threshold`` prompts and one summary line for the rest.
    Returns the number of prompts sent; no-op (0) when Telegram isn't configured.
    """
    from financial_dashboard.services import telegram

    # is_telegram_configured() already covers chat_id != 0; tg_app is a separate
    # concern — the bot Application may be uninitialized in this process.
    if not is_telegram_configured() or telegram.tg_app is None:
        return 0
    chat_id = get_telegram_chat_id()
    assistant_enabled = is_telegram_assistant_enabled()
    sent = 0
    async with async_session() as session:
        stmt = (
            select(Transaction)
            .where(
                Transaction.review_status == "pending",
                (Transaction.notify_attempts.is_(None))
                | (Transaction.notify_attempts < _MAX_NOTIFY_ATTEMPTS),
            )
            .order_by(Transaction.id)
            .limit(50)
        )
        rows = (await session.execute(stmt)).scalars().all()
        base_url = get_app_base_url()
        batch = await _cap_per_import(session, rows, assistant_enabled)
        failed: set[ImportKey | None] = set()
        for txn in batch.prompt:
            if await _notify_row(session, txn, chat_id, base_url, assistant_enabled):
                sent += 1
            else:
                failed.add(_import_key(txn))
        # A failed prompt keeps its slot. The summary waits until every prompt
        # of the import is sent, so the import gets one summary only.
        for key, overflow in batch.overflow.items():
            if key not in failed:
                await _send_overflow_summary(chat_id, key, overflow, base_url)
        await session.commit()
    return sent
