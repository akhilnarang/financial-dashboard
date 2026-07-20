"""The pending → notified → resolved lifecycle of a categorization review row,
and the retry cap that keeps a stranded row from looping forever.

The lifecycle crosses three modules, and a change to any of them can break it
silently:

* the engine sets ``review_status='pending'`` when the LLM is unsure;
* the sweep's ``run_review_notify`` pushes the row to Telegram and marks it
  ``'notified'`` (with a retry counter so a transient send failure cannot
  strand it);
* ``assign_category_manual`` flips it to ``'resolved'`` when the human reply
  arrives — never to ``'pending'`` or back to ``None``.

The retry cap is what keeps the sweep from looping forever on a row whose send
keeps failing: a row past ``max_attempts`` is skipped, so the queue drains.

The sweep opens its own session (``async with async_session()``), so these
tests stand up an in-memory engine and monkey-patch ``sweep.async_session`` to
a maker over it — the same pattern the sweep's own unit tests use, so the row
a test commits is the row the sweep reads.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.categorization.sweep as sweep
import financial_dashboard.services.telegram as tg
from financial_dashboard.db.models import Base, Transaction
from financial_dashboard.services.categorization.manual import assign_category_manual
from financial_dashboard.services.categorization.vocabulary import ensure_category

pytestmark = pytest.mark.anyio


@pytest.fixture
async def memdb(monkeypatch):
    """An in-memory engine + session maker the sweep is patched to use.

    The sweep opens its own session via ``sweep.async_session``; without
    patching it to this maker, the row a test commits on its own session would
    not be visible to the sweep's session (separate engine, separate
    :memory: DB). Patching here makes the two share one engine and one pool
    connection, which is what makes an in-memory DB visible across sessions.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(sweep, "async_session", maker)
    yield maker
    await engine.dispose()


@pytest.fixture
def telegram_send(monkeypatch):
    """Wire the sweep's Telegram surface to a recording no-op sender.

    The sweep reaches the bot through ``tg_app`` and ``_send_with_retry``; both
    are module-level on ``services.telegram``, and the sweep imports them by
    name, so monkey-patching them in place is enough to run the lifecycle
    without a real bot. Records every send so a test can count them.

    Does NOT flip the ``is_telegram_configured`` / chat_id settings — return
    value lets each test decide whether the sweep should think Telegram is
    reachable.
    """
    sent: list = []

    async def fake_send(app, *, chat_id, text, parse_mode=None, **kw):
        sent.append((text, parse_mode))

    monkeypatch.setattr(tg, "_send_with_retry", fake_send)
    monkeypatch.setattr(tg, "tg_app", object())
    return sent


def _configure_telegram(monkeypatch, *, base_url: str = "") -> None:
    """Flip the sweep's Telegram settings to 'configured'."""
    monkeypatch.setattr(sweep, "is_telegram_configured", lambda: True)
    monkeypatch.setattr(sweep, "get_telegram_chat_id", lambda: 123)
    monkeypatch.setattr(sweep, "get_app_base_url", lambda: base_url)


async def _seed_pending(memdb, *, review_reason="low confidence", attempts=0) -> int:
    """Insert one pending review row on the sweep's engine; return its id."""
    async with memdb() as s:
        txn = Transaction(
            bank="testbank",
            email_type="x",
            direction="debit",
            amount=Decimal("50"),
            counterparty="MYSTERY MERCHANT",
            raw_description="MYSTERY MERCHANT",
            review_status="pending",
            review_reason=review_reason,
            notify_attempts=attempts,
        )
        s.add(txn)
        await s.commit()
        return txn.id


async def _fetch(memdb, txn_id: int) -> Transaction:
    async with memdb() as s:
        return await s.get(Transaction, txn_id)


# ---------------------------------------------------------------------------
# pending → notified
# ---------------------------------------------------------------------------


async def test_pending_row_is_notified_and_advances_status(
    memdb, telegram_send, monkeypatch
):
    """A 'pending' row becomes 'notified' after a successful send, and the
    attempt counter advances. ``run_review_notify`` returns 1 (one row sent)."""
    _configure_telegram(monkeypatch)
    txn_id = await _seed_pending(memdb)

    n = await sweep.run_review_notify()

    assert n == 1
    assert len(telegram_send) == 1
    txn = await _fetch(memdb, txn_id)
    assert txn.review_status == "notified"
    assert txn.last_notified_at is not None
    assert txn.notify_attempts == 1


async def test_already_notified_row_is_not_resent(memdb, telegram_send, monkeypatch):
    """A row already 'notified' is not in the queue — the sweep selects only
    ``review_status == 'pending'``, so a second run does not duplicate the
    message."""
    _configure_telegram(monkeypatch)
    async with memdb() as s:
        s.add(
            Transaction(
                bank="b",
                email_type="x",
                direction="debit",
                amount=Decimal("1"),
                review_status="notified",
                notify_attempts=1,
            )
        )
        await s.commit()

    n = await sweep.run_review_notify()
    assert n == 0
    assert telegram_send == []


async def test_resolved_row_is_not_in_the_notify_queue(
    memdb, telegram_send, monkeypatch
):
    """A 'resolved' row never reaches the queue — once the human reply has
    landed, the row is out of the review loop."""
    _configure_telegram(monkeypatch)
    async with memdb() as s:
        s.add(
            Transaction(
                bank="b",
                email_type="x",
                direction="debit",
                amount=Decimal("1"),
                review_status="resolved",
            )
        )
        await s.commit()

    n = await sweep.run_review_notify()
    assert n == 0
    assert telegram_send == []


# ---------------------------------------------------------------------------
# notified → resolved (manual category assignment)
# ---------------------------------------------------------------------------


async def test_manual_assign_moves_notified_to_resolved(memdb):
    """``assign_category_manual`` is the terminal transition: a 'notified' row
    (or any review_status) becomes 'resolved' when the human reply lands, never
    the reverse. Confidence is 1.0 (authoritative) and the category_model
    records who applied it."""
    async with memdb() as s:
        await ensure_category(s, "groceries")
        txn = Transaction(
            bank="testbank",
            email_type="x",
            direction="debit",
            amount=Decimal("50"),
            review_status="notified",
            notify_attempts=1,
            review_reason="low confidence",
        )
        s.add(txn)
        await s.commit()
        txn_id = txn.id

    # Reopen on the same engine and apply — manual uses the CALLER's session,
    # so a test owns the transition the same way the API/Telegram path does.
    async with memdb() as s:
        ok, slug = await assign_category_manual(s, txn_id, "Groceries")
        assert ok is True
        assert slug == "groceries"

    txn = await _fetch(memdb, txn_id)
    assert txn.category == "groceries"
    assert txn.category_method == "manual"
    assert txn.category_confidence == 1.0
    assert txn.review_status == "resolved"
    assert (txn.category_model or "").startswith("manual:")


async def test_manual_clear_keeps_row_resolved_with_none_category(memdb):
    """Clearing the category (empty string) keeps the row 'resolved' — clearing
    is itself a terminal manual decision, not a return to the review queue."""
    async with memdb() as s:
        txn = Transaction(
            bank="testbank",
            email_type="x",
            direction="debit",
            amount=Decimal("50"),
            category="groceries",
            category_method="manual",
            review_status="resolved",
        )
        s.add(txn)
        await s.commit()
        txn_id = txn.id

    async with memdb() as s:
        ok, slug = await assign_category_manual(s, txn_id, "")
        assert ok is True
        assert slug is None

    txn = await _fetch(memdb, txn_id)
    assert txn.category is None
    assert txn.category_method == "manual"
    assert txn.category_confidence == 1.0
    assert txn.review_status == "resolved"


# ---------------------------------------------------------------------------
# Retry max — a row past max_attempts is skipped, so the queue drains
# ---------------------------------------------------------------------------


async def test_send_failure_advances_attempts_without_marking_notified(
    memdb, monkeypatch
):
    """A failed send advances notify_attempts but leaves the row 'pending' — so
    the next sweep picks it up again. No message was sent."""
    _configure_telegram(monkeypatch)

    async def failing_send(*a, **kw):
        raise RuntimeError("transient send failure")

    monkeypatch.setattr(tg, "_send_with_retry", failing_send)
    monkeypatch.setattr(tg, "tg_app", object())

    txn_id = await _seed_pending(memdb)

    n = await sweep.run_review_notify()
    assert n == 0
    txn = await _fetch(memdb, txn_id)
    assert txn.review_status == "pending"  # still pending — not advanced
    assert txn.notify_attempts == 1
    assert txn.last_notified_at is None


async def test_row_past_max_attempts_is_skipped_so_the_queue_drains(
    memdb, telegram_send, monkeypatch
):
    """A row whose notify_attempts has reached the cap is not selected again,
    even though it is still 'pending'. The retry cap is what stops a
    permanently-unsatisfiable row from monopolizing the queue."""
    _configure_telegram(monkeypatch)
    txn_id = await _seed_pending(memdb, attempts=5)  # at the default cap

    n = await sweep.run_review_notify(max_attempts=5)
    assert n == 0
    assert telegram_send == []
    txn = await _fetch(memdb, txn_id)
    assert txn.review_status == "pending"  # untouched — not selected


async def test_row_just_under_max_attempts_is_still_tried(
    memdb, telegram_send, monkeypatch
):
    """The cap is exclusive: a row at max_attempts - 1 is still selected, and
    after a successful send sits at max_attempts — so the next run skips it."""
    _configure_telegram(monkeypatch)
    txn_id = await _seed_pending(memdb, attempts=4)  # one under the cap of 5

    n = await sweep.run_review_notify(max_attempts=5)
    assert n == 1
    assert len(telegram_send) == 1
    txn = await _fetch(memdb, txn_id)
    assert txn.review_status == "notified"
    assert txn.notify_attempts == 5

    # A second run skips it: at the cap now, not selected.
    n2 = await sweep.run_review_notify(max_attempts=5)
    assert n2 == 0


async def test_a_changed_failure_is_retried_until_the_cap(memdb, monkeypatch):
    """A row keeps being re-tried (advancing attempts each time) until it hits
    the cap — a transient send failure does not strand it before then."""
    _configure_telegram(monkeypatch)
    attempts = {"count": 0}

    async def failing_send(*a, **kw):
        attempts["count"] += 1
        raise RuntimeError("still failing")

    monkeypatch.setattr(tg, "_send_with_retry", failing_send)
    monkeypatch.setattr(tg, "tg_app", object())

    txn_id = await _seed_pending(memdb, attempts=0)

    # Each run advances notify_attempts by 1, status stays 'pending'.
    await sweep.run_review_notify(max_attempts=3)
    txn = await _fetch(memdb, txn_id)
    assert txn.review_status == "pending"
    assert txn.notify_attempts == 1

    await sweep.run_review_notify(max_attempts=3)
    txn = await _fetch(memdb, txn_id)
    assert txn.notify_attempts == 2

    # Third run: still under the cap of 3, so tried once more.
    await sweep.run_review_notify(max_attempts=3)
    txn = await _fetch(memdb, txn_id)
    assert txn.notify_attempts == 3

    # Fourth run: at the cap, skipped.
    n = await sweep.run_review_notify(max_attempts=3)
    assert n == 0
    assert attempts["count"] == 3  # three send attempts, then queue drained


# ---------------------------------------------------------------------------
# Engine re-categorization respects the resolved row
# ---------------------------------------------------------------------------


def test_resolved_row_is_not_re_categorized_by_the_llm_sweep():
    """A 'resolved' row is out of the LLM sweep's eligibility window: the
    ``_needs_llm`` guard refuses it, so a stale-vocab requeue or a pending_llm
    sweep cannot overwrite the human's decision."""
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal("50"),
        category="groceries",
        category_method="manual",
        review_status="resolved",
    )
    assert sweep._needs_llm(txn) is False


def test_pending_unknown_row_stays_eligible_until_finalised():
    """The eligibility window keeps a pending 'unknown' row open for the LLM
    sweep to revisit (enrichment may have populated text, a vocab bump may
    have added a fitting slug). Eligibility ends at a confident 'llm'/'rule'
    answer or a 'manual' one, never at the 'pending' status itself."""
    pending_unknown = Transaction(
        bank="b",
        email_type="x",
        direction="debit",
        amount=Decimal("1"),
        category="unknown",
        category_method="llm",
        review_status="pending",
    )
    assert sweep._needs_llm(pending_unknown) is True

    pending_llm = Transaction(
        bank="b",
        email_type="x",
        direction="debit",
        amount=Decimal("1"),
        category_method="pending_llm",
    )
    assert sweep._needs_llm(pending_llm) is True

    never = Transaction(
        bank="b",
        email_type="x",
        direction="debit",
        amount=Decimal("1"),
        category_method=None,
    )
    assert sweep._needs_llm(never) is True

    # A confident 'llm' answer with a non-'unknown' slug is finalised, even
    # if its review_status is 'pending' for a different reason — the LLM pass
    # already had its say on this slug.
    confident_pending = Transaction(
        bank="b",
        email_type="x",
        direction="debit",
        amount=Decimal("1"),
        category="groceries",
        category_method="llm",
        review_status="pending",
    )
    assert sweep._needs_llm(confident_pending) is False
