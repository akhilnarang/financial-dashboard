import json
from decimal import Decimal

import pytest

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import financial_dashboard.services.telegram as tg
from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization import sweep
from tests.conftest import new_test_engine

pytestmark = pytest.mark.anyio


@pytest.fixture
async def memdb(monkeypatch):
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(
        "financial_dashboard.services.categorization.sweep.async_session", maker
    )
    yield maker
    await engine.dispose()
    holder.close()


@pytest.mark.parametrize("prompt", ["sent", "failed", "assistant", "assistant_failed"])
async def test_sweeps_retry_review_once_after_vocabulary_changes(
    memdb, monkeypatch, prompt
):
    prompt_failed = prompt in ("failed", "assistant_failed")
    from sqlalchemy import select

    from financial_dashboard.db.models import (
        AuditInteraction,
        Category,
        CategoryReviewDecision,
        TelegramOutboundDelivery,
    )
    from financial_dashboard.services import settings
    from financial_dashboard.services.categorization import engine, llm

    monkeypatch.setitem(settings._cache, "category_vocab_version", "1")
    monkeypatch.setitem(settings._cache, "categorization.enabled", "true")
    monkeypatch.setattr(sweep, "get_active_llm_key", lambda: "test-key")
    # Each run returns the same categories with a slightly different confidence.
    runs = iter([0.40, 0.41])

    async def classify(**kwargs):
        confidence = next(runs)
        return llm.LlmResult(
            "groceries",
            confidence,
            "ambiguous",
            (llm.LlmCandidate("groceries", confidence), llm.LlmCandidate("food", 0.2)),
        )

    monkeypatch.setattr(engine, "_llm_classify", classify)
    # An unmatched row becomes 'pending_llm' after the rule sweep, and a second
    # sweep finds zero never-touched rows (returns 0) — this is what lets the
    # backfill loop terminate with full coverage instead of re-evaluating forever.
    async with memdb() as s:
        s.add(
            Transaction(
                bank="testbank",
                email_type="x",
                direction="debit",
                amount=Decimal("99"),
                counterparty="ACME STORE",
                raw_description="ACME STORE MUMBAI",
            )
        )
        await s.commit()

    first = await sweep.run_rule_sweep()
    assert first == 1  # one row processed

    second = await sweep.run_rule_sweep()
    assert second == 0  # nothing left untouched → backfill loop would terminate

    async with memdb() as s:
        row = (await s.execute(select(Transaction))).scalars().one()
        assert row.category_method == "pending_llm"
        assert row.category is None

    assert await sweep.run_llm_sweep() == 1
    assert await sweep.run_llm_sweep() == 0
    async with memdb() as s:
        row = (await s.scalars(select(Transaction))).one()
        assert row.category == "expense"
        assert row.review_status == "pending"
        row.review_status = "notified"
        s.add(Category(slug="groceries", active=True))
        decision = await s.scalar(select(CategoryReviewDecision))
        assert decision.proposed_slug == "groceries"
        # Another transaction's prompt failed. It must not look like this
        # row's prompt.
        other = CategoryReviewDecision(
            transaction_id=row.id + 1, category_input_hash="other", candidates_json="[]"
        )
        s.add(other)
        await s.flush()
        s.add(
            TelegramOutboundDelivery(
                category_review_decision_id=other.id,
                transaction_id=row.id + 1,
                recipient_chat_id=7,
                text="Needs a category",
                delivery_token="other-dead-prompt",
                status="abandoned",
            )
        )
        owner = {"category_review_decision_id": decision.id}
        if prompt.startswith("assistant"):
            # An assistant proposal stores its candidates with "slug" keys.
            # The interaction that made it owns its delivery.
            decision.candidates_json = json.dumps(
                [
                    {"slug": "groceries", "reason": "r", "confidence": 0.4},
                    {"slug": "food", "reason": "r", "confidence": 0.2},
                ]
            )
            interaction = AuditInteraction(
                inbound_chat_id=7, trigger="reply", status="delivery_failed"
            )
            s.add(interaction)
            await s.flush()
            decision.source_interaction_id = interaction.id
            owner = {"interaction_id": interaction.id}
        if prompt_failed:
            s.add(
                TelegramOutboundDelivery(
                    **owner,
                    transaction_id=row.id,
                    recipient_chat_id=7,
                    text="Needs a category",
                    delivery_token="dead-prompt",
                    status="abandoned",
                )
            )
        await s.commit()
    monkeypatch.setitem(settings._cache, "category_vocab_version", "2")
    assert await sweep.run_llm_sweep() == 1
    assert await sweep.run_llm_sweep() == 0
    async with memdb() as s:
        row = (await s.scalars(select(Transaction))).one()
        assert row.category == "expense"
        # The same candidates reuse a sent prompt. A prompt that was never
        # sent must be sent again.
        assert row.review_status == ("pending" if prompt_failed else "notified")
        assert row.category_vocab_version == 2
        decisions = (
            await s.scalars(
                select(CategoryReviewDecision)
                .where(CategoryReviewDecision.transaction_id == row.id)
                .order_by(CategoryReviewDecision.id)
            )
        ).all()
        assert [decision.status for decision in decisions] == (
            ["superseded", "active"] if prompt_failed else ["active"]
        )
        row.category_method = "manual"
        row.category_vocab_version = 1
        row.review_status = "notified"
        await s.commit()
    assert await sweep.run_llm_sweep() == 0


@pytest.fixture
def telegram_send(monkeypatch):
    """Configure Telegram and record every review message the sweep sends."""
    monkeypatch.setattr(sweep, "is_telegram_configured", lambda: True)
    monkeypatch.setattr(sweep, "get_telegram_chat_id", lambda: 123)
    # Adversarial base_url (stray quote) must not break out of the href attribute.
    monkeypatch.setattr(sweep, "get_app_base_url", lambda: 'http://host:8000"x')
    sent: list = []

    async def fake_send(app, *, chat_id, text, parse_mode=None, **kw):
        sent.append((text, parse_mode))

    monkeypatch.setattr(tg, "_send_with_retry", fake_send)
    monkeypatch.setattr(tg, "tg_app", object())
    return sent


async def _seed_pending(memdb, **kw) -> int:
    async with memdb() as s:
        fields = dict(
            bank="testbank",
            email_type="x",
            direction="debit",
            amount=Decimal("50"),
            counterparty="MYSTERY MERCHANT",
            review_status="pending",
            review_reason="low confidence",
        )
        txn = Transaction(**(fields | kw))
        s.add(txn)
        await s.commit()
        return txn.id


async def test_pending_row_is_notified_once_with_escaped_fields(memdb, telegram_send):
    """A 'pending' row is sent once and becomes 'notified'. A 'resolved' row is
    not sent. The message links #id and html-escapes free-text fields."""
    txn_id = await _seed_pending(
        memdb,
        direction="deb<it",
        currency="IN&R",
        counterparty="A & B <x>",
        review_reason="unclear <tag>",
    )
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

    assert await sweep.run_review_notify() == 1
    assert len(telegram_send) == 1
    async with memdb() as s:
        txn = await s.get(Transaction, txn_id)
    assert txn.review_status == "notified"
    assert txn.last_notified_at is not None
    assert txn.notify_attempts == 1
    text, mode = telegram_send[0]
    assert mode == "HTML"
    assert 'href="http://host:8000&quot;x/transactions/' in text
    assert "A &amp; B &lt;x&gt;" in text
    assert "unclear &lt;tag&gt;" in text
    assert "deb&lt;it" in text
    assert "IN&amp;R" in text
    assert "&lt;note&gt;" in text and "&lt;category&gt;" in text

    assert await sweep.run_review_notify() == 0
    assert len(telegram_send) == 1

    # One statement import sends at most five prompts and one summary line.
    # A small import sends one prompt per row.
    big = [await _seed_pending(memdb, bank_statement_upload_id=41) for _ in range(7)]
    small = [await _seed_pending(memdb, statement_upload_id=42) for _ in range(2)]
    telegram_send.clear()
    assert await sweep.run_review_notify() == 7
    prompts = [text for text, _ in telegram_send if "Needs a category" in text]
    assert len(prompts) == 7
    assert all(f'/transactions/{i}">' in t for i, t in zip(big[:5] + small, prompts))
    summaries = [text for text, _ in telegram_send if "Needs a category" not in text]
    assert summaries == [
        '\U0001f50d 2 more rows need a category\nhttp://host:8000"x/api/transactions'
        "?bank_statement_upload_id=41&review_status=notified"
    ]
    async with memdb() as s:
        for txn_id in big + small:
            assert (await s.get(Transaction, txn_id)).review_status == "notified"

    # Resolved or re-opened prompts still use up the budget of their import. One summary
    # covers all new rows, also those past the sweep batch limit.
    async with memdb() as s:
        for txn_id in big[:5]:
            (await s.get(Transaction, txn_id)).review_status = "resolved"
        # A new vocabulary can send notified rows back to review.
        for txn_id in big[:3]:
            (await s.get(Transaction, txn_id)).review_status = "pending"
        await s.commit()
    late = [await _seed_pending(memdb, bank_statement_upload_id=41) for _ in range(55)]
    telegram_send.clear()
    assert await sweep.run_review_notify() == 0
    assert [text.split("\n")[0] for text, _ in telegram_send] == [
        "\U0001f50d 58 more rows need a category"
    ]
    async with memdb() as s:
        assert (await s.get(Transaction, late[-1])).review_status == "notified"


async def test_failed_send_is_retried_until_the_cap(memdb, telegram_send, monkeypatch):
    """A transient send failure does not strand a row before the cap. A row at
    the cap is skipped, so the queue drains. A failed prompt keeps its import
    slot and holds back the summary of the import."""
    texts: list[str] = []

    async def failing_send(*a, text, **kw):
        texts.append(text)
        raise RuntimeError("still failing")

    monkeypatch.setattr(tg, "_send_with_retry", failing_send)
    from financial_dashboard.services import settings

    monkeypatch.setitem(settings._cache, "telegram.bulk_threshold", "1")
    txn_id = await _seed_pending(memdb, bank_statement_upload_id=41)
    await _seed_pending(memdb, bank_statement_upload_id=41)

    for expected in range(1, sweep._MAX_NOTIFY_ATTEMPTS + 1):
        await sweep.run_review_notify()
        async with memdb() as s:
            txn = await s.get(Transaction, txn_id)
        assert txn.review_status == "pending"
        assert txn.notify_attempts == expected
        assert txn.last_notified_at is None

    # The abandoned row frees its slot for the next row of the import.
    assert await sweep.run_review_notify() == 0
    assert len(texts) == sweep._MAX_NOTIFY_ATTEMPTS + 1
    assert all("Needs a category" in text for text in texts)
