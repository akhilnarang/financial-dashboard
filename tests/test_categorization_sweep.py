# tests/test_categorization_sweep.py
import json
from decimal import Decimal

import pytest

from financial_dashboard.db.models import Base, Transaction
from financial_dashboard.services.categorization import sweep

pytestmark = pytest.mark.anyio


@pytest.fixture
async def memdb(monkeypatch):
    from sqlalchemy.ext.asyncio import (
        async_sessionmaker,
        create_async_engine,
        AsyncSession,
    )

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(
        "financial_dashboard.services.categorization.sweep.async_session", maker
    )
    yield maker
    await engine.dispose()


async def test_rule_sweep_categorizes_interest_rows(memdb):
    async with memdb() as s:
        s.add(
            Transaction(
                bank="testbank",
                email_type="x",
                direction="credit",
                amount=Decimal("10"),
                channel="interest",
            )
        )
        await s.commit()

    n = await sweep.run_rule_sweep()
    assert n == 1

    async with memdb() as s:
        from sqlalchemy import select

        row = (await s.execute(select(Transaction))).scalars().one()
        assert row.category == "interest"
        assert row.category_method == "rule"


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
        decisions = (await s.scalars(select(CategoryReviewDecision))).all()
        assert [decision.status for decision in decisions] == (
            ["superseded", "active"] if prompt_failed else ["active"]
        )
        row.category_method = "manual"
        row.category_vocab_version = 1
        row.review_status = "notified"
        await s.commit()
    assert await sweep.run_llm_sweep() == 0


async def test_review_notify_links_id_and_escapes_fields(memdb, monkeypatch):
    """Review messages link #id to the transaction page (when base_url set) and
    html-escape counterparty/reason and the literal <note>/<category> hint."""
    import financial_dashboard.services.telegram as tg
    from financial_dashboard.db.models import Transaction

    monkeypatch.setattr(sweep, "is_telegram_configured", lambda: True)
    monkeypatch.setattr(sweep, "get_telegram_chat_id", lambda: 123)
    # Adversarial base_url (stray quote) — must not break out of the href attribute.
    monkeypatch.setattr(sweep, "get_app_base_url", lambda: 'http://host:8000"x')

    sent: list = []

    async def fake_send(app, *, chat_id, text, parse_mode=None, **kw):
        sent.append((text, parse_mode))

    monkeypatch.setattr(tg, "tg_app", object())
    monkeypatch.setattr(tg, "_send_with_retry", fake_send)

    async with memdb() as s:
        s.add(
            Transaction(
                # direction/currency are free-text parser fields — include markup.
                bank="b",
                email_type="x",
                direction="deb<it",
                currency="IN&R",
                amount=Decimal("5"),
                counterparty="A & B <x>",
                review_status="pending",
                review_reason="unclear <tag>",
            )
        )
        await s.commit()

    n = await sweep.run_review_notify()
    assert n == 1
    text, mode = sent[0]
    assert mode == "HTML"
    # href attribute value escaped — the stray quote can't terminate the attribute
    assert 'href="http://host:8000&quot;x/transactions/' in text
    assert "A &amp; B &lt;x&gt;" in text  # counterparty escaped
    assert "unclear &lt;tag&gt;" in text  # reason escaped
    assert "deb&lt;it" in text  # direction escaped
    assert "IN&amp;R" in text  # currency escaped
    assert "&lt;note&gt;" in text and "&lt;category&gt;" in text
