import pytest

from financial_dashboard.services.categorization import backfill

pytestmark = pytest.mark.anyio


async def test_backfill_runs_rules_then_llm(monkeypatch):
    order = []

    async def fake_rule(**k):
        order.append(("rule", k.get("batch_limit")))
        return 0

    async def fake_llm(**k):
        order.append(("llm", k.get("batch_limit")))
        return 0

    monkeypatch.setattr(backfill, "run_rule_sweep", fake_rule)
    monkeypatch.setattr(backfill, "run_llm_sweep", fake_llm)

    rules_total, llm_total = await backfill.run_backfill(batch_size=50)
    assert order[0][0] == "rule"
    assert ("llm", 50) in order
    assert rules_total == 0 and llm_total == 0


async def test_backfill_rules_only(monkeypatch):
    order = []

    async def fake_rule(**k):
        order.append("rule")
        return 0

    async def fake_llm(**k):
        order.append("llm")
        return 0

    monkeypatch.setattr(backfill, "run_rule_sweep", fake_rule)
    monkeypatch.setattr(backfill, "run_llm_sweep", fake_llm)

    await backfill.run_backfill(rules_only=True)
    assert "llm" not in order
