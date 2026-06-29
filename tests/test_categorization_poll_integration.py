import pytest

from financial_dashboard.services import fetch as fetch_mod

pytestmark = pytest.mark.anyio


async def test_run_categorization_cycle_invokes_all(monkeypatch):
    calls = []

    async def fake_rule(**k):
        calls.append("rule")
        return 0

    async def fake_llm(**k):
        calls.append("llm")
        return 0

    async def fake_notify(**k):
        calls.append("notify")
        return 0

    async def fake_refresh(_session=None):
        calls.append("refresh")

    monkeypatch.setattr(fetch_mod, "load_merchant_rules", fake_refresh)
    monkeypatch.setattr(fetch_mod, "run_rule_sweep", fake_rule)
    monkeypatch.setattr(fetch_mod, "run_llm_sweep", fake_llm)
    monkeypatch.setattr(fetch_mod, "run_review_notify", fake_notify)

    await fetch_mod.run_categorization_cycle()
    assert calls == ["refresh", "rule", "llm", "notify"]
