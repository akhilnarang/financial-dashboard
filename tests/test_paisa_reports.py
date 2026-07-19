"""Curated Paisa report normalization + TTL cache tests.

Covers:
* Typed normalization of every report kind against fixtures modeled on the
  upstream v0.7.4 response shapes (server.go handlers).
* The PaisaReportCache: TTL expiry, cache-hit call counting, concurrent
  request coalescing (one fetch for N concurrent readers), TTL=0 disabling,
  bounded eviction, and per-app scoping via get_report_cache.
* Report-route dispatch: disabled mode → zero upstream calls; failure isolation
  → typed ok=False (never 500); cache hit reports cached=True.
"""

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from financial_dashboard.integrations.paisa import (
    PaisaError,
    REPORT_ALLOCATION,
    REPORT_ASSETS_BALANCE,
    REPORT_BUDGET,
    REPORT_INCOME_STATEMENT,
    REPORT_LIABILITIES,
    REPORT_RECURRING,
    normalize_report,
)
from financial_dashboard.services.paisa import surface
from financial_dashboard.services.paisa.report_cache import PaisaReportCache

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------- #
# Upstream-shape fixtures (modeled on internal/server/*.go v0.7.4 responses)
# --------------------------------------------------------------------------- #

BUDGET_PAYLOAD = {
    "budgetsByMonth": {
        "2026-01": {
            "date": "2026-01-01T00:00:00Z",
            "accounts": [
                {
                    "account": "Expenses:Food",
                    "forecast": "100",
                    "actual": "40",
                    "rollover": "0",
                    "available": "60",
                    "date": "2026-01-01T00:00:00Z",
                    "expenses": [],
                },
                {
                    "account": "Expenses:Rent",
                    "forecast": "120",
                    "actual": "60",
                    "rollover": "0",
                    "available": "60",
                    "date": "2026-01-01T00:00:00Z",
                    "expenses": [],
                },
                # Malformed rows are safely ignored, never summed.
                {"account": "Expenses:Junk", "actual": "not-a-number"},
                "not-a-dict",
            ],
            "availableThisMonth": "60",
            "endOfMonthBalance": "940",
            "forecast": "100",
        },
        "2026-02": {
            "date": "2026-02-01T00:00:00Z",
            "accounts": [],
            "availableThisMonth": "0",
            "endOfMonthBalance": "940",
            "forecast": "0",
        },
    },
    "checkingBalance": "1000",
    "availableForBudgeting": "940",
}

ALLOCATION_PAYLOAD = {
    "aggregates": {
        # Upstream v0.7.4 zeroes the *parent* rollup aggregate's market_amount
        # (only leaf accounts carry an amount), so "Assets" is 0 here while its
        # children carry the real balances — mirroring the real response shape
        # rather than a client-side sum.
        "Assets": {"date": "x", "account": "Assets", "market_amount": "0"},
        "Assets:Bank": {"date": "x", "account": "Assets:Bank", "market_amount": "5000"},
        "Assets:Stocks": {
            "date": "x",
            "account": "Assets:Stocks",
            "market_amount": "7000",
        },
        # A non-dict value is ignored (never decoded as a dict-as-decimal).
        "Assets:Malformed": "12000",
        # A dict missing market_amount falls back to 0.
        "Assets:NoAmount": {"account": "Assets:NoAmount", "date": "x"},
    },
    "aggregates_timeline": [],  # deliberately large; never proxied
    "allocation_targets": [
        {"name": "Equity", "target": "60", "current": "55.5", "aggregates": {}},
        {"name": "Debt", "target": "40", "current": "44.5", "aggregates": {}},
    ],
}

RECURRING_PAYLOAD = {
    "transaction_sequences": [
        {
            "transactions": [{}, {}, {}],
            "key": "netflix",
            "period": "monthly",
            "interval": 30,
        },
        {
            "transactions": [{}],
            "key": "lonely",
            "period": "",
            "interval": 0,
        },
    ]
}

INCOME_STATEMENT_PAYLOAD = {
    "yearly": {
        "2025-2026": {
            "startingBalance": "1000",
            "endingBalance": "2000",
            "date": "2025-04-01T00:00:00Z",
            "income": {"Income:Salary": "-50000"},
            "interest": {"Income:Interest:Savings": "-200"},
            "equity": {},
            "pnl": {"Assets:Stocks": "300"},
            "liabilities": {},
            "tax": {"Expenses:Tax:Income": "5000"},
            "expenses": {"Expenses:Food": "8000", "Expenses:Rent": "12000"},
        }
    }
}

LIABILITIES_PAYLOAD = {
    "liability_breakdowns": {
        "Liabilities:HomeLoan": {
            "group": "Liabilities:HomeLoan",
            "drawn_amount": "5000000",
            "repaid_amount": "1000000",
            "interest_amount": "200000",
            "balance_amount": "4200000",
            "apr": "8.5",
        }
    }
}


def test_normalize_budget_curates_monthly_totals_only():
    r = normalize_report(REPORT_BUDGET, BUDGET_PAYLOAD)
    assert len(r.months) == 2
    assert r.months[0].month == "2026-01"
    assert r.months[0].forecast == "100.00"
    # actual is the Decimal sum of accounts[].actual (40 + 60); malformed rows
    # (a non-numeric actual, a non-dict entry) are ignored, never summed.
    assert r.months[0].actual == "100.00"
    assert r.months[0].available_this_month == "60.00"
    assert r.months[0].end_of_month_balance == "940.00"
    # A month with no account rows sums to zero (v0.7.4 has no month-level actual).
    assert r.months[1].actual == "0.00"
    assert r.checking_balance == "1000.00"
    assert r.available_for_budgeting == "940.00"
    # Per-account expense postings are NOT proxied.
    assert not hasattr(r.months[0], "accounts")


def test_normalize_budget_months_sorted_numerically():
    payload = {
        "budgetsByMonth": {
            "2026-10": {
                "availableThisMonth": "1",
                "endOfMonthBalance": "1",
                "forecast": "1",
                "actual": "1",
            },
            "2026-02": {
                "availableThisMonth": "1",
                "endOfMonthBalance": "1",
                "forecast": "1",
                "actual": "1",
            },
        }
    }
    r = normalize_report(REPORT_BUDGET, payload)
    assert [m.month for m in r.months] == ["2026-02", "2026-10"]


def test_normalize_allocation_curates_targets_and_snapshot():
    r = normalize_report(REPORT_ALLOCATION, ALLOCATION_PAYLOAD)
    assert len(r.targets) == 2
    assert r.targets[0].name == "Equity"
    assert r.targets[0].current_percent == "55.50"
    # The daily aggregate_timeline is not proxied; only the current snapshot.
    # Each aggregate's market_amount is read from the Aggregate struct (verified
    # casing), not the struct itself decoded as a decimal. The parent "Assets"
    # rollup is zero upstream (only leaves carry amounts); children are intact.
    by_name = {m.account: m.amount for m in r.aggregate_accounts}
    assert by_name["Assets"] == "0.00"
    assert by_name["Assets:Bank"] == "5000.00"
    assert by_name["Assets:Stocks"] == "7000.00"
    # A struct missing market_amount falls back to 0; a non-dict value is dropped.
    assert by_name["Assets:NoAmount"] == "0"
    assert "Assets:Malformed" not in by_name


def test_normalize_recurring_summarizes_counts():
    r = normalize_report(REPORT_RECURRING, RECURRING_PAYLOAD)
    assert len(r.sequences) == 2
    netflix = next(s for s in r.sequences if s.key == "netflix")
    assert netflix.count == 3
    assert netflix.interval_days == 30
    assert netflix.period == "monthly"
    lonely = next(s for s in r.sequences if s.key == "lonely")
    assert lonely.count == 1


def test_normalize_income_statement_sections():
    r = normalize_report(REPORT_INCOME_STATEMENT, INCOME_STATEMENT_PAYLOAD)
    assert len(r.periods) == 1
    p = r.periods[0]
    assert p.period == "2025-2026"
    assert p.starting_balance == "1000.00"
    assert p.ending_balance == "2000.00"
    assert any(a.account == "Income:Salary" for a in p.income)
    assert any(a.account == "Expenses:Rent" for a in p.expenses)
    assert any(a.account == "Expenses:Tax:Income" for a in p.tax)
    assert any(a.account == "Assets:Stocks" for a in p.pnl)


def test_normalize_liabilities():
    r = normalize_report(REPORT_LIABILITIES, LIABILITIES_PAYLOAD)
    assert len(r.breakdowns) == 1
    b = r.breakdowns[0]
    assert b.group == "Liabilities:HomeLoan"
    assert b.balance_amount == "4200000.00"
    assert b.apr == "8.50"


def test_normalize_handles_non_dict_and_bad_numbers():
    # Non-dict payloads and unparseable numbers degrade to zeros/empty, never raise.
    r = normalize_report(REPORT_BUDGET, [])
    assert r.months == ()
    assert r.checking_balance == "0"
    r2 = normalize_report(
        REPORT_LIABILITIES,
        {"liability_breakdowns": {"X": {"balance_amount": "not-a-number"}}},
    )
    assert r2.breakdowns[0].balance_amount == "0"


# --------------------------------------------------------------------------- #
# PaisaReportCache
# --------------------------------------------------------------------------- #


async def test_cache_ttl_hits_do_not_call_upstream():
    cache = PaisaReportCache()
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        return {"v": calls}

    a = await cache.read("budget", ttl_seconds=60, fetch=fetch)
    b = await cache.read("budget", ttl_seconds=60, fetch=fetch)
    # Same value both times; the hit flag distinguishes fetch vs cache serve.
    assert a.value == b.value == {"v": 1}
    assert a.hit is False  # miss → fetched
    assert b.hit is True  # served from the fresh cache entry
    assert calls == 1
    assert cache.upstream_calls == 1


async def test_cache_ttl_zero_always_fetches_but_coalesces():
    cache = PaisaReportCache()

    async def fetch():
        return object()

    a = await cache.read("budget", ttl_seconds=0, fetch=fetch)
    b = await cache.read("budget", ttl_seconds=0, fetch=fetch)
    # TTL 0 disables caching → 2 fetches, neither reported as a hit.
    assert a.hit is False
    assert b.hit is False
    assert cache.upstream_calls == 2


async def test_cache_ttl_zero_coalesces_only_while_fetch_is_in_flight():
    cache = PaisaReportCache()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow_fetch():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return f"result-{calls}"

    tasks = [
        asyncio.create_task(cache.read("budget", ttl_seconds=0, fetch=slow_fetch))
        for _ in range(10)
    ]
    await started.wait()
    # Let every created reader join while the one fetch is deliberately held.
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*tasks)

    assert [result.value for result in results] == ["result-1"] * 10
    assert all(result.hit is False for result in results)
    assert calls == cache.upstream_calls == 1
    assert cache.size == 0

    # The completed value was not retained: a later caller performs a new
    # upstream read rather than reusing the finished in-flight task.
    later = await cache.read("budget", ttl_seconds=0, fetch=slow_fetch)
    assert later.value == "result-2"
    assert later.hit is False
    assert calls == cache.upstream_calls == 2
    assert cache.size == 0


async def test_cache_coalesces_concurrent_readers_into_one_call():
    cache = PaisaReportCache()
    started = 0

    async def slow_fetch():
        nonlocal started
        started += 1
        await asyncio.sleep(0.02)
        return "result"

    results = await asyncio.gather(
        *(cache.read("budget", ttl_seconds=60, fetch=slow_fetch) for _ in range(10))
    )
    assert all(r.value == "result" for r in results)
    assert started == 1
    assert cache.upstream_calls == 1
    # Exactly one reader (the lock winner) missed and fetched; every reader it
    # coalesced behind it observes the freshly stored entry → a hit.
    misses = [r for r in results if not r.hit]
    hits = [r for r in results if r.hit]
    assert len(misses) == 1
    assert len(hits) == 9


async def test_cache_propagates_fetch_error_without_caching():
    cache = PaisaReportCache()
    attempts = 0

    async def fetch():
        nonlocal attempts
        attempts += 1
        raise PaisaError("unreachable", "down")

    with pytest.raises(PaisaError):
        await cache.read("budget", ttl_seconds=60, fetch=fetch)
    # Error was not cached → a second read calls again.
    with pytest.raises(PaisaError):
        await cache.read("budget", ttl_seconds=60, fetch=fetch)
    assert attempts == 2


async def test_cache_evicts_oldest_beyond_cap():
    cache = PaisaReportCache(max_entries=2)
    await cache.read("a", 60, _const("a"))
    await cache.read("b", 60, _const("b"))
    await cache.read("c", 60, _const("c"))  # evicts "a" (oldest)
    assert cache.size == 2
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        return "fresh"

    # "a" was evicted → a fresh fetch.
    await cache.read("a", 60, fetch)
    assert calls == 1


def _const(value):
    async def fetch():
        return value

    return fetch


async def test_cache_concurrent_different_key_reads_are_independent():
    """Concurrent reads of distinct keys neither coalesce nor race each other's
    hit flag: each key misses on its first read and hits on the second, and one
    key's fetch never flips another key's hit (the cross-key race the old
    shared-counter snapshot had)."""
    cache = PaisaReportCache()
    keys = ("budget", "allocation", "recurring", "liabilities")

    async def fetch_for(key):
        await asyncio.sleep(0.01)
        return f"value-{key}"

    # All four keys' first reads in parallel — each must miss (fetch).
    first = await asyncio.gather(
        *(cache.read(k, 60, lambda k=k: fetch_for(k)) for k in keys)
    )
    assert [r.value for r in first] == [f"value-{k}" for k in keys]
    assert all(r.hit is False for r in first)
    assert cache.upstream_calls == len(keys)

    # Second round, again in parallel — every key now serves from cache.
    second = await asyncio.gather(
        *(cache.read(k, 60, lambda k=k: fetch_for(k)) for k in keys)
    )
    assert [r.value for r in second] == [f"value-{k}" for k in keys]
    assert all(r.hit is True for r in second)
    # No new upstream calls: all four were cache hits.
    assert cache.upstream_calls == len(keys)


async def test_cache_same_key_hit_flag_survives_concurrent_other_key_fetch():
    """The regression target for the cache contract: a same-key cache hit's
    ``hit`` flag is decided under the per-key lock, so a concurrent
    *different-key* fetch (which bumps the shared upstream_calls counter) cannot
    flip it. Under the old shared-counter snapshot this was a race."""
    cache = PaisaReportCache()

    async def slow_other():
        await asyncio.sleep(0.05)
        return "other"

    # Prime "budget" so the same-key read below is a guaranteed cache hit.
    primed = await cache.read("budget", 60, _const("budget"))
    assert primed.hit is False

    # Run a slow different-key fetch concurrently with a same-key (cached) read.
    other_task = asyncio.create_task(cache.read("allocation", 60, slow_other))
    cached_read = await cache.read("budget", 60, _const("budget"))
    other = await other_task
    assert cached_read.hit is True  # the regression: must stay True
    assert other.hit is False


# --------------------------------------------------------------------------- #
# Report route dispatch: disabled, failure isolation, cache hit
# --------------------------------------------------------------------------- #


class _AppState:
    """Minimal stand-in for Starlette's app.state (attributes)."""


async def test_report_disabled_mode_makes_no_upstream_calls(monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _cfg(mode="disabled"))
    state = _AppState()
    summary = await surface.report_summary(state, REPORT_BUDGET)
    assert summary.ok is False
    assert summary.reason == "disabled"
    cache = surface.get_report_cache(state)
    assert cache.upstream_calls == 0


async def test_report_failure_isolated_to_typed_body(monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _cfg(mode="connect"))
    state = _AppState()

    async def boom(config, report):
        raise PaisaError("http_error", "500")

    monkeypatch.setattr(surface, "_fetch_report", boom)
    summary = await surface.report_summary(state, REPORT_BUDGET)
    assert summary.ok is False
    assert summary.reason == "http_error"


async def test_report_cache_hit_marks_cached(monkeypatch):
    monkeypatch.setattr(surface, "load_config", lambda: _cfg(mode="connect"))

    async def fetch(config, report):
        return normalize_report(report, BUDGET_PAYLOAD)

    monkeypatch.setattr(surface, "_fetch_report", fetch)
    state = _AppState()
    first = await surface.report_summary(state, REPORT_BUDGET)
    second = await surface.report_summary(state, REPORT_BUDGET)
    assert first.ok is True
    assert first.cached is False
    assert second.ok is True
    assert second.cached is True
    cache = surface.get_report_cache(state)
    assert cache.upstream_calls == 1


def test_report_cache_key_uses_normalized_connection_identity_without_secret():
    base = _cfg(
        mode="connect",
        base_url="HTTP://LOCALHOST:80/",
        auth_username="alice",
        auth_password="correct horse battery staple",
        ledger_cli="ledger",
    )
    key = surface._report_cache_key(base, REPORT_BUDGET)
    equivalent = surface._report_cache_key(
        _cfg(
            mode="connect",
            base_url="http://localhost",
            auth_username="alice",
            auth_password="correct horse battery staple",
            ledger_cli="LEDGER",
        ),
        REPORT_BUDGET,
    )
    assert key == equivalent
    assert key.base_url == "http://localhost/"
    assert key.auth_username == "alice"
    assert "correct horse battery staple" not in repr(key)

    switches = [
        _cfg(**{**base._asdict(), "base_url": "http://localhost:7500"}),
        _cfg(**{**base._asdict(), "auth_username": "bob"}),
        _cfg(**{**base._asdict(), "auth_password": "different-secret"}),
        _cfg(**{**base._asdict(), "ledger_cli": "beancount"}),
    ]
    assert all(surface._report_cache_key(cfg, REPORT_BUDGET) != key for cfg in switches)


async def test_report_config_switch_never_serves_prior_instance_cache(monkeypatch):
    current = _cfg(
        mode="connect",
        base_url="http://127.0.0.1:7500",
        auth_username="alice",
        auth_password="first-secret",
        ledger_cli="ledger",
    )
    monkeypatch.setattr(surface, "load_config", lambda: current)
    calls = 0

    async def fetch(config, report):
        nonlocal calls
        calls += 1
        payload = dict(BUDGET_PAYLOAD)
        payload["checkingBalance"] = str(calls)
        return normalize_report(report, payload)

    monkeypatch.setattr(surface, "_fetch_report", fetch)
    state = _AppState()

    first = await surface.report_summary(state, REPORT_BUDGET)
    assert first.budget is not None
    assert first.budget.checking_balance == "1.00"

    configs = [
        _cfg(**{**current._asdict(), "base_url": "http://127.0.0.1:7501"}),
        _cfg(**{**current._asdict(), "auth_username": "bob"}),
        _cfg(**{**current._asdict(), "auth_password": "second-secret"}),
        _cfg(**{**current._asdict(), "ledger_cli": "beancount"}),
    ]
    for expected, config in enumerate(configs, start=2):
        current = config
        summary = await surface.report_summary(state, REPORT_BUDGET)
        assert summary.cached is False
        assert summary.budget is not None
        assert summary.budget.checking_balance == f"{expected}.00"

    assert calls == 5
    repeated = await surface.report_summary(state, REPORT_BUDGET)
    assert repeated.cached is True
    assert calls == 5


async def test_report_via_route_returns_typed_json_on_failure(monkeypatch):
    """An unexpected error in the surface still yields a typed JSON body, not 500."""
    app = FastAPI()

    async def boom(app_state, report):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(surface, "report_summary", boom)

    # Mount only the paisa report route by importing the api router.
    from financial_dashboard.api.extensions import paisa_router

    app.include_router(paisa_router, prefix="/api/extensions")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/extensions/paisa/reports/budget")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "paisa_report_failed"


async def test_report_cached_flag_unraced_by_concurrent_other_key(monkeypatch):
    """Regression for the cached-flag race: a same-key cache hit must report
    ``cached=True`` even while a concurrent *different-key* read fetches
    upstream. The old counter snapshot (``upstream_calls_before == after``)
    flipped a real hit into a reported miss when the other key bumped the shared
    counter mid-read; the per-key ``CacheRead.hit`` is decided under the lock
    and is immune."""
    monkeypatch.setattr(surface, "load_config", lambda: _cfg(mode="connect"))

    async def fetch(config, report):
        if report == REPORT_ALLOCATION:
            # Make the cross-key fetch overlap the cached budget read so the
            # shared counter bumps during its window.
            await asyncio.sleep(0.03)
        payload = BUDGET_PAYLOAD if report == REPORT_BUDGET else ALLOCATION_PAYLOAD
        return normalize_report(report, payload)

    monkeypatch.setattr(surface, "_fetch_report", fetch)
    state = _AppState()

    # Prime the budget cache with a first (miss) read.
    primed = await surface.report_summary(state, REPORT_BUDGET)
    assert primed.ok and primed.cached is False

    # Concurrently: a cached budget re-read (must be a hit) and a fresh
    # allocation fetch (a miss that bumps the shared counter).
    budget_again, allocation = await asyncio.gather(
        surface.report_summary(state, REPORT_BUDGET),
        surface.report_summary(state, REPORT_ALLOCATION),
    )
    assert budget_again.ok and budget_again.cached is True  # the regression
    assert allocation.ok and allocation.cached is False


# --------------------------------------------------------------------------- #
# _report_to_dto exhaustiveness
# --------------------------------------------------------------------------- #


def test_report_to_dto_rejects_assets_balance_and_unknown_kinds():
    """_report_to_dto is exhaustive over the report-page kinds; assets_balance
    (which flows through reconciliation) and any unknown kind hit an
    AssertionError, never a fall-through ValueError a caller could catch."""
    from financial_dashboard.services.paisa.surface import _report_to_dto

    assets = normalize_report(REPORT_ASSETS_BALANCE, {"asset_breakdowns": {}})
    with pytest.raises(AssertionError):
        _report_to_dto(REPORT_ASSETS_BALANCE, assets)
    with pytest.raises(AssertionError):
        _report_to_dto("not-a-real-report", object())


def test_report_to_dto_shapes_every_dto_kind():
    """Every member of _DTO_REPORT_KINDS round-trips through _report_to_dto
    without raising — pins the exhaustiveness contract from the positive side
    so a future kind added to the set without a branch fails this test."""
    from financial_dashboard.services.paisa.surface import (
        _DTO_REPORT_KINDS,
        _report_to_dto,
    )

    fixtures = {
        REPORT_BUDGET: BUDGET_PAYLOAD,
        REPORT_ALLOCATION: ALLOCATION_PAYLOAD,
        REPORT_RECURRING: RECURRING_PAYLOAD,
        REPORT_INCOME_STATEMENT: INCOME_STATEMENT_PAYLOAD,
        REPORT_LIABILITIES: LIABILITIES_PAYLOAD,
    }
    assert set(fixtures) == set(_DTO_REPORT_KINDS)
    for kind, payload in fixtures.items():
        dto = _report_to_dto(kind, normalize_report(kind, payload))
        assert dto.ok is True
        assert dto.report == kind


def test_report_to_dto_kind_set_excludes_assets_balance():
    """assets_balance is a supported report but is NOT in the DTO set — it flows
    through reconciliation, not the report page. Pin that so a future routing
    change cannot silently start sending it to _report_to_dto."""
    from financial_dashboard.integrations.paisa import SUPPORTED_REPORTS
    from financial_dashboard.services.paisa.surface import _DTO_REPORT_KINDS

    assert REPORT_ASSETS_BALANCE in SUPPORTED_REPORTS
    assert REPORT_ASSETS_BALANCE not in _DTO_REPORT_KINDS
    assert _DTO_REPORT_KINDS <= SUPPORTED_REPORTS


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _cfg(mode="connect", **overrides):
    import datetime as dt

    from financial_dashboard.services.paisa.config import PaisaProjectionConfig

    base = dict(
        mode=mode,
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=(),
        cutover_date=dt.date(2026, 1, 1),
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
    )
    base.update(overrides)
    return PaisaProjectionConfig(**base)
