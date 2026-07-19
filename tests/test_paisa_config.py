"""FX rate parsing and lookup for the ``paisa.fx_rates`` setting.

Pins the deterministic-historical contract: validate positive Decimal, normalize
uppercase, choose latest rate on/before the transaction date, and drop anything
malformed so a bad entry never reaches the journal.
"""

from decimal import Decimal

import pytest

from financial_dashboard.services.paisa.config import (
    FxRate,
    PaisaProjectionConfig,
    _parse_fx_rates,
)
from financial_dashboard.services.settings import _cache as settings_cache

pytestmark = pytest.mark.anyio

import datetime as dt  # noqa: E402


def _cfg(fx):
    return PaisaProjectionConfig(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=(1,),
        cutover_date=dt.date(2026, 1, 1),
        account_mappings={},
        category_mappings={},
        non_inr_policy="priced",
        request_timeout_seconds=15,
        fx_rates=fx,
    )


def test_parse_normalizes_currency_uppercase_and_sorts():
    fx = _parse_fx_rates(
        {
            "usd": [
                {"date": "2026-02-01", "rate": "84"},
                {"date": "2026-01-01", "rate": "83.00"},
            ]
        }
    )
    assert list(fx.keys()) == ["USD"]
    rates = fx["USD"]
    # Sorted ascending by date regardless of input order.
    assert [r.date for r in rates] == [dt.date(2026, 1, 1), dt.date(2026, 2, 1)]
    assert rates[0].rate == Decimal("83.0000")
    assert rates[1].rate == Decimal("84.0000")


def test_parse_drops_non_positive_and_unparseable():
    fx = _parse_fx_rates(
        {
            "USD": [
                {"date": "2026-01-01", "rate": "83.00"},  # ok
                {"date": "2026-01-02", "rate": "0"},  # non-positive → drop
                {"date": "2026-01-03", "rate": "-5"},  # negative → drop
                {"date": "2026-01-04", "rate": "not-a-number"},  # junk → drop
                {"date": "not-a-date", "rate": "10"},  # junk date → drop
                {"date": "2026-01-05"},  # missing rate → drop
                {"date": "2026-01-06", "rate": "1.5"},  # ok
            ],
            "EUR": "not-a-list",  # ignored entirely
            "": [],  # empty currency ignored
        }
    )
    assert list(fx.keys()) == ["USD"]
    assert [r.date for r in fx["USD"]] == [
        dt.date(2026, 1, 1),
        dt.date(2026, 1, 6),
    ]


def test_parse_accepts_numeric_rate_values():
    # A JSON number (int/float) round-trips through Decimal(str(...)).
    fx = _parse_fx_rates({"USD": [{"date": "2026-01-01", "rate": 83}]})
    assert fx["USD"][0].rate == Decimal("83.0000")


@pytest.mark.parametrize(
    "rate",
    ["NaN", "sNaN", "Infinity", "-Infinity", "1e999999", "1e-999999"],
)
def test_parse_drops_non_finite_and_extreme_rates(rate):
    fx = _parse_fx_rates({"USD": [{"date": "2026-01-01", "rate": rate}]})
    assert fx == {}


def test_parse_empty_and_garbage_return_empty():
    assert _parse_fx_rates({}) == {}
    assert _parse_fx_rates("nope") == {}  # type: ignore[arg-type]
    assert _parse_fx_rates(None) == {}  # type: ignore[arg-type]
    assert _parse_fx_rates({"USD": []}) == {}


def test_fx_rate_for_chooses_latest_on_or_before():
    fx = {
        "USD": (
            FxRate(dt.date(2026, 1, 1), Decimal("82.0000")),
            FxRate(dt.date(2026, 2, 10), Decimal("84.5000")),
            FxRate(dt.date(2026, 3, 1), Decimal("85.0000")),
        )
    }
    cfg = _cfg(fx)
    # Before any rate → None (missing_fx_rate downstream).
    assert cfg.fx_rate_for("USD", dt.date(2025, 12, 31)) is None
    # Exactly on a rate date → that rate.
    assert cfg.fx_rate_for("USD", dt.date(2026, 1, 1)).rate == Decimal("82.0000")
    # Between two rates → the earlier (latest on/before).
    assert cfg.fx_rate_for("USD", dt.date(2026, 2, 15)).rate == Decimal("84.5000")
    # After the last → the last.
    assert cfg.fx_rate_for("USD", dt.date(2026, 12, 31)).rate == Decimal("85.0000")


def test_fx_rate_for_unknown_currency_is_none():
    cfg = _cfg({})
    assert cfg.fx_rate_for("USD", dt.date(2026, 2, 1)) is None
    # Case-insensitive lookup (currency normalized at load).
    cfg2 = _cfg({"USD": (FxRate(dt.date(2026, 1, 1), Decimal("83.0000")),)})
    assert cfg2.fx_rate_for("usd", dt.date(2026, 2, 1)).rate == Decimal("83.0000")


async def test_load_config_reads_paisa_fx_rates_and_ledger_cli():
    # load_config reads the live settings cache directly. Populate it for the
    # Paisa keys and assert the typed resolution. The autouse conftest fixture
    # snapshots/restores settings_cache around the test.
    settings_cache["paisa.fx_rates"] = (
        '{"USD": [{"date": "2026-01-15", "rate": "83.00"}]}'
    )
    settings_cache["paisa.ledger_cli"] = "hledger"
    settings_cache["paisa.non_inr_policy"] = "priced"
    from financial_dashboard.services.paisa.config import load_config

    cfg = load_config()
    assert cfg.ledger_cli == "hledger"
    assert cfg.non_inr_policy == "priced"
    assert cfg.fx_rate_for("USD", dt.date(2026, 2, 1)).rate == Decimal("83.0000")


def test_load_config_defaults_ledger_cli_to_ledger():
    from financial_dashboard.services.paisa.config import load_config

    settings_cache.pop("paisa.ledger_cli", None)
    settings_cache.pop("paisa.fx_rates", None)
    assert load_config().ledger_cli == "ledger"
    assert load_config().fx_rates == {}


def test_coerce_policy_only_accepts_skip_or_priced():
    from financial_dashboard.services.paisa.config import _coerce_policy

    assert _coerce_policy("priced") == "priced"
    assert _coerce_policy("skip") == "skip"
    assert _coerce_policy("include") == "skip"  # v1 value coerced to skip
    assert _coerce_policy("") == "skip"
    assert _coerce_policy("garbage") == "skip"
