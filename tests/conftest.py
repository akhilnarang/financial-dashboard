from decimal import Decimal  # noqa: F401

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db.models import Base


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _restore_settings_cache():
    """Snapshot/restore the global settings._cache around each test so tests
    that mutate it (e.g. CAS feature tests) can't leak into unrelated ones."""
    from financial_dashboard.services import settings as settings_mod

    snapshot = dict(settings_mod._cache)
    try:
        yield
    finally:
        settings_mod._cache.clear()
        settings_mod._cache.update(snapshot)


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def cas_statement_payload():
    return {
        "file": "sample.pdf",
        "meta": {
            "source": "cdsl",
            "investor_name": "Example Investor",
            "pan": "ABCDE1234F",
            "statement_period_start": "2026-04-01",
            "statement_period_end": "2026-04-30",
            "generated_on": "2026-05-02",
        },
        "accounts": [
            {
                "depository": "CDSL",
                "dp_id": "12088700",
                "client_id": "00000001",
                "dp_name": "Example DP",
                "total_value": "150000.00",
                "holdings": [
                    {
                        "name": "Equity A",
                        "isin": "INE000A01012",
                        "asset_class": "equity",
                        "quantity": "100",
                        "price": "1000.00",
                        "value": "100000.00",
                        "flags": [],
                        "notes": None,
                    },
                    {
                        "name": "ETF B",
                        "isin": "INF000B01012",
                        "asset_class": "etf",
                        "quantity": "50",
                        "price": "1000.00",
                        "value": "50000.00",
                        "flags": [],
                        "notes": None,
                    },
                ],
            }
        ],
        "folios": [
            {
                "folio_number": "9999999999",
                "amc": "Example AMC",
                "total_value": "50000.00",
                "schemes": [
                    {
                        "scheme_name": "Example Fund",
                        "isin": "INF000M01018",
                        "units": "1000",
                        "nav": "50.00",
                        "value": "50000.00",
                        "cost": "40000.00",
                        "flags": [],
                        "notes": None,
                    }
                ],
            }
        ],
        "transactions": [],
        "summary": {
            "asset_class_totals": {
                "Equity": "100000.00",
                "Mutual Funds": "100000.00",
            },
            "grand_total": "200000.00",
        },
        "reconciliation": {
            "portfolio_ok": True,
            "portfolio_delta": "0.00",
            "holdings": [],
            "warnings": [],
        },
    }
