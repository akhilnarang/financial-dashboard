from decimal import Decimal  # noqa: F401
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard
from financial_dashboard.api import router as api_router
from financial_dashboard.core.deps import get_session
from financial_dashboard.db.models import Account, Base
from financial_dashboard.web import get_router

STATIC_DIR = Path(financial_dashboard.__file__).resolve().parent / "static"


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


# The cashflow report is scoped to the accounts a row is linked to, so an
# unlinked row reaches none of its figures. These give a test one place to say
# which side of that boundary a seeded transaction is on. The ids are fixed so a
# helper can link a row without threading an Account object through every call,
# and MISSING_ACCOUNT_ID deliberately names no row at all: the test engine does
# not enforce foreign keys, which is what makes a dangling link — the only way to
# reach a NULL account type through the ORM's non-null column — reachable here.
BANK_ACCOUNT_ID = 1
CARD_ACCOUNT_ID = 2
MISSING_ACCOUNT_ID = 9999


async def ensure_account(
    session: AsyncSession, account_id: int, account_type: str
) -> int:
    """Create the account with this id and type once; return its id either way."""
    if await session.get(Account, account_id) is None:
        session.add(
            Account(
                id=account_id,
                bank="hdfc",
                label=f"test {account_type}",
                type=account_type,
            )
        )
        await session.flush()
    return account_id


async def bank_account(session: AsyncSession) -> int:
    """The bank account a seeded cashflow row belongs to unless it says otherwise."""
    return await ensure_account(session, BANK_ACCOUNT_ID, "bank_account")


async def card_account(session: AsyncSession) -> int:
    """A credit-card account: its rows are out of every bank-scoped figure."""
    return await ensure_account(session, CARD_ACCOUNT_ID, "credit_card")


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
async def client(session):
    """HTTP client over an app mounting both routers, bound to the ``session`` fixture.

    Both the JSON (``/api/...``) and HTML routers are mounted, as the real app
    factory does, so a single fixture serves page tests and endpoint tests alike.
    The app is built directly rather than via ``create_app`` to skip the lifespan
    (DB init, pollers, auth), and ``get_session`` yields the test session so
    writes made in a test are visible to the request that follows.

    ``/static`` is mounted from the same directory the real app serves, so a page's
    ``<script src="/static/...">`` resolves under test: without it every asset a
    page loads would 404 here and a renamed or deleted module would still pass.
    """
    app = FastAPI()

    async def _override_session():
        yield session

    app.dependency_overrides[get_session] = _override_session
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(api_router)
    app.include_router(get_router())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


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
