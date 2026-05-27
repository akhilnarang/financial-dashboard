import datetime as dt
from decimal import Decimal

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from financial_dashboard.core.deps import get_session
from financial_dashboard.db.enums import SnapshotCategory, SnapshotKind, SnapshotSource
from financial_dashboard.db.models import BalanceSnapshot, CasUpload
from financial_dashboard.web import router as web_router

pytestmark = pytest.mark.anyio


def _build_app(session_factory):
    app = FastAPI()
    app.dependency_overrides[get_session] = session_factory
    app.include_router(web_router)
    return app


async def test_networth_page_renders_current_figure(session):
    upload = CasUpload(
        portfolio_key="ABCDE1234F",
        depository_source="cdsl",
        investor_name="Example Investor",
        statement_date=dt.date(2026, 4, 30),
        grand_total=Decimal("200000.00"),
        portfolio_ok=True,
        raw_holdings_json="{}",
    )
    session.add(upload)
    await session.flush()
    session.add(
        BalanceSnapshot(
            cas_upload_id=upload.id,
            portfolio_key=upload.portfolio_key,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.investment.value,
            as_of_date=dt.date(2026, 4, 30),
            value=Decimal("200000.00"),
            source=SnapshotSource.cas.value,
        )
    )
    await session.commit()

    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        response = await client.get("/networth")

    assert response.status_code == 200
    assert "Net Worth" in response.text
    assert "2L" in response.text


async def test_cas_upload_rejects_oversize_file(session):
    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        oversize = b"%PDF-" + b"x" * (10 * 1024 * 1024 + 1)
        response = await client.post(
            "/cas/upload",
            data={"password": "", "force_replace": "false"},
            files={"file": ("big.pdf", oversize, "application/pdf")},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert "exceeds" in response.headers["location"].lower()


async def test_api_cas_upload_rejects_oversize_file(session):
    from financial_dashboard.api import router as api_router

    app = FastAPI()
    app.dependency_overrides[get_session] = lambda: (yield session)
    app.include_router(api_router)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        oversize = b"%PDF-" + b"x" * (10 * 1024 * 1024 + 1)
        response = await client.post(
            "/api/cas/upload",
            data={"password": "", "force_replace": "false"},
            files={"file": ("big.pdf", oversize, "application/pdf")},
        )

    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]
