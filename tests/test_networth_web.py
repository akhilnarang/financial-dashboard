import datetime as dt
from decimal import Decimal

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from financial_dashboard.core.deps import get_session
from financial_dashboard.db.enums import (
    ManualCategory,
    ManualKind,
    SnapshotCategory,
    SnapshotKind,
    SnapshotSource,
)
from financial_dashboard.db.models import BalanceSnapshot, CasUpload
from financial_dashboard.services import manual_items
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


async def test_manual_edit_collision_redirects_with_error(session):
    item = await manual_items.create_item(
        session,
        name="Cash",
        kind=ManualKind.asset,
        category=ManualCategory.cash,
        value=Decimal("5000.00"),
        as_of_date=dt.date(2026, 4, 1),
    )
    await manual_items.update_value(
        session,
        item_id=item.id,
        value=Decimal("7000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.commit()
    april = (
        await session.execute(
            select(BalanceSnapshot).where(
                BalanceSnapshot.as_of_date == dt.date(2026, 4, 1)
            )
        )
    ).scalar_one()

    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/networth/manual/snapshot/{april.id}/edit",
            data={"value": "9999.00", "as_of_date": "2026-05-01"},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]


async def test_manual_delete_redirects_clean(session):
    await manual_items.create_item(
        session,
        name="Cash",
        kind=ManualKind.asset,
        category=ManualCategory.cash,
        value=Decimal("5000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.commit()
    snap = await session.get(BalanceSnapshot, 1)

    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/networth/manual/snapshot/{snap.id}/delete", follow_redirects=False
        )

    assert resp.status_code == 303
    assert resp.headers["location"] == "/networth/manual"
    assert "error=" not in resp.headers["location"]


async def test_manual_page_renders_history_newest_first(session):
    item = await manual_items.create_item(
        session,
        name="Gold",
        kind=ManualKind.asset,
        category=ManualCategory.gold,
        value=Decimal("100000.00"),
        as_of_date=dt.date(2026, 4, 1),
    )
    await manual_items.update_value(
        session,
        item_id=item.id,
        value=Decimal("120000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.commit()

    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.get(
            "/networth/manual?error=An+entry+already+exists+for+that+date"
        )

    assert resp.status_code == 200
    body = resp.text
    assert "01 May 2026" in body
    assert "01 Apr 2026" in body
    assert body.index("01 May 2026") < body.index("01 Apr 2026")
    assert "/delete" in body
    assert "An entry already exists for that date" in body


async def test_manual_snapshot_edit_unknown_id_redirects_with_error(session):
    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/networth/manual/snapshot/999/edit",
            data={"value": "1.00", "as_of_date": "2026-05-01"},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]


async def test_manual_snapshot_delete_unknown_id_redirects_with_error(session):
    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/networth/manual/snapshot/999/delete", follow_redirects=False
        )

    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]


async def test_manual_edit_collision_renders_banner_after_redirect(session):
    item = await manual_items.create_item(
        session,
        name="Cash",
        kind=ManualKind.asset,
        category=ManualCategory.cash,
        value=Decimal("5000.00"),
        as_of_date=dt.date(2026, 4, 1),
    )
    await manual_items.update_value(
        session,
        item_id=item.id,
        value=Decimal("7000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.commit()
    april = (
        await session.execute(
            select(BalanceSnapshot).where(
                BalanceSnapshot.as_of_date == dt.date(2026, 4, 1)
            )
        )
    ).scalar_one()

    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/networth/manual/snapshot/{april.id}/edit",
            data={"value": "9999.00", "as_of_date": "2026-05-01"},
            follow_redirects=True,
        )

    assert resp.status_code == 200
    assert "An entry already exists for that date" in resp.text


async def test_inactive_item_history_has_no_edit_delete_forms(session):
    item = await manual_items.create_item(
        session,
        name="Sold property",
        kind=ManualKind.asset,
        category=ManualCategory.property,
        value=Decimal("1000000.00"),
        as_of_date=dt.date(2026, 4, 1),
    )
    await manual_items.deactivate(session, item_id=item.id)
    await session.commit()

    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.get("/networth/manual")

    assert resp.status_code == 200
    body = resp.text
    # History date still shown (read-only), but no mutate forms for this item.
    assert "01 Apr 2026" in body
    assert "/snapshot/" not in body
