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


async def test_networth_page_carries_its_chart_and_the_app_serves_it(client):
    """The page's own tests assert the figure and the heading, and every one of them
    would still pass with the trend chart gone: the module tag deleted, networth.js
    renamed under the mount, the SVG's data-trend-url dropped. This is the assertion
    that fails when that happens — the tag is on the page, its src resolves through
    the app, and the hooks the module reads are where it looks for them."""
    page = await client.get("/networth")
    assert page.status_code == 200
    assert '<script type="module" src="/static/js/networth.js"></script>' in page.text
    assert 'id="nw-trend"' in page.text
    assert 'data-trend-url="/api/networth/trend"' in page.text

    module = await client.get("/static/js/networth.js")
    assert module.status_code == 200, "the page loads networth.js, the app 404s it"
    assert "charts.js" in module.text, "networth.js does not import the chart module"

    # The URL the SVG names is one the app answers, so the chart has data to draw.
    trend = await client.get("/api/networth/trend")
    assert trend.status_code == 200


async def test_networth_page_renders_stale_banner_and_row_badge(session):
    """A bank snapshot older than the 45-day bank staleness threshold surfaces the
    page-level 'stale' banner and the per-row stale badge."""
    from financial_dashboard.db.models import Account

    account = Account(
        bank="Example", label="Old Bank", type="bank_account", active=True
    )
    session.add(account)
    await session.flush()
    old = dt.date.today() - dt.timedelta(days=60)  # 60 > 45-day bank threshold
    session.add(
        BalanceSnapshot(
            account_id=account.id,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=old,
            value=Decimal("100000.00"),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await session.commit()

    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.get("/networth")

    assert resp.status_code == 200
    body = resp.text
    # Page-level banner (summary.has_stale) and per-row badge both present.
    assert "stale" in body
    assert "badge-pending" in body


async def test_networth_page_renders_unreconciled_badge_for_failed_cas(session):
    """An investment snapshot whose CAS upload failed reconciliation surfaces the
    per-row 'unreconciled' (danger) badge on the net worth page."""
    recent = dt.date.today() - dt.timedelta(days=5)  # recent -> not stale
    upload = CasUpload(
        portfolio_key="PANBAD1",
        depository_source="cdsl",
        investor_name="Example Investor",
        statement_date=recent,
        grand_total=Decimal("50000.00"),
        portfolio_ok=False,
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
            as_of_date=recent,
            value=Decimal("50000.00"),
            source=SnapshotSource.cas.value,
        )
    )
    await session.commit()

    async def override():
        yield session

    async with AsyncClient(
        transport=ASGITransport(app=_build_app(override)), base_url="http://test"
    ) as client:
        resp = await client.get("/networth")

    assert resp.status_code == 200
    body = resp.text
    assert "unreconciled" in body
    assert "badge danger" in body
