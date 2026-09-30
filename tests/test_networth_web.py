import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db.enums import (
    ManualCategory,
    ManualKind,
    SnapshotCategory,
    SnapshotKind,
    SnapshotSource,
)
from financial_dashboard.db.models import Account, BalanceSnapshot, CasUpload
from financial_dashboard.services import manual_items

pytestmark = pytest.mark.anyio

OVERSIZE_PDF = b"%PDF-" + b"x" * (10 * 1024 * 1024 + 1)


async def test_networth_page_renders_figure_and_flags_stale_and_unreconciled(
    client, session
):
    """A bank snapshot past the 45-day threshold shows as stale. An investment
    snapshot from a CAS that failed reconciliation shows as unreconciled."""
    account = Account(
        bank="Example", label="Old Bank", type="bank_account", active=True
    )
    upload = CasUpload(
        portfolio_key="PANBAD1",
        depository_source="cdsl",
        investor_name="Example Investor",
        statement_date=dt.date.today() - dt.timedelta(days=5),
        grand_total=Decimal("100000.00"),
        portfolio_ok=False,
        raw_holdings_json="{}",
    )
    session.add_all([account, upload])
    await session.flush()
    session.add_all(
        [
            BalanceSnapshot(
                account_id=account.id,
                kind=SnapshotKind.asset.value,
                category=SnapshotCategory.bank_balance.value,
                as_of_date=dt.date.today() - dt.timedelta(days=60),
                value=Decimal("100000.00"),
                source=SnapshotSource.bank_statement.value,
            ),
            BalanceSnapshot(
                cas_upload_id=upload.id,
                portfolio_key=upload.portfolio_key,
                kind=SnapshotKind.asset.value,
                category=SnapshotCategory.investment.value,
                as_of_date=upload.statement_date,
                value=Decimal("100000.00"),
                source=SnapshotSource.cas.value,
            ),
        ]
    )
    await session.commit()

    resp = await client.get("/networth")

    assert resp.status_code == 200
    assert "2L" in resp.text
    assert "stale" in resp.text
    assert "unreconciled" in resp.text


async def test_cas_upload_rejects_oversize_file(client):
    upload = {
        "data": {"password": "", "force_replace": "false"},
        "files": {"file": ("big.pdf", OVERSIZE_PDF, "application/pdf")},
    }
    html = await client.post("/cas/upload", follow_redirects=False, **upload)
    assert html.status_code == 303
    assert "error=" in html.headers["location"]

    api = await client.post("/api/cas/upload", **upload)
    assert api.status_code == 413


async def test_manual_edit_collision_keeps_both_entries(client, session):
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

    resp = await client.post(
        f"/networth/manual/snapshot/{april.id}/edit",
        data={"value": "9999.00", "as_of_date": "2026-05-01"},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]
    session.expire_all()
    values = (
        await session.execute(
            select(BalanceSnapshot.as_of_date, BalanceSnapshot.value).order_by(
                BalanceSnapshot.as_of_date
            )
        )
    ).all()
    assert values == [
        (dt.date(2026, 4, 1), Decimal("5000.00")),
        (dt.date(2026, 5, 1), Decimal("7000.00")),
    ]


async def test_manual_snapshot_delete_removes_it_and_rejects_unknown_id(
    client, session
):
    await manual_items.create_item(
        session,
        name="Cash",
        kind=ManualKind.asset,
        category=ManualCategory.cash,
        value=Decimal("5000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.commit()
    snap = (await session.execute(select(BalanceSnapshot))).scalar_one()

    resp = await client.post(
        f"/networth/manual/snapshot/{snap.id}/delete", follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/networth/manual"
    session.expire_all()
    assert (await session.execute(select(BalanceSnapshot))).first() is None

    again = await client.post(
        f"/networth/manual/snapshot/{snap.id}/delete", follow_redirects=False
    )
    assert again.status_code == 303
    assert "error=" in again.headers["location"]


async def test_manual_page_lists_history_newest_first_read_only_when_inactive(
    client, session
):
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

    body = (await client.get("/networth/manual")).text
    assert body.index("01 May 2026") < body.index("01 Apr 2026")
    assert "/snapshot/" in body

    await manual_items.deactivate(session, item_id=item.id)
    await session.commit()

    body = (await client.get("/networth/manual")).text
    # The history stays visible, with no forms that change it.
    assert "01 Apr 2026" in body
    assert "/snapshot/" not in body
