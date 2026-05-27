import datetime as dt
from decimal import Decimal

import pytest

from financial_dashboard.db.enums import ManualCategory, ManualKind, SnapshotCategory
from financial_dashboard.db.models import BalanceSnapshot
from financial_dashboard.services import manual_items, networth

pytestmark = pytest.mark.anyio


async def test_create_item_writes_first_value_snapshot(session):
    item = await manual_items.create_item(
        session,
        name="Emergency cash",
        kind=ManualKind.asset,
        category=ManualCategory.cash,
        value=Decimal("50000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.flush()

    snapshot = await session.get(BalanceSnapshot, 1)
    assert item.id is not None
    assert snapshot.manual_item_id == item.id
    assert snapshot.category == SnapshotCategory.manual_asset.value
    assert snapshot.value == Decimal("50000.00")


async def test_update_value_preserves_history_and_latest_wins(session):
    item = await manual_items.create_item(
        session,
        name="Loan",
        kind=ManualKind.liability,
        category=ManualCategory.loan,
        value=Decimal("100000.00"),
        as_of_date=dt.date(2026, 4, 1),
    )
    await manual_items.update_value(
        session,
        item_id=item.id,
        value=Decimal("90000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_liabilities == Decimal("90000.00")
    assert summary.net_worth == Decimal("-90000.00")


async def test_deactivate_excludes_item_from_networth(session):
    item = await manual_items.create_item(
        session,
        name="Property",
        kind=ManualKind.asset,
        category=ManualCategory.property,
        value=Decimal("1000000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await manual_items.deactivate(session, item_id=item.id)
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("0.00")
