import datetime as dt
from decimal import Decimal

from sqlalchemy import select
import pytest

from financial_dashboard.db.enums import ManualCategory, ManualKind
from financial_dashboard.db.models import BalanceSnapshot
from financial_dashboard.services import manual_items, networth

pytestmark = pytest.mark.anyio


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


async def test_edit_snapshot_colliding_date_raises_and_writes_nothing(session):
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
    await session.flush()
    april = await session.get(BalanceSnapshot, 1)

    with pytest.raises(ValueError):
        await manual_items.edit_snapshot(
            session,
            snapshot_id=april.id,
            value=Decimal("9999.00"),
            as_of_date=dt.date(2026, 5, 1),
        )

    unchanged = await session.get(BalanceSnapshot, april.id)
    assert unchanged.value == Decimal("5000.00")
    assert unchanged.as_of_date == dt.date(2026, 4, 1)


async def test_delete_snapshot_falls_back_then_drops_item(session):
    item = await manual_items.create_item(
        session,
        name="Property",
        kind=ManualKind.asset,
        category=ManualCategory.property,
        value=Decimal("1000000.00"),
        as_of_date=dt.date(2026, 4, 1),
    )
    await manual_items.update_value(
        session,
        item_id=item.id,
        value=Decimal("1100000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.flush()
    april, may = (
        (await session.execute(select(BalanceSnapshot).order_by("as_of_date")))
        .scalars()
        .all()
    )

    with pytest.raises(ValueError):
        await manual_items.delete_snapshot(session, snapshot_id=999)

    await manual_items.delete_snapshot(session, snapshot_id=may.id)
    await session.flush()
    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("1000000.00")

    await manual_items.delete_snapshot(session, snapshot_id=april.id)
    await session.flush()
    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("0.00")


async def test_edit_snapshot_in_place_and_by_date_changes_latest(session):
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
        value=Decimal("80000.00"),
        as_of_date=dt.date(2026, 4, 10),
    )
    await session.flush()
    april1 = (
        await session.execute(
            select(BalanceSnapshot).where(
                BalanceSnapshot.as_of_date == dt.date(2026, 4, 1)
            )
        )
    ).scalar_one()

    # Editing a snapshot on its own date is not a collision.
    await manual_items.edit_snapshot(
        session,
        snapshot_id=april1.id,
        value=Decimal("90000.00"),
        as_of_date=dt.date(2026, 4, 1),
    )
    await session.flush()
    assert (await session.get(BalanceSnapshot, april1.id)).value == Decimal("90000.00")

    await manual_items.edit_snapshot(
        session,
        snapshot_id=april1.id,
        value=Decimal("100000.00"),
        as_of_date=dt.date(2026, 4, 20),
    )
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("100000.00")
