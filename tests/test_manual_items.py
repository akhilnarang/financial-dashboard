import datetime as dt
from decimal import Decimal

from sqlalchemy import select
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


async def test_edit_snapshot_updates_value_and_date_in_place(session):
    item = await manual_items.create_item(
        session,
        name="Gold",
        kind=ManualKind.asset,
        category=ManualCategory.gold,
        value=Decimal("100000.00"),
        as_of_date=dt.date(2026, 4, 1),
    )
    await session.flush()
    snap = await session.get(BalanceSnapshot, 1)

    await manual_items.edit_snapshot(
        session,
        snapshot_id=snap.id,
        value=Decimal("120000.00"),
        as_of_date=dt.date(2026, 4, 15),
    )
    await session.flush()

    refreshed = await session.get(BalanceSnapshot, snap.id)
    assert refreshed.value == Decimal("120000.00")
    assert refreshed.as_of_date == dt.date(2026, 4, 15)
    assert refreshed.manual_item_id == item.id


async def test_edit_snapshot_same_date_is_not_a_collision(session):
    await manual_items.create_item(
        session,
        name="Cash",
        kind=ManualKind.asset,
        category=ManualCategory.cash,
        value=Decimal("5000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.flush()
    snap = await session.get(BalanceSnapshot, 1)

    await manual_items.edit_snapshot(
        session,
        snapshot_id=snap.id,
        value=Decimal("6000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.flush()
    assert (await session.get(BalanceSnapshot, snap.id)).value == Decimal("6000.00")


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


async def test_edit_snapshot_missing_id_raises(session):
    with pytest.raises(ValueError):
        await manual_items.edit_snapshot(
            session,
            snapshot_id=999,
            value=Decimal("1.00"),
            as_of_date=dt.date(2026, 5, 1),
        )


async def test_delete_snapshot_removes_target_and_keeps_others(session):
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
    april = await session.get(BalanceSnapshot, 1)

    await manual_items.delete_snapshot(session, snapshot_id=april.id)
    await session.flush()

    assert await session.get(BalanceSnapshot, april.id) is None
    remaining = (
        (
            await session.execute(
                select(BalanceSnapshot).where(BalanceSnapshot.manual_item_id == item.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(remaining) == 1
    assert remaining[0].as_of_date == dt.date(2026, 5, 1)


async def test_delete_snapshot_missing_id_raises(session):
    with pytest.raises(ValueError):
        await manual_items.delete_snapshot(session, snapshot_id=999)


async def test_delete_latest_falls_back_to_older_snapshot(session):
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
    may = (
        await session.execute(
            select(BalanceSnapshot).where(
                BalanceSnapshot.as_of_date == dt.date(2026, 5, 1)
            )
        )
    ).scalar_one()

    await manual_items.delete_snapshot(session, snapshot_id=may.id)
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("1000000.00")


async def test_delete_only_snapshot_drops_item_from_total(session):
    await manual_items.create_item(
        session,
        name="Cash",
        kind=ManualKind.asset,
        category=ManualCategory.cash,
        value=Decimal("5000.00"),
        as_of_date=dt.date(2026, 5, 1),
    )
    await session.flush()
    only = await session.get(BalanceSnapshot, 1)

    await manual_items.delete_snapshot(session, snapshot_id=only.id)
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("0.00")


async def test_edit_date_changes_which_snapshot_is_latest(session):
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

    await manual_items.edit_snapshot(
        session,
        snapshot_id=april1.id,
        value=Decimal("100000.00"),
        as_of_date=dt.date(2026, 4, 20),
    )
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("100000.00")
