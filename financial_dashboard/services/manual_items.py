"""Manual asset and liability service."""

import datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.enums import (
    ManualCategory,
    ManualKind,
    SnapshotCategory,
    SnapshotKind,
    SnapshotSource,
)
from financial_dashboard.db.models import BalanceSnapshot, ManualItem
from financial_dashboard.services.snapshots import upsert_manual_snapshot


def _value(value: str | StrEnum) -> str:
    return value.value if isinstance(value, StrEnum) else str(value)


def _snapshot_category(kind: str) -> str:
    return (
        SnapshotCategory.manual_asset.value
        if kind == ManualKind.asset.value
        else SnapshotCategory.manual_liability.value
    )


async def create_item(
    session: AsyncSession,
    *,
    name: str,
    kind: str | ManualKind,
    category: str | ManualCategory,
    value: Decimal,
    as_of_date: datetime.date,
    notes: str | None = None,
) -> ManualItem:
    kind_value = _value(kind)
    item = ManualItem(
        name=name,
        kind=kind_value,
        category=_value(category),
        notes=notes or None,
        active=True,
    )
    session.add(item)
    await session.flush()
    await upsert_manual_snapshot(
        session,
        manual_item_id=item.id,
        kind=SnapshotKind(kind_value),
        category=_snapshot_category(kind_value),
        as_of_date=as_of_date,
        value=value,
        source=SnapshotSource.manual,
    )
    return item


async def update_value(
    session: AsyncSession,
    *,
    item_id: int,
    value: Decimal,
    as_of_date: datetime.date,
) -> ManualItem:
    item = await session.get(ManualItem, item_id)
    if item is None:
        raise ValueError(f"Manual item {item_id} not found")
    await upsert_manual_snapshot(
        session,
        manual_item_id=item.id,
        kind=SnapshotKind(item.kind),
        category=_snapshot_category(item.kind),
        as_of_date=as_of_date,
        value=value,
        source=SnapshotSource.manual,
    )
    return item


async def edit_snapshot(
    session: AsyncSession,
    *,
    snapshot_id: int,
    value: Decimal,
    as_of_date: datetime.date,
) -> BalanceSnapshot:
    """Update a manual snapshot's value and date in place.

    Raises ValueError if the snapshot is missing/not manual, or if moving it to
    a new date would collide with another snapshot of the same item (the date is
    that item's natural key, so the move is rejected rather than overwriting).
    """
    snapshot = await session.get(BalanceSnapshot, snapshot_id)
    if snapshot is None or snapshot.manual_item_id is None:
        raise ValueError(f"Manual snapshot {snapshot_id} not found")

    if as_of_date != snapshot.as_of_date:
        clash = (
            await session.execute(
                select(BalanceSnapshot.id).where(
                    BalanceSnapshot.manual_item_id == snapshot.manual_item_id,
                    BalanceSnapshot.as_of_date == as_of_date,
                    BalanceSnapshot.id != snapshot.id,
                )
            )
        ).first()
        if clash is not None:
            raise ValueError("An entry already exists for that date")

    snapshot.value = value
    snapshot.as_of_date = as_of_date
    await session.flush()
    return snapshot


async def delete_snapshot(session: AsyncSession, *, snapshot_id: int) -> None:
    snapshot = await session.get(BalanceSnapshot, snapshot_id)
    if snapshot is None or snapshot.manual_item_id is None:
        raise ValueError(f"Manual snapshot {snapshot_id} not found")

    await session.delete(snapshot)
    await session.flush()


async def deactivate(session: AsyncSession, *, item_id: int) -> bool:
    item = await session.get(ManualItem, item_id)
    if item is None:
        return False
    item.active = False
    await session.flush()
    return True
