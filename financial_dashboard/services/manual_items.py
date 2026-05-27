"""Manual asset and liability service."""

from __future__ import annotations

import datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.enums import (
    ManualCategory,
    ManualKind,
    SnapshotCategory,
    SnapshotKind,
    SnapshotSource,
)
from financial_dashboard.db.models import ManualItem
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


async def deactivate(session: AsyncSession, *, item_id: int) -> bool:
    item = await session.get(ManualItem, item_id)
    if item is None:
        return False
    item.active = False
    await session.flush()
    return True
