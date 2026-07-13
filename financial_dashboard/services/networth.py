"""Net-worth aggregation over balance snapshots."""

import datetime
from collections import defaultdict
from decimal import Decimal

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import month_end
from financial_dashboard.db.enums import SnapshotCategory, SnapshotKind
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    CasUpload,
    ManualItem,
    SnapshotHolding,
)
from financial_dashboard.schemas.networth import (
    CategoryGroup,
    HoldingBreakdown,
    NetWorthSummary,
    SourceBalance,
    TrendPoint,
)

STALE_DAYS = {
    SnapshotCategory.bank_balance.value: 45,
    SnapshotCategory.cc_outstanding.value: 35,
    SnapshotCategory.investment.value: 90,
    SnapshotCategory.manual_asset.value: 180,
    SnapshotCategory.manual_liability.value: 180,
}


def _source_key(snapshot: BalanceSnapshot) -> tuple:
    if snapshot.account_id is not None:
        return ("acct", snapshot.account_id, snapshot.category)
    if snapshot.portfolio_key is not None:
        return ("inv", snapshot.portfolio_key, snapshot.category)
    return ("man", snapshot.manual_item_id)


def _latest_per_source(
    snapshots: list[BalanceSnapshot], as_of: datetime.date
) -> list[BalanceSnapshot]:
    best: dict[tuple, BalanceSnapshot] = {}
    for snapshot in snapshots:
        if snapshot.as_of_date > as_of:
            continue
        key = _source_key(snapshot)
        current = best.get(key)
        if current is None or (snapshot.as_of_date, snapshot.id) > (
            current.as_of_date,
            current.id,
        ):
            best[key] = snapshot
    return list(best.values())


def _label(snapshot: BalanceSnapshot) -> str:
    if snapshot.account is not None:
        return snapshot.account.label
    if snapshot.manual_item is not None:
        return snapshot.manual_item.name
    if snapshot.cas_upload is not None:
        return snapshot.cas_upload.investor_name or snapshot.portfolio_key or "CAS"
    return snapshot.portfolio_key or "Snapshot"


async def current_networth(
    session: AsyncSession, *, today: datetime.date | None = None
) -> NetWorthSummary:
    as_of = today or datetime.date.today()
    snapshots = (
        (
            await session.execute(
                select(BalanceSnapshot)
                .outerjoin(Account, BalanceSnapshot.account_id == Account.id)
                .outerjoin(CasUpload, BalanceSnapshot.cas_upload_id == CasUpload.id)
                .outerjoin(ManualItem, BalanceSnapshot.manual_item_id == ManualItem.id)
                .where(
                    or_(
                        BalanceSnapshot.account_id.is_(None),
                        Account.active.is_not(False),
                    ),
                    or_(
                        BalanceSnapshot.manual_item_id.is_(None),
                        ManualItem.active.is_not(False),
                    ),
                )
                .order_by(BalanceSnapshot.as_of_date, BalanceSnapshot.id)
            )
        )
        .unique()
        .scalars()
        .all()
    )

    current = _latest_per_source(list(snapshots), as_of)
    total_assets = Decimal("0.00")
    total_liabilities = Decimal("0.00")
    grouped: dict[tuple[str, str], list[SourceBalance]] = defaultdict(list)
    investment_snapshot_ids: list[int] = []

    for snapshot in current:
        if snapshot.currency != "INR":
            continue
        value = Decimal(snapshot.value)
        if snapshot.kind == SnapshotKind.asset.value:
            total_assets += value
        elif snapshot.kind == SnapshotKind.liability.value:
            total_liabilities += value

        stale = (as_of - snapshot.as_of_date).days > STALE_DAYS.get(
            snapshot.category, 9999
        )
        row = SourceBalance(
            label=_label(snapshot),
            category=snapshot.category,
            kind=snapshot.kind,
            value=value,
            as_of_date=snapshot.as_of_date,
            stale=stale,
            unreconciled=(
                snapshot.cas_upload is not None and not snapshot.cas_upload.portfolio_ok
            ),
        )
        grouped[(snapshot.category, snapshot.kind)].append(row)
        if snapshot.category == SnapshotCategory.investment.value:
            investment_snapshot_ids.append(snapshot.id)

    groups = [
        CategoryGroup(
            category=category,
            kind=kind,
            total=sum((row.value for row in rows), Decimal("0.00")),
            rows=sorted(rows, key=lambda row: row.label.lower()),
        )
        for (category, kind), rows in sorted(grouped.items())
    ]

    investment_breakdown: list[HoldingBreakdown] = []
    if investment_snapshot_ids:
        holdings = (
            (
                await session.execute(
                    select(SnapshotHolding).where(
                        SnapshotHolding.snapshot_id.in_(investment_snapshot_ids)
                    )
                )
            )
            .scalars()
            .all()
        )
        breakdown_totals: dict[tuple[str, str], Decimal] = defaultdict(
            lambda: Decimal("0.00")
        )
        for holding in holdings:
            breakdown_totals[(holding.asset_class, holding.label)] += Decimal(
                holding.value
            )
        investment_breakdown = [
            HoldingBreakdown(asset_class=asset_class, label=label, value=value)
            for (asset_class, label), value in sorted(breakdown_totals.items())
        ]

    return NetWorthSummary(
        net_worth=total_assets - total_liabilities,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        groups=groups,
        investment_breakdown=investment_breakdown,
        has_stale=any(row.stale for group in groups for row in group.rows),
        as_of=as_of,
    )


async def monthly_trend(
    session: AsyncSession, *, today: datetime.date | None = None
) -> list[TrendPoint]:
    as_of_today = today or datetime.date.today()

    snapshots = list(
        (
            (
                await session.execute(
                    select(BalanceSnapshot)
                    .outerjoin(Account, BalanceSnapshot.account_id == Account.id)
                    .outerjoin(
                        ManualItem, BalanceSnapshot.manual_item_id == ManualItem.id
                    )
                    .where(
                        or_(
                            BalanceSnapshot.account_id.is_(None),
                            Account.active.is_not(False),
                        ),
                        or_(
                            BalanceSnapshot.manual_item_id.is_(None),
                            ManualItem.active.is_not(False),
                        ),
                    )
                    .order_by(BalanceSnapshot.as_of_date, BalanceSnapshot.id)
                )
            )
            .scalars()
            .all()
        )
    )

    if not snapshots:
        return []
    # Earliest from the filtered set, not the global table — otherwise
    # snapshots that belong only to deactivated accounts produce phantom
    # leading zero months.
    earliest = snapshots[0].as_of_date
    # Future-only snapshots (e.g. a manual item dated next year) would
    # otherwise produce a single misleading zero-valued point at the
    # current month. Treat that as "no history yet".
    if earliest > as_of_today:
        return []

    # Single-pass bucketing: snapshots are sorted by (as_of_date, id) ASC, so
    # we walk a pointer forward through the list and maintain `best` per source
    # across month boundaries — last write wins per key, matching the
    # `_latest_per_source` semantics but O(snapshots + months) instead of
    # O(snapshots * months).
    best: dict[tuple, BalanceSnapshot] = {}
    snap_idx = 0
    points: list[TrendPoint] = []
    year, month = earliest.year, earliest.month
    while True:
        last_day = month_end(datetime.date(year, month, 1))
        cutoff = min(last_day, as_of_today)
        while snap_idx < len(snapshots) and snapshots[snap_idx].as_of_date <= cutoff:
            snapshot = snapshots[snap_idx]
            best[_source_key(snapshot)] = snapshot
            snap_idx += 1
        assets = Decimal("0.00")
        liabilities = Decimal("0.00")
        for snapshot in best.values():
            if snapshot.currency != "INR":
                continue
            value = Decimal(snapshot.value)
            if snapshot.kind == SnapshotKind.asset.value:
                assets += value
            elif snapshot.kind == SnapshotKind.liability.value:
                liabilities += value
        points.append(
            TrendPoint(month=cutoff.strftime("%Y-%m"), value=assets - liabilities)
        )
        if last_day >= as_of_today:
            break
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return points
