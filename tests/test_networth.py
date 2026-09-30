import datetime as dt
from decimal import Decimal

import pytest

from financial_dashboard.db.enums import (
    SnapshotCategory,
    SnapshotKind,
    SnapshotSource,
)
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    CasUpload,
    ManualItem,
)
from financial_dashboard.services import networth

pytestmark = pytest.mark.anyio


async def _account(session, *, account_type: str = "bank_account", active: bool = True):
    account = Account(
        bank="Example Bank",
        label=f"{account_type}-{active}",
        type=account_type,
        active=active,
    )
    session.add(account)
    await session.flush()
    return account


async def _cas_upload(
    session, *, portfolio_key: str, statement_date: dt.date, portfolio_ok: bool = True
):
    upload = CasUpload(
        portfolio_key=portfolio_key,
        depository_source="cdsl",
        investor_name="Example Investor",
        statement_date=statement_date,
        grand_total=Decimal("0.00"),
        portfolio_ok=portfolio_ok,
        raw_holdings_json="{}",
    )
    session.add(upload)
    await session.flush()
    return upload


async def _bank(session, account_id: int, d: dt.date, v: str):
    session.add(
        BalanceSnapshot(
            account_id=account_id,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=d,
            value=Decimal(v),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await session.flush()


async def _cc(session, account_id: int, d: dt.date, v: str):
    session.add(
        BalanceSnapshot(
            account_id=account_id,
            kind=SnapshotKind.liability.value,
            category=SnapshotCategory.cc_outstanding.value,
            as_of_date=d,
            value=Decimal(v),
            source=SnapshotSource.cc_statement.value,
        )
    )
    await session.flush()


async def _investment(
    session, *, upload: CasUpload, d: dt.date, v: str
) -> BalanceSnapshot:
    snapshot = BalanceSnapshot(
        cas_upload_id=upload.id,
        portfolio_key=upload.portfolio_key,
        kind=SnapshotKind.asset.value,
        category=SnapshotCategory.investment.value,
        as_of_date=d,
        value=Decimal(v),
        source=SnapshotSource.cas.value,
    )
    session.add(snapshot)
    await session.flush()
    return snapshot


async def _manual_item(session, *, kind: str, name: str = "Item") -> ManualItem:
    item = ManualItem(name=name, kind=kind, category="other", active=True)
    session.add(item)
    await session.flush()
    return item


async def _manual(
    session, *, item: ManualItem, kind: str, category: str, d: dt.date, v: str
) -> BalanceSnapshot:
    snapshot = BalanceSnapshot(
        manual_item_id=item.id,
        kind=kind,
        category=category,
        as_of_date=d,
        value=Decimal(v),
        source=SnapshotSource.manual.value,
    )
    session.add(snapshot)
    await session.flush()
    return snapshot


async def test_current_networth_counts_the_latest_inr_value_of_each_active_source(
    session,
):
    """Each decoy moves the total if its guard breaks: an older period summed
    with the latest, an inactive source, a non-INR snapshot, or a second PAN on
    the same date dropped as a duplicate."""
    bank = await _account(session, account_type="bank_account")
    card = await _account(session, account_type="credit_card")
    inactive = await _account(session, active=False)
    await _bank(session, bank.id, dt.date(2026, 5, 20), "100000.00")
    await _cc(session, card.id, dt.date(2026, 5, 21), "250000.00")
    await _bank(session, inactive.id, dt.date(2026, 5, 20), "999999.00")
    inactive_item = ManualItem(
        name="Inactive asset", kind="asset", category="cash", active=False
    )
    session.add(inactive_item)
    await session.flush()
    await _manual(
        session,
        item=inactive_item,
        kind=SnapshotKind.asset.value,
        category=SnapshotCategory.manual_asset.value,
        d=dt.date(2026, 5, 20),
        v="999999.00",
    )
    usd_bank = await _account(session)
    session.add(
        BalanceSnapshot(
            account_id=usd_bank.id,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=dt.date(2026, 5, 22),
            value=Decimal("777.00"),
            source=SnapshotSource.bank_statement.value,
            currency="USD",
        )
    )
    # PAN A: two periods, only the latest counts. PAN B: same date, a separate
    # source, from a CAS that failed reconciliation.
    for d, v in [
        (dt.date(2026, 3, 31), "1000000.00"),
        (dt.date(2026, 4, 30), "1100000.00"),
    ]:
        upload = await _cas_upload(
            session, portfolio_key="PANAAA1111A", statement_date=d
        )
        await _investment(session, upload=upload, d=d, v=v)
    upload_b = await _cas_upload(
        session,
        portfolio_key="PANBBB2222B",
        statement_date=dt.date(2026, 4, 30),
        portfolio_ok=False,
    )
    await _investment(session, upload=upload_b, d=dt.date(2026, 4, 30), v="50000.00")

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.total_assets == Decimal("1250000.00")
    assert summary.total_liabilities == Decimal("250000.00")
    assert summary.net_worth == Decimal("1000000.00")
    investments = {
        r.value: r for g in summary.groups for r in g.rows if g.category == "investment"
    }
    assert investments[Decimal("1100000.00")].unreconciled is False
    assert investments[Decimal("50000.00")].unreconciled is True
    assert summary.has_stale is False


async def test_net_worth_can_go_negative(session):
    bank = await _account(session, account_type="bank_account")
    card = await _account(session, account_type="credit_card")
    await _bank(session, bank.id, dt.date(2026, 5, 20), "100000.00")
    await _cc(session, card.id, dt.date(2026, 5, 21), "125000.00")

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.net_worth == Decimal("-25000.00")


async def test_empty_or_future_only_history_is_zero_with_no_trend(session):
    today = dt.date(2026, 5, 24)
    summary = await networth.current_networth(session, today=today)
    assert (summary.total_assets, summary.total_liabilities, summary.net_worth) == (
        Decimal("0.00"),
        Decimal("0.00"),
        Decimal("0.00"),
    )
    assert summary.groups == []
    assert await networth.monthly_trend(session, today=today) == []

    account = await _account(session)
    await _bank(session, account.id, dt.date(2027, 1, 15), "100000.00")
    assert await networth.monthly_trend(session, today=today) == []


async def test_monthly_trend_forward_fills_uncapped_and_ends_on_the_headline(session):
    """The trend spans every month from the earliest snapshot, with no 12-month
    cap. Each month end carries the latest INR snapshot on or before it."""
    account = await _account(session)
    usd_account = await _account(session)
    await _bank(session, account.id, dt.date(2025, 1, 15), "100000.00")
    await _bank(session, account.id, dt.date(2026, 4, 10), "120000.00")
    session.add(
        BalanceSnapshot(
            account_id=usd_account.id,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=dt.date(2025, 6, 1),
            value=Decimal("5.00"),
            source=SnapshotSource.bank_statement.value,
            currency="USD",
        )
    )
    await session.flush()

    today = dt.date(2026, 5, 12)
    points = await networth.monthly_trend(session, today=today)

    assert len(points) == 17
    assert (points[0].month, points[-1].month) == ("2025-01", "2026-05")
    by_month = {p.month: p.value for p in points}
    assert by_month["2025-06"] == Decimal("100000.00")
    assert by_month["2026-03"] == Decimal("100000.00")
    assert by_month["2026-04"] == Decimal("120000.00")
    summary = await networth.current_networth(session, today=today)
    assert points[-1].value == summary.net_worth == Decimal("120000.00")


async def test_staleness_boundary_exact_threshold_fresh_next_day_stale(session):
    """At exactly the threshold age a snapshot is fresh; one day older is stale."""
    as_of = dt.date(2026, 5, 31)
    fresh_date = as_of - dt.timedelta(days=45)
    fresh = await _account(session)
    stale = await _account(session)
    await _bank(session, fresh.id, fresh_date, "1000.00")
    await _bank(session, stale.id, fresh_date - dt.timedelta(days=1), "2000.00")

    summary = await networth.current_networth(session, today=as_of)

    rows = {r.value: r for g in summary.groups for r in g.rows}
    assert rows[Decimal("1000.00")].stale is False
    assert rows[Decimal("2000.00")].stale is True
    assert summary.has_stale is True
