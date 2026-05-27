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


async def _cas_upload(session, *, portfolio_key: str, statement_date: dt.date):
    upload = CasUpload(
        portfolio_key=portfolio_key,
        depository_source="cdsl",
        investor_name="Example Investor",
        statement_date=statement_date,
        grand_total=Decimal("0.00"),
        portfolio_ok=True,
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


async def test_current_networth_assets_minus_liabilities(session):
    bank = await _account(session, account_type="bank_account")
    card = await _account(session, account_type="credit_card")
    await _bank(session, bank.id, dt.date(2026, 5, 20), "100000.00")
    await _cc(session, card.id, dt.date(2026, 5, 21), "25000.00")

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.total_assets == Decimal("100000.00")
    assert summary.total_liabilities == Decimal("25000.00")
    assert summary.net_worth == Decimal("75000.00")


async def test_forward_fill_latest_per_source_and_stale_flag(session):
    account = await _account(session)
    await _bank(session, account.id, dt.date(2026, 3, 31), "90000.00")
    await _bank(session, account.id, dt.date(2026, 5, 1), "110000.00")

    summary = await networth.current_networth(session, today=dt.date(2026, 7, 1))

    assert summary.total_assets == Decimal("110000.00")
    assert summary.has_stale is True
    assert summary.groups[0].rows[0].stale is True


async def test_deactivated_sources_are_excluded(session):
    active = await _account(session, active=True)
    inactive = await _account(session, active=False)
    item = ManualItem(
        name="Inactive asset",
        kind="asset",
        category="cash",
        active=False,
    )
    session.add(item)
    await session.flush()

    await _bank(session, active.id, dt.date(2026, 5, 20), "1000.00")
    await _bank(session, inactive.id, dt.date(2026, 5, 20), "999999.00")
    session.add(
        BalanceSnapshot(
            manual_item_id=item.id,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.manual_asset.value,
            as_of_date=dt.date(2026, 5, 20),
            value=Decimal("999999.00"),
            source=SnapshotSource.manual.value,
        )
    )
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.total_assets == Decimal("1000.00")


async def test_bank_and_cc_on_distinct_accounts_are_not_double_counted(session):
    bank = await _account(session, account_type="bank_account")
    card = await _account(session, account_type="credit_card")
    await _bank(session, bank.id, dt.date(2026, 5, 20), "50000.00")
    await _cc(session, card.id, dt.date(2026, 5, 20), "10000.00")

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.net_worth == Decimal("40000.00")


async def test_empty_db_returns_zero_summary(session):
    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.total_assets == Decimal("0.00")
    assert summary.total_liabilities == Decimal("0.00")
    assert summary.net_worth == Decimal("0.00")
    assert summary.groups == []


async def test_monthly_trend_empty_db(session):
    points = await networth.monthly_trend(session, today=dt.date(2026, 5, 24))
    assert points == []


async def test_monthly_trend_future_only_snapshots_returns_empty(session):
    account = await _account(session)
    await _bank(session, account.id, dt.date(2027, 1, 15), "100000.00")

    points = await networth.monthly_trend(session, today=dt.date(2026, 5, 24))
    assert points == []


async def test_monthly_trend_forward_fills_and_last_point_matches_headline(session):
    account = await _account(session)
    # Irregular-date snapshots across three months.
    await _bank(session, account.id, dt.date(2026, 2, 15), "100000.00")
    await _bank(session, account.id, dt.date(2026, 4, 10), "120000.00")

    today = dt.date(2026, 5, 12)
    points = await networth.monthly_trend(session, today=today)

    # Spine: 2026-02 through 2026-05 (earliest month → current month).
    assert [p.month for p in points] == ["2026-02", "2026-03", "2026-04", "2026-05"]
    assert points[0].value == Decimal(
        "100000.00"
    )  # 2026-02-28 forward-fills the Feb 15 snapshot
    assert points[1].value == Decimal(
        "100000.00"
    )  # 2026-03-31 still on the Feb snapshot
    assert points[2].value == Decimal(
        "120000.00"
    )  # 2026-04-30 picks up the Apr 10 snapshot
    assert points[3].value == Decimal(
        "120000.00"
    )  # current month capped at today, still latest

    summary = await networth.current_networth(session, today=today)
    assert points[-1].value == summary.net_worth


async def test_investment_latest_period_not_summed(session):
    portfolio_key = "ABCDE1234F"
    for d, v in [
        (dt.date(2026, 3, 31), "1000000.00"),
        (dt.date(2026, 4, 30), "1100000.00"),
    ]:
        upload = await _cas_upload(
            session, portfolio_key=portfolio_key, statement_date=d
        )
        session.add(
            BalanceSnapshot(
                cas_upload_id=upload.id,
                portfolio_key=portfolio_key,
                kind=SnapshotKind.asset.value,
                category=SnapshotCategory.investment.value,
                as_of_date=d,
                value=Decimal(v),
                source=SnapshotSource.cas.value,
            )
        )
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.total_assets == Decimal("1100000.00")
