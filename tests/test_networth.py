import datetime as dt
from decimal import Decimal

import pytest
from dateutil.relativedelta import relativedelta

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


# ---------------------------------------------------------------------------
# Two PANs / latest-per-source / currency / staleness / unreconciled
# ---------------------------------------------------------------------------


async def test_two_pans_same_date_both_counted(session):
    """Two distinct portfolio keys (PANs) on the same date are separate sources
    (``("inv", portfolio_key, category)``) so both contribute to net worth."""
    upload_a = await _cas_upload(
        session, portfolio_key="PANAAA1111A", statement_date=dt.date(2026, 4, 30)
    )
    upload_b = await _cas_upload(
        session, portfolio_key="PANBBB2222B", statement_date=dt.date(2026, 4, 30)
    )
    await _investment(session, upload=upload_a, d=dt.date(2026, 4, 30), v="100000.00")
    await _investment(session, upload=upload_b, d=dt.date(2026, 4, 30), v="200000.00")

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.total_assets == Decimal("300000.00")
    inv_group = next(g for g in summary.groups if g.category == "investment")
    assert len(inv_group.rows) == 2


async def test_latest_per_source_wins_tie_broken_by_id(session):
    """For one source, the latest as_of wins; an exact-date tie is broken by id
    so the most-recently inserted snapshot wins. Other sources are independent."""
    acct_a = await _account(session)
    acct_b = await _account(session)
    await _bank(session, acct_a.id, dt.date(2026, 4, 10), "1000.00")
    await _bank(session, acct_a.id, dt.date(2026, 4, 20), "2000.00")  # latest for A
    await _bank(session, acct_b.id, dt.date(2026, 4, 20), "5000.00")  # only B

    summary = await networth.current_networth(session, today=dt.date(2026, 4, 24))

    assert summary.total_assets == Decimal("7000.00")


async def test_asset_and_liability_signs_keep_net_worth_signed(session):
    """Assets add, liabilities subtract; net worth can go negative."""
    bank = await _account(session, account_type="bank_account")
    card = await _account(session, account_type="credit_card")
    await _bank(session, bank.id, dt.date(2026, 5, 20), "30000.00")
    await _cc(session, card.id, dt.date(2026, 5, 20), "90000.00")

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))

    assert summary.total_assets == Decimal("30000.00")
    assert summary.total_liabilities == Decimal("90000.00")
    assert summary.net_worth == Decimal("-60000.00")


async def test_inr_default_currency_is_counted(session):
    """The balance_snapshots.currency column is NOT NULL with default 'INR', so a
    snapshot created without an explicit currency resolves to INR and is counted.
    This is the practical 'NULL' path: NULL cannot occur, the default fills it."""
    account = await _account(session)
    snapshot = BalanceSnapshot(
        account_id=account.id,
        kind=SnapshotKind.asset.value,
        category=SnapshotCategory.bank_balance.value,
        as_of_date=dt.date(2026, 5, 20),
        value=Decimal("1000.00"),
        source=SnapshotSource.bank_statement.value,
    )
    session.add(snapshot)
    await session.flush()

    assert snapshot.currency == "INR"  # column default applied
    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("1000.00")


async def test_non_inr_currency_excluded_from_totals_and_groups(session):
    """A non-INR snapshot is skipped entirely: not in totals, not in any group."""
    account = await _account(session)
    session.add(
        BalanceSnapshot(
            account_id=account.id,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=dt.date(2026, 5, 20),
            value=Decimal("1000.00"),
            source=SnapshotSource.bank_statement.value,
            currency="USD",
        )
    )
    await session.flush()

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    assert summary.total_assets == Decimal("0.00")
    assert summary.groups == []


async def test_non_inr_excluded_in_monthly_trend_too(session):
    """The trend applies the same INR filter, so a non-INR source contributes 0."""
    account = await _account(session)
    session.add(
        BalanceSnapshot(
            account_id=account.id,
            kind=SnapshotKind.asset.value,
            category=SnapshotCategory.bank_balance.value,
            as_of_date=dt.date(2026, 4, 20),
            value=Decimal("1000.00"),
            source=SnapshotSource.bank_statement.value,
            currency="USD",
        )
    )
    await session.flush()

    points = await networth.monthly_trend(session, today=dt.date(2026, 5, 12))
    assert [p.value for p in points] == [Decimal("0.00"), Decimal("0.00")]


@pytest.mark.parametrize(
    ("category", "kind", "threshold"),
    [
        (SnapshotCategory.bank_balance, SnapshotKind.asset, 45),
        (SnapshotCategory.cc_outstanding, SnapshotKind.liability, 35),
        (SnapshotCategory.investment, SnapshotKind.asset, 90),
        (SnapshotCategory.manual_asset, SnapshotKind.asset, 180),
        (SnapshotCategory.manual_liability, SnapshotKind.liability, 180),
    ],
)
async def test_staleness_boundary_exact_threshold_fresh_next_day_stale(
    session, category, kind, threshold
):
    """At exactly the threshold age a snapshot is fresh; one day older is stale,
    for every category (the boundary is exclusive: age > threshold)."""
    as_of = dt.date(2026, 5, 31)
    fresh_date = as_of - dt.timedelta(days=threshold)
    stale_date = fresh_date - dt.timedelta(days=1)

    # Build one snapshot of this category at each boundary age.
    if category is SnapshotCategory.investment:
        upload_fresh = await _cas_upload(
            session, portfolio_key="PAN1", statement_date=fresh_date
        )
        upload_stale = await _cas_upload(
            session, portfolio_key="PAN2", statement_date=stale_date
        )
        await _investment(session, upload=upload_fresh, d=fresh_date, v="1000.00")
        await _investment(session, upload=upload_stale, d=stale_date, v="2000.00")
    elif category in (SnapshotCategory.manual_asset, SnapshotCategory.manual_liability):
        item_fresh = await _manual_item(session, kind=kind.value, name="Fresh Item")
        item_stale = await _manual_item(session, kind=kind.value, name="Stale Item")
        await _manual(
            session,
            item=item_fresh,
            kind=kind.value,
            category=category.value,
            d=fresh_date,
            v="1000.00",
        )
        await _manual(
            session,
            item=item_stale,
            kind=kind.value,
            category=category.value,
            d=stale_date,
            v="2000.00",
        )
    else:
        account = await _account(
            session,
            account_type=(
                "bank_account"
                if category is SnapshotCategory.bank_balance
                else "credit_card"
            ),
        )
        session.add(
            BalanceSnapshot(
                account_id=account.id,
                kind=kind.value,
                category=category.value,
                as_of_date=fresh_date,
                value=Decimal("1000.00"),
                source=(
                    SnapshotSource.bank_statement.value
                    if category is SnapshotCategory.bank_balance
                    else SnapshotSource.cc_statement.value
                ),
            )
        )
        # Second distinct account so it is a separate source.
        account2 = await _account(
            session,
            account_type=(
                "bank_account"
                if category is SnapshotCategory.bank_balance
                else "credit_card"
            ),
        )
        session.add(
            BalanceSnapshot(
                account_id=account2.id,
                kind=kind.value,
                category=category.value,
                as_of_date=stale_date,
                value=Decimal("2000.00"),
                source=(
                    SnapshotSource.bank_statement.value
                    if category is SnapshotCategory.bank_balance
                    else SnapshotSource.cc_statement.value
                ),
            )
        )
        await session.flush()

    summary = await networth.current_networth(session, today=as_of)
    rows = [r for g in summary.groups for r in g.rows if g.category == category.value]
    fresh_row = next(r for r in rows if r.value == Decimal("1000.00"))
    stale_row = next(r for r in rows if r.value == Decimal("2000.00"))
    assert fresh_row.stale is False, f"{category}: threshold age must be fresh"
    assert stale_row.stale is True, f"{category}: threshold+1 must be stale"
    assert summary.has_stale is True


async def test_unreconciled_cas_badge_propagates_to_source_balance(session):
    """An investment snapshot whose CAS upload failed reconciliation carries the
    unreconciled badge; a reconciled one does not."""
    upload_ok = await _cas_upload(
        session,
        portfolio_key="PANOK1",
        statement_date=dt.date(2026, 4, 30),
        portfolio_ok=True,
    )
    upload_bad = await _cas_upload(
        session,
        portfolio_key="PANBAD1",
        statement_date=dt.date(2026, 4, 30),
        portfolio_ok=False,
    )
    await _investment(session, upload=upload_ok, d=dt.date(2026, 4, 30), v="100000.00")
    await _investment(session, upload=upload_bad, d=dt.date(2026, 4, 30), v="50000.00")

    summary = await networth.current_networth(session, today=dt.date(2026, 5, 24))
    rows = {
        r.value: r for g in summary.groups for r in g.rows if g.category == "investment"
    }
    assert rows[Decimal("100000.00")].unreconciled is False
    assert rows[Decimal("50000.00")].unreconciled is True


async def test_monthly_trend_returns_full_uncapped_history(session):
    """The trend is intentionally uncapped: it spans every month from the
    earliest snapshot to the current month — there is no DEFAULT_TREND_MONTHS
    cap applied server-side (that would be a breaking API change)."""
    account = await _account(session)
    # 15 monthly snapshots: 2025-02 .. 2026-04.
    for i in range(15):
        await _bank(
            session,
            account.id,
            dt.date(2025, 2, 1) + relativedelta(months=i),
            "1000.00",
        )

    points = await networth.monthly_trend(session, today=dt.date(2026, 4, 30))

    assert len(points) == 15  # uncapped — exceeds DEFAULT_TREND_MONTHS (12)
    assert points[0].month == "2025-02"
    assert points[-1].month == "2026-04"
