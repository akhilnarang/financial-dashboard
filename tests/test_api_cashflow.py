import datetime
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from tests.conftest import bank_account

pytestmark = pytest.mark.anyio
D = Decimal
TODAY = datetime.date.today()


async def _add(session: AsyncSession, **kw) -> None:
    """Seed one transaction, linked to the bank account unless ``account_id`` says else.

    The summary counts bank rows, so a row seeded without a link would be
    unaccounted and would reach none of the figures asserted below.
    """
    base = dict(
        bank="hdfc",
        email_type="x",
        currency="INR",
        transaction_date=datetime.date(2026, 6, 15),
        account_id=await bank_account(session),
    )
    base.update(kw)
    session.add(Transaction(**base))
    await session.flush()


def _dec(value: object) -> Decimal:
    return Decimal(str(value))


async def test_summary_defaults_to_current_month(client, session: AsyncSession):
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(
        session,
        direction="debit",
        amount=D("300"),
        category="groceries",
        transaction_date=TODAY,
    )

    r = await client.get("/api/cashflow/summary")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "income",
        "expense",
        "investment",
        "transfers_in",
        "uncategorized",
        "net_cash_retained",
        "footnotes",
    ):
        assert key in body

    # Missing bounds → first of the current month through today: the June row is
    # outside that window, the row dated today is inside it.
    assert body["date_from"] == TODAY.replace(day=1).isoformat()
    assert body["date_to"] == TODAY.isoformat()
    assert _dec(body["income"]["total"]) == D("0")
    assert _dec(body["expense"]["total"]) == D("300")
    assert _dec(body["net_cash_retained"]) == D("-300")


async def test_summary_accepts_range(client, session: AsyncSession):
    await _add(session, direction="credit", amount=D("1000"), category="salary")
    await _add(session, direction="debit", amount=D("250"), category="groceries")
    await _add(
        session,
        direction="credit",
        amount=D("500"),
        category="repayment",
        counterparty="MOM",
    )

    r = await client.get(
        "/api/cashflow/summary?date_from=2026-06-01&date_to=2026-06-30"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["date_from"] == "2026-06-01"
    assert body["date_to"] == "2026-06-30"
    assert _dec(body["income"]["total"]) == D("1000")
    assert body["income"]["lines"][0]["slug"] == "salary"
    assert _dec(body["expense"]["total"]) == D("250")
    assert _dec(body["transfers_in"]["total"]) == D("500")
    assert body["transfers_in"]["lines"][0]["counterparty"] == "MOM"
    assert _dec(body["net_cash_retained"]) == D("1250")


async def test_summary_invalid_bound_does_not_reset_the_other(client):
    r = await client.get(
        "/api/cashflow/summary?date_from=not-a-date&date_to=2026-06-30"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["date_from"] == TODAY.replace(day=1).isoformat()
    assert body["date_to"] == "2026-06-30"


async def test_trend_shape(client, session: AsyncSession):
    await _add(
        session,
        direction="credit",
        amount=D("900"),
        category="salary",
        transaction_date=TODAY,
    )
    await _add(
        session,
        direction="debit",
        amount=D("100"),
        category="investment",
        transaction_date=TODAY,
    )

    r = await client.get("/api/cashflow/trend?months=6")
    assert r.status_code == 200
    pts = r.json()
    assert len(pts) == 6
    assert {"month", "income", "expense", "net_invested", "salary_count"} <= set(pts[0])

    current = pts[-1]
    assert current["month"] == f"{TODAY.year:04d}-{TODAY.month:02d}"
    assert _dec(current["income"]) == D("900")
    assert _dec(current["net_invested"]) == D("100")
    assert current["salary_count"] == 1


async def test_trend_defaults_to_twelve_months(client):
    r = await client.get("/api/cashflow/trend")
    assert r.status_code == 200
    assert len(r.json()) == 12


async def test_trend_months_is_clamped(client):
    low = await client.get("/api/cashflow/trend?months=0")
    assert low.status_code == 200
    assert len(low.json()) == 1

    high = await client.get("/api/cashflow/trend?months=999")
    assert high.status_code == 200
    assert len(high.json()) == 60
