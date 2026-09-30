import datetime
import html
import re
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest

from financial_dashboard.db import Account, Transaction
from financial_dashboard.web.transactions import _date_presets

pytestmark = pytest.mark.anyio


def _currency_summary(page: str, currency: str) -> str:
    match = re.search(
        rf'<article[^>]+data-currency="{re.escape(currency)}".*?</article>',
        page,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def _summary_total(summary: str, key: str) -> str:
    match = re.search(
        rf'<dd[^>]+data-total="{re.escape(key)}"[^>]*>(.*?)</dd>',
        summary,
        flags=re.DOTALL,
    )
    assert match is not None
    return "".join(html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).split())


async def test_filtered_transaction_totals_cover_all_pages(client, session):
    account = Account(
        bank="synthetic-bank",
        type="credit_card",
        label="Synthetic card",
        active=True,
    )
    session.add(account)
    await session.flush()
    session.add_all(
        [
            Transaction(
                account_id=account.id,
                bank=account.bank,
                email_type="synthetic_alert",
                direction="debit",
                amount=Decimal("1.00"),
                currency="INR",
                transaction_date=datetime.date(2030, 4, 10),
            )
            for _ in range(51)
        ]
        + [
            Transaction(
                account_id=account.id,
                bank=account.bank,
                email_type="synthetic_alert",
                direction="credit",
                amount=Decimal("5.00"),
                currency="INR",
                transaction_date=datetime.date(2030, 4, 11),
            ),
            Transaction(
                account_id=account.id,
                bank=account.bank,
                email_type="synthetic_alert",
                direction="credit",
                amount=Decimal("10.00"),
                currency="EUR",
                transaction_date=datetime.date(2030, 4, 12),
            ),
            Transaction(
                account_id=account.id,
                bank=account.bank,
                email_type="synthetic_alert",
                direction="debit",
                amount=Decimal("2.00"),
                currency=" eur ",
                transaction_date=datetime.date(2030, 4, 13),
            ),
            Transaction(
                account_id=account.id,
                bank=account.bank,
                email_type="synthetic_alert",
                direction="debit",
                amount=Decimal("999.00"),
                currency="INR",
                transaction_date=datetime.date(2030, 5, 1),
            ),
        ]
    )
    await session.commit()

    response = await client.get(
        "/transactions",
        params={
            "account_id": account.id,
            "date_from": "2030-04-01",
            "date_to": "2030-04-30",
        },
    )

    assert response.status_code == 200
    summary = _currency_summary(response.text, "INR")
    assert _summary_total(summary, "credits") == "+₹5.00"
    assert _summary_total(summary, "debits") == "−₹51.00"
    assert _summary_total(summary, "net") == "−₹46.00"
    foreign_summary = _currency_summary(response.text, "EUR")
    assert _summary_total(foreign_summary, "credits") == "+EUR10.00"
    assert _summary_total(foreign_summary, "debits") == "−EUR2.00"
    assert _summary_total(foreign_summary, "net") == "+EUR8.00"
    assert response.text.count('data-currency="EUR"') == 1
    assert 'data-currency="eur"' not in response.text
    assert "₹10.00" not in foreign_summary
    assert "54 results" in response.text
    assert "Showing 1&ndash;50 of 54" in response.text
    assert "999.00" not in summary


async def test_overflowing_query_day_redirects_to_month_end(client):
    response = await client.get(
        "/transactions?account_id=30&date_from=2026-04-01&date_to=2026-04-31",
        follow_redirects=False,
    )

    assert response.status_code == 307
    location = response.headers["location"]
    query = parse_qs(urlsplit(location).query)
    assert query["account_id"] == ["30"]
    assert query["date_from"] == ["2026-04-01"]
    assert query["date_to"] == ["2026-04-30"]

    corrected = await client.get(location)
    assert corrected.status_code == 200
    assert 'value="30-04-2026"' in corrected.text


async def test_unrecoverable_query_date_returns_validation_error(client):
    response = await client.get("/transactions?date_from=2026-13-01")

    assert response.status_code == 422


async def test_date_presets_preserve_active_non_date_filters(client):
    response = await client.get(
        "/transactions",
        params={
            "bank": "synthetic-bank",
            "account_id": "30",
            "direction": "debit",
            "counterparty": "",
            "scope": "bank",
            "sort": "amount",
            "order": "asc",
            "date_from": "2020-01-01",
            "date_to": "2020-01-31",
        },
    )

    assert response.status_code == 200
    preset = re.search(
        r'<a href="([^"]+)" class="date-preset[^>]*>Current month</a>',
        response.text,
    )
    assert preset is not None
    href = html.unescape(preset.group(1))
    query = parse_qs(urlsplit(href).query, keep_blank_values=True)
    assert query["bank"] == ["synthetic-bank"]
    assert query["account_id"] == ["30"]
    assert query["direction"] == ["debit"]
    assert query["counterparty"] == [""]
    assert query["scope"] == ["bank"]
    assert query["sort"] == ["amount"]
    assert query["order"] == ["asc"]
    assert query["date_from"] != ["2020-01-01"]
    assert query["date_to"] != ["2020-01-31"]


async def test_financial_year_presets_use_april_to_march():
    presets = {
        item["label"]: item for item in _date_presets(datetime.date(2026, 7, 21))
    }

    assert presets["Current financial year"] == {
        "label": "Current financial year",
        "date_from": "2026-04-01",
        "date_to": "2027-03-31",
    }
    assert presets["Previous financial year"] == {
        "label": "Previous financial year",
        "date_from": "2025-04-01",
        "date_to": "2026-03-31",
    }
    assert presets["Previous month"]["date_from"] == "2026-06-01"
    assert presets["Previous month"]["date_to"] == "2026-06-30"
