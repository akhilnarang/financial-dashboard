"""What one cashflow page load costs the database.

The page renders every figure server-side from ``cashflow_summary``. If the
breakdown chart then fetched ``/api/cashflow/summary``, the *same* range would be
aggregated a second time — eight aggregate queries run twice over ``transactions``
for numbers already in the HTML. These tests pin the cost so that regression
cannot come back in quietly: they count the statements a request actually issues,
rather than asserting on a comment about them.

Scoping a figure to the bank costs no statement of its own: the scope is a
correlated ``EXISTS`` inside the aggregate that needed it, not a second read.
"""

import datetime
from contextlib import contextmanager
from decimal import Decimal

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from financial_dashboard.db.models import Transaction
from tests.conftest import bank_account

pytestmark = pytest.mark.anyio

RANGE = "date_from=2026-06-01&date_to=2026-06-30"

# What the summary costs: the grouped bank-side bucket scan, the all-account
# expense detail, transfers-in, uncategorized, and the four footnote reads
# (internal, non-INR, undated, unaccounted).
SUMMARY_QUERIES = 8
# What the trend costs: the month/category/direction scan and the salary counts.
TREND_QUERIES = 2


async def _seed(session):
    """One in-range bank row, so the page draws its charts rather than an empty state.

    Linked to the bank account: the figures are bank-scoped, so an unlinked row
    would be unaccounted and the page would render its empty state — and an empty
    state is not the page whose cost these tests mean to pin.
    """
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            amount=Decimal("90000"),
            direction="credit",
            currency="INR",
            category="salary",
            transaction_date=datetime.date(2026, 6, 15),
            account_id=await bank_account(session),
        )
    )
    await session.commit()


@contextmanager
def count_transaction_reads():
    """Count the statements issued against ``transactions`` inside the block."""
    seen: list[str] = []

    def before_cursor_execute(conn, cursor, statement, params, context, executemany):
        if "FROM transactions" in statement:
            seen.append(statement)

    event.listen(Engine, "before_cursor_execute", before_cursor_execute)
    try:
        yield seen
    finally:
        event.remove(Engine, "before_cursor_execute", before_cursor_execute)


async def test_page_load_aggregates_the_range_once(client, session):
    """The page aggregates the selected range exactly once.

    Eight queries, not sixteen: the breakdown chart is handed the summary the page
    was rendered from instead of fetching it back.
    """
    await _seed(session)
    with count_transaction_reads() as queries:
        page = await client.get(f"/cashflow?{RANGE}")
    assert page.status_code == 200
    assert len(queries) == SUMMARY_QUERIES

    # The page carries the summary, so there is nothing for the chart to re-read.
    assert "/api/cashflow/summary" not in page.text
    assert 'id="cf-summary"' in page.text


async def test_full_page_load_is_summary_plus_trend_and_nothing_more(client, session):
    """Everything one page load costs, counted end to end.

    The document is one summary; the only fetch it then makes is the trend, whose
    trailing-twelve-month window really is a different question from the selected
    range. Ten aggregate queries in total — where a page that also re-fetched the
    summary would spend eighteen.
    """
    await _seed(session)
    with count_transaction_reads() as page_queries:
        page = await client.get(f"/cashflow?{RANGE}")
    with count_transaction_reads() as trend_queries:
        trend = await client.get("/api/cashflow/trend?months=12")
    assert trend.status_code == 200

    assert len(page_queries) == SUMMARY_QUERIES
    assert len(trend_queries) == TREND_QUERIES
    assert len(page_queries) + len(trend_queries) == 10

    # The figure the removed fetch would have added back, measured rather than
    # asserted from memory: it is a second full summary.
    with count_transaction_reads() as summary_queries:
        await client.get(f"/api/cashflow/summary?{RANGE}")
    assert len(summary_queries) == SUMMARY_QUERIES
    assert len(page_queries) + len(trend_queries) + len(summary_queries) == 18
    assert page.status_code == 200
