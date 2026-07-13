import datetime
import html
import re
from decimal import Decimal

import pytest

from financial_dashboard.db.models import Transaction

pytestmark = pytest.mark.anyio

# Every seeded row is dated, so undated=1 only matches a row that sets it NULL.
DATED = datetime.date(2026, 6, 15)


async def _seed(session):
    rows = [
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("100"),
            category="groceries",
            currency="INR",
            transaction_date=DATED,
        ),
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="credit",
            amount=Decimal("500"),
            category="repayment",
            counterparty="MOM",
            currency="INR",
            transaction_date=DATED,
        ),
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("10"),
            category="self_transfer",
            currency="INR",
            transaction_date=DATED,
        ),
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("7"),
            category="unknown",
            currency="INR",
            transaction_date=DATED,
        ),
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("9"),
            category=None,
            currency="INR",
            transaction_date=DATED,
        ),
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("3"),
            category="dining",
            currency="USD",
            transaction_date=DATED,
        ),
    ]
    session.add_all(rows)
    await session.flush()


def _count_rows(html: str) -> int:
    # Each rendered row links to /transactions/<id>/detail, and nothing else on
    # the page does, so this is a stable per-row marker.
    return html.count("/detail")


async def test_unfiltered_list_shows_every_row(client, session):
    await _seed(session)
    r = await client.get("/transactions")
    assert r.status_code == 200
    assert _count_rows(r.text) == 6


async def test_category_filter(client, session):
    await _seed(session)
    r = await client.get("/transactions?category=groceries")
    assert r.status_code == 200
    assert _count_rows(r.text) == 1


async def test_uncategorized_filter_matches_report_definition(client, session):
    await _seed(session)
    r = await client.get("/transactions?uncategorized=1")
    assert r.status_code == 200
    assert _count_rows(r.text) == 2  # NULL + 'unknown'


async def test_uncategorized_includes_non_inr(client, session):
    # The uncategorized drill has no currency clause, so a non-INR uncategorized
    # row is included — keeping the drill count equal to the report's tile count.
    await _seed(session)
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("2"),
            category=None,
            currency="USD",
            transaction_date=DATED,
        )
    )
    await session.flush()
    r = await client.get("/transactions?uncategorized=1")
    assert _count_rows(r.text) == 3


async def test_uncategorized_includes_unmapped_slug(client, session):
    # A runtime slug the code map does not know must surface in the drill, not
    # vanish, so it matches the report line that also treats it as uncategorized.
    await _seed(session)
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("5"),
            category="brand_new_slug",
            currency="INR",
            transaction_date=DATED,
        )
    )
    await session.flush()
    r = await client.get("/transactions?uncategorized=1")
    assert _count_rows(r.text) == 3


async def test_category_null_is_narrower_than_uncategorized(client, session):
    # Two different questions: "no category at all" (the NULL rows) versus "no
    # category any bucket can use" (those, plus the 'unknown' sentinel, plus slugs
    # the map does not know). The report has a line for the first and a tile for
    # the second, so each needs a filter that returns its own population and no
    # more — one filter serving both makes the line contradict its own count.
    await _seed(session)
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("5"),
            category="brand_new_slug",
            currency="INR",
            transaction_date=DATED,
        )
    )
    await session.flush()

    wide = await client.get("/transactions?uncategorized=1")
    assert _count_rows(wide.text) == 3  # NULL + 'unknown' + unmapped

    narrow = await client.get("/transactions?category_null=1")
    assert narrow.status_code == 200
    assert _count_rows(narrow.text) == 1  # the NULL row alone
    assert "9.00" in narrow.text
    assert "7.00" not in narrow.text  # the 'unknown' sentinel row
    assert "5.00" not in narrow.text  # the unmapped-slug row


async def test_internal_filter(client, session):
    await _seed(session)
    r = await client.get("/transactions?internal=1")
    assert _count_rows(r.text) == 1  # only self_transfer


async def test_non_inr_filter(client, session):
    await _seed(session)
    r = await client.get("/transactions?non_inr=1")
    assert _count_rows(r.text) == 1  # only the USD dining row


async def _seed_null_currency(session):
    # A NULL currency is a rupee row that predates the column's default, so the
    # rupee drill must list it and the non-INR one must not.
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("11"),
            category="dining",
            currency=None,
            transaction_date=DATED,
        )
    )
    await session.flush()


async def test_non_inr_zero_lists_inr_and_null_currency_rows(client, session):
    await _seed(session)
    await _seed_null_currency(session)
    r = await client.get("/transactions?non_inr=0")
    assert r.status_code == 200
    # The 5 INR seed rows plus the NULL-currency one; the USD row is excluded.
    assert _count_rows(r.text) == 6
    assert "3.00" not in r.text  # the USD dining row


async def test_non_inr_zero_and_one_are_complements(client, session):
    await _seed(session)
    await _seed_null_currency(session)
    rupee = await client.get("/transactions?non_inr=0")
    foreign = await client.get("/transactions?non_inr=1")
    everything = await client.get("/transactions")
    assert _count_rows(rupee.text) + _count_rows(foreign.text) == _count_rows(
        everything.text
    )


async def test_absent_non_inr_applies_no_currency_filter(client, session):
    await _seed(session)
    await _seed_null_currency(session)
    r = await client.get("/transactions?category=dining")
    # USD, NULL and nothing else: an omitted non_inr must stay a non-filter.
    assert _count_rows(r.text) == 2


async def test_undated_filter(client, session):
    await _seed(session)
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("4"),
            category="dining",
            currency="INR",
            transaction_date=None,
        )
    )
    await session.flush()
    r = await client.get("/transactions?undated=1")
    assert _count_rows(r.text) == 1  # only the transaction_date IS NULL row


async def test_repayment_counterparty_filter(client, session):
    await _seed(session)
    r = await client.get("/transactions?category=repayment&counterparty=MOM")
    assert _count_rows(r.text) == 1  # only the MOM repayment row


async def test_blank_counterparty_groups_null_and_empty(client, session):
    # Blank drill (counterparty=) must match BOTH NULL and empty-string
    # counterparty rows, so it equals the transfers-in "(no counterparty)" group.
    for cp in (None, ""):
        session.add(
            Transaction(
                bank="hdfc",
                email_type="x",
                direction="credit",
                amount=Decimal("50"),
                category="repayment",
                counterparty=cp,
                currency="INR",
                transaction_date=DATED,
            )
        )
    await session.flush()
    r = await client.get("/transactions?category=repayment&counterparty=")
    assert _count_rows(r.text) == 2  # NULL + empty-string, one group


async def test_blank_counterparty_also_matches_whitespace_only_rows(client, session):
    # A tab-only counterparty names nobody, so it belongs to the blank group on
    # both sides of the drill-through: the report collapses it into the
    # "(no counterparty)" line, and this filter has to list it there. It is also
    # what a whitespace-only value in the query string selects, rather than an
    # exact match on a string no row is ever meant to carry.
    for cp in (None, "", "\t", "  "):
        session.add(
            Transaction(
                bank="hdfc",
                email_type="x",
                direction="credit",
                amount=Decimal("50"),
                category="repayment",
                counterparty=cp,
                currency="INR",
                transaction_date=DATED,
            )
        )
    await session.flush()

    for query in ("counterparty=", "counterparty=%09", "counterparty=+"):
        r = await client.get(f"/transactions?category=repayment&{query}")
        assert _count_rows(r.text) == 4, f"{query} listed the wrong blank group"


async def test_blank_counterparty_page_two_link_keeps_filter_and_result_set(
    client, session
):
    # A present-but-empty counterparty is a real filter, so the page links must
    # carry it. If it were dropped for being falsy, page 2 would silently widen
    # to every repayment row — including the named-counterparty ones.
    for i in range(55):
        session.add(
            Transaction(
                bank="hdfc",
                email_type="x",
                direction="credit",
                amount=Decimal(50 + i),
                category="repayment",
                # Both blank spellings belong to the same "(no counterparty)" group.
                counterparty=None if i % 2 else "",
                currency="INR",
                transaction_date=DATED,
            )
        )
    # Decoys: the same category with a real counterparty, and a blank
    # counterparty under a different category. Neither may ever be listed.
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="credit",
            amount=Decimal("999"),
            category="repayment",
            counterparty="MOM",
            currency="INR",
            transaction_date=DATED,
        )
    )
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("998"),
            category="groceries",
            counterparty=None,
            currency="INR",
            transaction_date=DATED,
        )
    )
    await session.flush()

    r = await client.get("/transactions?category=repayment&counterparty=")
    assert r.status_code == 200
    assert _count_rows(r.text) == 50  # a full first page, so pagination renders

    hrefs = [html.unescape(h) for h in re.findall(r'href="([^"]+)"', r.text)]
    page_two = [h for h in hrefs if "page=2" in h]
    assert page_two, "pagination nav did not render a page-2 link"
    assert all("counterparty=" in h for h in page_two)

    r2 = await client.get(page_two[0])
    assert r2.status_code == 200
    assert _count_rows(r2.text) == 5  # 55 matching rows - a full page of 50
    assert "MOM" not in r2.text
    assert "998" not in r2.text


async def test_existing_filters_unchanged_without_drill_params(client, session):
    await _seed(session)
    r = await client.get("/transactions?direction=credit")
    assert _count_rows(r.text) == 1  # only the repayment credit
