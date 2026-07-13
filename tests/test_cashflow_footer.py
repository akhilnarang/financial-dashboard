"""Integrity tests for the /cashflow footnotes, footer and empty state.

Every test here asks the same question of a different figure: does the number the
page prints describe exactly the rows the link *next to that number* lists? So
none of them assert on a page-wide substring — each one pulls the anchor out of
the element under test, follows that very href, and counts what comes back. A
figure and its drill-through can then never drift apart without a failure here.
"""

import datetime
import re
from decimal import Decimal

import pytest

from financial_dashboard.core.templating import format_inr_compact, format_inr_exact
from tests.test_web_cashflow import (
    RANGE,
    RANGE_HTML,
    _add,
    _month_start,
    _region,
    _tile,
)

pytestmark = pytest.mark.anyio

# The listing renders one "/detail" link per row, so counting them counts rows.
DETAIL = "/detail"


def _href(markup: str) -> str:
    """The single anchor inside one rendered row/tile, as a followable URL."""
    match = re.search(r'href="([^"]+)"', markup)
    assert match, f"no anchor in {markup!r}"
    # Autoescaping wrote the separators as entities; a client needs them back.
    return match.group(1).replace("&amp;", "&")


def _count(markup: str) -> int:
    """The transaction count a footnote row or tile prints ("N txns")."""
    match = re.search(r"(\d+) txns", markup)
    assert match, f"no count in {markup!r}"
    return int(match.group(1))


def _rows(section: str) -> list[str]:
    return re.findall(r"<tr>.*?</tr>", section, flags=re.DOTALL)


def _row_with(section: str, needle: str) -> str:
    """The one table row of a rendered section whose label contains ``needle``."""
    matching = [row for row in _rows(section) if needle in row]
    assert len(matching) == 1, f"expected exactly one row containing {needle!r}"
    return matching[0]


def _line_count(row: str) -> int:
    """The count cell of a category/counterparty line."""
    match = re.search(r'<td class="text-sm text-muted">(\d+)</td>', row)
    assert match, f"no count cell in {row!r}"
    return int(match.group(1))


def _listed_amounts(listing: str) -> list[Decimal]:
    """Every amount a /transactions listing prints, signed by its own direction.

    The listing renders the sign as an entity and prefixes the currency symbol, so
    the number has to be pulled back out of the cell to be summed. Summing these is
    what lets a figure on the cashflow page be proved *from the rows its link
    returned*, rather than from the seed data the test happens to remember writing.
    """
    cells = re.findall(
        r'class="amt amt-\w+">(&minus;|\+)[^\d]*([\d,]+\.\d{2})</td>', listing
    )
    return [
        -Decimal(body.replace(",", ""))
        if sign == "&minus;"
        else Decimal(body.replace(",", ""))
        for sign, body in cells
    ]


async def _listed(client, markup: str) -> str:
    """Follow the drill-through rendered inside ``markup`` and return the listing."""
    response = await client.get(_href(markup))
    assert response.status_code == 200
    return response.text


async def test_empty_db_renders_zero_tiles_not_a_broken_page(client):
    """An empty range still has a shape: five ₹0 tiles and the empty-state card.

    Hiding the tiles outright would make "no data" and "page failed to build"
    look the same, and the tiles are the thing a reader looks for first.
    """
    r = await client.get(f"/cashflow?{RANGE}")
    assert r.status_code == 200
    assert "Traceback" not in r.text

    assert re.findall(r'data-tile="([a-z_]+)"', r.text) == [
        "income",
        "expense",
        "net_invested",
        "transfers_in",
        "uncategorized",
    ]
    for name in ("income", "expense", "net_invested", "transfers_in", "uncategorized"):
        tile = _tile(r.text, name)
        assert "₹0.00" in tile, f"{name} tile does not read zero"
    assert "No transactions in this range" in r.text


async def test_income_line_agrees_with_the_rows_its_own_link_lists(client, session):
    """A core bucket line: its count and its exact total must both be provable
    from the listing its anchor points at, decoys included."""
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="10000", direction="credit", category="salary", day=20)
    await _add(session, amount="4321", direction="credit", category="interest")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _row_with(_region(page, "data-section", "income"), ">Salary<")
    assert "₹1,00,000.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _line_count(row) == 2
    assert "90,000.00" in listing
    assert "10,000.00" in listing
    # The other income line's row is not in this line's list.
    assert "4,321.00" not in listing


async def test_no_counterparty_group_lists_every_blank_spelling(client, session):
    """NULL, empty and whitespace-only are the same absence of a counterparty.

    They must collapse into one "(no counterparty)" line *and* that line's link
    must list all of them: a tab-only counterparty that gets its own line, or is
    counted in the group but missing from the group's listing, is a figure whose
    own link contradicts it.
    """
    await _add(session, amount="100", direction="credit", category="repayment")
    await _add(
        session, amount="200", direction="credit", category="repayment", counterparty=""
    )
    await _add(
        session,
        amount="400",
        direction="credit",
        category="repayment",
        counterparty="\t",
    )
    await _add(
        session,
        amount="800",
        direction="credit",
        category="repayment",
        counterparty="MOM",
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text
    section = _region(page, "data-section", "transfers_in")
    # One line per real counterparty: the blank spellings are not three of them.
    assert len(_rows(section)) == 2

    row = _row_with(section, "(no counterparty)")
    assert "₹700.00" in row
    assert _line_count(row) == 3

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == 3
    for amount in ("100.00", "200.00", "400.00"):
        assert amount in listing, f"the group's link omits the row it counted: {amount}"
    assert "800.00" not in listing


async def test_uncategorized_lines_label_null_and_unmapped_slugs(client, session):
    """The two placeholder labels the bucket map produces are the only thing that
    tells a reader *why* a row is uncategorized, and each one drills through.

    The categorized decoy is what makes the tile's link worth testing: seed only
    uncategorized rows and an entirely unfiltered link would return them all and
    still count three. The tile's *total* is checked too, against the amounts the
    listing itself printed — a count alone cannot catch a link that lists the right
    number of the wrong rows.
    """
    await _add(session, amount="800", direction="debit", category=None)
    await _add(session, amount="250", direction="debit", category="crypto_yield")
    await _add(session, amount="90", direction="debit", category="unknown")
    await _add(session, amount="5555", direction="debit", category="rent")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    section = _region(page, "data-section", "uncategorized")
    assert "(uncategorized)" in section
    assert "unmapped: crypto_yield" in section

    # A runtime slug is a real category value, so its line lists exactly its rows.
    unmapped = _row_with(section, "unmapped: crypto_yield")
    listing = await _listed(client, unmapped)
    assert listing.count(DETAIL) == _line_count(unmapped) == 1
    assert "250.00" in listing
    assert "800.00" not in listing

    # The tile counts the whole population, and its link lists exactly that: the
    # three uncategorized rows and not the categorized one sharing their range.
    tile = _tile(page, "uncategorized")
    tile_listing = await _listed(client, tile)
    assert tile_listing.count(DETAIL) == _count(tile) == 3
    assert "5,555.00" not in tile_listing, "the tile's link lists a categorized row"

    # And the figure the tile prints is the sum of the rows that link returned.
    listed = _listed_amounts(tile_listing)
    assert sorted(listed) == [Decimal("-800"), Decimal("-250"), Decimal("-90")]
    assert format_inr_compact(sum(listed)) in tile
    assert format_inr_exact(sum(listed)) in section


async def test_null_category_line_lists_only_the_rows_it_counts(client, session):
    """The "(uncategorized)" line is the NULL-category rows alone.

    Its drill-through must be the NULL rows alone as well. The whole-bucket filter
    is a superset — it also takes in the 'unknown' sentinel and unmapped slugs — so
    pointing this line at it would print "1 txn" above a list of three.
    """
    await _add(session, amount="800", direction="debit", category=None)
    await _add(session, amount="250", direction="debit", category="crypto_yield")
    await _add(session, amount="90", direction="debit", category="unknown")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    section = _region(page, "data-section", "uncategorized")
    row = _row_with(section, ">(uncategorized)<")

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _line_count(row) == 1
    assert _listed_amounts(listing) == [Decimal("-800")]
    # The other two members of the bucket are not this line's rows.
    assert "250.00" not in listing
    assert "90.00" not in listing


async def test_trend_survives_a_range_with_no_activity(client, session):
    """The trend is a trailing-12-month series, so the selected range does not
    scope it: an empty range is not a reason to hide months of real history.

    Only the range-scoped parts — breakdown, tables, footer — go away.
    """
    # The history has to be inside the trend's trailing-12-month window for the
    # test to mean what it says, and that window ends today: seeding a fixed month
    # would put the row outside it once that month aged out. Last month is always
    # in the window, and the month before it is an empty range inside it.
    today = datetime.date.today()
    history = _month_start(today, 1)
    await _add(
        session,
        amount="90000",
        direction="credit",
        category="salary",
        day=15,
        month=history,
    )

    # A range the seeded month is not in: empty here, history immediately after it.
    empty = _month_start(today, 2)
    end = history - datetime.timedelta(days=1)
    page = (
        await client.get(
            f"/cashflow?date_from={empty.isoformat()}&date_to={end.isoformat()}"
        )
    ).text
    assert "No transactions in this range" in page

    # The seeded month really is in the trend's window, so the trend below has
    # history to draw even though the range above it has none.
    api = await client.get("/api/cashflow/trend?months=12")
    seeded = f"{history.year:04d}-{history.month:02d}"
    assert [p["month"] for p in api.json() if Decimal(p["income"])] == [seeded]

    trend = _region(page, "data-trend")
    assert 'id="cf-trend"' in trend
    # The range-scoped regions are the ones an empty range removes.
    assert "data-reconciliation" not in page
    assert 'id="cf-breakdown"' not in page


async def test_internal_footnote_count_agrees_with_its_drill(client, session):
    """The internal footnote is the only place these rows are visible at all, so
    its count has to be the number of rows its link lists — nothing else checks."""
    await _add(session, amount="5000", direction="debit", category="self_transfer")
    await _add(
        session, amount="12000", direction="debit", category="credit_card_payment"
    )
    await _add(session, amount="7", direction="debit", category="rent")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _region(page, "data-footnote", "internal")
    assert f"internal=1&amp;{RANGE_HTML}" in row
    # Gross, not net: two outflows add up rather than cancelling.
    assert "₹17,000.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 2
    assert "7.00" not in listing


async def test_non_inr_footnote_count_agrees_with_its_drill(client, session):
    await _add(
        session, amount="100", direction="debit", category="rent", currency="USD"
    )
    await _add(
        session, amount="60", direction="debit", category="dining", currency="EUR"
    )
    await _add(session, amount="20000", direction="debit", category="rent")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _region(page, "data-footnote", "non_inr")
    assert f"non_inr=1&amp;{RANGE_HTML}" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 2
    # The rupee row belongs to a bucket above, never to this footnote's list.
    assert "20,000.00" not in listing


async def test_undated_footnote_count_agrees_with_its_unscoped_drill(client, session):
    """The undated link carries no range because an undated row matches none: the
    figure is range-independent, so following the link must return the same rows
    whatever range the page is showing."""
    await _add(session, amount="333", direction="debit", category="rent", dated=False)
    await _add(session, amount="90", direction="credit", category="salary", dated=False)
    await _add(session, amount="20000", direction="debit", category="rent")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _region(page, "data-footnote", "undated")
    assert _href(row) == "/transactions?undated=1"
    # Signed net: a 90 credit against a 333 debit.
    assert "-₹243.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 2
    assert "333.00" in listing
    assert "20,000.00" not in listing


async def test_reconciliation_footer_adds_up_from_the_tiles_it_names(client, session):
    """The footer is informational, not an invariant — but it must at least be
    arithmetic over the very figures the tiles above it print."""
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="2000", direction="credit", category="repayment")
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(session, amount="500", direction="credit", category="refund")
    await _add(session, amount="10000", direction="debit", category="investment")
    # Excluded populations must not move the identity.
    await _add(session, amount="5000", direction="debit", category="self_transfer")
    await _add(session, amount="70", direction="debit", category="rent", currency="USD")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    footer = _region(page, "data-reconciliation")
    # 90,000 + 2,000 - 19,500 - 10,000 = 62,500.
    assert "₹90,000.00" in footer
    assert "₹2,000.00" in footer
    assert "₹19,500.00" in footer
    assert "₹10,000.00" in footer
    assert "net cash retained ₹62,500.00" in footer
    assert "₹5,000.00" not in footer
    assert "70.00" not in footer


async def test_reconciliation_footer_carries_its_error_bar_and_its_caveat(
    client, session
):
    """The identity is not an invariant, and the footer has to say so.

    Two things make it honest rather than a balance that silently fails to balance:
    the uncategorized net printed beside it — the money the buckets could not place,
    and therefore the margin the retained figure may be wrong by — and wording that
    it is informational. Both assertions are scoped to the footer, so neither can be
    satisfied by the tile or the section that print the same figure elsewhere.
    """
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(session, amount="1234", direction="debit", category=None)
    await _add(session, amount="4000", direction="debit", category="crypto_yield")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    footer = _region(page, "data-reconciliation")

    assert "net cash retained ₹70,000.00" in footer
    # The error bar: the uncategorized net, exact and Indian-grouped, not the
    # compact figure the tile shows.
    assert "uncategorized -₹5,234.00" in footer
    assert format_inr_exact(Decimal("-5234")) == "-₹5,234.00"

    lowered = footer.lower()
    assert "informational" in lowered
    assert "not an enforced invariant" in lowered
    assert "error bar" in lowered
