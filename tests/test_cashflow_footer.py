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
from tests.conftest import MISSING_ACCOUNT_ID, card_account, ensure_account
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
    its count has to be the number of rows its link lists — nothing else checks.

    On the cash basis the footnote counts self-transfers alone: a card bill is
    money leaving the bank, so it reads as expense and is counted there instead.

    The bank-side card bill is the decoy the whole test turns on. It is exactly the
    row the old, unscoped ``internal=1`` link listed and the new footnote does not
    count — the drift these scopes exist to close — so without it here the link
    could go back to listing card bills under a self-transfers-only count and no
    test would notice. It must be absent from this listing *and* present in the
    Card bills expense line, which is where the money actually went.
    """
    await _add(session, amount="5000", direction="debit", category="self_transfer")
    await _add(session, amount="12000", direction="debit", category="self_transfer")
    await _add(
        session, amount="9100", direction="debit", category="credit_card_payment"
    )
    await _add(session, amount="7", direction="debit", category="rent")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _region(page, "data-footnote", "internal")
    # Scoped, so `internal=1` here means self-transfers and not the two-slug set.
    assert f"internal=1&amp;{RANGE_HTML}&amp;scope=bank" in row
    # Gross, not net: two outflows add up rather than cancelling. The bill is not
    # in it.
    assert "₹17,000.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 2
    assert "9,100.00" not in listing, (
        "the internal drill lists the card bill the footnote counts as expense"
    )
    assert "7.00" not in listing

    # And the bill is not lost between the two: it is the Card bills expense line,
    # whose own link lists it and nothing else.
    bills = _row_with(_region(page, "data-section", "expense"), ">Card bills<")
    bill_listing = await _listed(client, bills)
    assert bill_listing.count(DETAIL) == _line_count(bills) == 1
    assert "9,100.00" in bill_listing
    assert "5,000.00" not in bill_listing


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


# ---------------------------------------------------------------------------
# The bank scope, as the page renders it: every figure's link carries exactly the
# scope of the figure it sits next to, and following it returns exactly its rows.
# Every test below seeds a card row, an unlinked row or both as decoys, so a link
# that lost its scope — or gained one it should not have — lists them and fails.
# ---------------------------------------------------------------------------


async def test_core_lines_drill_into_bank_rows_alone(client, session):
    """A card row can carry any slug — a manual override mints one — and none of
    them reach a bank figure. So no core link may list one either.

    The card salary is the decoy that matters: it is an *income* slug on an
    out-of-scope account, and a link that dropped the scope would list it beneath
    an income figure that never counted it.
    """
    card = await card_account(session)
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(
        session,
        amount="4444",
        direction="credit",
        category="salary",
        account_id=card,
    )
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(
        session, amount="777", direction="debit", category="rent", account_id=card
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text
    assert "1 txns" in _tile(page, "income")

    for section, label, mine, theirs in (
        ("income", ">Salary<", "90,000.00", "4,444.00"),
        ("expense", ">Rent<", "20,000.00", "777.00"),
    ):
        row = _row_with(_region(page, "data-section", section), label)
        assert "scope=bank" in row
        assert "non_inr=0" in row
        listing = await _listed(client, row)
        assert listing.count(DETAIL) == _line_count(row) == 1
        assert mine in listing
        assert theirs not in listing, f"the {section} link lists a card row"


async def test_uncategorized_drill_is_bank_scoped_but_still_currency_agnostic(
    client, session
):
    """Two rules on one link, and they pull in opposite directions.

    Only a bank-side uncategorized row can distort a bank-basis identity, so the
    tile is scoped; but a foreign-currency row with no category is still
    uncategorized, so the tile is *not* currency-filtered. The link has to carry
    the first and not the second, or its count and its listing disagree.
    """
    card = await card_account(session)
    await _add(session, amount="800", direction="debit", category=None)
    await _add(session, amount="7", direction="debit", category=None, currency="USD")
    await _add(
        session, amount="6543", direction="debit", category=None, account_id=card
    )

    tile = _tile((await client.get(f"/cashflow?{RANGE}")).text, "uncategorized")
    assert "scope=bank" in tile
    assert "non_inr=0" not in tile

    listing = await _listed(client, tile)
    assert listing.count(DETAIL) == _count(tile) == 2
    assert "800.00" in listing
    assert "7.00" in listing, "the currency filter crept back onto the link"
    assert "6,543.00" not in listing, "the uncategorized drill lists a card row"


async def test_unaccounted_footnote_count_agrees_with_its_drill(client, session):
    """The rows on no known account: unlinked, dangling, or an account type nothing
    recognizes. They reach no figure above, so this footnote is the only place they
    are visible, and its link is the only way to see which rows they are.

    The bank and card rows are the decoys: the footnote is the *complement* of both
    scopes, so a link that dropped its scope would list them too.
    """
    dangling = MISSING_ACCOUNT_ID
    unknown_type = await ensure_account(session, 7, "wallet")
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(
        session,
        amount="4444",
        direction="debit",
        category="dining",
        account_id=await card_account(session),
    )
    await _add(
        session, amount="800", direction="debit", category="rent", account_id=None
    )
    await _add(
        session, amount="250", direction="debit", category="rent", account_id=dangling
    )
    await _add(
        session,
        amount="60",
        direction="credit",
        category="refund",
        account_id=unknown_type,
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _region(page, "data-footnote", "unaccounted")
    assert f"scope=unaccounted&amp;{RANGE_HTML}" in row or (
        f"{RANGE_HTML}&amp;scope=unaccounted" in row
    )
    # Signed net: a 60 credit against 800 + 250 of debits.
    assert "-₹990.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 3
    # And the figure is the sum of the rows that link returned, not of the seed.
    listed = _listed_amounts(listing)
    assert sorted(listed) == [Decimal("-800"), Decimal("-250"), Decimal("60")]
    assert format_inr_exact(sum(listed)) in row
    assert "90,000.00" not in listing
    assert "4,444.00" not in listing, "the unaccounted drill lists a card row"


async def test_expense_detail_counts_the_swipes_over_every_account(client, session):
    """The other question — what was *bought* — and the one figure here that is not
    bank-scoped, so it is the one link that must NOT say ``scope=bank``.

    A card swipe never touches the bank, so an all-account figure whose link carried
    the bank scope would list a fraction of the rows it summed. The card bill is the
    decoy on the other side: over every account it is internal churn, because it
    settles the very swipes this figure already counted, so it must not appear as a
    line here at all — while the headline above counts it as expense. The two figures
    disagreeing is the point, and the caveat has to say so.
    """
    card = await card_account(session)
    await _add(
        session, amount="4000", direction="debit", category="dining", account_id=card
    )
    await _add(
        session,
        amount="55",
        direction="debit",
        category="dining",
        currency="USD",
        account_id=card,
    )
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(
        session, amount="9100", direction="debit", category="credit_card_payment"
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text
    detail = _region(page, "data-section", "expense_detail")

    # 4,000 of swipes + 20,000 of rent. The card bill is internal here, so it is
    # neither a line nor part of the total...
    assert "₹24,000.00" in detail
    assert "Card bills" not in detail
    # ...but it *is* the headline, which counts the bill and not the swipe.
    assert "₹29,100.00" in _region(page, "data-section", "expense")

    row = _row_with(detail, ">Dining<")
    assert "scope=" not in row, "the all-account detail's link claims an account scope"
    assert "non_inr=0" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _line_count(row) == 1
    assert "4,000.00" in listing, "the unscoped link did not reach the card swipe"
    assert "55.00" not in listing  # the detail is INR-or-null, and so is its link

    # The caveat, next to the figure it is about: these two do not reconcile, and a
    # reader who adds them double-counts.
    lowered = detail.lower()
    assert "not" in lowered and "reconcile" in lowered
    assert "swipes" in lowered


async def test_a_card_only_range_is_not_an_empty_range(client, session):
    """A range holding only card swipes has an expense detail to show, so a page
    that calls it empty contradicts the table it is printing directly below."""
    await _add(
        session,
        amount="4000",
        direction="debit",
        category="dining",
        account_id=await card_account(session),
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text
    assert "No transactions in this range" not in page
    detail = _region(page, "data-section", "expense_detail")
    assert "₹4,000.00" in detail
    # And the bank tiles are honestly zero: no card row reached them.
    assert "₹0.00" in _tile(page, "expense")


async def test_an_unaccounted_only_range_is_not_an_empty_range(client, session):
    """Same lie, other population: a range whose only rows are on no known account
    still has a footnote counting them, so it is not an empty range."""
    await _add(
        session, amount="800", direction="debit", category="rent", account_id=None
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text
    assert "No transactions in this range" not in page

    row = _region(page, "data-footnote", "unaccounted")
    assert _count(row) == 1
    listing = await _listed(client, row)
    assert listing.count(DETAIL) == 1
    assert "800.00" in listing


async def test_perimeter_caveat_carries_the_internal_net_and_its_drill(client, session):
    """`net_cash_retained` is not the change in the tracked bank balances, and the
    page has to say so where it prints that figure.

    Self-transfers inside the tracked accounts cancel, so a non-zero *signed* net is
    money that crossed the tracked perimeter — it left the figures above without
    being spent. The gross cannot show that (two opposite legs add up in it), which
    is why the caveat carries the net, its count and a drill a reader can follow.
    """
    await _add(session, amount="5000", direction="debit", category="self_transfer")
    await _add(session, amount="1000", direction="credit", category="self_transfer")
    await _add(session, amount="90000", direction="credit", category="salary")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    caveat = _region(page, "data-perimeter")

    lowered = caveat.lower()
    assert "not the change in your tracked bank balances" in lowered
    assert "2 internal movements" in caveat
    # The signed net, not the ₹6,000 gross the footnote prints.
    assert format_inr_exact(Decimal("-4000")) in caveat
    assert "₹6,000.00" not in caveat

    listing = await _listed(client, caveat)
    assert listing.count(DETAIL) == 2
    assert "5,000.00" in listing
    assert "1,000.00" in listing
    assert "90,000.00" not in listing


# ---------------------------------------------------------------------------
# The bank perimeter, link by link. A scope that is only tested where the seed
# data happens to be bank-only is not tested at all: the link would list the same
# rows with the scope stripped off it. So each test below seeds one row on *every*
# account a row can sit on, all carrying the figure's own filter — same category,
# same direction, same currency — and differing in nothing but the account. The
# two inside the perimeter must be listed and the four outside it must not, which
# is an assertion that fails the moment `scope=bank` leaves the href.
# ---------------------------------------------------------------------------

#: A debit card spends the bank's money, so it is *inside* the bank scope: the one
#: account type whose rows a bank figure counts without being a `bank_account`.
DEBIT_CARD_ID = 3
#: An account type the report knows nothing about — neither bank nor card.
UNKNOWN_TYPE_ID = 7

#: What the four out-of-perimeter rows below print in a listing. Each is unique and
#: none is a substring of another, so "not in listing" means the row is absent
#: rather than the digits being spelled somewhere else.
OUTSIDE = ("4,444.00", "3,333.00", "2,222.00", "1,111.00")


async def _seed_outside_the_bank(session, **row) -> None:
    """Seed the same row on each of the four accounts no bank figure may count.

    ``row`` is the figure's own filter — the category, direction and currency the
    figure selects on — so these rows differ from the ones it counts in the account
    alone. A credit card, an account type nothing recognizes, an ``account_id``
    naming no account row, and no account at all: between them they are every way a
    row can be out of the bank scope.
    """
    await _add(session, amount="4444", account_id=await card_account(session), **row)
    await _add(
        session,
        amount="3333",
        account_id=await ensure_account(session, UNKNOWN_TYPE_ID, "wallet"),
        **row,
    )
    await _add(session, amount="2222", account_id=MISSING_ACCOUNT_ID, **row)
    await _add(session, amount="1111", account_id=None, **row)


async def test_investment_line_drills_into_the_bank_perimeter_alone(client, session):
    """The Net Invested lines, followed as rendered.

    The four decoys are contributions like the two the line counts and sit outside
    the bank, so a link that lost its scope lists six rows under a count of two.
    """
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(
        session,
        amount="2500",
        direction="debit",
        category="investment",
        account_id=await ensure_account(session, DEBIT_CARD_ID, "debit_card"),
    )
    await _seed_outside_the_bank(session, direction="debit", category="investment")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _row_with(_region(page, "data-section", "investment"), ">Investment<")
    assert "scope=bank" in row
    assert "₹12,500.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _line_count(row) == 2
    assert "10,000.00" in listing
    # The debit card spends the bank's money, so its row is one the figure counted.
    assert "2,500.00" in listing, "the bank scope dropped the debit-card row"
    for amount in OUTSIDE:
        assert amount not in listing, (
            f"the investment link lists {amount}, out of scope"
        )


async def test_both_transfers_in_anchors_drill_into_the_bank_perimeter_alone(
    client, session
):
    """Transfers In prints its figure twice — once on the tile, once per counterparty
    line — so it has two anchors, and each is a place the scope can be lost alone.

    The four decoys are repayments too, so either link without its scope lists them.
    The second bank counterparty is what keeps the two anchors apart: the tile is the
    whole bucket and the line is one counterparty of it, and a line pointed at the
    tile's own filter would print "2" above a list of three.
    """
    await _add(
        session,
        amount="1500",
        direction="credit",
        category="repayment",
        counterparty="MOM",
    )
    await _add(
        session,
        amount="900",
        direction="credit",
        category="repayment",
        counterparty="MOM",
        account_id=await ensure_account(session, DEBIT_CARD_ID, "debit_card"),
    )
    await _add(
        session,
        amount="700",
        direction="credit",
        category="repayment",
        counterparty="DAD",
    )
    await _seed_outside_the_bank(
        session, direction="credit", category="repayment", counterparty="MOM"
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text

    # The tile: the whole bucket over the bank, its three rows and no others.
    tile = _tile(page, "transfers_in")
    assert "scope=bank" in tile
    tile_listing = await _listed(client, tile)
    assert tile_listing.count(DETAIL) == _count(tile) == 3
    for amount in ("1,500.00", "900.00", "700.00"):
        assert amount in tile_listing
    for amount in OUTSIDE:
        assert amount not in tile_listing, f"the transfers-in tile lists {amount}"

    # The line: one counterparty of that bucket, over the same perimeter.
    row = _row_with(_region(page, "data-section", "transfers_in"), ">MOM<")
    assert "scope=bank" in row
    assert "₹2,400.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _line_count(row) == 2
    assert "1,500.00" in listing
    assert "900.00" in listing, "the bank scope dropped the debit-card row"
    assert "700.00" not in listing  # the other counterparty is not this line's row
    for amount in OUTSIDE:
        assert amount not in listing, f"the transfers-in line lists {amount}"


async def test_non_inr_footnote_drills_into_the_bank_perimeter_alone(client, session):
    """The non-INR footnote counts the foreign rows the rupee buckets left out — of
    the *bank*, because those are the buckets it is a footnote to.

    The decoys are foreign rows on the other accounts: they were never in a bank
    bucket, so they are not what this footnote is excusing, and an unscoped link
    would list them beneath a count that never had them.
    """
    await _add(
        session, amount="100", direction="debit", category="rent", currency="USD"
    )
    await _add(
        session,
        amount="60",
        direction="debit",
        category="dining",
        currency="EUR",
        account_id=await ensure_account(session, DEBIT_CARD_ID, "debit_card"),
    )
    await _seed_outside_the_bank(
        session, direction="debit", category="dining", currency="USD"
    )
    # And the rupee row the footnote is not about, on the perimeter it is about.
    await _add(session, amount="20000", direction="debit", category="rent")

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _region(page, "data-footnote", "non_inr")
    assert "scope=bank" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 2
    assert "100.00" in listing
    assert "60.00" in listing, "the bank scope dropped the debit-card row"
    assert "20,000.00" not in listing
    for amount in OUTSIDE:
        assert amount not in listing, f"the non-INR footnote lists {amount}"
