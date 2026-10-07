"""HTML page tests for /cashflow.

The page server-renders every figure and every drill-through link. Each test
reads the element under test, follows the href rendered inside it, and counts
what comes back. A figure and its drill-through then cannot drift apart without
a failure here.
"""

import datetime
import json
import re
from contextlib import contextmanager
from decimal import Decimal

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from financial_dashboard.db.models import Transaction
from tests.conftest import (
    MISSING_ACCOUNT_ID,
    bank_account,
    card_account,
    ensure_account,
)

pytestmark = pytest.mark.anyio


RANGE = "date_from=2026-06-01&date_to=2026-06-30"

# The month the seed helper writes into by default.
SEED_MONTH = datetime.date(2026, 6, 1)

# The listing renders one "/detail" link per row, so counting them counts rows.
DETAIL = "/detail"

EMPTY = "No transactions in this range"


def _tile(page: str, name: str) -> str:
    """The markup of one headline tile.

    A tile's compact figure is not unique on the page, so an assertion has to be
    scoped to the tile itself or it would still pass with the tile deleted.
    """
    match = re.search(
        rf'<article[^>]*data-tile="{name}".*?</article>', page, flags=re.DOTALL
    )
    assert match, f"no tile named {name!r} on the page"
    return match.group(0)


def _region(page: str, attribute: str, name: str | None = None) -> str:
    """The markup of one server-rendered table/footnote/footer region."""
    selector = f'{attribute}="{name}"' if name else attribute
    match = re.search(
        rf"<(article|tr|section)[^>]*{selector}.*?</\1>", page, flags=re.DOTALL
    )
    assert match, f"no region matching {selector!r} on the page"
    return match.group(0)


def _href(markup: str) -> str:
    """The single anchor inside one rendered row/tile, as a followable URL."""
    match = re.search(r'href="([^"]+)"', markup)
    assert match, f"no anchor in {markup!r}"
    return match.group(1).replace("&amp;", "&")


def _month_start(today: datetime.date, back: int) -> datetime.date:
    """The first of the month ``back`` months before ``today``'s month."""
    absolute = today.year * 12 + (today.month - 1) - back
    return datetime.date(absolute // 12, absolute % 12 + 1, 1)


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
    """Every amount a /transactions listing prints, signed by its own direction."""
    cells = re.findall(
        r'class="amt amt-\w+">(&minus;|\+)[^\d]*([\d,]+\.\d{2})</td>', listing
    )
    return [
        -Decimal(body.replace(",", ""))
        if sign == "&minus;"
        else Decimal(body.replace(",", ""))
        for sign, body in cells
    ]


async def _get(client, href: str) -> str:
    """Follow a rendered href exactly as a browser would."""
    r = await client.get(href.replace("&amp;", "&"))
    assert r.status_code == 200
    return r.text


async def _listed(client, markup: str) -> str:
    """Follow the drill-through rendered inside ``markup`` and return the listing."""
    return await _get(client, _href(markup))


def _lines(page: str, tile: str) -> list[tuple[str, str]]:
    """The (key, href) of every drill anchor the given tile's lines rendered.

    The breakdown chart's bars take these hrefs, so what they point at is what
    the bars point at.
    """
    return re.findall(rf'<a data-line="{tile}" data-key="([^"]*)" href="([^"]*)"', page)


#: ``_add`` links the row to the bank account. Anything else — a card, or no
#: account at all — is the caller's explicit choice, because it changes which of
#: the page's figures the row can reach.
LINKED_TO_BANK = object()


async def _add(
    session,
    *,
    amount,
    direction,
    category,
    day=15,
    counterparty=None,
    currency="INR",
    dated=True,
    month=SEED_MONTH,
    account_id=LINKED_TO_BANK,
    exclude_from_cashflow=False,
):
    """Seed one transaction the page can count.

    The page's figures are bank-scoped, so a row with no account is *unaccounted*
    and lands in none of them. Linking by default keeps each test below about the
    figure it names rather than about the linker.
    """
    if account_id is LINKED_TO_BANK:
        account_id = await bank_account(session)
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            amount=Decimal(amount),
            direction=direction,
            category=category,
            counterparty=counterparty,
            currency=currency,
            transaction_date=month.replace(day=day) if dated else None,
            account_id=account_id,
            exclude_from_cashflow=exclude_from_cashflow,
        )
    )
    await session.commit()


async def test_empty_range_keeps_zero_tiles_and_the_trend(client, session):
    """An empty range still shows five zero tiles and the empty-state card.

    The trend is a trailing-12-month series, so the selected range does not
    scope it. Only the range-scoped parts go away.
    """
    # Last month is always inside the trend window. The month before it is an
    # empty range inside that window.
    today = datetime.date.today()
    history = _month_start(today, 1)
    await _add(
        session,
        amount="90000",
        direction="credit",
        category="salary",
        month=history,
    )
    empty = _month_start(today, 2)
    end = history - datetime.timedelta(days=1)
    page = (
        await client.get(
            f"/cashflow?date_from={empty.isoformat()}&date_to={end.isoformat()}"
        )
    ).text

    assert EMPTY in page
    names = ["income", "expense", "net_invested", "transfers_in", "uncategorized"]
    assert re.findall(r'data-tile="([a-z_]+)"', page) == names
    for name in names:
        assert "₹0.00" in _tile(page, name), f"{name} tile does not read zero"

    api = await client.get("/api/cashflow/trend?months=12")
    seeded = f"{history.year:04d}-{history.month:02d}"
    assert [p["month"] for p in api.json() if Decimal(p["income"])] == [seeded]
    assert 'id="cf-trend"' in _region(page, "data-trend")
    assert "data-reconciliation" not in page
    assert 'id="cf-breakdown"' not in page


@pytest.mark.parametrize("population", ["card", "unaccounted", "non_inr"])
async def test_a_range_of_rows_outside_the_bank_tiles_is_not_empty(
    client, session, population
):
    """A row no bank tile counts still shows in a table or footnote on the page,
    so the page must not call the range empty."""
    row = {"amount": "800", "direction": "debit", "category": "rent"}
    if population == "card":
        row["account_id"] = await card_account(session)
    elif population == "unaccounted":
        row["account_id"] = None
    else:
        row["currency"] = "USD"
    await _add(session, **row)

    page = (await client.get(f"/cashflow?{RANGE}")).text
    assert EMPTY not in page
    assert "₹0.00" in _tile(page, "expense")


async def test_tiles_tables_and_footer_show_the_summary_figures(client, session):
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="2000", direction="credit", category="repayment")
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(session, amount="500", direction="credit", category="refund")
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(session, amount="4000", direction="credit", category="investment")
    # Excluded populations must not move the identity.
    await _add(session, amount="5000", direction="debit", category="self_transfer")
    await _add(session, amount="70", direction="debit", category="rent", currency="USD")
    # Uncategorized rows move no term. The footer prints their net as its error bar.
    await _add(session, amount="1234", direction="debit", category=None)
    await _add(session, amount="4000", direction="debit", category="crypto_yield")

    page = (await client.get(f"/cashflow?{RANGE}")).text

    # Tiles carry the compact figure: expense is 20,000 - 500 contra.
    income = _tile(page, "income")
    assert "₹90K" in income
    assert _count(income) == 1
    assert "₹19.5K" in _tile(page, "expense")
    invested = _tile(page, "net_invested")
    assert _count(invested) == 2
    assert "₹10K" in invested
    assert "₹4K" in invested

    # The tables carry the exact figure.
    assert "₹90,000.00" in _region(page, "data-section", "income")
    assert "₹19,500.00" in _region(page, "data-section", "expense")
    assert "₹6,000.00" in _region(page, "data-section", "investment")

    # 90,000 + 2,000 - 19,500 - 6,000 = 66,500.
    footer = _region(page, "data-reconciliation")
    for term in ("₹90,000.00", "₹2,000.00", "₹19,500.00", "₹6,000.00"):
        assert term in footer
    assert "net cash retained ₹66,500.00" in footer
    assert "₹5,000.00" not in footer
    assert "70.00" not in footer
    assert "uncategorized -₹5,234.00" in footer


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


async def test_internal_footnote_and_perimeter_caveat_list_self_transfers_alone(
    client, session
):
    """Under the bank scope the internal footnote counts self-transfers alone. A
    card bill is money that leaves the bank, so it is the Card bills expense line.

    The perimeter caveat carries the signed internal net: a non-zero net is money
    that crossed the tracked perimeter without being spent.
    """
    await _add(session, amount="5000", direction="debit", category="self_transfer")
    await _add(session, amount="1000", direction="credit", category="self_transfer")
    await _add(
        session, amount="9100", direction="debit", category="credit_card_payment"
    )
    await _add(session, amount="90000", direction="credit", category="salary")
    # A card-side self-transfer is out of the bank scope.
    await _add(
        session,
        amount="3700",
        direction="debit",
        category="self_transfer",
        account_id=await card_account(session),
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text
    row = _region(page, "data-footnote", "internal")
    # Gross, not net: the two legs add up rather than cancel.
    assert "₹6,000.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 2
    assert "9,100.00" not in listing, (
        "the internal drill lists the card bill the footnote counts as expense"
    )
    assert "90,000.00" not in listing
    assert "3,700.00" not in listing, "the internal drill lists a card row"

    bills = _row_with(_region(page, "data-section", "expense"), ">Card bills<")
    bill_listing = await _listed(client, bills)
    assert bill_listing.count(DETAIL) == _line_count(bills) == 1
    assert "9,100.00" in bill_listing

    caveat = _region(page, "data-perimeter")
    assert "2 internal movements" in caveat
    assert "-₹4,000.00" in caveat
    assert "₹6,000.00" not in caveat
    caveat_listing = await _listed(client, caveat)
    assert caveat_listing.count(DETAIL) == 2
    assert "90,000.00" not in caveat_listing
    assert "3,700.00" not in caveat_listing


async def test_excluded_and_undated_footnotes_count_what_the_buckets_drop(
    client, session
):
    """A flagged row leaves every figure above and the Excluded footnote counts it.
    An undated row is in no range, so the Undated footnote counts it with a
    rangeless drill, flagged or not."""
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="5000", direction="debit", category="dining")
    await _add(
        session,
        amount="24995",
        direction="debit",
        category="shopping",
        exclude_from_cashflow=True,
    )
    await _add(
        session,
        amount="333",
        direction="debit",
        category="rent",
        dated=False,
        exclude_from_cashflow=True,
    )
    await _add(session, amount="90", direction="credit", category="salary", dated=False)

    page = (await client.get(f"/cashflow?{RANGE}")).text

    expense = _region(page, "data-section", "expense")
    assert "₹5,000.00" in expense
    assert "24,995" not in expense
    assert "₹90,000.00" in _region(page, "data-section", "income")

    excluded = _region(page, "data-footnote", "excluded")
    listing = await _listed(client, excluded)
    assert listing.count(DETAIL) == _count(excluded) == 1
    assert "24,995.00" in listing
    assert "5,000.00" not in listing
    assert "90,000.00" not in listing

    undated = _region(page, "data-footnote", "undated")
    # Signed net: a 90 credit against a 333 debit.
    assert "-₹243.00" in undated
    listing = await _listed(client, undated)
    assert listing.count(DETAIL) == _count(undated) == 2
    assert "333.00" in listing
    assert "24,995.00" not in listing


async def test_family_excluded_from_buckets_and_counted_in_its_footnote(
    client, session
):
    """A `family` row is excluded from income and expense and counted only in the
    family footnote, whose drill lists exactly those rows."""
    # Decoy amounts are chosen so none is a substring of a family amount.
    await _add(session, amount="40000", direction="credit", category="family")
    await _add(session, amount="13000", direction="debit", category="family")
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="6500", direction="debit", category="rent")
    # A family row outside the range.
    await _add(
        session,
        amount="3100",
        direction="debit",
        category="family",
        month=datetime.date(2026, 7, 1),
    )

    page = (await client.get(f"/cashflow?{RANGE}")).text

    # Excluded from the money buckets entirely.
    assert "Family" not in _region(page, "data-section", "income")
    assert "Family" not in _region(page, "data-section", "expense")

    # Counted in its own footnote: signed net +40,000 credit -13,000 debit.
    row = _region(page, "data-footnote", "family")
    assert "₹27,000.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 2
    # The salary/rent decoys are not in the family drill.
    assert "90,000.00" not in listing
    assert "6,500.00" not in listing
    assert "3,100.00" not in listing

    # Family is not "internal": it must not inflate the internal footnote (that
    # figure only holds self-transfers, which family flows are not).
    assert _count(_region(page, "data-footnote", "internal")) == 0


async def test_reimbursement_credit_nets_against_spend(client, session):
    """A reimbursement credit reduces Spent (contra-expense) and is not income."""
    await _add(session, amount="5000", direction="debit", category="dining")
    await _add(session, amount="4000", direction="credit", category="reimbursement")
    # An income decoy so the income section renders (to prove reimbursement is not
    # in it).
    await _add(session, amount="90000", direction="credit", category="salary")

    page = (await client.get(f"/cashflow?{RANGE}")).text

    expense = _region(page, "data-section", "expense")
    # 5,000 spent minus 4,000 reimbursed nets to 1,000.
    assert "₹1,000.00" in expense
    assert "Reimbursement" in expense
    # Not income.
    assert "Reimbursement" not in _region(page, "data-section", "income")


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
    # Signed net: a 60 credit against 800 + 250 of debits.
    assert "-₹990.00" in row

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 3
    assert sorted(_listed_amounts(listing)) == [
        Decimal("-800"),
        Decimal("-250"),
        Decimal("60"),
    ]
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

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _line_count(row) == 1
    assert "4,000.00" in listing, "the unscoped link did not reach the card swipe"
    assert "55.00" not in listing  # the detail is INR-or-null, and so is its link


# The bank perimeter, link by link. Each test seeds one row on every account a
# row can sit on, all with the figure's own filter. The two inside the perimeter
# must be listed and the four outside it must not.

#: A debit card spends the bank's money, so it is inside the bank scope.
DEBIT_CARD_ID = 3
#: An account type the report knows nothing about.
UNKNOWN_TYPE_ID = 7

#: What the four out-of-perimeter rows print in a listing. None is a substring
#: of another.
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
    tile_listing = await _listed(client, tile)
    assert tile_listing.count(DETAIL) == _count(tile) == 3
    for amount in ("1,500.00", "900.00", "700.00"):
        assert amount in tile_listing
    for amount in OUTSIDE:
        assert amount not in tile_listing, f"the transfers-in tile lists {amount}"

    # The line: one counterparty of that bucket, over the same perimeter.
    row = _row_with(_region(page, "data-section", "transfers_in"), ">MOM<")
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

    listing = await _listed(client, row)
    assert listing.count(DETAIL) == _count(row) == 2
    assert "100.00" in listing
    assert "60.00" in listing, "the bank scope dropped the debit-card row"
    assert "20,000.00" not in listing
    for amount in OUTSIDE:
        assert amount not in listing, f"the non-INR footnote lists {amount}"


async def test_every_tiles_breakdown_lines_drill_to_exactly_their_own_rows(
    client, session
):
    """Each selectable tile's bars carry that line's own drill-through predicate.

    Every seeded row is a decoy for every other line, so an unfiltered — or
    wrongly-filtered — link cannot pass: each href is followed and must list its
    own rows and no others.
    """
    await _add(session, amount="90000", direction="credit", category="salary")
    # A foreign row no rupee bucket summed: a core link that lists it contradicts
    # the figure it sits under.
    await _add(
        session, amount="555", direction="credit", category="salary", currency="USD"
    )
    await _add(
        session,
        amount="20000",
        direction="debit",
        category="rent",
        counterparty="Alice",
    )
    # Card rows carry the same slugs but sit outside the bank scope.
    card = await card_account(session)
    await _add(
        session, amount="4444", direction="credit", category="salary", account_id=card
    )
    await _add(
        session, amount="777", direction="debit", category="rent", account_id=card
    )
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(session, amount="4000", direction="credit", category="investment")
    await _add(
        session,
        amount="1500",
        direction="credit",
        category="repayment",
        counterparty="Alice",
    )
    await _add(
        session,
        amount="333",
        direction="credit",
        category="repayment",
        counterparty="Alice",
        currency="USD",
    )
    await _add(session, amount="700", direction="credit", category="repayment")
    await _add(session, amount="800", direction="debit", category=None)
    await _add(session, amount="600", direction="debit", category="")
    await _add(session, amount="200", direction="debit", category="unknown")
    await _add(session, amount="90", direction="debit", category="crypto")

    r = await client.get(f"/cashflow?{RANGE}")
    assert _count(_tile(r.text, "income")) == 1

    # tile -> its lines, each as (key, the rows the link must list, the rows it
    # must not).
    expected = {
        "income": {"salary": (["90,000.00"], ["555.00", "4,444.00", "20,000.00"])},
        "expense": {"rent": (["20,000.00"], ["777.00", "90,000.00", "10,000.00"])},
        "net_invested": {
            "investment:contribution": (["10,000.00"], ["4,000.00"]),
            "investment:redemption": (["4,000.00"], ["10,000.00"]),
        },
        "transfers_in": {
            "Alice": (["1,500.00"], ["333.00", "700.00", "20,000.00"]),
            "": (["700.00"], ["1,500.00"]),
        },
        "uncategorized": {
            # NULL and empty-string are one line, so its link lists both rows.
            "": (["800.00", "600.00"], ["200.00", "90.00", "20,000.00"]),
            "unknown": (["200.00"], ["800.00", "600.00", "90.00"]),
            "crypto": (["90.00"], ["800.00", "200.00"]),
        },
    }

    for tile, lines in expected.items():
        rendered = _lines(r.text, tile)
        assert {key for key, _ in rendered} == set(lines), (
            f"{tile} rendered lines {rendered}"
        )
        for key, href in rendered:
            listed = await _get(client, href)
            present, absent = lines[key]
            for amount in present:
                assert amount in listed, f"{tile}/{key or '(blank)'} lost {amount}"
            for amount in absent:
                assert amount not in listed, (
                    f"{tile}/{key or '(blank)'} listed {amount}"
                )
            assert listed.count("/detail") == len(present)

    # NULL, blank, 'unknown' and unmapped rows form one bucket of four.
    assert "4 txns" in _tile(r.text, "uncategorized")
    everything = await _get(client, f"/transactions?uncategorized=1&{RANGE}")
    assert everything.count("/detail") == 4


# Text a counterparty can really carry, every character of which is a way out of a
# <script> block or out of the JSON string inside it.
HOSTILE = (
    "</script><script>alert(1)</script>"
    "<!-- --> & \"quoted\" 'single' \\ </SCRIPT >"
    # U+2028/U+2029 terminate a JS line but not a JSON string.
    "\u2028\u2029"
)


async def test_embedded_json_cannot_break_out_of_its_script_block(client, session):
    """The summary is embedded in the page as a JSON island, and a counterparty is
    text a bank hands us — so it is text an attacker can influence. The escaping has
    to survive it: no premature close of the <script>, and the value the chart parses
    back has to be the text that went in, byte for byte.

    U+2028/U+2029 are in there because they terminate a line in JavaScript but not in
    JSON: a serializer that leaves them raw produces a page that is valid JSON inside
    a <script> that no longer parses.
    """
    await _add(
        session,
        amount="1500",
        direction="credit",
        category="repayment",
        counterparty=HOSTILE,
    )

    r = await client.get(f"/cashflow?{RANGE}")
    assert r.status_code == 200

    block = re.search(
        r'<script type="application/json" id="cf-summary">(.*?)</script>',
        r.text,
        flags=re.DOTALL,
    )
    assert block, "no summary island on the page"

    # Nothing in the payload closed the block early: what the regex captured is the
    # whole island, and the hostile close tag is not sitting in the document raw.
    assert "</script>" not in block.group(1)
    assert "<script>alert(1)</script>" not in r.text
    assert "<!--" not in block.group(1)

    # The escaping is lossless, not lossy: the chart reads back exactly what a bank
    # sent, so the defence cannot be quietly costing the reader their data.
    payload = json.loads(block.group(1))
    # The island is the API's own shape, so the chart reads one contract.
    assert payload == (await client.get(f"/api/cashflow/summary?{RANGE}")).json()
    counterparties = [line["counterparty"] for line in payload["transfers_in"]["lines"]]
    assert counterparties == [HOSTILE]

    # The line separators are escaped *inside the island*, which is the only place
    # they are dangerous: they end a line in JavaScript, so a raw one there closes
    # nothing but breaks the parse. In the HTML body they are ordinary text, and the
    # table below the chart is free to carry them.
    assert "\u2028" not in block.group(1)
    assert "\u2029" not in block.group(1)


# The summary: the grouped bank-side bucket scan, the all-account expense
# detail, transfers-in, uncategorized, and the six footnote reads.
SUMMARY_QUERIES = 10
# The cash bridge: one ordered read of the bank rows.
BRIDGE_QUERIES = 1
# The trend: the month/category/direction scan and the salary counts.
TREND_QUERIES = 2


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


async def test_page_load_is_one_summary_plus_trend(client, session):
    """The page aggregates the range once. The breakdown chart reads the summary
    from the page, the bridge reuses it, and the trend is the only fetch."""
    await _add(session, amount="90000", direction="credit", category="salary")
    with count_transaction_reads() as page_queries:
        page = await client.get(f"/cashflow?{RANGE}")
    assert page.status_code == 200
    assert len(page_queries) == SUMMARY_QUERIES + BRIDGE_QUERIES
    assert "/api/cashflow/summary" not in page.text

    with count_transaction_reads() as trend_queries:
        trend = await client.get("/api/cashflow/trend?months=12")
    assert trend.status_code == 200
    assert len(trend_queries) == TREND_QUERIES
