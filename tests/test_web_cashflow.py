"""HTML page tests for /cashflow.

The page server-renders every figure and every drill-through link, so the
assertions here read the rendered markup rather than the JSON the charts
hydrate from: the charts are progressive enhancement over what these tests
already prove is on the page.
"""

import datetime
import json
import re
from decimal import Decimal

import pytest

from financial_dashboard.db.models import Transaction
from tests.conftest import bank_account

pytestmark = pytest.mark.anyio


async def _chart_js(client) -> str:
    """The chart code the page loads, fetched the way the browser fetches it.

    The charts live in static ES modules rather than inline in the template, so an
    assertion about what the chart *does* has to read the module — the page itself
    only carries the hooks and the data. Going through the app rather than the
    filesystem means a module the app does not actually serve — renamed, moved out
    from under the mount, deleted — fails here instead of passing on a file the
    browser would never reach.
    """
    sources = []
    for name in ("cashflow.js", "charts.js"):
        r = await client.get(f"/static/js/{name}")
        assert r.status_code == 200, f"the page loads /static/js/{name}, app 404s it"
        sources.append(r.text)
    return "".join(sources)


# The query string as requested, and as it appears inside a rendered href,
# where the separator is the "&amp;" entity.
RANGE = "date_from=2026-06-01&date_to=2026-06-30"
RANGE_HTML = "date_from=2026-06-01&amp;date_to=2026-06-30"
# The rupee buckets sum INR-and-null rows only, so their drill-throughs pin the
# currency; the currency-agnostic links (uncategorized, internal, undated) do not.
CORE_HTML = f"{RANGE_HTML}&amp;non_inr=0"
CORE = f"{RANGE}&non_inr=0"

# The month the seed helper writes into by default. Tests that pass an explicit
# range to the report pin both ends themselves, so a fixed month is stable for
# them; anything reading a window derived from today() must seed relative to
# today instead — see `_month_start`.
SEED_MONTH = datetime.date(2026, 6, 1)


def _month_start(today: datetime.date, back: int) -> datetime.date:
    """The first of the month ``back`` months before ``today``'s month."""
    absolute = today.year * 12 + (today.month - 1) - back
    return datetime.date(absolute // 12, absolute % 12 + 1, 1)


def _tile(page: str, name: str) -> str:
    """The markup of one headline tile.

    A tile's compact figure ("₹90K") is not unique on the page, and its exact
    twin sits in the table below it, so an assertion has to be scoped to the tile
    itself or it would still pass with the tile deleted.
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


def _links(page: str, prefix: str) -> list[str]:
    return re.findall(rf'href="({re.escape(prefix)}[^"]*)"', page)


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
        )
    )
    await session.commit()


async def test_cashflow_page_renders_default_range(client):
    r = await client.get("/cashflow")
    assert r.status_code == 200
    assert "Cashflow" in r.text
    # The default range is resolved server-side and pinned into the form.
    today = datetime.date.today()
    assert f'value="{today.replace(day=1).isoformat()}"' in r.text
    assert f'value="{today.isoformat()}"' in r.text


async def test_cashflow_page_renders_tiles_and_lines(client, session):
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(session, amount="500", direction="credit", category="refund")
    await _add(session, amount="10000", direction="debit", category="investment")

    r = await client.get(f"/cashflow?{RANGE}")
    assert r.status_code == 200

    # Tiles carry the compact figure, each asserted inside its own tile: income
    # 90,000; expense 20,000 - 500 contra = 19,500; net invested 10,000.
    assert "₹90K" in _tile(r.text, "income")
    assert "1 txns" in _tile(r.text, "income")
    assert "₹19.5K" in _tile(r.text, "expense")
    assert "₹10K" in _tile(r.text, "net_invested")

    # The tables carry the exact figure.
    assert "₹90,000.00" in _region(r.text, "data-section", "income")
    assert "₹19,500.00" in _region(r.text, "data-section", "expense")
    assert "₹10,000.00" in _region(r.text, "data-section", "investment")

    # Category lines drill through, range-scoped.
    assert f"/transactions?category=salary&amp;{CORE_HTML}" in r.text
    assert f"/transactions?category=rent&amp;{CORE_HTML}" in r.text
    assert (
        f"/transactions?category=investment&amp;direction=debit&amp;{CORE_HTML}"
        in r.text
    )
    # The contra credit is a negative expense line, not an income line.
    assert "Refund" in r.text


async def test_headline_tiles_are_exactly_the_five_specified(client, session):
    """Net cash retained is a derived identity, not a headline: it reads in the
    reconciliation footer that shows the terms it comes from."""
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(session, amount="10000", direction="debit", category="investment")

    r = await client.get(f"/cashflow?{RANGE}")
    assert re.findall(r'data-tile="([a-z_]+)"', r.text) == [
        "income",
        "expense",
        "net_invested",
        "transfers_in",
        "uncategorized",
    ]

    # 90,000 - 20,000 - 10,000, exact and spelled out from its terms.
    footer = _region(r.text, "data-reconciliation")
    assert "net cash retained ₹60,000.00" in footer
    assert "₹90,000.00" in footer
    assert "₹20,000.00" in footer
    assert "₹10,000.00" in footer


async def test_exact_amounts_use_indian_digit_grouping(client, session):
    """Above a lakh the grouping is the whole point: ₹12,34,567.89, never the
    Western ₹1,234,567.89 that a plain "{:,}" would print."""
    await _add(session, amount="1234567.89", direction="credit", category="salary")

    r = await client.get(f"/cashflow?{RANGE}")
    assert "₹12,34,567.89" in _region(r.text, "data-section", "income")
    assert "₹12,34,567.89" in _region(r.text, "data-reconciliation")
    assert "1,234,567.89" not in r.text


async def test_uncategorized_caveat_sits_with_its_own_figure(client, session):
    """Each rough figure carries the caveat next to it — the phrase appearing
    somewhere else on the page is not the same promise."""
    await _add(session, amount="800", direction="debit", category=None)
    await _add(session, amount="5000", direction="debit", category="self_transfer")
    await _add(session, amount="333", direction="debit", category="rent", dated=False)

    r = await client.get(f"/cashflow?{RANGE}")
    assert "may mix currencies" in _tile(r.text, "uncategorized")
    assert "may mix currencies" in _region(r.text, "data-section", "uncategorized")
    assert "may mix currencies" in _region(r.text, "data-footnote", "internal")
    assert "may mix currencies" in _region(r.text, "data-footnote", "undated")


async def test_investment_lines_show_contribution_and_redemption_kinds(client, session):
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(
        session, amount="4000", direction="credit", category="investment_redemption"
    )

    r = await client.get(f"/cashflow?{RANGE}")
    assert "contribution" in r.text
    assert "redemption" in r.text
    # Net invested = 10,000 contributed - 4,000 redeemed.
    assert "₹6,000.00" in r.text


async def test_investment_links_carry_direction_to_split_the_slug(client, session):
    """One slug, two lines: only the direction tells the contribution from the
    redemption, so a link without it lists both and contradicts its own figure."""
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(session, amount="4000", direction="credit", category="investment")

    r = await client.get(f"/cashflow?{RANGE}")
    assert (
        f"/transactions?category=investment&amp;direction=debit&amp;{RANGE_HTML}"
        in r.text
    )
    assert (
        f"/transactions?category=investment&amp;direction=credit&amp;{RANGE_HTML}"
        in r.text
    )

    # Following the contribution link must list the debit alone.
    listed = await client.get(
        f"/transactions?category=investment&direction=debit&{RANGE}"
    )
    assert "10,000.00" in listed.text
    assert "4,000.00" not in listed.text


async def test_transfers_in_link_is_scoped_to_repayments(client, session):
    """The line groups *repayment* rows by counterparty; counterparty alone would
    list everything else that person appears on."""
    await _add(
        session,
        amount="1500",
        direction="credit",
        category="repayment",
        counterparty="Alice",
    )
    await _add(
        session,
        amount="9999",
        direction="debit",
        category="groceries",
        counterparty="Alice",
    )

    r = await client.get(f"/cashflow?{RANGE}")
    assert (
        f"/transactions?category=repayment&amp;counterparty=Alice&amp;{RANGE_HTML}"
        in r.text
    )

    listed = await client.get(
        f"/transactions?category=repayment&counterparty=Alice&{RANGE}"
    )
    assert "1,500.00" in listed.text
    assert "9,999.00" not in listed.text


async def test_non_inr_only_range_is_not_an_empty_range(client, session):
    """No bucket sums a foreign-currency row, but the page must not claim the
    range is empty while the footnote below it counts that very row."""
    await _add(
        session, amount="100", direction="debit", category="rent", currency="USD"
    )

    r = await client.get(f"/cashflow?{RANGE}")
    assert "No transactions in this range" not in r.text
    assert f"/transactions?non_inr=1&amp;{RANGE_HTML}" in r.text


async def test_transfers_in_tile_and_counterparty_drill(client, session):
    await _add(
        session,
        amount="1500",
        direction="credit",
        category="repayment",
        counterparty="Alice",
    )
    await _add(session, amount="700", direction="credit", category="repayment")

    r = await client.get(f"/cashflow?{RANGE}")
    assert (
        f"/transactions?category=repayment&amp;counterparty=Alice&amp;{RANGE_HTML}"
        in r.text
    )
    # A blank counterparty is its own filter value, not an omitted param.
    assert (
        f"/transactions?category=repayment&amp;counterparty=&amp;{RANGE_HTML}" in r.text
    )
    assert "₹2,200.00" in r.text


async def test_uncategorized_drill_stays_currency_agnostic(client, session):
    """The uncategorized total sums every currency, so its link must not pin one:
    the tile's count and the listed row count have to agree."""
    await _add(session, amount="800", direction="debit", category=None)
    await _add(session, amount="200", direction="debit", category="unknown")
    await _add(session, amount="7", direction="debit", category=None, currency="USD")

    r = await client.get(f"/cashflow?{RANGE}")
    assert f"/transactions?uncategorized=1&amp;{RANGE_HTML}" in r.text
    assert "non_inr=0" not in _tile(r.text, "uncategorized")
    assert "3 txns" in _tile(r.text, "uncategorized")

    listed = await client.get(f"/transactions?uncategorized=1&{RANGE}")
    assert listed.text.count("/detail") == 3


async def test_core_drill_links_exclude_the_foreign_row_their_totals_exclude(
    client, session
):
    """A rupee bucket that lists a row it never summed contradicts itself: the
    USD salary is absent from the tile's ₹ figure, so it must be absent from the
    list the tile links to."""
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(
        session, amount="500", direction="credit", category="salary", currency="USD"
    )
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(
        session, amount="700", direction="debit", category="rent", currency="USD"
    )
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(
        session,
        amount="800",
        direction="debit",
        category="investment",
        currency="USD",
    )

    r = await client.get(f"/cashflow?{RANGE}")
    # The tiles count only the rupee rows they sum.
    assert "1 txns" in _tile(r.text, "income")
    assert "1 txns" in _tile(r.text, "expense")

    # Every rupee-bucket link pins the currency; follow each rendered href and
    # the foreign row is not in the list.
    for slug, foreign, rupee in (
        ("salary", "500.00", "90,000.00"),
        ("rent", "700.00", "20,000.00"),
        ("investment", "800.00", "10,000.00"),
    ):
        hrefs = _links(r.text, f"/transactions?category={slug}")
        assert hrefs, f"no drill-through rendered for {slug}"
        for href in hrefs:
            assert "non_inr=0" in href
            listed = await client.get(href.replace("&amp;", "&"))
            assert listed.status_code == 200
            assert rupee in listed.text
            assert foreign not in listed.text
            assert listed.text.count("/detail") == 1


async def test_transfers_in_drill_excludes_the_foreign_row_its_total_excludes(
    client, session
):
    await _add(
        session,
        amount="1500",
        direction="credit",
        category="repayment",
        counterparty="Alice",
    )
    await _add(
        session,
        amount="900",
        direction="credit",
        category="repayment",
        counterparty="Alice",
        currency="USD",
    )

    r = await client.get(f"/cashflow?{RANGE}")
    assert "1 txns" in _tile(r.text, "transfers_in")
    assert "₹1,500.00" in _region(r.text, "data-section", "transfers_in")
    assert (
        f"/transactions?category=repayment&amp;counterparty=Alice&amp;{CORE_HTML}"
        in r.text
    )

    listed = await client.get(
        f"/transactions?category=repayment&counterparty=Alice&{CORE}"
    )
    assert "1,500.00" in listed.text
    assert "900.00" not in listed.text
    assert listed.text.count("/detail") == 1


async def test_footnotes_drill_through_with_undated_unscoped(client, session):
    await _add(session, amount="5000", direction="debit", category="self_transfer")
    await _add(
        session, amount="100", direction="debit", category="rent", currency="USD"
    )
    await _add(session, amount="333", direction="debit", category="rent", dated=False)

    r = await client.get(f"/cashflow?{RANGE}")
    assert f"/transactions?internal=1&amp;{RANGE_HTML}" in r.text
    assert f"/transactions?non_inr=1&amp;{RANGE_HTML}" in r.text
    # Undated rows match no range by definition, so their link carries none.
    assert '/transactions?undated=1"' in r.text
    assert f"/transactions?undated=1&amp;{RANGE_HTML}" not in r.text


async def test_undated_link_actually_lists_the_undated_row(client, session):
    """The Undated link is unscoped because a scoped one would list nothing.

    Following both proves it: the range-scoped URL misses the row the footnote
    counts, which is the whole reason that one link omits the range.
    """
    await _add(session, amount="333", direction="debit", category="rent", dated=False)

    listed = await client.get("/transactions?undated=1")
    assert listed.status_code == 200
    assert "333.00" in listed.text

    scoped = await client.get(f"/transactions?undated=1&{RANGE}")
    assert "333.00" not in scoped.text


async def test_category_link_lists_only_that_category(client, session):
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(session, amount="777", direction="debit", category="groceries")

    listed = await client.get(f"/transactions?category=rent&{RANGE}")
    assert listed.status_code == 200
    assert "20,000.00" in listed.text
    assert "777.00" not in listed.text


async def test_empty_range_shows_empty_state(client):
    r = await client.get(f"/cashflow?{RANGE}")
    assert r.status_code == 200
    assert "No transactions in this range" in r.text


async def test_breakdown_hydrates_from_the_page_not_a_second_summary_query(
    client, session
):
    """The breakdown draws the range the page already aggregated, so the summary is
    serialized into the page and the chart reads it there.

    Re-fetching /api/cashflow/summary would be the same eight aggregate queries run a
    second time to recompute figures that are already in this response, so the page
    must not carry that URL at all. The trend *is* fetched: its window is the
    trailing twelve months, which is not the selected range and so is genuinely not
    on the page."""
    await _add(session, amount="90000", direction="credit", category="salary")

    r = await client.get(f"/cashflow?{RANGE}")
    assert "/api/cashflow/summary" not in r.text
    assert "/api/cashflow/trend?months=12" in r.text

    block = re.search(
        r'<script type="application/json" id="cf-summary">(.*?)</script>', r.text
    )
    assert block, "the breakdown chart has no summary to draw"
    payload = json.loads(block.group(1))

    # The payload is the API's own shape, so the chart reads one contract whichever
    # way the numbers reached it.
    api = await client.get(f"/api/cashflow/summary?{RANGE}")
    assert payload == api.json()
    assert payload["income"]["lines"][0]["slug"] == "salary"
    assert Decimal(payload["income"]["total"]) == Decimal("90000")

    # And the chart hydrates from that block rather than from the network.
    assert 'jsonBlock("cf-summary"' in await _chart_js(client)


async def test_nav_has_cashflow_link(client):
    r = await client.get("/transactions")
    assert '<a href="/cashflow"' in r.text


async def test_cashflow_page_marks_its_nav_item_active(client):
    r = await client.get("/cashflow")
    assert '<a href="/cashflow" aria-current="page"' in r.text


def _tile_names(page: str) -> list[str]:
    return re.findall(r'data-tile="([a-z_]+)"', page)


def _lines(page: str, tile: str) -> list[tuple[str, str]]:
    """The (key, href) of every drill anchor the given tile's lines rendered.

    These are the anchors the breakdown chart looks a bar's link up in, so what
    they point at is what the bars point at: the bars take the row's href rather
    than rebuilding the URL a second time in JavaScript.
    """
    return re.findall(rf'<a data-line="{tile}" data-key="([^"]*)" href="([^"]*)"', page)


async def _listed(client, href: str) -> str:
    """Follow a rendered href exactly as a browser would."""
    r = await client.get(href.replace("&amp;", "&"))
    assert r.status_code == 200
    return r.text


async def test_all_five_tiles_carry_the_selection_hooks_the_chart_needs(
    client, session
):
    """Every tile is selectable, or a tile's lines can never be charted at all —
    which is precisely what kept Transfers-in and Uncategorized off the breakdown."""
    await _add(session, amount="90000", direction="credit", category="salary")

    r = await client.get(f"/cashflow?{RANGE}")
    assert _tile_names(r.text) == [
        "income",
        "expense",
        "net_invested",
        "transfers_in",
        "uncategorized",
    ]
    for name in _tile_names(r.text):
        tile = _tile(r.text, name)
        assert f'data-select="{name}"' in tile
        assert 'role="button"' in tile
        assert 'tabindex="0"' in tile
        assert "aria-pressed" in tile

    # The chart's selection listener binds to that hook, and its bars take their
    # links from the rows asserted below rather than rebuilding them.
    js = await _chart_js(client)
    assert 'querySelectorAll("[data-select]")' in js
    assert 'querySelectorAll("a[data-line]")' in js
    # And the page loads the module that binds them.
    assert '<script type="module" src="/static/js/cashflow.js"></script>' in r.text


async def test_net_invested_tile_shows_its_transaction_count(client, session):
    """The other four tiles say how many rows they are made of; this one said only
    gross in/out, so the one number that makes the tile comparable was missing."""
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(session, amount="4000", direction="credit", category="investment")
    await _add(session, amount="90000", direction="credit", category="salary")

    r = await client.get(f"/cashflow?{RANGE}")
    tile = _tile(r.text, "net_invested")
    assert "2 txns" in tile
    # Still the gross figures, next to the count rather than instead of it.
    assert "₹10K" in tile
    assert "₹4K" in tile


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
    await _add(session, amount="20000", direction="debit", category="rent")
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(session, amount="4000", direction="credit", category="investment")
    await _add(
        session,
        amount="1500",
        direction="credit",
        category="repayment",
        counterparty="Alice",
    )
    await _add(session, amount="700", direction="credit", category="repayment")
    await _add(session, amount="800", direction="debit", category=None)
    await _add(session, amount="600", direction="debit", category="")
    await _add(session, amount="200", direction="debit", category="unknown")
    await _add(session, amount="90", direction="debit", category="crypto")

    r = await client.get(f"/cashflow?{RANGE}")

    # tile -> its lines, each as (key, the rows the link must list, the rows it
    # must not).
    expected = {
        "income": {"salary": (["90,000.00"], ["555.00", "20,000.00"])},
        "expense": {"rent": (["20,000.00"], ["90,000.00", "10,000.00"])},
        "net_invested": {
            "investment:contribution": (["10,000.00"], ["4,000.00"]),
            "investment:redemption": (["4,000.00"], ["10,000.00"]),
        },
        "transfers_in": {
            "Alice": (["1,500.00"], ["700.00", "90,000.00"]),
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
            listed = await _listed(client, href)
            present, absent = lines[key]
            for amount in present:
                assert amount in listed, f"{tile}/{key or '(blank)'} lost {amount}"
            for amount in absent:
                assert amount not in listed, (
                    f"{tile}/{key or '(blank)'} listed {amount}"
                )
            assert listed.count("/detail") == len(present)


async def test_uncategorized_line_counts_and_its_link_agree_on_every_spelling(
    client, session
):
    """A blank category is the same absence a NULL is, on BOTH sides of the drill.

    Treated as a slug of its own it forms its own line, whose link — a
    category-less filter — then lists the NULL row instead of the row the line
    counted: a line that links to somebody else's money.
    """
    await _add(session, amount="800", direction="debit", category=None)
    await _add(session, amount="600", direction="debit", category="")
    await _add(session, amount="200", direction="debit", category="unknown")
    await _add(session, amount="90", direction="debit", category="crypto")
    await _add(session, amount="20000", direction="debit", category="rent")  # decoy

    r = await client.get(f"/cashflow?{RANGE}")

    # Three lines over four rows: NULL and "" are one.
    assert [key for key, _ in _lines(r.text, "uncategorized")] == [
        "",
        "unknown",
        "crypto",
    ]
    assert "4 txns" in _tile(r.text, "uncategorized")

    blank = dict(_lines(r.text, "uncategorized"))[""]
    listed = await _listed(client, blank)
    assert "800.00" in listed
    assert "600.00" in listed  # the row a NULL-only filter would have dropped
    assert "20,000.00" not in listed
    assert listed.count("/detail") == 2

    # And the whole-bucket link still lists all four, rent excluded.
    everything = await _listed(client, f"/transactions?uncategorized=1&{RANGE}")
    assert everything.count("/detail") == 4
    assert "20,000.00" not in everything


async def test_trend_keeps_the_paycheck_count_the_api_sends(client, session):
    """A calendar month holds 2 or 3 paychecks on a ~14-day cycle, so income swings
    by half with nothing having changed. The tooltip's paycheck count is what tells
    that apart from a real swing — dropping it from the page discards the one field
    that explains the chart."""
    # The trend window is the twelve months ending today, so the rows have to be
    # seeded relative to today or the test expires the day the seeded month falls
    # out of the window. Last month is always in the window and always wholly in
    # the past, which the current month's rows would not be: the trend's upper
    # bound is today, so a row dated later this month is outside it.
    month = _month_start(datetime.date.today(), 1)
    key = f"{month.year:04d}-{month.month:02d}"
    for day in (1, 15):
        await _add(
            session,
            amount="45000",
            direction="credit",
            category="salary",
            day=day,
            month=month,
        )

    api = await client.get("/api/cashflow/trend?months=12")
    points = {p["month"]: p for p in api.json()}
    assert key in points, f"{key} is not in the trend window {list(points)}"
    # Two paychecks that month, and only that month.
    assert points[key]["salary_count"] == 2
    assert [p["month"] for p in api.json() if p["salary_count"]] == [key]

    js = await _chart_js(client)
    # The field survives into the client's own value objects...
    assert "salary_count: Number(p.salary_count)" in js
    # ...and reaches the reader, in the tooltip the spec puts it in.
    assert "paycheck" in js


async def test_trend_months_each_carry_the_range_clicking_them_selects(client):
    """Clicking a month sets the range bar to that month, so every month drawn needs
    its own bounds on the page — and they must be the days that month really has."""
    r = await client.get("/cashflow")
    block = re.search(
        r'<script type="application/json" id="cf-trend-ranges">(.*?)</script>', r.text
    )
    assert block, "the trend months carry no ranges, so a month click has nowhere to go"
    ranges = json.loads(block.group(1))

    today = datetime.date.today()
    assert len(ranges) == 12
    this_month = f"{today.year:04d}-{today.month:02d}"
    # The current month is partial: selecting it means today, not a future month end.
    assert ranges[this_month] == [
        today.replace(day=1).isoformat(),
        today.isoformat(),
    ]

    # The month click is a navigation to those bounds; following one lands on the
    # report for that month, with the range bar set to it.
    first_month = next(iter(ranges))
    start, end = ranges[first_month]
    clicked = await client.get(f"/cashflow?date_from={start}&date_to={end}")
    assert clicked.status_code == 200
    assert f'value="{start}"' in clicked.text
    assert f'value="{end}"' in clicked.text

    # The chart builds each month's link out of exactly that data.
    js = await _chart_js(client)
    assert 'jsonBlock("cf-trend-ranges")' in js
    assert "/cashflow?date_from=${esc(range[0])}&amp;date_to=${esc(range[1])}" in js


async def test_trend_empty_state_is_gated_on_zero_values_not_array_length(client):
    """The trend endpoint pre-seeds every month in the window, so an empty history
    comes back as twelve zeros and never as an empty array — an empty state gated on
    the array's length is unreachable, and the chart draws a row of zero-height bars
    where the message belongs."""
    api = await client.get("/api/cashflow/trend?months=12")
    points = api.json()
    assert len(points) == 12  # not empty, even against an empty DB
    assert all(Decimal(p["income"]) == 0 for p in points)

    r = await client.get("/cashflow")
    assert 'id="cf-trend-empty"' in r.text
    # The reachable gate: every point zero, not "no points".
    js = await _chart_js(client)
    assert "const allZero = values.every(" in js
    assert "if (!values.length || allZero) {" in js


async def test_every_chart_module_the_pages_load_is_actually_served(client):
    """The pages are inert without these three, and a page that names a module the
    app does not serve is a silent 404 in the browser: nothing on the server errors,
    the figures still render, the charts just never appear. Fetching each one through
    the app is the only assertion that catches a rename, a move out from under the
    /static mount, or a deletion."""
    for name in ("charts.js", "cashflow.js", "networth.js"):
        r = await client.get(f"/static/js/{name}")
        assert r.status_code == 200, f"/static/js/{name} is not served"
        assert r.text.strip(), f"/static/js/{name} is served empty"

    # And the shared module both pages build on is imported rather than copied.
    for page in ("cashflow.js", "networth.js"):
        source = (await client.get(f"/static/js/{page}")).text
        assert "charts.js" in source, f"{page} does not import the shared chart module"


async def test_cashflow_page_carries_the_module_tag_and_the_chart_config(
    client, session
):
    """The template's side of the contract: the module tag whose src must resolve,
    and the hooks the module reads. A renamed file or a lost data-* attribute leaves
    a page that serves 200 and draws nothing.

    A row is seeded because the breakdown island is behind the range-activity gate:
    an empty range has nothing to break down and correctly omits it.
    """
    await _add(session, amount="90000", direction="credit", category="salary")

    r = await client.get(f"/cashflow?{RANGE}")
    assert '<script type="module" src="/static/js/cashflow.js"></script>' in r.text
    assert 'id="cf-trend"' in r.text
    assert 'data-trend-url="/api/cashflow/trend?months=12"' in r.text
    assert '<script type="application/json" id="cf-summary">' in r.text
    assert '<script type="application/json" id="cf-trend-ranges">' in r.text


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
    counterparties = [line["counterparty"] for line in payload["transfers_in"]["lines"]]
    assert counterparties == [HOSTILE]

    # The line separators are escaped *inside the island*, which is the only place
    # they are dangerous: they end a line in JavaScript, so a raw one there closes
    # nothing but breaks the parse. In the HTML body they are ordinary text, and the
    # table below the chart is free to carry them.
    assert "\u2028" not in block.group(1)
    assert "\u2029" not in block.group(1)


# ---------------------------------------------------------------------------
# Layout: the breakdown tables must stay inside their own grid track.
#
# These are structural assertions, not layout ones. Nothing here lays the page
# out — there is no browser in this suite — so what they prove is that the
# selectors and the overriding declarations the fix depends on are still on the
# page, not that the pixels land where they should. That is exactly the gap that
# let the bug ship: the whole suite passed while the Investments table painted
# over the Transfers In links. A test that could have *caught* it needs a DOM
# harness with real layout (jsdom does not compute it; a headless browser does),
# which is a known, filed gap and a new dependency this suite does not take.
#
# What they do catch is the regression by which the bug arrived and would arrive
# again: someone deletes the page-scoped override, base.html's global rule wins
# back, and the tables stop clipping.
# ---------------------------------------------------------------------------

# The bucket each rendered table charts under. The investment table is the odd
# one: its section is "investment" but its rows drill under the "net_invested"
# tile, because that is the tile they are the breakdown of.
SECTION_LINES = {
    "income": "income",
    "expense": "expense",
    "investment": "net_invested",
    "transfers_in": "transfers_in",
    "uncategorized": "uncategorized",
}


def _css(page: str) -> str:
    """Every <style> block on the page, whitespace-normalized for matching."""
    return re.sub(
        r"\s+", " ", "\n".join(re.findall(r"<style>(.*?)</style>", page, re.DOTALL))
    )


async def _seed_all_five_buckets(session):
    """One line in every bucket, so all five tables render side by side."""
    await _add(session, amount="90000", direction="credit", category="salary")
    await _add(session, amount="19500", direction="debit", category="rent")
    await _add(session, amount="10000", direction="debit", category="investment")
    await _add(session, amount="4000", direction="credit", category="investment")
    await _add(
        session,
        amount="1500",
        direction="credit",
        category="repayment",
        counterparty="Alice",
    )
    await _add(session, amount="700", direction="debit", category=None)


async def test_breakdown_tables_are_scoped_so_they_cannot_overflow_their_grid_track(
    client, session
):
    """The tables sit in a grid, and base.html stops clipping `.table` at >=1400px.

    That rule is right for the ledger's wide stacked tables and wrong for these:
    a table that will not clip does not run wide harmlessly here, it paints into
    the *next grid track*. The Investments table is 4 columns and the widest, so
    it drew over the Transfers In links beside it — and being on top, its anchors
    took the clicks, sending a reader who clicked a counterparty to the wrong
    transactions entirely.

    So the page must override that rule for its own tables. If the override goes,
    the global rule wins again and the overlap comes back.
    """
    await _seed_all_five_buckets(session)

    r = await client.get(f"/cashflow?{RANGE}")
    assert r.status_code == 200
    page = r.text
    css = _css(page)

    # The hazard this override exists to answer is really on the page: without
    # base.html's wide-screen rule, none of what follows would be load-bearing,
    # and this test would be asserting against nothing.
    assert re.search(
        r"@media \(min-width: 1400px\) \{ \.table \{ overflow-x: visible;", css
    ), "base.html's global wide-screen rule is gone; this override may be obsolete"

    # The tables are the grid, and every table region is inside it.
    grid = _region(page, 'class="cf-tables"')
    for section in SECTION_LINES:
        assert f'data-section="{section}"' in grid, (
            f"the {section} table is outside .cf-tables, so the scoping below "
            f"does not reach it"
        )

    # A grid item defaults to min-width: auto and refuses to shrink below its
    # content's intrinsic width — which is what forced the table wider than its
    # track in the first place. Without this, the overflow rule below cannot fire.
    assert ".cf-tables > article { min-width: 0; }" in css

    # And with shrinking allowed, the table clips inside the track it was given.
    # `.cf-tables .table` outranks base's bare `.table` on specificity, so this
    # holds *inside* the >=1400px media query too, which is the entire point.
    assert re.search(
        r"\.cf-tables \.table \{ width: 100%; max-width: 100%; overflow-x: auto;", css
    ), "the cashflow tables no longer override the global overflow-x: visible"

    # A track that cannot go below 320px on a narrow phone would itself overflow
    # the viewport; min() lets it fall back to the full width instead.
    assert (
        "grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr));" in css
    )

    # The charts are sized from their SVG's measured width, so a chart that
    # overflowed its card would feed the next resize a width it had just grown to.
    assert ".cf-chart-card { overflow: hidden; }" in css
    assert re.search(
        r"#cf-breakdown, #cf-trend \{ display: block; width: 100%; max-width: 100%; overflow: hidden;",
        css,
    )
    # Both chart cards actually carry the clipping class — a rule matching nothing
    # is not a fix.
    assert page.count('class="card cf-chart-card"') == 2

    # The override must come *after* base.html's rule as well as outrank it: a
    # page-level <style> that the base stylesheet followed would lose on order.
    assert page.index(".cf-tables .table") > page.index("overflow-x: visible")


async def test_each_tables_drill_anchors_are_namespaced_to_its_own_tile(
    client, session
):
    """No table's anchors can be mistaken for another's.

    The chart looks a bar's link up as DRILL[tile][key], so the anchors have to
    partition cleanly by tile. This is also the invariant the overlap bug made a
    liar of *visually*: an Income anchor sat on top of a Transfers In row and took
    its click. The markup was never the problem — the anchors were correctly
    namespaced all along — and this pins that down so a future refactor of the
    tables cannot quietly merge two tiles' links into one namespace.
    """
    await _seed_all_five_buckets(session)

    r = await client.get(f"/cashflow?{RANGE}")
    page = r.text

    seen = set()
    for section, line in SECTION_LINES.items():
        region = _region(page, "data-section", section)
        lines_in_region = set(re.findall(r"<a data-line=\"([^\"]+)\"", region))
        assert lines_in_region == {line}, (
            f"the {section} table renders anchors for {lines_in_region}, not just "
            f"{line!r}: a bar in one tile would drill into another tile's rows"
        )
        # Each key is unique within its tile, so DRILL[tile][key] cannot collide.
        keys = [key for key, _ in _lines(region, line)]
        assert keys, f"the {section} table rendered no drill anchors at all"
        assert len(keys) == len(set(keys)), f"duplicate drill keys in {section}: {keys}"
        seen.add(line)

    # Every anchor on the page belongs to one of the five, so none is orphaned
    # into a namespace no tile selects.
    assert set(re.findall(r"<a data-line=\"([^\"]+)\"", page)) == seen

    # The Income and Transfers In anchors — the two the overlap confused — carry
    # genuinely different predicates, so landing on the wrong one lists the wrong
    # rows. That is what made the overlap damaging rather than merely ugly.
    (income_href,) = [href for _, href in _lines(page, "income")]
    (transfers_href,) = [href for _, href in _lines(page, "transfers_in")]
    assert income_href != transfers_href
    assert "category=salary" in income_href
    assert "category=repayment&amp;counterparty=Alice" in transfers_href
