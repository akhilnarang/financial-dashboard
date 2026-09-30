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


# The query string as requested, and as it appears inside a rendered href,
# where the separator is the "&amp;" entity.
RANGE = "date_from=2026-06-01&date_to=2026-06-30"
RANGE_HTML = "date_from=2026-06-01&amp;date_to=2026-06-30"
# The rupee buckets sum INR-and-null rows only, so their drill-throughs pin the
# currency; the currency-agnostic links (uncategorized, internal, undated) do not.
CORE_HTML = f"{RANGE_HTML}&amp;non_inr=0"

# The month the seed helper writes into by default.
SEED_MONTH = datetime.date(2026, 6, 1)


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

    # The undated link lists the undated row and no dated one.
    listed = await _listed(client, "/transactions?undated=1")
    assert "333.00" in listed
    assert "5,000.00" not in listed
    assert listed.count("/detail") == 1


async def test_empty_range_shows_empty_state(client):
    r = await client.get(f"/cashflow?{RANGE}")
    assert r.status_code == 200
    assert "No transactions in this range" in r.text


async def test_breakdown_hydrates_from_the_page_not_a_second_summary_query(
    client, session
):
    """The breakdown draws the range the page already aggregated, so the summary is
    serialized into the page and the chart reads it there.

    Re-fetching /api/cashflow/summary would be the same aggregate queries run a
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
    # Net invested = 10,000 contributed - 4,000 redeemed.
    assert "₹6,000.00" in _region(r.text, "data-section", "investment")


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
            "Alice": (["1,500.00"], ["333.00", "700.00", "90,000.00"]),
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

    # NULL, blank, 'unknown' and unmapped rows form one bucket of four.
    assert "4 txns" in _tile(r.text, "uncategorized")
    everything = await _listed(client, f"/transactions?uncategorized=1&{RANGE}")
    assert everything.count("/detail") == 4


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
