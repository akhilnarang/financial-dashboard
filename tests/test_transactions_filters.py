import datetime
import html
import re
from decimal import Decimal

import pytest
from sqlalchemy import null, select

from financial_dashboard.db.models import Category, Transaction
from tests.conftest import MISSING_ACCOUNT_ID, ensure_account

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


def _transaction_rows_html(page: str) -> str:
    """Limit amount assertions to rows, excluding the new totals cards."""
    match = re.search(r"<tbody>(.*?)</tbody>", page, flags=re.DOTALL)
    assert match is not None
    return match.group(1)


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


async def _seed_null_currency(session):
    """A row whose currency really is SQL NULL — the branch INR_OR_NULL exists for.

    ``Transaction.currency`` carries ``default="INR"``, so constructing the row with
    ``currency=None`` stores the string ``"INR"``: the fixture would seed an ordinary
    rupee row and every assertion below it would pass without the NULL branch ever
    being reached. ``null()`` is what forces the real SQL NULL, and the value is read
    back out of the database rather than off the instance, so the fixture cannot go
    back to seeding an ``"INR"`` row without this failing.
    """
    row = Transaction(
        bank="hdfc",
        email_type="x",
        direction="debit",
        amount=Decimal("11"),
        category="dining",
        currency=null(),
        transaction_date=DATED,
    )
    session.add(row)
    await session.flush()
    stored = (
        await session.execute(
            select(Transaction.currency).where(Transaction.id == row.id)
        )
    ).scalar_one()
    assert stored is None, f"the fixture stored {stored!r}, not NULL"


async def test_non_inr_zero_lists_inr_and_null_currency_rows(client, session):
    await _seed(session)
    await _seed_null_currency(session)
    r = await client.get("/transactions?non_inr=0")
    assert r.status_code == 200
    # The 5 INR seed rows plus the NULL-currency one; the USD row is excluded.
    assert _count_rows(r.text) == 6
    assert "3.00" not in _transaction_rows_html(r.text)  # the USD dining row

    foreign = await client.get("/transactions?non_inr=1")
    assert _count_rows(foreign.text) == 1  # the USD row alone

    # An omitted non_inr is no filter: USD and NULL dining rows both list.
    r = await client.get("/transactions?category=dining")
    assert _count_rows(r.text) == 2


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


# ---------------------------------------------------------------------------
# Category column, filter dropdown and uncategorized toggle on /transactions.
#
# The category the enricher/seed wrote onto a row is now visible on the list as a
# badge, selectable from a dropdown built off the active categories table, and
# the uncategorized population — rows no bucket can place — is reachable through
# an explicit checkbox wired to the existing ?uncategorized= drill param.
# ---------------------------------------------------------------------------


async def _seed_categories(session, *slugs):
    for slug in slugs:
        session.add(Category(slug=slug, active=True))
    await session.flush()


async def test_category_dropdown_lists_active_categories_only(client, session):
    await _seed(session)
    await _seed_categories(session, "groceries", "rent")
    session.add(Category(slug="dining", active=False))
    await session.flush()
    r = await client.get("/transactions")
    assert r.status_code == 200
    assert '<option value="groceries"' in r.text
    assert '<option value="rent"' in r.text
    assert '<option value="dining"' not in r.text


async def test_list_uses_label_override_for_known_slug(client, session):
    # 'credit_card_payment' has a report override ("Card bills"); the list badge
    # uses the same override so the column and the cashflow tile agree.
    await _seed_categories(session, "credit_card_payment")
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction="debit",
            amount=Decimal("100"),
            category="credit_card_payment",
            currency="INR",
            transaction_date=DATED,
        )
    )
    await session.flush()
    r = await client.get("/transactions")
    assert "Card bills" in r.text


async def test_list_shows_uncategorized_badge_for_unknown_sentinel(client, session):
    await _seed_categories(session, "unknown")
    await _seed(session)
    r = await client.get("/transactions?category=unknown")
    assert _count_rows(r.text) == 1
    assert "Uncategorized" in r.text


async def test_detail_shows_category_method_and_review_labels(client, session):
    txn = Transaction(
        bank="hdfc",
        email_type="x",
        direction="debit",
        amount=Decimal("100"),
        category="groceries",
        category_method="llm",
        category_confidence=0.85,
        review_status="pending",
        review_reason="low confidence",
        currency="INR",
        transaction_date=DATED,
    )
    session.add(txn)
    await session.flush()

    r = await client.get(f"/transactions/{txn.id}/detail")
    assert r.status_code == 200
    assert "AI" in r.text  # category_method 'llm' -> 'AI'
    assert "Needs review" in r.text  # review_status 'pending'
    assert "85%" in r.text  # confidence rendered as a percentage


async def test_detail_hides_method_badge_when_never_categorized(client, session):
    # A row that the pipeline never touched has category_method=None and
    # review_status=None; neither badge should render.
    txn = Transaction(
        bank="hdfc",
        email_type="x",
        direction="debit",
        amount=Decimal("100"),
        category=None,
        category_method=None,
        review_status=None,
        currency="INR",
        transaction_date=DATED,
    )
    session.add(txn)
    await session.flush()

    r = await client.get(f"/transactions/{txn.id}/detail")
    assert "Needs review" not in r.text
    assert "Reviewed" not in r.text
    # Uncategorized IS shown (the category is NULL), but no method badge.
    assert "Uncategorized" in r.text


# ---------------------------------------------------------------------------
# ?scope= — the account perimeter a cashflow figure was drawn over.
#
# The report is bank-scoped, so every one of its drill-throughs has to be able to
# say so, or the rows behind a figure's link are not the rows it counted. The
# three scopes partition the table, so each test below seeds a row in *every*
# scope: each is a decoy for the other two, and an unfiltered listing cannot pass.
# ---------------------------------------------------------------------------

DEBIT_CARD_ACCOUNT_ID = 3
UNKNOWN_TYPE_ACCOUNT_ID = 4


async def _add(session, *, amount, account_id, category="groceries", direction="debit"):
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction=direction,
            amount=Decimal(amount),
            category=category,
            currency="INR",
            transaction_date=DATED,
            account_id=account_id,
        )
    )
    await session.flush()


async def _seed_every_scope(session):
    """One row in each scope, each with an amount no other row shares.

    A debit card is bank money moving immediately, so it is *bank* scope even
    though nothing in prod is typed that way yet; an account type nothing
    recognizes and a link to an account row that does not exist are both
    *unaccounted*, which is the complement and cannot be reached any other way
    (``Account.type`` is non-null in the ORM).
    """
    await _add(
        session,
        amount="1000",
        account_id=await ensure_account(session, 1, "bank_account"),
    )
    await _add(
        session,
        amount="2000",
        account_id=await ensure_account(session, DEBIT_CARD_ACCOUNT_ID, "debit_card"),
    )
    await _add(
        session,
        amount="3000",
        account_id=await ensure_account(session, 2, "credit_card"),
    )
    await _add(
        session,
        amount="4000",
        account_id=await ensure_account(session, UNKNOWN_TYPE_ACCOUNT_ID, "wallet"),
    )
    await _add(session, amount="5000", account_id=MISSING_ACCOUNT_ID)
    await _add(session, amount="6000", account_id=None)


async def test_the_three_scopes_partition_the_table(client, session):
    """Every row is in exactly one scope: the three listings add up to all of them.

    A row in two scopes is a row two figures could both count, which is the double
    count the cash basis exists to avoid; a row in none is money the page drops.
    """
    await _seed_every_scope(session)
    everything = _count_rows((await client.get("/transactions")).text)
    # A debit card is bank cash. Unknown-type, dangling and unlinked rows are
    # unaccounted.
    expected = {
        "bank": {"1,000.00", "2,000.00"},
        "card": {"3,000.00"},
        "unaccounted": {"4,000.00", "5,000.00", "6,000.00"},
    }
    every_amount = set().union(*expected.values())
    counted = 0
    for scope, amounts in expected.items():
        r = await client.get(f"/transactions?scope={scope}")
        rows = _transaction_rows_html(r.text)
        assert _count_rows(r.text) == len(amounts)
        for amount in every_amount:
            assert (amount in rows) == (amount in amounts), f"{scope}: {amount}"
        counted += len(amounts)
    assert counted == everything == 6


async def test_an_unknown_scope_is_rejected_rather_than_silently_ignored(
    client, session
):
    """A typo in a scope must not fall back to listing every account.

    That is the failure the whole param exists to prevent: a figure's link that
    quietly widens to every row while still sitting under a bank-only number.
    """
    await _seed_every_scope(session)
    r = await client.get("/transactions?scope=banck")
    assert r.status_code == 422


async def test_internal_under_bank_scope_is_self_transfers_alone(client, session):
    """The one composite filter whose meaning changes with the scope.

    Over the bank a card bill IS the expense — the day the money leaves — so the
    internal footnote counts self-transfers alone and its link must list exactly
    those. The unscoped filter keeps both slugs, because over every account the
    bill settles swipes that view already counted.
    """
    bank = await ensure_account(session, 1, "bank_account")
    card = await ensure_account(session, 2, "credit_card")
    await _add(session, amount="10", account_id=bank, category="self_transfer")
    await _add(session, amount="20", account_id=bank, category="credit_card_payment")
    await _add(session, amount="30", account_id=card, category="self_transfer")
    await _add(session, amount="40", account_id=bank, category="rent")

    scoped = await client.get("/transactions?internal=1&scope=bank")
    assert _count_rows(scoped.text) == 1
    assert "10.00" in scoped.text
    assert "20.00" not in scoped.text, (
        "a card bill is expense over the bank, not internal"
    )
    assert "30.00" not in scoped.text  # a card row is out of the bank scope entirely

    # No scope: today's meaning, preserved. Both slugs, on every account.
    unscoped = await client.get("/transactions?internal=1")
    assert _count_rows(unscoped.text) == 3
    for amount in ("10.00", "20.00", "30.00"):
        assert amount in unscoped.text
    assert "40.00" not in unscoped.text


async def test_scope_survives_pagination(client, session):
    """Page 2 of a scoped listing is still the scope's rows.

    A scope dropped from the pagination links would widen page 2 to every account
    while the figure above it still said "bank".
    """
    bank = await ensure_account(session, 1, "bank_account")
    card = await ensure_account(session, 2, "credit_card")
    for i in range(55):
        await _add(session, amount=Decimal(100 + i), account_id=bank)
    await _add(session, amount="9999", account_id=card)

    r = await client.get("/transactions?scope=bank")
    assert _count_rows(r.text) == 50  # a full first page, so pagination renders
    assert "9,999.00" not in r.text

    hrefs = [html.unescape(h) for h in re.findall(r'href="([^"]+)"', r.text)]
    page_two = [h for h in hrefs if "page=2" in h]
    assert page_two, "pagination nav did not render a page-2 link"
    assert all("scope=bank" in h for h in page_two)

    r2 = await client.get(page_two[0])
    assert _count_rows(r2.text) == 5  # 55 bank rows - a full page of 50
    assert "9,999.00" not in r2.text, "page 2 of the bank listing shows a card row"


async def test_scope_survives_sorting(client, session):
    """Re-sorting a scoped listing must not widen it either: the sort links carry
    the scope, and following one returns the same population in a new order."""
    bank = await ensure_account(session, 1, "bank_account")
    card = await ensure_account(session, 2, "credit_card")
    await _add(session, amount="100", account_id=bank)
    await _add(session, amount="300", account_id=bank)
    await _add(session, amount="9999", account_id=card)

    r = await client.get("/transactions?scope=bank")
    sort_links = [
        html.unescape(h)
        for h in re.findall(r'href="([^"]+)"', r.text)
        if "sort=amount" in h
    ]
    assert sort_links, "no sortable column header rendered a link"
    assert all("scope=bank" in h for h in sort_links)

    sorted_page = await client.get(sort_links[0])
    assert _count_rows(sorted_page.text) == 2
    assert "9,999.00" not in sorted_page.text, (
        "sorting widened the listing to a card row"
    )
