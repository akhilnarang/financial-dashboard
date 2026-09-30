import datetime
import html
import re
from decimal import Decimal

import pytest
from sqlalchemy import null, select

from financial_dashboard.db.models import Transaction
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


async def _add(
    session,
    *,
    amount,
    account_id,
    category="groceries",
    direction="debit",
    counterparty=None,
):
    session.add(
        Transaction(
            bank="hdfc",
            email_type="x",
            direction=direction,
            amount=Decimal(amount),
            category=category,
            counterparty=counterparty,
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


async def test_paging_and_sorting_keep_every_drill_filter(client, session):
    """Page 2 and a re-sorted page list the same rows as page 1.

    A blank counterparty is a real filter, so it must survive even though it is
    falsy. A scope dropped from a link would widen the listing to every account.
    """
    bank = await ensure_account(session, 1, "bank_account")
    card = await ensure_account(session, 2, "credit_card")
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
                account_id=bank,
            )
        )
    await session.flush()
    # Decoys: a named counterparty, another category, and a card row.
    repayment = {"category": "repayment", "direction": "credit"}
    await _add(session, amount="999", account_id=bank, counterparty="MOM", **repayment)
    await _add(session, amount="998", account_id=bank)
    await _add(session, amount="997", account_id=card, **repayment)

    query = "/transactions?category=repayment&counterparty=&scope=bank"
    r = await client.get(query)
    assert r.status_code == 200
    assert _count_rows(r.text) == 50  # a full first page, so pagination renders

    hrefs = [html.unescape(h) for h in re.findall(r'href="([^"]+)"', r.text)]
    page_two = await client.get(next(h for h in hrefs if "page=2" in h))
    assert _count_rows(page_two.text) == 5  # 55 matching rows - a full page of 50
    # The link sorts by amount descending, so a lost filter puts the decoys first.
    by_amount = await client.get(next(h for h in hrefs if "sort=amount" in h))
    assert _count_rows(by_amount.text) == 50
    for listing in (page_two.text, by_amount.text):
        for decoy in ("999.00", "998.00", "997.00"):
            assert decoy not in _transaction_rows_html(listing)
