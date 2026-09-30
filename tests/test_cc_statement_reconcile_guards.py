"""Card-identity and contention guards on the CC-statement reconciler.

``reconcile_statement`` has no reference numbers: the card is its only hard
signal, and everything else it buckets on (date, amount, direction) is
shared by unrelated transactions. Both failure directions cost real data —
matching across two different cards mis-assigns one row and silently never
imports the other, while refusing two spellings of the same card imports a
duplicate. So the card is an identity question (``core.masks``, compared
positionally, wildcards absorbing what a mask cannot see), never a last-4
lookup, and absent data is not evidence of a conflict.

The same candidate sets that answer "could this statement row be this DB
row?" for the picker also answer "did this unmatched row merely lose a
race?" for the importer. A row that could have claimed a DB candidate and
got none is held back; a row whose candidate set is *empty* must still
import — these tests drive the real importer and assert on row count in
both directions.
"""

from decimal import Decimal
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from cc_parser.parsers.models import Transaction as CcTransaction
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db import (
    Account,
    Card,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.statements import cc as cc_module
from financial_dashboard.services.statement_settlement import (
    Reconciliation,
    answer,
    held_rows,
    row_digest,
)
from financial_dashboard.services.statements.cc import (
    reconciliation_to_json,
    import_missing_cc_txns,
    load_account_card_masks,
    parse_cc_date,
    reconcile_statement,
)
from tests.conftest import new_test_engine

ACCOUNT_ID = 1
CARD_NUMBER = "4111XXXXXXXX9012"

NARRATION = "SWIGGY LIMITED           BANGALORE   IN"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch):
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(cc_module, "async_session", maker)
    yield maker
    await engine.dispose()
    holder.close()


def _stmt_txn(
    *,
    date: str,
    amount: str,
    narration: str,
    direction: Literal["debit", "credit"] = "debit",
) -> CcTransaction:
    return CcTransaction(
        date=date,
        narration=narration,
        amount=amount,
        card_number=CARD_NUMBER,
        transaction_type=direction,
    )


def _parsed(transactions: list[CcTransaction]):
    """The slice of a cc-parser ``ParsedStatement`` the reconciler and the
    importer actually read."""
    return SimpleNamespace(
        bank="hdfc",
        transactions=transactions,
        payments_refunds=[],
        payments_refunds_total="0.00",
        card_summaries=[],
        possible_adjustment_pairs=[],
        overall_total="0.00",
        overall_reward_points="0",
    )


async def _seed_account(maker) -> None:
    async with maker() as session:
        session.add(
            Account(
                id=ACCOUNT_ID,
                bank="hdfc",
                label="HDFC Credit Card",
                type="credit_card",
                account_number=CARD_NUMBER,
            )
        )
        await session.commit()


async def _seed_txn(maker, **overrides) -> int:
    """Insert one DB transaction — a card-alert email row for a ₹450 purchase
    on the statement's card, unless overridden."""
    fields = {
        "account_id": ACCOUNT_ID,
        "bank": "hdfc",
        "email_type": "transaction",
        "direction": "debit",
        "amount": Decimal("450.00"),
        "currency": "INR",
        "transaction_date": parse_cc_date("07/04/2026"),
        "counterparty": "MERCHANT A",
        "raw_description": "MERCHANT A",
        "channel": "card",
        "card_mask": "9012",
    }
    fields.update(overrides)
    async with maker() as session:
        txn = Transaction(**fields)
        session.add(txn)
        await session.commit()
        return txn.id


async def _reconcile(maker, parsed) -> dict:
    async with maker() as session:
        db_txns = list(
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id == ACCOUNT_ID)
                )
            )
            .scalars()
            .all()
        )
        # Loaded the way the real callers load it, so a test cannot pass by
        # feeding the reconciler a card list production would never build.
        card_masks = await load_account_card_masks(session, ACCOUNT_ID)
    return reconcile_statement(parsed, db_txns, ACCOUNT_ID, card_masks)


async def _import(maker, parsed, recon, due_date=None) -> tuple[list, list]:
    """Runs ``import_missing_cc_txns`` for a test upload.

    Returns the imported transactions and all database rows.
    """
    async with maker() as session:
        upload = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="parsed",
            due_date=due_date,
        )
        session.add(upload)
        await session.flush()
        account = await session.get(Account, ACCOUNT_ID)
        imported = await import_missing_cc_txns(session, upload, parsed, account, recon)
        await session.commit()

    async with maker() as session:
        rows = list((await session.execute(select(Transaction))).scalars().all())
    return imported, rows


@pytest.mark.parametrize(
    ("db_mask", "matches"),
    [
        # A card the account does not list is a hard no to the pairing — but
        # the registry drifts, so the row stays a candidate and the statement
        # row is held back rather than imported over it.
        pytest.param("1111", False, id="conflicting-card"),
        # Two cards sharing a last-4: the visible BIN digits conflict, so a
        # flatten-to-digits rule would wrongly match here.
        pytest.param("5100XXXXXXXX9012", False, id="shared-last4-different-bin"),
        # A short mask is blind, not empty: XX34 still conflicts on a shown digit.
        pytest.param("XX34", False, id="short-mask-conflicts-on-visible-digit"),
        # XX12 agrees everywhere it shows a digit — unknown, not a conflict.
        pytest.param("XX12", True, id="partial-mask-is-unknown"),
    ],
)
@pytest.mark.anyio
async def test_db_card_mask_decides_pairing_or_holdback(
    session_factory, db_mask, matches
):
    """One DB row whose card mask agrees or conflicts with the statement's
    card (``4111XXXXXXXX9012``), compared positionally.

    A conflict refuses the *pairing* but is never read as "the transaction is
    absent": the row is held back as ambiguous, not imported. A compatible
    mask pairs, and nothing is imported. Either way the table must not grow.
    """
    await _seed_account(session_factory)
    txn_id = await _seed_txn(session_factory, card_mask=db_mask)

    parsed = _parsed(
        [_stmt_txn(date="07/04/2026", amount="450.00", narration=NARRATION)]
    )
    recon = await _reconcile(session_factory, parsed)

    if matches:
        assert [entry["db_txn_id"] for entry in recon["matched"]] == [txn_id]
        assert recon["missing"] == []
    else:
        assert recon["matched"] == []
        assert [entry["ambiguous"] for entry in recon["missing"]] == [True]

    imported, rows = await _import(session_factory, parsed, recon)

    if matches:
        assert imported == []
        assert [row.id for row in rows] == [txn_id]
    else:
        assert len(imported) == 1
        assert txn_id in [row.id for row in rows]
        assert next(row for row in rows if row.id == txn_id).counterparty == (
            "MERCHANT A"
        )


@pytest.mark.anyio
async def test_an_addon_cards_transaction_matches_the_primarys_statement(
    session_factory,
):
    """The card question is asked of the account, not of the statement header.

    The DB holds the add-on card's purchase; the statement lists it stamped
    with the *header* mask (the primary's), as banks overwhelmingly do. Read
    per-row that header is a card conflict, the candidate set empties, and the
    purchase is re-imported as a confidently-wrong duplicate. An account
    answers to every card it holds, so the row must match.
    """
    await _seed_account(session_factory)
    async with session_factory() as session:
        session.add(
            Card(account_id=ACCOUNT_ID, card_mask="4111XXXXXXXX7788", label="Add-on")
        )
        await session.commit()
    addon_txn_id = await _seed_txn(session_factory, card_mask="4111XXXXXXXX7788")

    parsed = _parsed(
        [_stmt_txn(date="07/04/2026", amount="450.00", narration=NARRATION)]
    )
    recon = await _reconcile(session_factory, parsed)

    assert len(recon["matched"]) == 1
    assert recon["matched"][0]["db_txn_id"] == addon_txn_id
    assert recon["missing"] == []

    imported, rows = await _import(session_factory, parsed, recon)

    assert imported == []
    assert [row.id for row in rows] == [addon_txn_id]


@pytest.mark.anyio
async def test_a_card_on_another_account_is_still_not_this_accounts_card(
    session_factory,
):
    """Only the account's own cards count. A mask from another account's card
    refuses the pairing, and the statement row is held back."""
    await _seed_account(session_factory)
    async with session_factory() as session:
        session.add(
            Account(
                id=2,
                bank="hdfc",
                label="HDFC Credit Card 2",
                type="credit_card",
                account_number="5100XXXXXXXX0002",
            )
        )
        session.add(Card(account_id=2, card_mask="5100XXXXXXXX3344", label="Other"))
        await session.commit()
    other_id = await _seed_txn(
        session_factory,
        card_mask="5100XXXXXXXX3344",
        counterparty="MERCHANT ON OTHER ACCOUNT",
        raw_description="MERCHANT ON OTHER ACCOUNT",
    )

    parsed = _parsed(
        [_stmt_txn(date="07/04/2026", amount="450.00", narration=NARRATION)]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True]

    imported, rows = await _import(session_factory, parsed, recon)

    assert len(imported) == 1
    other = next(row for row in rows if row.id == other_id)
    assert other.counterparty == "MERCHANT ON OTHER ACCOUNT"


@pytest.mark.anyio
async def test_an_account_recording_no_cards_does_not_refuse_everything(
    session_factory,
):
    """An empty card list means "no cards known", not "no cards match" —
    reading it the other way would refuse every row on the statement and
    import the lot as duplicates."""
    async with session_factory() as session:
        session.add(
            Account(
                id=ACCOUNT_ID,
                bank="hdfc",
                label="HDFC Credit Card",
                type="credit_card",
                account_number=None,
            )
        )
        await session.commit()
    txn_id = await _seed_txn(session_factory, card_mask="4111XXXXXXXX9012")

    parsed = _parsed(
        [_stmt_txn(date="07/04/2026", amount="450.00", narration=NARRATION)]
    )
    recon = await _reconcile(session_factory, parsed)

    assert len(recon["matched"]) == 1
    assert recon["matched"][0]["db_txn_id"] == txn_id

    imported, rows = await _import(session_factory, parsed, recon)

    assert imported == []
    assert [row.id for row in rows] == [txn_id]


@pytest.mark.anyio
async def test_the_same_card_matches_and_an_unknown_card_does_not_block(
    session_factory,
):
    """The card rule must not over-refuse. A matching last-four still pairs,
    and a DB row whose card is simply unknown stays claimable — absent data is
    not evidence of a conflict. Neither row may be imported a second time."""
    await _seed_account(session_factory)
    same_card_id = await _seed_txn(session_factory, card_mask="9012")
    unknown_card_id = await _seed_txn(
        session_factory, amount=Decimal("770.00"), card_mask=None
    )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="450.00", narration=NARRATION),
            _stmt_txn(date="07/04/2026", amount="770.00", narration="OTHER SHOP"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert sorted(entry["db_txn_id"] for entry in recon["matched"]) == sorted(
        [same_card_id, unknown_card_id]
    )
    assert recon["missing"] == []

    imported, rows = await _import(session_factory, parsed, recon)

    assert imported == []
    assert len(rows) == 2


@pytest.mark.anyio
async def test_statement_side_collision_is_not_imported_as_a_duplicate(
    session_factory,
):
    """Contention runs both ways; counting only the DB side misses half of it.

    ONE DB row (MERCHANT A) and TWO statement rows, ordered B then A. The
    greedy matcher hands A's DB row to B — a DB-side-only ambiguity check
    calls that unambiguous, and A's own row would import as a duplicate. Nor
    may B's win be committed: it won on statement order, which is not
    evidence, and enrichment would rewrite A's row to say MERCHANT B. Both
    are held back and the row count does not grow.
    """
    await _seed_account(session_factory)
    a_id = await _seed_txn(
        session_factory, amount=Decimal("100.00"), counterparty="MERCHANT A"
    )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="100.00", narration="MERCHANT B"),
            _stmt_txn(date="07/04/2026", amount="100.00", narration="MERCHANT A"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["narration"] for entry in recon["missing"]] == [
        "MERCHANT B",
        "MERCHANT A",
    ]
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True]

    imported, rows = await _import(session_factory, parsed, recon)

    assert len(imported) == len(recon["missing"])
    for entry in recon["missing"]:
        assert entry["imported"] is True
    assert [row.id for row in rows] == [a_id] + [row.id for row in imported]


@pytest.mark.anyio
async def test_statement_rows_two_days_apart_still_contend_for_the_row_between_them(
    session_factory,
):
    """Rivalry is about reaching the same DB row, not about sitting near each
    other: rows on 06 and 08 Apr both reach the 07 Apr DB row through the
    ±1-day window, so neither the loser's import nor the winner's
    order-decided pairing may be committed."""
    await _seed_account(session_factory)
    a_id = await _seed_txn(
        session_factory, amount=Decimal("100.00"), counterparty="MERCHANT A"
    )

    parsed = _parsed(
        [
            _stmt_txn(date="06/04/2026", amount="100.00", narration="MERCHANT B"),
            _stmt_txn(date="08/04/2026", amount="100.00", narration="MERCHANT A"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["narration"] for entry in recon["missing"]] == [
        "MERCHANT B",
        "MERCHANT A",
    ]
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True]

    imported, rows = await _import(session_factory, parsed, recon)

    assert len(imported) == 2
    assert [row.id for row in rows] == [a_id] + [row.id for row in imported]


@pytest.mark.anyio
async def test_distinct_addon_cards_disambiguate_contended_rows(session_factory):
    """Two same-amount rows a day apart, each on its own add-on card, are not
    rivals. Each DB row's card mask pairs it with exactly one statement row, so
    the ``+/-1``-day windows overlap yet the cards resolve the contention and
    both rows match instead of being demoted."""
    await _seed_account(session_factory)
    async with session_factory() as session:
        session.add(
            Card(account_id=ACCOUNT_ID, card_mask="4111XXXXXXXX1111", label="Add-on 1")
        )
        session.add(
            Card(account_id=ACCOUNT_ID, card_mask="4111XXXXXXXX2222", label="Add-on 2")
        )
        await session.commit()

    db1 = await _seed_txn(
        session_factory,
        amount=Decimal("100.00"),
        transaction_date=parse_cc_date("07/04/2026"),
        counterparty="MERCHANT X",
        card_mask="4111XXXXXXXX1111",
    )
    db2 = await _seed_txn(
        session_factory,
        amount=Decimal("100.00"),
        transaction_date=parse_cc_date("08/04/2026"),
        counterparty="MERCHANT X",
        card_mask="4111XXXXXXXX2222",
    )

    parsed = _parsed(
        [
            CcTransaction(
                date="07/04/2026",
                narration="MERCHANT X BANGALORE",
                amount="100.00",
                card_number="4111XXXXXXXX1111",
                transaction_type="debit",
            ),
            CcTransaction(
                date="08/04/2026",
                narration="MERCHANT X BANGALORE IN",
                amount="100.00",
                card_number="4111XXXXXXXX2222",
                transaction_type="debit",
            ),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["missing"] == []
    assert {entry["stmt_idx"]: entry["db_txn_id"] for entry in recon["matched"]} == {
        0: db1,
        1: db2,
    }


@pytest.mark.anyio
async def test_ambiguous_short_suffix_does_not_confirm_a_card(session_factory):
    """A statement mask that matches two of the account's cards singles out
    neither, so it cannot prune a rival on the other card. Both rows stay
    conservatively demoted rather than one falsely claiming a match."""
    await _seed_account(session_factory)
    async with session_factory() as session:
        # A second card sharing the last four digits, distinguished only by BIN,
        # so "XX9012" below denotes either card.
        session.add(
            Card(account_id=ACCOUNT_ID, card_mask="5100XXXXXXXX9012", label="Other BIN")
        )
        await session.commit()

    await _seed_txn(
        session_factory,
        amount=Decimal("100.00"),
        transaction_date=parse_cc_date("07/04/2026"),
        counterparty="MERCHANT X",
        card_mask="4111XXXXXXXX9012",
    )
    await _seed_txn(
        session_factory,
        amount=Decimal("100.00"),
        transaction_date=parse_cc_date("08/04/2026"),
        counterparty="MERCHANT X",
        card_mask="5100XXXXXXXX9012",
    )

    parsed = _parsed(
        [
            CcTransaction(
                date="07/04/2026",
                narration="CHARGE A",
                amount="100.00",
                card_number="XX9012",
                transaction_type="debit",
            ),
            CcTransaction(
                date="08/04/2026",
                narration="CHARGE B",
                amount="100.00",
                card_number="5100XXXXXXXX9012",
                transaction_type="debit",
            ),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True]


@pytest.mark.anyio
async def test_new_rows_with_no_db_candidate_still_import(session_factory):
    """The ordinary-miss guarantee: two same-day, same-amount statement rows
    with nothing in the DB to claim have empty candidate sets, so neither is
    contended and both import. Refusing them would silently drop real
    spending from the ledger."""
    await _seed_account(session_factory)
    await _seed_txn(session_factory)  # unrelated ₹450 row

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="999.00", narration="NEW MERCHANT ONE"),
            _stmt_txn(date="07/04/2026", amount="999.00", narration="NEW MERCHANT TWO"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)
    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [False, False]

    imported, rows = await _import(session_factory, parsed, recon)

    assert sorted(txn.counterparty for txn in imported) == [
        "NEW MERCHANT ONE",
        "NEW MERCHANT TWO",
    ]
    assert len(rows) == 3


async def _seed_cc_account(
    maker, *, account_id: int, account_number: str | None
) -> None:
    async with maker() as session:
        session.add(
            Account(
                id=account_id,
                bank="hdfc",
                label=f"HDFC Credit Card {account_id}",
                type="credit_card",
                account_number=account_number,
                active=True,
            )
        )
        await session.commit()


@pytest.mark.parametrize(
    ("account_numbers", "stmt_card", "expected_id"),
    [
        # Two cards sharing a last-4 and a statement mask that hides the BIN:
        # nothing can choose, so nothing is chosen.
        pytest.param(
            ["5100XXXXXXXX9012", "4111XXXXXXXX9012"],
            "XXXX XXXX XXXX 9012",
            None,
            id="shared-last4-refused",
        ),
        # Exactly one account answers: refusal must not be the default.
        pytest.param(
            ["5100XXXXXXXX9012", "4111XXXXXXXX7788"],
            "XXXX XXXX XXXX 7788",
            2,
            id="sole-match-returned",
        ),
        # A visible BIN resolves the shared last-4 — the evidence a last-4
        # lookup throws away.
        pytest.param(
            ["5100XXXXXXXX9012", "4111XXXXXXXX9012"],
            "4111XXXXXXXX9012",
            2,
            id="visible-bin-resolves-shared-last4",
        ),
        # Flattened to digits, "1234XXXXXXXX" would read as the suffix of an
        # account ending 1234 — one clean, completely wrong hit that the
        # multiplicity refusal never gets a chance to save.
        pytest.param(
            ["5100XXXXXXXX1234"],
            "1234XXXXXXXX",
            None,
            id="bin-lookalike-suffix-refused",
        ),
        # SBI prints only two digits: weak evidence, but reaching exactly one
        # account there is nothing to confuse it with. A digit-count floor
        # would drop every SBI statement.
        pytest.param(
            ["5100XXXXXXXX9067", "4111XXXXXXXX9012"],
            "XXXX XXXX XXXX XX67",
            1,
            id="sbi-short-suffix-sole-reach",
        ),
        # An account recording no mask never wins a statement no other
        # account claims — silence is not a wildcard. Zero matches is zero.
        pytest.param(
            ["5100XXXXXXXX9012", None],
            "XXXX XXXX XXXX 7788",
            None,
            id="silent-account-does-not-absorb",
        ),
        # ...and it is not a rival either: it must not block the account that
        # does match.
        pytest.param(
            ["4111XXXXXXXX9012", None],
            "XXXX XXXX XXXX 9012",
            1,
            id="silent-account-does-not-block",
        ),
    ],
)
@pytest.mark.anyio
async def test_find_account_selects_by_positional_mask(
    session_factory, account_numbers, stmt_card, expected_id
):
    """``_find_account`` routes an *entire* statement, so it selects an
    account iff exactly one answers to the statement's mask, compared
    positionally. Accounts are seeded with ids 1..n from ``account_numbers``.
    """
    for idx, number in enumerate(account_numbers, start=1):
        await _seed_cc_account(session_factory, account_id=idx, account_number=number)

    parsed = SimpleNamespace(card_number=stmt_card)
    account = await cc_module._find_account("hdfc", parsed)

    if expected_id is None:
        assert account is None
    else:
        assert account is not None
        assert account.id == expected_id


@pytest.mark.anyio
async def test_find_account_aggregates_conflicts_across_both_routes(session_factory):
    """Account 1 records a card ending ``5566`` as its account_number;
    account 2 records one in the cards table. Each route alone sees a single
    clean hit, so stopping at the first route to produce something would hide
    the conflict — this is two accounts and no answer."""
    await _seed_cc_account(
        session_factory, account_id=1, account_number="4111XXXXXXXX5566"
    )
    await _seed_cc_account(
        session_factory, account_id=2, account_number="5100XXXXXXXX0002"
    )
    async with session_factory() as session:
        session.add(Card(account_id=2, card_mask="XXXX XXXX XXXX 5566", label="Addon"))
        await session.commit()

    parsed = SimpleNamespace(card_number="XXXX XXXX XXXX 5566")

    assert await cc_module._find_account("hdfc", parsed) is None


@pytest.mark.anyio
async def test_find_account_matches_a_non_canonical_stored_mask(session_factory):
    """The stored side is a mask too: dashes, spaces and lowercase ``x`` are
    cosmetic. Compared raw, the separators shift every digit out of alignment
    and the account silently stops matching its own statements."""
    await _seed_cc_account(
        session_factory, account_id=1, account_number="4000-xxxx xxxx-1234"
    )

    parsed = SimpleNamespace(card_number="4000XXXXXXXX1234")

    account = await cc_module._find_account("hdfc", parsed)
    assert account is not None
    assert account.id == 1


@pytest.mark.anyio
async def test_interchangeable_rivals_leave_the_winners_match_alone(session_factory):
    """A pairing with no wrong answer is not worth refusing: two identical
    autopay rows contend for one DB row, but the narration — the only thing a
    refresh writes here — is the same either way, so the winner keeps its
    match. The loser is still held back; interchangeable rivals do not make a
    second copy any less of a duplicate."""
    await _seed_account(session_factory)
    a_id = await _seed_txn(
        session_factory, amount=Decimal("1000.00"), counterparty="SLICE AUTOPAY"
    )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="1000.00", narration="SLICE AUTOPAY"),
            _stmt_txn(date="07/04/2026", amount="1000.00", narration="SLICE AUTOPAY"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert [entry["stmt_idx"] for entry in recon["matched"]] == [0]
    assert recon["matched"][0]["db_txn_id"] == a_id
    assert [entry["stmt_idx"] for entry in recon["missing"]] == [1]
    assert recon["missing"][0]["ambiguous"] is True

    imported, rows = await _import(session_factory, parsed, recon)

    assert len(imported) == 1
    assert [row.id for row in rows] == [a_id] + [row.id for row in imported]


@pytest.mark.anyio
async def test_counterparty_tiebreak_overrides_greedy_insertion_order(session_factory):
    """The greedy pick pairs rows to candidates in DB insertion order, so it
    can cross the correct pairing. The group-injective counterparty assignment
    reassigns each row to the candidate whose counterparty it names — proving
    the tiebreak is not just greedy luck.
    """
    await _seed_account(session_factory)
    # DB insertion order is SGST then CGST — the reverse of the statement rows.
    sgst_id = await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        counterparty="SGST ON FEE",
        raw_description="SGST ON FEE",
    )
    cgst_id = await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        counterparty="CGST ON FEE",
        raw_description="CGST ON FEE",
    )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="34.00", narration="CGST ON FEE"),
            _stmt_txn(date="07/04/2026", amount="34.00", narration="SGST ON FEE"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["missing"] == []
    assert {entry["stmt_idx"]: entry["db_txn_id"] for entry in recon["matched"]} == {
        0: cgst_id,
        1: sgst_id,
    }


@pytest.mark.anyio
async def test_a_missing_duplicate_row_spoils_the_counterparty_tiebreak(
    session_factory,
):
    """The tiebreak weighs the whole tie, not only the greedy winners.

    Two DB rows tie with no reference, but three statement rows name them: two
    say ``CGST ON FEE`` and one says ``SGST ON FEE``. The CGST candidate is
    named by two rows, so which one is really it stays genuinely ambiguous. The
    winners must not be kept matched just because the third row lost the greedy
    race — every row stays demoted.
    """
    await _seed_account(session_factory)
    await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        counterparty="CGST ON FEE",
        raw_description="CGST ON FEE",
    )
    await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        counterparty="SGST ON FEE",
        raw_description="SGST ON FEE",
    )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="34.00", narration="CGST ON FEE"),
            _stmt_txn(date="07/04/2026", amount="34.00", narration="SGST ON FEE"),
            _stmt_txn(date="07/04/2026", amount="34.00", narration="CGST ON FEE"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True, True]


@pytest.mark.parametrize(
    "db_order",
    [
        ("CGST ON FEE", "SGST ON FEE", "SGST"),
        ("CGST ON FEE", "SGST", "SGST ON FEE"),
    ],
)
@pytest.mark.anyio
async def test_an_unclaimed_tied_candidate_spoils_the_tiebreak(
    session_factory, db_order
):
    """A row that also names an unclaimed tied candidate stays ambiguous.

    Three DB rows tie with no reference: counterparties ``CGST ON FEE``,
    ``SGST ON FEE``, and the terser ``SGST``. Two statement rows name them.
    ``SGST ON FEE`` word-matches both ``SGST ON FEE`` and ``SGST``, so which
    candidate it is stays genuinely ambiguous — even though only two of the
    three candidates are ever claimed. The pair must stay demoted, and the
    outcome must not flip with DB insertion order.
    """
    await _seed_account(session_factory)
    for counterparty in db_order:
        await _seed_txn(
            session_factory,
            amount=Decimal("34.00"),
            counterparty=counterparty,
            raw_description=counterparty,
        )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="34.00", narration="CGST ON FEE"),
            _stmt_txn(date="07/04/2026", amount="34.00", narration="SGST ON FEE"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True]


@pytest.mark.parametrize(
    ("counterparty", "narration"),
    [
        pytest.param("CRED", "CREDIT MANTRA", id="latin-word"),
        pytest.param("CAFE", "CAFE\u0301TERIA CENTRAL", id="combining-accent"),
        pytest.param("राम", "रामा CENTRAL", id="devanagari-matra"),
        pytest.param("SHOP", "SHOP\u200cLIFT CENTRAL", id="zero-width-non-joiner"),
    ],
)
@pytest.mark.anyio
async def test_counterparty_containment_needs_a_whole_word(
    session_factory, counterparty, narration
):
    """Containment matches a whole counterparty, not a fragment of a longer
    word. A combining mark, a matra, or a join control continues the word.
    The first row singles out no candidate, so both rows stay demoted.
    """
    await _seed_account(session_factory)
    for name in (counterparty, "BOOKS"):
        await _seed_txn(
            session_factory,
            amount=Decimal("34.00"),
            counterparty=name,
            raw_description=name,
        )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="34.00", narration=narration),
            _stmt_txn(date="07/04/2026", amount="34.00", narration="BOOKS STORE"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True]


@pytest.mark.anyio
async def test_counterparty_tiebreak_ignores_a_different_card_namer(session_factory):
    """A row confirmed on a different card cannot be a candidate's transaction,
    so it must not spoil the tiebreak.

    Two same-card tax rows resolve by counterparty even though a third row on
    another card shares the amount and date and repeats one narration. Were the
    third row counted as a rival namer, the pair would demote on a card the
    candidates cannot belong to.
    """
    await _seed_account(session_factory)
    async with session_factory() as session:
        session.add(
            Card(account_id=ACCOUNT_ID, card_mask="4111XXXXXXXX1111", label="Add-on 1")
        )
        session.add(
            Card(account_id=ACCOUNT_ID, card_mask="4111XXXXXXXX2222", label="Add-on 2")
        )
        await session.commit()

    cgst_id = await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        counterparty="CGST ON FEE",
        raw_description="CGST ON FEE",
        card_mask="4111XXXXXXXX1111",
    )
    sgst_id = await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        counterparty="SGST ON FEE",
        raw_description="SGST ON FEE",
        card_mask="4111XXXXXXXX1111",
    )

    parsed = _parsed(
        [
            CcTransaction(
                date="07/04/2026",
                narration="CGST ON FEE",
                amount="34.00",
                card_number="4111XXXXXXXX1111",
                transaction_type="debit",
            ),
            CcTransaction(
                date="07/04/2026",
                narration="SGST ON FEE",
                amount="34.00",
                card_number="4111XXXXXXXX1111",
                transaction_type="debit",
            ),
            CcTransaction(
                date="07/04/2026",
                narration="CGST ON FEE",
                amount="34.00",
                card_number="4111XXXXXXXX2222",
                transaction_type="debit",
            ),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert {entry["narration"]: entry["db_txn_id"] for entry in recon["matched"]} == {
        "CGST ON FEE": cgst_id,
        "SGST ON FEE": sgst_id,
    }
    assert [entry["stmt_idx"] for entry in recon["missing"]] == [2]
    assert recon["missing"][0]["ambiguous"] is True


@pytest.mark.anyio
async def test_two_candidates_without_narration_evidence_still_demote(session_factory):
    """The tiebreak only adds resolutions. Two DB rows tie with no reference
    and their counterparties appear in neither statement narration, so nothing
    singles a candidate out — the pair stays demoted rather than guessed.
    """
    await _seed_account(session_factory)
    await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        counterparty="MERCHANT A",
        raw_description="MERCHANT A",
    )
    await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        counterparty="MERCHANT B",
        raw_description="MERCHANT B",
    )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="34.00", narration="UNRELATED ONE"),
            _stmt_txn(date="07/04/2026", amount="34.00", narration="UNRELATED TWO"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True]


@pytest.mark.anyio
async def test_reassignment_refreshes_the_decision_reason(session_factory):
    """A cross-date reassignment must refresh the evidence reason.

    The greedy pick recorded the candidate it first won; after the tiebreak
    swaps each row to a candidate one day away, the reason must describe the
    reassigned candidate (offset), not the greedily-won one (exact).
    """
    await _seed_account(session_factory)
    cgst_id = await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        transaction_date=parse_cc_date("07/04/2026"),
        counterparty="CGST ON FEE",
        raw_description="CGST ON FEE",
    )
    sgst_id = await _seed_txn(
        session_factory,
        amount=Decimal("34.00"),
        transaction_date=parse_cc_date("08/04/2026"),
        counterparty="SGST ON FEE",
        raw_description="SGST ON FEE",
    )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="34.00", narration="SGST ON FEE"),
            _stmt_txn(date="08/04/2026", amount="34.00", narration="CGST ON FEE"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    by_narration = {entry["narration"]: entry for entry in recon["matched"]}
    assert by_narration["SGST ON FEE"]["db_txn_id"] == sgst_id
    assert by_narration["CGST ON FEE"]["db_txn_id"] == cgst_id
    assert by_narration["SGST ON FEE"]["decision_reason"] == "matched_date_offset"
    assert by_narration["CGST ON FEE"]["decision_reason"] == "matched_date_offset"


@pytest.mark.anyio
async def test_a_reprocess_keeps_the_question_open(session_factory):
    """Verifies that reprocessing a statement keeps held rows open for user resolution.

    Without this check, repeated processing drops unresolved rows and leaves duplicate
    records.
    """
    await _seed_account(session_factory)
    await _seed_txn(session_factory, amount=Decimal("90.00"), counterparty="MERCHANT A")
    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="90.00", narration="MERCHANT A"),
            _stmt_txn(date="07/04/2026", amount="90.00", narration="MERCHANT A BRANCH"),
        ]
    )

    for _ in range(3):
        recon = await _reconcile(session_factory, parsed)
        _imported, rows = await _import(
            session_factory, parsed, recon, due_date="20/05/2026"
        )

    held = held_rows(cast(Reconciliation, recon))
    recorded = [row.id for row in rows if row.statement_upload_id is not None]

    assert len(rows) == 3
    assert len(held) == 2
    held_ids = [i for entry in held if (i := entry["imported_txn_id"]) is not None]
    assert sorted(held_ids) == sorted(recorded)


@pytest.mark.anyio
async def test_a_line_a_parser_missed_before_is_recorded(session_factory):
    """Verifies that newly discovered statement lines import as distinct transactions.

    Without this protection, new lines reuse existing copies and miss legitimate
    purchases.
    """
    await _seed_account(session_factory)
    before = _parsed(
        [_stmt_txn(date="07/04/2026", amount="90.00", narration=NARRATION)]
    )
    after = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="90.00", narration=NARRATION),
            _stmt_txn(date="07/04/2026", amount="90.00", narration=NARRATION),
            _stmt_txn(date="07/04/2026", amount="90.00", narration=NARRATION),
        ]
    )

    recon = await _reconcile(session_factory, before)
    await _import(session_factory, before, recon)

    sizes = []
    for _ in range(3):
        recon = await _reconcile(session_factory, after)
        _imported, rows = await _import(session_factory, after, recon)
        sizes.append(len(rows))

    assert sizes == [3, 3, 3]


@pytest.mark.anyio
async def test_a_held_row_takes_its_own_copy_not_a_neighbours(session_factory):
    """Verifies that a held row matches only its own recorded copy.

    Without this check, the row binds to an unrelated transaction with a lower ID.
    """
    await _seed_account(session_factory)

    # An earlier statement records a refund and another merchant's purchase,
    # both at this amount on this day. Their ids are the low ones.
    earlier = _parsed(
        [_stmt_txn(date="07/04/2026", amount="500.00", narration="OTHER SHOP PUNE")]
    )
    earlier.payments_refunds = [
        _stmt_txn(date="07/04/2026", amount="500.00", narration="SAMPLE FUEL PUNE")
    ]
    recon = await _reconcile(session_factory, earlier)
    await _import(session_factory, earlier, recon)

    # This statement prints one purchase, held because an alert names it.
    await _seed_txn(
        session_factory,
        amount=Decimal("500.00"),
        counterparty="SAMPLE FUEL",
        raw_description="SAMPLE FUEL",
    )
    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="500.00", narration="SAMPLE FUEL PUNE"),
            _stmt_txn(
                date="07/04/2026", amount="500.00", narration="SAMPLE FUEL STN PUNE"
            ),
        ]
    )

    for _ in range(3):
        recon = await _reconcile(session_factory, parsed)
        _imported, rows = await _import(session_factory, parsed, recon)

    by_id = {row.id: row for row in rows}
    held = [
        entry
        for entry in recon["missing"]
        if entry.get("ambiguous") and entry.get("imported_txn_id") is not None
    ]

    assert held, "the shape must hold a row, or it tests nothing"
    for entry in held:
        copy = by_id[entry["imported_txn_id"]]
        assert copy.direction == entry["direction"]
        assert copy.counterparty == entry["narration"]


@pytest.mark.anyio
async def test_a_fuel_surcharge_is_held_for_a_person(session_factory):
    """Verifies that the reconciler holds surcharged transactions for user review when
    merchants match.

    Without this check, the reconciler does not hold the row or name the stored row
    as its candidate.
    """
    await _seed_account(session_factory)
    stored_id = await _seed_txn(
        session_factory,
        amount=Decimal("2500.00"),
        counterparty="SAMPLE ENTERPRISES",
        raw_description=None,
    )
    parsed = _parsed(
        [
            _stmt_txn(
                date="07/04/2026",
                amount="2,529.00",
                narration="MW SAMPLE ENTERPRISES Pune",
            )
        ]
    )

    recon = await _reconcile(session_factory, parsed)

    assert recon["missing"][0]["ambiguous"] is True
    assert recon["missing"][0]["candidate_transaction_ids"] == [stored_id]


@pytest.mark.anyio
async def test_a_folded_alert_is_never_handed_out_as_a_copy(session_factory):
    """Verifies that a folded card alert never counts as a statement's recorded copy.

    Without this rule, a later fold can delete the purchase itself.
    """
    await _seed_account(session_factory)
    alert_id = await _seed_txn(
        session_factory,
        amount=Decimal("2500.00"),
        counterparty="SAMPLE ENTERPRISES",
        raw_description=None,
        card_mask="1111",
    )
    parsed = _parsed(
        [
            _stmt_txn(
                date="07/04/2026",
                amount="2,529.00",
                narration="MW SAMPLE ENTERPRISES Pune",
            )
        ]
    )

    recon = await _reconcile(session_factory, parsed)
    await _import(session_factory, parsed, recon)
    async with session_factory() as session:
        upload = (
            await session.scalars(
                select(StatementUpload).order_by(StatementUpload.id.desc())
            )
        ).first()
        upload.reconciliation_data = reconciliation_to_json(recon)
        await session.commit()
        upload_id = upload.id

    [entry] = held_rows(cast(Reconciliation, recon))
    async with session_factory() as session:
        folded = await answer(
            session, upload_id, entry["stmt_idx"], row_digest(entry), alert_id
        )
    assert folded.outcome == "merged"

    recon = await _reconcile(session_factory, parsed)
    _imported, rows = await _import(session_factory, parsed, recon)

    assert alert_id in [row.id for row in rows]
    assert alert_id not in [entry.get("imported_txn_id") for entry in recon["missing"]]


@pytest.mark.anyio
async def test_another_merchant_is_a_second_purchase(session_factory):
    """Verifies that statement rows for different merchants import as distinct
    purchases.

    Without this check, unrelated purchases with similar amounts prompt unnecessary
    merges.
    """
    await _seed_account(session_factory)
    stored_id = await _seed_txn(
        session_factory,
        amount=Decimal("2500.00"),
        counterparty="SAMPLE PHARMACY",
        raw_description="SAMPLE PHARMACY",
    )
    parsed = _parsed(
        [_stmt_txn(date="07/04/2026", amount="2,529.00", narration=NARRATION)]
    )

    recon = await _reconcile(session_factory, parsed)
    imported, rows = await _import(session_factory, parsed, recon)

    assert recon["missing"][0]["ambiguous"] is False
    assert recon["missing"][0]["candidate_transaction_ids"] == []
    assert len(imported) == 1
    assert sorted(row.amount for row in rows) == [
        Decimal("2500.00"),
        Decimal("2529.00"),
    ]
    assert stored_id in [row.id for row in rows]


@pytest.mark.anyio
async def test_a_banded_row_does_not_outrank_an_exact_match_a_day_away(
    session_factory,
):
    """Verifies that exact amount matches take priority over nearby amounts from the
    same day.

    Without this rule, a nearby surcharge steals the match and leaves the real purchase
    unlinked.
    """
    await _seed_account(session_factory)
    near_id = await _seed_txn(session_factory, amount=Decimal("995.00"))
    exact_id = await _seed_txn(
        session_factory,
        amount=Decimal("1000.00"),
        transaction_date=parse_cc_date("06/04/2026"),
    )
    parsed = _parsed(
        [_stmt_txn(date="07/04/2026", amount="1,000.00", narration=NARRATION)]
    )

    recon = await _reconcile(session_factory, parsed)
    imported, rows = await _import(session_factory, parsed, recon)

    assert [entry["db_txn_id"] for entry in recon["matched"]] == [exact_id]
    amounts = {row.id: row.amount for row in rows}
    assert amounts == {near_id: Decimal("995.00"), exact_id: Decimal("1000.00")}
    assert imported == []


@pytest.mark.anyio
async def test_two_exact_matches_inside_the_band_both_match(session_factory):
    """Verifies that distinct transactions with exact matches pair directly when their
    amounts sit inside the band.

    Without this rule, contention demotes valid matches to ambiguous prompts.
    """
    await _seed_account(session_factory)
    low_id = await _seed_txn(
        session_factory, amount=Decimal("149.00"), counterparty="SAMPLE CAFE"
    )
    high_id = await _seed_txn(
        session_factory, amount=Decimal("150.00"), counterparty="SAMPLE CAFE"
    )
    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="149.00", narration="SAMPLE CAFE Pune"),
            _stmt_txn(
                date="07/04/2026", amount="150.00", narration="SAMPLE CAFE Mumbai"
            ),
        ]
    )

    recon = await _reconcile(session_factory, parsed)
    imported, rows = await _import(session_factory, parsed, recon)

    assert sorted(entry["db_txn_id"] for entry in recon["matched"]) == [
        low_id,
        high_id,
    ]
    assert recon["missing"] == []
    assert imported == []
    assert {row.id: row.amount for row in rows} == {
        low_id: Decimal("149.00"),
        high_id: Decimal("150.00"),
    }


@pytest.mark.anyio
async def test_a_row_in_another_currency_is_not_a_candidate(session_factory):
    """Verifies that foreign currency transactions are not candidates for domestic
    statement rows.

    Without this guard, mismatched currencies trigger invalid settlement prompts.
    """
    await _seed_account(session_factory)
    await _seed_txn(
        session_factory,
        amount=Decimal("2500.00"),
        currency="USD",
        counterparty="SWIGGY LIMITED",
    )
    parsed = _parsed(
        [_stmt_txn(date="07/04/2026", amount="2,529.00", narration=NARRATION)]
    )

    recon = await _reconcile(session_factory, parsed)
    imported, rows = await _import(session_factory, parsed, recon)

    # The statement row is new, so it imports. The foreign row is untouched.
    assert [entry["ambiguous"] for entry in recon["missing"]] == [False]
    assert len(imported) == 1
    assert {row.currency for row in rows} == {"USD", "INR"}
