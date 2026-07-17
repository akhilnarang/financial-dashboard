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
from typing import Literal

import pytest
from cc_parser.parsers.models import Transaction as CcTransaction
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import (
    Account,
    Base,
    Card,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.statements import cc as cc_module
from financial_dashboard.services.statements.cc import (
    import_missing_cc_txns,
    load_account_card_masks,
    parse_cc_date,
    reconcile_statement,
)

ACCOUNT_ID = 1
CARD_NUMBER = "4111XXXXXXXX9012"

NARRATION = "SWIGGY LIMITED           BANGALORE   IN"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(cc_module, "async_session", maker)
    yield maker
    await engine.dispose()


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


async def _import(maker, parsed, recon) -> tuple[list, list]:
    """Drive the real ``import_missing_cc_txns`` and return
    (imported transactions, every DB row afterwards)."""
    async with maker() as session:
        upload = StatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="parsed",
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
        # A short mask is blind, not empty: XX34 still says the card ends 34,
        # and this account's card ends 12 — a real conflict on a shown digit.
        pytest.param("XX34", False, id="short-mask-conflicts-on-visible-digit"),
        # Non-canonical spelling of the same card must not read as a
        # conflict; refusing here would import a duplicate.
        pytest.param("XX9012", True, id="non-canonical-spelling-same-card"),
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

    assert imported == []
    assert [row.id for row in rows] == [txn_id]
    if not matches:
        assert "ambiguous" in recon["missing"][0]["import_error"]
        assert rows[0].counterparty == "MERCHANT A"


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
    """Account grain widens the rule to the account's own cards — no further.

    The DB row carries a mask belonging to a card on a *different* account:
    the picker refuses the pairing, and the statement row is held back rather
    than imported, since a refusal is not evidence the transaction is absent.
    """
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
    assert len(recon["missing"]) == 1
    assert recon["missing"][0]["ambiguous"] is True

    imported, rows = await _import(session_factory, parsed, recon)

    assert imported == []
    assert [row.id for row in rows] == [other_id]
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

    assert imported == []
    for entry in recon["missing"]:
        assert entry["imported"] is False
        assert "ambiguous" in entry["import_error"]
    assert [row.id for row in rows] == [a_id]


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

    assert imported == []
    assert [row.id for row in rows] == [a_id]


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
        # No trailing visible digits: a BIN denotes no particular card.
        pytest.param(
            ["5100XXXXXXXX9012"], "5100XXXXXXXX", None, id="bin-only-mask-refused"
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
        # ...and the multiplicity refusal is the half of the bargain that
        # makes the short suffix safe to trust at all.
        pytest.param(
            ["5100XXXXXXXX9067", "4111XXXXXXXX1167"],
            "XXXX XXXX XXXX XX67",
            None,
            id="sbi-short-suffix-two-reaches-refused",
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
async def test_find_account_matches_through_the_cards_table(session_factory):
    """A card the account carries in the cards table identifies it just as an
    account_number does — both routes are gathered before either decides."""
    await _seed_cc_account(
        session_factory, account_id=1, account_number="5100XXXXXXXX0001"
    )
    async with session_factory() as session:
        session.add(Card(account_id=1, card_mask="4111XXXXXXXX5566", label="Addon"))
        await session.commit()

    parsed = SimpleNamespace(card_number="XXXX XXXX XXXX 5566")

    account = await cc_module._find_account("hdfc", parsed)
    assert account is not None
    assert account.id == 1


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


@pytest.mark.parametrize(
    "stored",
    [
        "4000-XXXX-XXXX-1234",
        "4000 xxxx xxxx 1234",
        "4000XXXXXXXX1234",
    ],
)
@pytest.mark.anyio
async def test_find_account_matches_a_non_canonical_stored_mask(
    session_factory, stored
):
    """The stored side is a mask too: dashes, spaces and lowercase ``x`` are
    cosmetic. Compared raw, the separators shift every digit out of alignment
    and the account silently stops matching its own statements."""
    await _seed_cc_account(session_factory, account_id=1, account_number=stored)

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

    assert imported == []
    assert "ambiguous" in recon["missing"][0]["import_error"]
    assert [row.id for row in rows] == [a_id]


@pytest.mark.anyio
async def test_rivals_with_differing_narrations_still_demote_the_winner(
    session_factory,
):
    """Rows that disagree on the narration are a real choice: the winner
    would write ``CGST ON FEE`` onto a row the loser was going to call
    ``SGST ON FEE``, and nothing here says which is right — so neither is
    committed."""
    await _seed_account(session_factory)
    a_id = await _seed_txn(
        session_factory, amount=Decimal("90.00"), counterparty="MERCHANT A"
    )

    parsed = _parsed(
        [
            _stmt_txn(date="07/04/2026", amount="90.00", narration="CGST ON FEE"),
            _stmt_txn(date="07/04/2026", amount="90.00", narration="SGST ON FEE"),
        ]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True]

    imported, rows = await _import(session_factory, parsed, recon)

    assert imported == []
    assert [row.id for row in rows] == [a_id]


@pytest.mark.anyio
async def test_a_row_masked_with_a_deleted_card_is_held_back_not_reimported(
    session_factory,
):
    """Deleting a card must not make its history re-importable.

    The account had an add-on and the DB holds its purchase; the card is then
    removed exactly the way ``card_delete`` removes it — ``card_id`` cleared,
    ``card_mask`` left as it was. The card check now refuses the pairing, but
    that refusal must not be read as "the DB does not hold this transaction":
    the row stays in the candidate set, the statement row surfaces as
    ambiguous, and the count does not grow.
    """
    await _seed_account(session_factory)
    async with session_factory() as session:
        card = Card(account_id=ACCOUNT_ID, card_mask="4111XXXXXXXX7788", label="Add-on")
        session.add(card)
        await session.flush()
        card_id = card.id
        await session.commit()

    addon_txn_id = await _seed_txn(
        session_factory, card_mask="4111XXXXXXXX7788", card_id=card_id
    )

    async with session_factory() as session:
        await session.execute(
            update(Transaction)
            .where(Transaction.card_id == card_id)
            .values(card_id=None)
        )
        await session.delete(await session.get(Card, card_id))
        await session.commit()

    parsed = _parsed(
        [_stmt_txn(date="07/04/2026", amount="450.00", narration=NARRATION)]
    )
    recon = await _reconcile(session_factory, parsed)

    assert recon["matched"] == []
    assert len(recon["missing"]) == 1
    assert recon["missing"][0]["ambiguous"] is True

    imported, rows = await _import(session_factory, parsed, recon)

    assert imported == []
    assert "ambiguous" in recon["missing"][0]["import_error"]
    assert [row.id for row in rows] == [addon_txn_id]
