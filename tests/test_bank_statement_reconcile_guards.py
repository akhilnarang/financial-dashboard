"""Contention guards on the bank-statement reconciler.

``reconcile_bank_statement`` pairs statement rows with DB rows greedily, and
the rows it fails to pair are handed to the import path. That is safe only
for a row whose transaction really is new: a row that *could* have claimed a
DB candidate but ended up with none lost a race or was refused, and its
transaction is plausibly already in the DB under the row a rival took —
importing it would count the money twice. So the reconciler computes each
row's candidate set (the matcher's own ±1-day window and compatibility rule,
across both passes), tags an unmatched row with a non-empty set
``ambiguous``, and the import path holds it back.

The other half of the bargain is guarded hardest here: a row whose candidate
set is *empty* must still import — over-refusing silently shortens the
ledger. Every case is driven through the real importer and asserts on the
resulting row count.
"""

from decimal import Decimal
from typing import Literal

import pytest
from bank_statement_parser.models import BankTransaction, ParsedBankStatement
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import (
    Account,
    Base,
    BankStatementUpload,
    Transaction,
)
from financial_dashboard.services.statements import bank as bank_module
from financial_dashboard.services.statements import shared as shared_module
from financial_dashboard.services.statements.bank import reconcile_bank_statement

ACCOUNT_ID = 1
ACCOUNT_NUMBER = "9876500011122233"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session_factory(monkeypatch):
    """In-memory DB installed as the global ``async_session`` of both the
    reconciler module and the statement-retry path."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(bank_module, "async_session", maker)
    monkeypatch.setattr(shared_module, "async_session", maker)
    yield maker
    await engine.dispose()


def _stmt_txn(
    *,
    date: str,
    amount: str,
    narration: str,
    direction: Literal["debit", "credit"] = "debit",
    counterparty: str | None = None,
    ref: str | None = None,
    channel: str | None = None,
) -> BankTransaction:
    return BankTransaction(
        date=date,
        narration=narration,
        amount=amount,
        transaction_type=direction,
        counterparty=counterparty,
        reference_number=ref,
        channel=channel,
    )


def _parsed(transactions: list[BankTransaction]) -> ParsedBankStatement:
    return ParsedBankStatement(
        file="statement.pdf",
        bank="hdfc",
        account_number=ACCOUNT_NUMBER,
        transactions=transactions,
    )


async def _seed_account(maker) -> None:
    async with maker() as session:
        session.add(
            Account(
                id=ACCOUNT_ID,
                bank="hdfc",
                label="HDFC Savings",
                type="bank_account",
                account_number=ACCOUNT_NUMBER,
            )
        )
        await session.commit()


async def _seed_txn(maker, **overrides) -> int:
    """Insert one DB transaction (₹2,500 debit on 07 Apr unless overridden)."""
    fields = {
        "account_id": ACCOUNT_ID,
        "bank": "hdfc",
        "email_type": "bank_statement",
        "direction": "debit",
        "amount": Decimal("2500.00"),
        "currency": "INR",
        "transaction_date": bank_module._parse_date("07/04/2026"),
        "counterparty": "MERCHANT A",
        "raw_description": "MERCHANT A",
        "channel": "card",
        "reference_number": None,
    }
    fields.update(overrides)
    async with maker() as session:
        txn = Transaction(**fields)
        session.add(txn)
        await session.commit()
        return txn.id


async def _reconcile(maker, parsed: ParsedBankStatement) -> dict:
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
    return reconcile_bank_statement(parsed, db_txns, ACCOUNT_ID)


async def _run_real_reparse(maker, monkeypatch, parsed: ParsedBankStatement):
    """Drive ``retry_bank_statement_upload`` — the real import path — over
    ``parsed``, and return (every DB row afterwards, the upload row)."""
    monkeypatch.setattr(
        shared_module, "parse_bank_statement", lambda path, bank, password=None: parsed
    )
    async with maker() as session:
        upload = BankStatementUpload(
            account_id=ACCOUNT_ID,
            bank="hdfc",
            filename="statement.pdf",
            file_path="/nonexistent/statement.pdf",
            status="password_required",
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    assert await shared_module.retry_bank_statement_upload(upload_id, "secret") is True

    async with maker() as session:
        rows = list((await session.execute(select(Transaction))).scalars().all())
        upload = await session.get(BankStatementUpload, upload_id)
    return rows, upload


@pytest.mark.anyio
async def test_row_the_matcher_refused_is_not_imported(session_factory, monkeypatch):
    """A statement row the matcher declined to pair — two DB candidates it
    cannot be told apart from — is unresolved, not new: its transaction is
    plausibly already stored under one of them, and importing would make a
    third copy. Same rule as the race-loser case; contrast
    ``test_new_rows_import_beside_an_incompatible_db_row``, where the
    candidate set is empty and the row must import."""
    await _seed_account(session_factory)
    a_id = await _seed_txn(session_factory, counterparty="MERCHANT A")
    b_id = await _seed_txn(session_factory, counterparty="MERCHANT B")

    parsed = _parsed(
        [
            _stmt_txn(
                date="07/04/2026",
                amount="2,500.00",
                narration="MERCHANT B RETAIL",
                counterparty="MERCHANT B",
            )
        ]
    )
    recon = await _reconcile(session_factory, parsed)
    assert recon["matched"] == []
    assert recon["missing"][0]["ambiguous"] is True

    rows, upload = await _run_real_reparse(session_factory, monkeypatch, parsed)

    assert sorted(row.id for row in rows) == sorted([a_id, b_id])
    assert upload.imported_count == 0
    assert upload.missing_count == 1


@pytest.mark.parametrize(
    ("db_overrides", "stmt_rows"),
    [
        # One DB row, two same-day statement rows chasing it: the loser is
        # already in the DB under the row the winner took, and the winner won
        # on statement order, which is not evidence.
        pytest.param(
            {},
            [
                dict(
                    date="07/04/2026",
                    amount="2,500.00",
                    narration="MERCHANT B RETAIL",
                    counterparty="MERCHANT B",
                ),
                dict(
                    date="07/04/2026",
                    amount="2,500.00",
                    narration="MERCHANT A RETAIL",
                    counterparty="MERCHANT A",
                ),
            ],
            id="same-day-rivals",
        ),
        # Contention is about candidate sets, not proximity: rows on 06 and
        # 08 Apr both reach the 07 Apr DB row through the ±1-day window.
        pytest.param(
            {},
            [
                dict(
                    date="06/04/2026",
                    amount="2,500.00",
                    narration="MERCHANT B RETAIL",
                    counterparty="MERCHANT B",
                ),
                dict(
                    date="08/04/2026",
                    amount="2,500.00",
                    narration="MERCHANT A RETAIL",
                    counterparty="MERCHANT A",
                ),
            ],
            id="two-days-apart-rivals",
        ),
        # Counterparty is not the whole refresh identity: rows agreeing on it
        # but arriving through different channels are still a real choice.
        pytest.param(
            {"amount": Decimal("1000.00"), "counterparty": "SLICE AUTOPAY"},
            [
                dict(
                    date="07/04/2026",
                    amount="1,000.00",
                    narration="SLICE AUTOPAY",
                    counterparty="SLICE AUTOPAY",
                    channel="autopay",
                ),
                dict(
                    date="07/04/2026",
                    amount="1,000.00",
                    narration="SLICE AUTOPAY",
                    counterparty="SLICE AUTOPAY",
                    channel="upi",
                ),
            ],
            id="differ-only-on-channel",
        ),
        # Interchangeability is judged on the value a refresh lands, not the
        # raw field: with no parsed counterparty the refresh falls back to
        # the narration, and the narrations name different taxes.
        pytest.param(
            {"amount": Decimal("90.00"), "counterparty": None},
            [
                dict(date="07/04/2026", amount="90.00", narration="CGST ON FEE"),
                dict(date="07/04/2026", amount="90.00", narration="SGST ON FEE"),
            ],
            id="told-apart-by-narration-fallback",
        ),
    ],
)
@pytest.mark.anyio
async def test_distinguishable_rivals_hold_back_winner_and_loser(
    session_factory, monkeypatch, db_overrides, stmt_rows
):
    """One DB row, two statement rows that both reach it and would write
    different things onto it. Neither pairing can be trusted: the loser must
    not import as a duplicate, and the winner's order-decided match must not
    be committed. The DB finishes with exactly the row it started with."""
    await _seed_account(session_factory)
    a_id = await _seed_txn(session_factory, **db_overrides)

    parsed = _parsed([_stmt_txn(**row) for row in stmt_rows])
    recon = await _reconcile(session_factory, parsed)
    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True, True]

    rows, upload = await _run_real_reparse(session_factory, monkeypatch, parsed)

    assert [row.id for row in rows] == [a_id]
    assert upload.imported_count == 0
    assert upload.missing_count == 2


@pytest.mark.anyio
async def test_incompatible_reused_reference_does_not_contend_with_valid_match(
    session_factory, monkeypatch
):
    """A contradictory amount remains ambiguous but cannot demote a valid match."""
    await _seed_account(session_factory)
    a_id = await _seed_txn(
        session_factory, counterparty="MERCHANT A", reference_number="REF12345"
    )

    parsed = _parsed(
        [
            _stmt_txn(
                date="07/04/2026",
                amount="2,500.00",
                narration="MERCHANT B RETAIL",
                counterparty="MERCHANT B",
                ref="REF12345",
            ),
            # Same reference but a contradictory amount. Keep it as ambiguous
            # evidence without treating it as a rival for the valid winner.
            _stmt_txn(
                date="09/04/2026",
                amount="3,100.00",
                narration="MERCHANT A RETAIL",
                counterparty="MERCHANT A",
                ref="REF12345",
            ),
        ]
    )
    recon = await _reconcile(session_factory, parsed)
    assert [entry["stmt_idx"] for entry in recon["matched"]] == [0]
    assert [entry["stmt_idx"] for entry in recon["missing"]] == [1]
    assert recon["missing"][0]["ambiguous"] is True
    assert recon["missing"][0]["candidate_transaction_ids"] == [a_id]

    rows, upload = await _run_real_reparse(session_factory, monkeypatch, parsed)

    assert [row.id for row in rows] == [a_id]
    assert upload.imported_count == 0
    assert upload.missing_count == 1


@pytest.mark.anyio
async def test_new_rows_import_beside_an_incompatible_db_row(
    session_factory, monkeypatch
):
    """The ordinary-miss guarantee, and the thing contention must never cost.

    A DB row shares the date and amount but carries a different reference, so
    the matcher correctly refuses it. Judging contention by raw date/amount
    occupancy would see a crowded window and refuse to import the two
    genuinely new rows; judging by candidate sets — the matcher's own
    compatibility rule — leaves both sets empty, and both import.
    """
    await _seed_account(session_factory)
    old_id = await _seed_txn(
        session_factory,
        counterparty="MERCHANT OLD",
        raw_description="MERCHANT OLD",
        reference_number="OLDREF9",
    )

    parsed = _parsed(
        [
            _stmt_txn(
                date="07/04/2026",
                amount="2,500.00",
                narration="MERCHANT NEW ONE",
                counterparty="MERCHANT NEW ONE",
                ref="NEWREF1",
            ),
            _stmt_txn(
                date="07/04/2026",
                amount="2,500.00",
                narration="MERCHANT NEW TWO",
                counterparty="MERCHANT NEW TWO",
                ref="NEWREF2",
            ),
        ]
    )
    recon = await _reconcile(session_factory, parsed)
    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [False, False]

    rows, upload = await _run_real_reparse(session_factory, monkeypatch, parsed)

    assert upload.imported_count == 2
    assert upload.missing_count == 0
    assert sorted(row.reference_number for row in rows) == [
        "NEWREF1",
        "NEWREF2",
        "OLDREF9",
    ]
    # The incompatible DB row was neither matched nor rewritten.
    old_row = next(row for row in rows if row.id == old_id)
    assert old_row.counterparty == "MERCHANT OLD"


@pytest.mark.anyio
async def test_one_statement_row_facing_two_db_rows_on_one_reference(
    session_factory, monkeypatch
):
    """A reference bucket is not an identity, so it decides only when it
    decides alone: ``uq_transactions_ref`` keys on ``bank`` as well, so two
    DB rows from different banks may share a reference while agreeing on
    nothing else. Taking whichever the query returned first would reconcile
    against the wrong bank's row; instead the row falls through to the date
    pass, finds nothing, and surfaces for a human."""
    await _seed_account(session_factory)
    first_id = await _seed_txn(
        session_factory,
        bank="hdfc",
        counterparty="MERCHANT A",
        transaction_date=bank_module._parse_date("24/04/2026"),
        reference_number="REF10001",
    )
    second_id = await _seed_txn(
        session_factory,
        bank="icici",
        counterparty="MERCHANT Z",
        amount=Decimal("9900.00"),
        transaction_date=bank_module._parse_date("20/04/2026"),
        reference_number="REF10001",
    )

    parsed = _parsed(
        [
            _stmt_txn(
                date="25/04/2026",
                amount="7,700.00",
                narration="MERCHANT Q RETAIL",
                counterparty="MERCHANT Q",
                ref="REF10001",
            )
        ]
    )
    recon = await _reconcile(session_factory, parsed)
    assert recon["matched"] == []
    assert [entry["ambiguous"] for entry in recon["missing"]] == [True]

    rows, upload = await _run_real_reparse(session_factory, monkeypatch, parsed)

    assert sorted(row.id for row in rows) == sorted([first_id, second_id])
    assert upload.imported_count == 0
    assert upload.missing_count == 1


@pytest.mark.anyio
async def test_reference_claim_still_contends_with_a_fuzzy_row(
    session_factory, monkeypatch
):
    """Contention spans the two passes. The parser emitted the same ₹2,500
    transaction twice — once with the reference, once without — and pass 1
    claims the DB row for the first. If the second row's candidate set were
    built against what pass 1 consumed, it would read as a brand-new
    transaction and import a duplicate; instead the sets are evaluated as
    though nothing had been consumed and it is held back. The winner keeps
    its match — the two rows would write the same counterparty either way —
    but the loser is still a second copy of a transaction the DB holds once.
    """
    await _seed_account(session_factory)
    a_id = await _seed_txn(
        session_factory, counterparty="MERCHANT A", reference_number="REF12399"
    )

    parsed = _parsed(
        [
            _stmt_txn(
                date="07/04/2026",
                amount="2,500.00",
                narration="MERCHANT A RETAIL",
                counterparty="MERCHANT A",
                ref="REF12399",
            ),
            _stmt_txn(
                date="07/04/2026",
                amount="2,500.00",
                narration="MERCHANT A RETAIL",
                counterparty="MERCHANT A",
            ),
        ]
    )
    recon = await _reconcile(session_factory, parsed)
    assert [entry["stmt_idx"] for entry in recon["matched"]] == [0]
    assert recon["matched"][0]["db_txn_id"] == a_id
    assert [entry["stmt_idx"] for entry in recon["missing"]] == [1]
    assert recon["missing"][0]["ambiguous"] is True

    rows, upload = await _run_real_reparse(session_factory, monkeypatch, parsed)

    assert [row.id for row in rows] == [a_id]
    assert upload.imported_count == 0
    assert upload.missing_count == 1


@pytest.mark.anyio
async def test_interchangeable_rivals_leave_the_winners_match_alone(
    session_factory, monkeypatch
):
    """A pairing with no wrong answer is not worth refusing: two rows
    identical in everything a refresh would write contend for one DB row, so
    whichever won, the refresh lands the same values. The winner keeps its
    match; the loser is still held back, because interchangeable rivals do
    not make a duplicate any less of a duplicate."""
    await _seed_account(session_factory)
    a_id = await _seed_txn(
        session_factory,
        amount=Decimal("1000.00"),
        counterparty="SLICE AUTOPAY",
        raw_description="SLICE AUTOPAY",
    )

    parsed = _parsed(
        [
            _stmt_txn(
                date="07/04/2026",
                amount="1,000.00",
                narration="SLICE AUTOPAY",
                counterparty="SLICE AUTOPAY",
                channel="autopay",
            ),
            _stmt_txn(
                date="07/04/2026",
                amount="1,000.00",
                narration="SLICE AUTOPAY",
                counterparty="SLICE AUTOPAY",
                channel="autopay",
            ),
        ]
    )
    recon = await _reconcile(session_factory, parsed)
    assert [entry["stmt_idx"] for entry in recon["matched"]] == [0]
    assert recon["matched"][0]["db_txn_id"] == a_id
    assert [entry["stmt_idx"] for entry in recon["missing"]] == [1]
    assert recon["missing"][0]["ambiguous"] is True

    rows, upload = await _run_real_reparse(session_factory, monkeypatch, parsed)

    assert [row.id for row in rows] == [a_id]
    assert upload.imported_count == 0
    assert upload.missing_count == 1
