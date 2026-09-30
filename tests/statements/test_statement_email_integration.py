"""Real end-to-end integration tests for the statement email pipelines.

Exercises ``process_statement_email`` (CC) and ``process_bank_statement_email``
against an in-memory SQLite session factory, monkeypatching only the parser
adapter boundary (``_parse_pdf_bytes_sync``) — the reconciliation, import,
snapshot, and enrichment services all run for real.

Covers: subject/PDF/account gates; clean parse; encrypted (stored password,
single-account password_required, multi-account refusal); non-password
parse_error; card resolution via the cards table; exact + ±1 date;
same-reference contention; duplicate and generic per-row import errors;
malformed rows; balance verification; snapshot emission; counterparty
enrichment; notifications threshold branch without network.
"""

import datetime
import json
from decimal import Decimal
from email.message import EmailMessage

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

import financial_dashboard.services.statements.bank as bank_module
import financial_dashboard.services.statements.cc as cc_module
from financial_dashboard.db import (
    Account,
    BankStatementUpload,
    BalanceSnapshot,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.statements.bank import (
    BankStatementProcessingError,
    process_bank_statement_email,
)
from financial_dashboard.services.statements.cc import process_statement_email

from . import _helpers as h


# ---------------------------------------------------------------------------
# CC: gates
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_gates_return_none(maker, statements_dir, monkeypatch):
    monkeypatch.setattr(
        cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(h.cc_parsed())
    )
    await h.add_cc_account(maker)
    for subject in ("Your monthly offer", "Account statement for July 2026"):
        raw = h.email_with_pdf(subject=subject)
        assert await process_statement_email("hdfc", raw, subject) is None

    msg = EmailMessage()
    msg["Subject"] = "Credit card statement"
    msg["From"] = "x@hdfc.com"
    msg.set_content("no attachment")
    result = await process_statement_email(
        "hdfc", msg.as_bytes(), "Credit card statement"
    )
    assert result is None

    async with maker() as session:
        assert (await session.execute(select(StatementUpload))).first() is None


# ---------------------------------------------------------------------------
# CC: clean parse + reconciliation + import
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_clean_parse_imports_missing_and_emits_snapshot(
    maker, statements_dir, monkeypatch
):
    acc_id = await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="01/07/2026", amount="1,000.00", narration="AMAZON")
        ],
        payments_refunds=[
            h.cc_txn(
                date="02/07/2026",
                amount="5,000.00",
                narration="PAYMENT RECEIVED",
                transaction_type="credit",
            )
        ],
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")

    assert result is not None
    assert result["matched"] == 0
    assert result["missing"] == 2
    assert result["imported"] == 2

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert upload.account_id == acc_id
        assert upload.status == "imported"  # all missing imported
        assert upload.parsed_txn_count == 2
        assert upload.matched_count == 0
        assert upload.missing_count == 0
        assert upload.imported_count == 2
        assert upload.error is None

        txns = {
            t.direction: t
            for t in (await session.execute(select(Transaction))).scalars().all()
        }
        assert txns["credit"].counterparty == "PAYMENT RECEIVED"
        txn = txns["debit"]
        assert txn.account_id == acc_id
        assert txn.email_type == "cc_statement"
        assert txn.direction == "debit"
        assert txn.amount == Decimal("1000.00")
        assert txn.counterparty == "AMAZON"
        assert txn.channel == "cc_statement"
        assert txn.statement_upload_id == upload.id

        # CC snapshot (liability, cc_outstanding) emitted from total_amount_due.
        snaps = (await session.execute(select(BalanceSnapshot))).scalars().all()
        assert any(s.account_id == acc_id for s in snaps)


@pytest.mark.anyio
async def test_cc_exact_and_plus_minus_one_day_match(
    maker, statements_dir, monkeypatch
):
    """Exact and +1 day rows match. A generic placeholder counterparty on a
    matched row takes the statement narration."""
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        session.add_all(
            [
                Transaction(
                    account_id=acc_id,
                    bank="hdfc",
                    email_type="cc_txn",
                    direction="debit",
                    amount=Decimal("500.00"),
                    transaction_date=datetime.date(2026, 7, 5),
                    counterparty="payment received",
                ),
                Transaction(
                    account_id=acc_id,
                    bank="hdfc",
                    email_type="cc_txn",
                    direction="debit",
                    amount=Decimal("750.00"),
                    transaction_date=datetime.date(2026, 7, 11),
                ),
            ]
        )
        await session.commit()

    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="05/07/2026", amount="500.00", narration="EXACT"),
            h.cc_txn(date="10/07/2026", amount="750.00", narration="PLUSONE"),
        ]
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result["matched"] == 2
    assert result["missing"] == 0
    assert result["enriched"] == 2

    async with maker() as session:
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 2
        assert {t.counterparty for t in txns} == {"EXACT", "PLUSONE"}


@pytest.mark.anyio
async def test_cc_adjustment_pairs_high_low_confidence(
    maker, statements_dir, monkeypatch
):
    """Adjustment pairs are surfaced in reconciliation_data regardless of
    confidence; totals only sum high-confidence legs."""
    await h.add_cc_account(maker)
    debit = h.cc_txn(date="01/07/2026", amount="1,000.00", narration="AMAZON")
    credit = h.cc_txn(
        date="03/07/2026",
        amount="1,000.00",
        narration="AMAZON REFUND",
        transaction_type="credit",
    )
    low_debit = h.cc_txn(date="10/07/2026", amount="200.00", narration="OTHER")
    low_credit = h.cc_txn(
        date="12/07/2026",
        amount="150.00",
        narration="OTHER REFUND",
        transaction_type="credit",
    )
    parsed = h.cc_parsed(
        transactions=[debit, low_debit],
        payments_refunds=[credit, low_credit],
        adjustment_pairs=[
            h.cc_adjustment_pair(
                pair_id="p1", confidence="high", debit=debit, credit=credit
            ),
            h.cc_adjustment_pair(
                pair_id="p2", confidence="low", debit=low_debit, credit=low_credit
            ),
        ],
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result is not None

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        recon = json.loads(upload.reconciliation_data)
        confidences = {p["confidence"] for p in recon["adjustment_pairs"]}
        assert confidences == {"high", "low"}
        # High-confidence debit total = 1000; credit total = 1000.
        assert recon["adjustments_debit_total"] == "1,000.00"
        assert recon["adjustments_credit_total"] == "1,000.00"


@pytest.mark.anyio
async def test_cc_card_resolves_via_cards_table(maker, statements_dir, monkeypatch):
    """The statement card resolves through the account's cards table, also
    from a partial suffix (SBI prints only "XX67"). An unknown card resolves
    to no account."""
    acc_id = await h.add_cc_account(
        maker, account_number="0000000000000000", cards=["XXXX XXXX XXXX 4567"]
    )
    for idx, (card, imported) in enumerate(
        (
            ("XXXX XXXX XXXX 1234", False),
            ("XXXX XXXX XXXX 4567", True),
            ("XXXX XXXX XXXX XX67", True),
        )
    ):
        parsed = h.cc_parsed(
            card_number=card,
            due_date=f"1{idx}/08/2026",
            transactions=[
                h.cc_txn(date="01/07/2026", amount=f"{idx + 1}00.00", narration="X")
            ],
        )
        monkeypatch.setattr(
            cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed)
        )
        raw = h.email_with_pdf(
            subject="Credit card statement", pdf_bytes=f"%PDF {idx}".encode()
        )
        result = await process_statement_email("hdfc", raw, "Credit card statement")
        if imported:
            assert result["imported"] == 1
        else:
            assert result is None

    async with maker() as session:
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert [t.account_id for t in txns] == [acc_id, acc_id]


# ---------------------------------------------------------------------------
# CC: encrypted PDF paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_encrypted_password_required_then_stored_password(
    maker, statements_dir, monkeypatch
):
    """Without a stored password the upload waits as password_required and
    keeps the hint. After the password is saved, the next statement imports."""
    monkeypatch.setattr(
        cc_module, "extract_password_hint", lambda *a, **kw: "DOB in DDMMYYYY"
    )
    acc_id = await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[h.cc_txn(date="01/07/2026", amount="500.00", narration="X")]
    )
    monkeypatch.setattr(
        cc_module,
        "_parse_pdf_bytes_sync",
        h.make_cc_parser(parsed, password_required=True, correct_password="secret"),
    )

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result["imported"] == 0

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert upload.status == "password_required"
        assert upload.account_id == acc_id
        acc = await session.get(Account, acc_id)
        assert acc.statement_password_hint == "DOB in DDMMYYYY"
        acc.statement_password = h.encrypt_password("secret")
        await session.commit()

    raw = h.email_with_pdf(subject="Credit card statement", pdf_bytes=b"%PDF next")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result["imported"] == 1


@pytest.mark.anyio
async def test_cc_encrypted_multi_account_returns_none(
    maker, statements_dir, monkeypatch
):
    await h.add_cc_account(maker, label="A")
    await h.add_cc_account(maker, label="B")
    parsed = h.cc_parsed()
    monkeypatch.setattr(
        cc_module,
        "_parse_pdf_bytes_sync",
        h.make_cc_parser(parsed, password_required=True, correct_password="secret"),
    )

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result is None
    async with maker() as session:
        uploads = (await session.execute(select(StatementUpload))).scalars().all()
        assert uploads == []


@pytest.mark.anyio
async def test_cc_non_password_parse_error_returns_none(
    maker, statements_dir, monkeypatch
):
    await h.add_cc_account(maker)

    def _bad(pdf_bytes, password=None, bank="auto"):
        raise ValueError("Could not extract tables from PDF")

    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", _bad)
    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result is None


# ---------------------------------------------------------------------------
# CC: per-row import error tolerance
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_row_import_errors_tolerated(maker, statements_dir, monkeypatch):
    """A duplicate or unexpected error on one row must not abort the batch.
    The good rows still import and the failed entries are tagged."""
    await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="01/07/2026", amount="100.00", narration="GOOD"),
            h.cc_txn(date="02/07/2026", amount="200.00", narration="DUP"),
            h.cc_txn(date="03/07/2026", amount="300.00", narration="BOOM"),
        ]
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    real_link = cc_module.link_transaction

    def _flaky(ctx, txn):
        if txn.counterparty == "DUP":
            raise IntegrityError("simulated", {}, Exception("dup"))
        if txn.counterparty == "BOOM":
            raise RuntimeError("kaboom")
        real_link(ctx, txn)

    monkeypatch.setattr(cc_module, "link_transaction", _flaky)

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result["imported"] == 1

    async with maker() as session:
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert [t.counterparty for t in txns] == ["GOOD"]
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert "1 duplicate" in (upload.error or "")
        assert "1 unexpected error" in (upload.error or "")
        recon = json.loads(upload.reconciliation_data)
        dup = next(e for e in recon["missing"] if e["narration"] == "DUP")
        assert dup.get("duplicate") is True


# ---------------------------------------------------------------------------
# Bank: gates
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bank_gates_return_none(maker, statements_dir, monkeypatch):
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(h.bank_parsed())
    )
    raw = h.email_with_pdf(subject="Account statement")
    assert await process_bank_statement_email("hdfc", raw, "Account statement") is None

    await h.add_bank_account(maker)
    for subject in ("Your monthly offer", "Credit card statement"):
        raw = h.email_with_pdf(subject=subject)
        assert await process_bank_statement_email("hdfc", raw, subject) is None

    msg = EmailMessage()
    msg["Subject"] = "Account statement"
    msg["From"] = "x@hdfc.com"
    msg.set_content("no attachment")
    result = await process_bank_statement_email(
        "hdfc", msg.as_bytes(), "Account statement"
    )
    assert result is None


# ---------------------------------------------------------------------------
# Bank: clean parse + reconciliation + import
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bank_clean_parse_imports_and_verifies_balance(
    maker, statements_dir, monkeypatch
):
    acc_id = await h.add_bank_account(maker)
    parsed = h.bank_parsed(
        account_number="1234567890",
        opening_balance="10,000.00",
        closing_balance="9,000.00",
        statement_period_start="01/07/2026",
        statement_period_end="31/07/2026",
        debit_total="1,000.00",
        credit_total="0.00",
        transactions=[
            h.bank_txn(date="05/07/2026", amount="1,000.00", narration="UPI Debit"),
        ],
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )

    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result is not None
    assert result["imported"] == 1
    assert result["matched"] == 0
    assert result["missing"] == 1

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.status == "imported"
        assert upload.imported_count == 1
        assert upload.account_number == "1234567890"
        bv = json.loads(upload.reconciliation_data)["balance_verification"]
        assert bv["is_balanced"] is True

        txn = (await session.execute(select(Transaction))).scalars().one()
        assert txn.email_type == "bank_statement"
        assert txn.account_mask == "7890"

        # Bank balance snapshot emitted.
        snaps = (await session.execute(select(BalanceSnapshot))).scalars().all()
        assert any(s.account_id == acc_id for s in snaps)


@pytest.mark.anyio
async def test_bank_enrichment_marker_reaches_the_stored_reconciliation(
    maker, statements_dir, monkeypatch
):
    """The statement page reads its enrichment badge from the stored JSON.

    Enrichment marks the rows it changed, so it must run before the
    reconciliation is serialized. Otherwise the row is enriched but the page
    shows nothing.
    """
    acc_id = await h.add_bank_account(maker)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="sms",
                direction="credit",
                amount=Decimal("2000.00"),
                transaction_date=datetime.date(2026, 4, 14),
                counterparty="Mobile XXXXXXXXX006",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(
                date="14/04/2026",
                amount="2,000.00",
                transaction_type="credit",
                counterparty="ANJALIJY OTESHNA",
                narration="IMPS/1234/ANJALIJY OTESHNA/HDFC/Selftransfer",
            )
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result["enriched"] == 1

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        recon = json.loads(upload.reconciliation_data)
        assert any(m.get("enriched") for m in recon["matched"])

        txn = (await session.execute(select(Transaction))).scalars().first()
        assert txn is not None
        assert txn.counterparty == "ANJALIJY OTESHNA"
        assert txn.raw_description == "IMPS/1234/ANJALIJY OTESHNA/HDFC/Selftransfer"


@pytest.mark.anyio
async def test_bank_malformed_rows_stay_missing_and_unbalanced(
    maker, statements_dir, monkeypatch
):
    """Unparseable rows go to missing and are not imported. A closing
    balance that disagrees with the rows marks the statement unbalanced."""
    await h.add_bank_account(maker)
    parsed = h.bank_parsed(
        opening_balance="10,000.00",
        closing_balance="8,000.00",
        debit_total="100.00",
        credit_total="0.00",
        transactions=[
            h.bank_txn(date="not-a-date", amount="100.00", narration="BADDATE"),
            h.bank_txn(date="05/07/2026", amount="100.00", narration="OK"),
        ],
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result["missing"] == 2
    assert result["imported"] == 1

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.status == "partial_import"
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert [t.counterparty for t in txns] == ["OK"]
        bv = json.loads(upload.reconciliation_data)["balance_verification"]
        assert bv["is_balanced"] is False


# ---------------------------------------------------------------------------
# Bank: encrypted + parse-error paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bank_encrypted_password_required_then_stored_password(
    maker, statements_dir, monkeypatch
):
    monkeypatch.setattr(
        bank_module, "extract_password_hint", lambda *a, **kw: "PAN + DOB"
    )
    acc_id = await h.add_bank_account(maker)
    parsed = h.bank_parsed(
        transactions=[h.bank_txn(date="01/07/2026", amount="500.00", narration="X")]
    )
    monkeypatch.setattr(
        bank_module,
        "_parse_pdf_bytes_sync",
        h.make_bank_parser(parsed, password_required=True, correct_password="secret"),
    )
    raw = h.email_with_pdf(subject="Account statement")
    assert await process_bank_statement_email("hdfc", raw, "Account statement")
    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.status == "password_required"
        acc = await session.get(Account, acc_id)
        assert acc.statement_password_hint == "PAN + DOB"
        acc.statement_password = h.encrypt_password("secret")
        await session.commit()

    raw = h.email_with_pdf(subject="Account statement", pdf_bytes=b"%PDF next")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result["imported"] == 1


@pytest.mark.anyio
async def test_bank_encrypted_multi_account_raises(maker, statements_dir, monkeypatch):
    await h.add_bank_account(maker, label="A")
    await h.add_bank_account(maker, label="B")
    parsed = h.bank_parsed()
    monkeypatch.setattr(
        bank_module,
        "_parse_pdf_bytes_sync",
        h.make_bank_parser(parsed, password_required=True, correct_password="secret"),
    )
    raw = h.email_with_pdf(subject="Account statement")
    with pytest.raises(BankStatementProcessingError):
        await process_bank_statement_email("hdfc", raw, "Account statement")


@pytest.mark.anyio
async def test_bank_non_password_parse_error_raises(maker, statements_dir, monkeypatch):
    await h.add_bank_account(maker)

    def _bad(pdf_bytes, bank, password=None):
        raise ValueError("corrupt PDF structure")

    monkeypatch.setattr(bank_module, "_parse_pdf_bytes_sync", _bad)
    raw = h.email_with_pdf(subject="Account statement")
    with pytest.raises(BankStatementProcessingError):
        await process_bank_statement_email("hdfc", raw, "Account statement")


# ---------------------------------------------------------------------------
# Bank: per-row import error tolerance
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bank_same_ref_contention_is_held_back(
    maker, statements_dir, monkeypatch
):
    """Two statement rows reaching the same DB reference are ambiguous.

    Neither contender may be auto-imported; an unrelated clean row still
    imports. The different-account collision test below retains the real
    ``IntegrityError`` / SAVEPOINT coverage.
    """
    acc_id = await h.add_bank_account(maker)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="bank_statement",
                direction="debit",
                amount=Decimal("500.00"),
                transaction_date=datetime.date(2026, 7, 2),
                reference_number="DUPREF",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        transactions=[
            # Both rows can claim the pre-existing DB row by reference.
            h.bank_txn(
                date="02/07/2026",
                amount="500.00",
                reference_number="DUPREF",
                narration="first",
            ),
            # Neither statement-order winner is safe, so both are held back.
            h.bank_txn(
                date="03/07/2026",
                amount="500.00",
                reference_number="DUPREF",
                narration="dup",
            ),
            h.bank_txn(
                date="04/07/2026",
                amount="300.00",
                reference_number="NEWREF",
                narration="new",
            ),
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result["matched"] == 0
    assert result["imported"] == 1
    assert result["duplicates"] == 0

    async with maker() as session:
        # Pre-existing + the unrelated clean import = 2; neither contender
        # was inserted.
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 2
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        recon = json.loads(upload.reconciliation_data)
        ambiguous = [entry for entry in recon["missing"] if entry.get("ambiguous")]
        assert len(ambiguous) == 2
        assert all("ambiguous" in entry["import_error"] for entry in ambiguous)
        assert upload.error is None


@pytest.mark.anyio
async def test_bank_ref_collision_other_account_is_duplicate(
    maker, statements_dir, monkeypatch
):
    """A reference_number that already exists on a DIFFERENT account still
    violates the global ``uq_transactions_ref`` partial index — the import
    must tag it duplicate and continue, not abort the whole batch."""
    other_id = await h.add_bank_account(
        maker, label="Other", account_number="9999999999"
    )
    acc_id = await h.add_bank_account(maker, label="Main", account_number="1234567890")
    async with maker() as session:
        session.add(
            Transaction(
                account_id=other_id,
                bank="hdfc",
                email_type="bank_statement",
                direction="debit",
                amount=Decimal("500.00"),
                transaction_date=datetime.date(2026, 7, 2),
                reference_number="SHAREDREF",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        account_number="1234567890",
        transactions=[
            h.bank_txn(
                date="02/07/2026",
                amount="500.00",
                reference_number="SHAREDREF",
                narration="collides",
            ),
            h.bank_txn(
                date="03/07/2026",
                amount="300.00",
                reference_number="OKREF",
                narration="ok",
            ),
        ],
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result["imported"] == 1
    assert result["duplicates"] == 1
    assert result["import_errors"] == 0

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.account_id == acc_id
        assert "1 duplicate" in (upload.error or "")


# ---------------------------------------------------------------------------
# Bank: notifications threshold branch (no network)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bank_notifications_single_vs_bulk_threshold(
    maker, statements_dir, monkeypatch
):
    """Below ``telegram.bulk_threshold`` each txn fires a single notification;
    at/above it a single bulk summary is sent instead. No network — we patch
    the send functions to recorders."""
    await h.add_bank_account(maker)

    # 3 txns; threshold 5 → single path.
    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(date=f"0{i}/07/2026", amount="100.00", narration=f"T{i}")
            for i in range(1, 4)
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    monkeypatch.setattr(bank_module, "should_notify_transactions", lambda: True)
    monkeypatch.setattr(bank_module, "get_telegram_chat_id", lambda: 123)
    monkeypatch.setattr(bank_module, "get_setting_int", lambda *_a, **_kw: 5)

    singles = []
    bulks = []

    async def _single(txn_id, info, chat_id):
        singles.append(txn_id)

    async def _bulk(count, chat_id, **kw):
        bulks.append(count)

    monkeypatch.setattr(bank_module, "send_transaction_notification", _single)
    monkeypatch.setattr(bank_module, "send_bulk_summary", _bulk)
    raw = h.email_with_pdf(subject="Account statement")
    await process_bank_statement_email("hdfc", raw, "Account statement")
    assert len(singles) == 3
    assert bulks == []

    # Now 6 txns; threshold 5 → bulk path.
    await h.add_bank_account(maker, bank="icici", label="ICICI")
    parsed2 = h.bank_parsed(
        bank="icici",
        account_number="2222222222",
        transactions=[
            h.bank_txn(date=f"0{i}/07/2026", amount="100.00", narration=f"T{i}")
            for i in range(1, 7)
        ],
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed2)
    )
    singles.clear()
    bulks.clear()
    raw2 = h.email_with_pdf(subject="Account statement")
    await process_bank_statement_email("icici", raw2, "Account statement")
    assert singles == []
    assert bulks == [6]
