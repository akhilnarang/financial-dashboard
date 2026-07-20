"""Real end-to-end integration tests for the statement email pipelines.

Exercises ``process_statement_email`` (CC) and ``process_bank_statement_email``
against an in-memory SQLite session factory, monkeypatching only the parser
adapter boundary (``_parse_pdf_bytes_sync``) — the reconciliation, import,
snapshot, and enrichment services all run for real.

Covers: subject/PDF/account gates; clean parse; encrypted (stored password,
single-account password_required, multi-account refusal); non-password
parse_error; account/card exact + partial (add-on) resolution; exact + ±1
date; ref-first / narration-ref rescue / UPI token / ambiguous refusal;
duplicate and generic per-row import errors; malformed rows; balanced /
unbalanced verification; snapshot emission; status/count derivation; generic
counterparty enrichment; notifications threshold branch without network.
"""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

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

# Import fixtures into this module's namespace so pytest discovers them.
from . import _helpers as h


# ---------------------------------------------------------------------------
# CC: gates
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_subject_without_statement_returns_none(maker, statements_dir):
    raw = h.email_with_pdf(subject="Your monthly offer")
    result = await process_statement_email("hdfc", raw, "Your monthly offer")
    assert result is None


@pytest.mark.anyio
async def test_cc_bank_account_subject_without_card_returns_none(maker, statements_dir):
    await h.add_cc_account(maker)
    subject = "Account statement for July 2026"
    raw = h.email_with_pdf(subject=subject)
    result = await process_statement_email("hdfc", raw, subject)
    assert result is None


@pytest.mark.anyio
async def test_cc_no_pdf_returns_none(maker, statements_dir):
    await h.add_cc_account(maker)
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "Credit card statement"
    msg["From"] = "x@hdfc.com"
    msg.set_content("no attachment")
    result = await process_statement_email(
        "hdfc", msg.as_bytes(), "Credit card statement"
    )
    assert result is None


@pytest.mark.anyio
async def test_cc_no_cc_account_returns_none(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.cc as cc_module

    monkeypatch.setattr(
        cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(h.cc_parsed())
    )
    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result is None


# ---------------------------------------------------------------------------
# CC: clean parse + reconciliation + import
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_clean_parse_imports_missing_and_emits_snapshot(
    maker, statements_dir, monkeypatch
):
    import financial_dashboard.services.statements.cc as cc_module

    acc_id = await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="01/07/2026", amount="1,000.00", narration="AMAZON")
        ]
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")

    assert result is not None
    assert result["matched"] == 0
    assert result["missing"] == 1
    assert result["imported"] == 1

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert upload.account_id == acc_id
        assert upload.status == "imported"  # all missing imported
        assert upload.parsed_txn_count == 1
        assert upload.matched_count == 0
        assert upload.missing_count == 0
        assert upload.imported_count == 1
        assert upload.error is None

        txn = (await session.execute(select(Transaction))).scalars().one()
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
    import financial_dashboard.services.statements.cc as cc_module

    acc_id = await h.add_cc_account(maker)
    # Two DB txns: one exact-date, one +1 day off.
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


@pytest.mark.anyio
async def test_cc_transactions_vs_payments_refunds(maker, statements_dir, monkeypatch):
    """Debits live in ``transactions``, credits in ``payments_refunds``; both
    flow into reconciliation and import with the correct direction."""
    import financial_dashboard.services.statements.cc as cc_module

    await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="01/07/2026", amount="2,000.00", narration="DEBIT")
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
    assert result["missing"] == 2
    assert result["imported"] == 2

    async with maker() as session:
        txns = {
            (t.direction, t.counterparty): t
            for t in (await session.execute(select(Transaction))).scalars().all()
        }
        assert ("debit", "DEBIT") in txns
        assert ("credit", "PAYMENT RECEIVED") in txns


@pytest.mark.anyio
async def test_cc_adjustment_pairs_high_low_confidence(
    maker, statements_dir, monkeypatch
):
    """Adjustment pairs are surfaced in reconciliation_data regardless of
    confidence; totals only sum high-confidence legs."""
    import financial_dashboard.services.statements.cc as cc_module

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
        import json

        recon = json.loads(upload.reconciliation_data)
        confidences = {p["confidence"] for p in recon["adjustment_pairs"]}
        assert confidences == {"high", "low"}
        # High-confidence debit total = 1000; credit total = 1000.
        assert recon["adjustments_debit_total"] == "1,000.00"
        assert recon["adjustments_credit_total"] == "1,000.00"


@pytest.mark.anyio
async def test_cc_addon_card_resolution(maker, statements_dir, monkeypatch):
    """A statement entry whose card_number is a partial suffix (e.g. XX67)
    resolves to the account's registered add-on card."""
    import financial_dashboard.services.statements.cc as cc_module

    # Account number has NO last4 match; only the cards table carries 4567.
    await h.add_cc_account(maker, account_number="0000", cards=["XXXX XXXX XXXX 4567"])
    parsed = h.cc_parsed(
        card_number="XXXX XXXX XXXX 1234",
        transactions=[h.cc_txn(date="01/07/2026", amount="100.00", narration="X")],
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    # No account matches last4 1234 (account has 0000, card has 4567) → None.
    assert result is None


@pytest.mark.anyio
async def test_cc_addon_card_resolves_via_cards_table(
    maker, statements_dir, monkeypatch
):
    """A statement whose card last4 matches an add-on card registered on the
    account (but NOT the account_number itself) still resolves the account."""
    import financial_dashboard.services.statements.cc as cc_module

    # account_number has no matching last4; only the cards table carries 4567.
    acc_id = await h.add_cc_account(
        maker, account_number="0000000000000000", cards=["XXXX XXXX XXXX 4567"]
    )
    parsed = h.cc_parsed(
        card_number="XXXX XXXX XXXX 4567",
        transactions=[h.cc_txn(date="01/07/2026", amount="100.00", narration="X")],
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result is not None
    assert result["imported"] == 1
    async with maker() as session:
        txn = (await session.execute(select(Transaction))).scalars().one()
        assert txn.account_id == acc_id


@pytest.mark.anyio
async def test_cc_sbi_style_partial_suffix_resolves(maker, statements_dir, monkeypatch):
    """SBI prints only two trailing digits (e.g. "XX67"). The resolver must
    still match it to a card whose last4 ends with that suffix."""
    import financial_dashboard.services.statements.cc as cc_module

    acc_id = await h.add_cc_account(
        maker, account_number="0000000000000000", cards=["XXXX XXXX XXXX 4567"]
    )
    parsed = h.cc_parsed(
        card_number="XXXX XXXX XXXX XX67",  # only "67" trailing
        transactions=[h.cc_txn(date="01/07/2026", amount="100.00", narration="X")],
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result is not None
    assert result["imported"] == 1
    async with maker() as session:
        txn = (await session.execute(select(Transaction))).scalars().one()
        assert txn.account_id == acc_id


@pytest.mark.anyio
async def test_cc_generic_counterparty_enrichment(maker, statements_dir, monkeypatch):
    """A matched DB txn whose counterparty is a generic placeholder gets
    enriched with the statement narration."""
    import financial_dashboard.services.statements.cc as cc_module

    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="cc_txn",
                direction="debit",
                amount=Decimal("800.00"),
                transaction_date=datetime.date(2026, 7, 3),
                counterparty="payment received",
            )
        )
        await session.commit()

    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="03/07/2026", amount="800.00", narration="FLIPKART")
        ],
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result["enriched"] == 1

    async with maker() as session:
        txn = (await session.execute(select(Transaction))).scalars().one()
        assert txn.counterparty == "FLIPKART"


# ---------------------------------------------------------------------------
# CC: encrypted PDF paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cc_encrypted_uses_stored_password(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.cc as cc_module

    await h.add_cc_account(maker, statement_password=h.encrypt_password("secret"))
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
    assert result is not None
    assert result["imported"] == 1


@pytest.mark.anyio
async def test_cc_encrypted_single_account_password_required(
    maker, statements_dir, monkeypatch
):
    import financial_dashboard.services.statements.cc as cc_module

    monkeypatch.setattr(
        cc_module, "extract_password_hint", lambda *a, **kw: "DOB in DDMMYYYY"
    )
    acc_id = await h.add_cc_account(maker)  # no stored password
    parsed = h.cc_parsed()
    monkeypatch.setattr(
        cc_module,
        "_parse_pdf_bytes_sync",
        h.make_cc_parser(parsed, password_required=True, correct_password="secret"),
    )

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result is not None
    assert result["imported"] == 0

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert upload.status == "password_required"
        assert upload.account_id == acc_id
        assert "encrypted" in (upload.error or "").lower()
        acc = await session.get(Account, acc_id)
        assert acc.statement_password_hint == "DOB in DDMMYYYY"


@pytest.mark.anyio
async def test_cc_encrypted_multi_account_returns_none(
    maker, statements_dir, monkeypatch
):
    await h.add_cc_account(maker, label="A")
    await h.add_cc_account(maker, label="B")
    parsed = h.cc_parsed()
    import financial_dashboard.services.statements.cc as cc_module

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
    import financial_dashboard.services.statements.cc as cc_module

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
async def test_cc_duplicate_row_tolerated(maker, statements_dir, monkeypatch):
    """One row hitting IntegrityError must not abort the batch — the good
    rows still import and the failed entry is tagged."""
    import financial_dashboard.services.statements.cc as cc_module

    await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="01/07/2026", amount="100.00", narration="OK1"),
            h.cc_txn(date="02/07/2026", amount="200.00", narration="BAD"),
            h.cc_txn(date="03/07/2026", amount="300.00", narration="OK2"),
        ]
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    real_link = cc_module.link_transaction
    call = {"n": 0}

    def _flaky(ctx, txn):
        call["n"] += 1
        if txn.counterparty == "BAD":
            raise IntegrityError("simulated", {}, Exception("dup"))
        real_link(ctx, txn)

    monkeypatch.setattr(cc_module, "link_transaction", _flaky)

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result["imported"] == 2

    async with maker() as session:
        txns = {
            t.counterparty: t
            for t in (await session.execute(select(Transaction))).scalars().all()
        }
        assert set(txns) == {"OK1", "OK2"}
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert "1 duplicate" in (upload.error or "")
        import json

        recon = json.loads(upload.reconciliation_data)
        bad = next(e for e in recon["missing"] if e["narration"] == "BAD")
        assert bad.get("duplicate") is True


@pytest.mark.anyio
async def test_cc_generic_import_error_tolerated(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.cc as cc_module

    await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="01/07/2026", amount="100.00", narration="GOOD"),
            h.cc_txn(date="02/07/2026", amount="200.00", narration="BOOM"),
        ]
    )
    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", h.make_cc_parser(parsed))

    real_link = cc_module.link_transaction

    def _flaky(ctx, txn):
        if txn.counterparty == "BOOM":
            raise RuntimeError("kaboom")
        real_link(ctx, txn)

    monkeypatch.setattr(cc_module, "link_transaction", _flaky)

    raw = h.email_with_pdf(subject="Credit card statement")
    result = await process_statement_email("hdfc", raw, "Credit card statement")
    assert result["imported"] == 1

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert "1 unexpected error" in (upload.error or "")


# ---------------------------------------------------------------------------
# Bank: gates
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bank_subject_without_statement_returns_none(maker, statements_dir):
    await h.add_bank_account(maker)
    raw = h.email_with_pdf(subject="Your monthly offer")
    result = await process_bank_statement_email("hdfc", raw, "Your monthly offer")
    assert result is None


@pytest.mark.anyio
async def test_bank_cc_subject_returns_none(maker, statements_dir):
    await h.add_bank_account(maker)
    subject = "Credit card statement"
    raw = h.email_with_pdf(subject=subject)
    result = await process_bank_statement_email("hdfc", raw, subject)
    assert result is None


@pytest.mark.anyio
async def test_bank_no_pdf_returns_none(maker, statements_dir):
    await h.add_bank_account(maker)
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "Account statement"
    msg["From"] = "x@hdfc.com"
    msg.set_content("no attachment")
    result = await process_bank_statement_email(
        "hdfc", msg.as_bytes(), "Account statement"
    )
    assert result is None


@pytest.mark.anyio
async def test_bank_no_bank_account_returns_none(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.bank as bank_module

    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(h.bank_parsed())
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result is None


# ---------------------------------------------------------------------------
# Bank: clean parse + reconciliation + import
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bank_clean_parse_imports_and_verifies_balance(
    maker, statements_dir, monkeypatch
):
    import financial_dashboard.services.statements.bank as bank_module

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

        txn = (await session.execute(select(Transaction))).scalars().one()
        assert txn.email_type == "bank_statement"
        assert txn.account_mask == "7890"

        # Bank balance snapshot emitted.
        snaps = (await session.execute(select(BalanceSnapshot))).scalars().all()
        assert any(s.account_id == acc_id for s in snaps)


@pytest.mark.anyio
async def test_bank_ref_first_match_takes_priority(maker, statements_dir, monkeypatch):
    """A statement row whose ref appears in the DB must win the match even if
    an earlier fuzzy-only stmt row could have consumed it by date+amount."""
    import financial_dashboard.services.statements.bank as bank_module

    acc_id = await h.add_bank_account(maker)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="bank_statement",
                direction="debit",
                amount=Decimal("2000.00"),
                transaction_date=datetime.date(2026, 4, 14),
                reference_number="REF-X",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(
                date="13/04/2026",
                amount="2,000.00",
                reference_number="REF-Y",
                narration="earlier fuzzy",
            ),
            h.bank_txn(
                date="14/04/2026",
                amount="2,000.00",
                reference_number="REF-X",
                narration="ref match",
            ),
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    await process_bank_statement_email("hdfc", raw, "Account statement")

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        import json

        recon = json.loads(upload.reconciliation_data)
        matched_refs = {m["reference_number"] for m in recon["matched"]}
        missing_refs = {m["reference_number"] for m in recon["missing"]}
        assert matched_refs == {"REF-X"}
        assert missing_refs == {"REF-Y"}


@pytest.mark.anyio
async def test_bank_narration_ref_rescue(maker, statements_dir, monkeypatch):
    """DB ref embedded in statement narration rescues an otherwise
    ref-disagreement case."""
    import financial_dashboard.services.statements.bank as bank_module

    acc_id = await h.add_bank_account(maker)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="cc_txn",
                direction="credit",
                amount=Decimal("650.00"),
                transaction_date=datetime.date(2026, 4, 28),
                reference_number="100200300400",
                counterparty="Sample Payer",
                raw_description="UPI credit 100200300400",
                channel="upi",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(
                date="28/04/2026",
                amount="650.00",
                reference_number="20990428180878701",
                narration="UPI-Credit-100200300400-Sample Payer",
                channel="upi",
                transaction_type="credit",
            ),
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    await process_bank_statement_email("hdfc", raw, "Account statement")

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        import json

        recon = json.loads(upload.reconciliation_data)
        assert len(recon["matched"]) == 1


@pytest.mark.anyio
async def test_bank_upi_token_match_and_ambiguous_refusal(
    maker, statements_dir, monkeypatch
):
    """Two scenarios: distinctive UPI token overlap matches; ambiguous
    multiple-compatible-candidates is refused into missing."""
    import financial_dashboard.services.statements.bank as bank_module

    acc_id = await h.add_bank_account(maker)

    # UPI token match case.
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="cc_txn",
                direction="credit",
                amount=Decimal("500.00"),
                transaction_date=datetime.date(2026, 4, 14),
                reference_number="UTR-EMAIL",
                counterparty="SAMPLE MERCHANT",
                raw_description="UPI credit from SAMPLE MERCHANT",
                channel="upi",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(
                date="14/04/2026",
                amount="500.00",
                reference_number="STMT-INTERNAL",
                narration="UPI Credit-SAMPLE MERCHANT-x@okaxis",
                channel="upi",
                transaction_type="credit",
            ),
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    await process_bank_statement_email("hdfc", raw, "Account statement")
    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        import json

        recon = json.loads(upload.reconciliation_data)
        assert len(recon["matched"]) == 1, recon

    # Ambiguous refusal case: two ref-less candidates, same date+amount.
    async with maker() as session:
        session.add_all(
            [
                Transaction(
                    account_id=acc_id,
                    bank="hdfc",
                    email_type="cc_txn",
                    direction="debit",
                    amount=Decimal("1000.00"),
                    transaction_date=datetime.date(2026, 5, 1),
                    counterparty="MERCHANT A",
                    channel="upi",
                ),
                Transaction(
                    account_id=acc_id,
                    bank="hdfc",
                    email_type="cc_txn",
                    direction="debit",
                    amount=Decimal("1000.00"),
                    transaction_date=datetime.date(2026, 5, 1),
                    counterparty="MERCHANT B",
                    channel="upi",
                ),
            ]
        )
        await session.commit()

    parsed2 = h.bank_parsed(
        transactions=[
            h.bank_txn(
                date="01/05/2026",
                amount="1,000.00",
                narration="UPI Debit unknown",
                channel="upi",
            ),
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed2)
    )
    raw2 = h.email_with_pdf(subject="Account statement")
    await process_bank_statement_email("hdfc", raw2, "Account statement")
    async with maker() as session:
        uploads = (
            (
                await session.execute(
                    select(BankStatementUpload).order_by(BankStatementUpload.id)
                )
            )
            .scalars()
            .all()
        )
        import json

        recon = json.loads(uploads[-1].reconciliation_data)
        assert len(recon["matched"]) == 0
        assert len(recon["missing"]) == 1


@pytest.mark.anyio
async def test_bank_malformed_rows_stay_missing(maker, statements_dir, monkeypatch):
    """Unparseable date/amount rows go to missing (not matched, not imported)."""
    import financial_dashboard.services.statements.bank as bank_module

    await h.add_bank_account(maker)
    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(date="not-a-date", amount="100.00", narration="BADDATE"),
            h.bank_txn(date="05/07/2026", amount="100.00", narration="OK"),
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result["missing"] == 2  # both in missing
    assert result["imported"] == 1  # only the parseable one imported

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.status == "partial_import"
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 1


@pytest.mark.anyio
async def test_bank_unbalanced_verification(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.bank as bank_module

    await h.add_bank_account(maker)
    parsed = h.bank_parsed(
        opening_balance="10,000.00",
        closing_balance="8,000.00",  # real computed would be 9000 → delta 1000
        debit_total="1,000.00",
        credit_total="0.00",
        transactions=[
            h.bank_txn(date="05/07/2026", amount="1,000.00", narration="X"),
        ],
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    await process_bank_statement_email("hdfc", raw, "Account statement")
    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        import json

        recon = json.loads(upload.reconciliation_data)
        bv = recon["balance_verification"]
        assert bv["is_balanced"] is False
        assert bv["delta"] == "-1,000.00"


@pytest.mark.anyio
async def test_bank_generic_counterparty_enrichment(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.bank as bank_module

    acc_id = await h.add_bank_account(maker)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="cc_txn",
                direction="debit",
                amount=Decimal("800.00"),
                transaction_date=datetime.date(2026, 7, 3),
                counterparty="payment done",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(
                date="03/07/2026",
                amount="800.00",
                narration="UPI-Debit-FLIPKART",
                counterparty="FLIPKART",
            ),
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result["enriched"] == 1

    async with maker() as session:
        txn = (await session.execute(select(Transaction))).scalars().one()
        assert txn.counterparty == "FLIPKART"


# ---------------------------------------------------------------------------
# Bank: encrypted + parse-error paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bank_encrypted_uses_stored_password(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.bank as bank_module

    await h.add_bank_account(maker, statement_password=h.encrypt_password("secret"))
    parsed = h.bank_parsed(
        transactions=[h.bank_txn(date="01/07/2026", amount="500.00", narration="X")]
    )
    monkeypatch.setattr(
        bank_module,
        "_parse_pdf_bytes_sync",
        h.make_bank_parser(parsed, password_required=True, correct_password="secret"),
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result is not None
    assert result["imported"] == 1


@pytest.mark.anyio
async def test_bank_encrypted_single_account_password_required(
    maker, statements_dir, monkeypatch
):
    import financial_dashboard.services.statements.bank as bank_module

    monkeypatch.setattr(
        bank_module, "extract_password_hint", lambda *a, **kw: "PAN + DOB"
    )
    acc_id = await h.add_bank_account(maker)
    parsed = h.bank_parsed()
    monkeypatch.setattr(
        bank_module,
        "_parse_pdf_bytes_sync",
        h.make_bank_parser(parsed, password_required=True, correct_password="secret"),
    )
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result is not None
    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.status == "password_required"
        acc = await session.get(Account, acc_id)
        assert acc.statement_password_hint == "PAN + DOB"


@pytest.mark.anyio
async def test_bank_encrypted_multi_account_raises(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.bank as bank_module

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
    import financial_dashboard.services.statements.bank as bank_module

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
    import financial_dashboard.services.statements.bank as bank_module

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
                amount="700.00",
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
    import financial_dashboard.services.statements.bank as bank_module

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

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.account_id == acc_id


@pytest.mark.anyio
async def test_bank_generic_import_error_tolerated(maker, statements_dir, monkeypatch):
    import financial_dashboard.services.statements.bank as bank_module

    await h.add_bank_account(maker)
    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(date="01/07/2026", amount="100.00", narration="GOOD"),
            h.bank_txn(date="02/07/2026", amount="200.00", narration="BOOM"),
        ]
    )
    monkeypatch.setattr(
        bank_module, "_parse_pdf_bytes_sync", h.make_bank_parser(parsed)
    )

    real_link = bank_module.link_transaction

    def _flaky(ctx, txn):
        if txn.counterparty == "BOOM":
            raise RuntimeError("kaboom")
        real_link(ctx, txn)

    monkeypatch.setattr(bank_module, "link_transaction", _flaky)
    raw = h.email_with_pdf(subject="Account statement")
    result = await process_bank_statement_email("hdfc", raw, "Account statement")
    assert result["imported"] == 1
    assert result["import_errors"] == 1

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert "1 unexpected error" in (upload.error or "")


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
    import financial_dashboard.services.statements.bank as bank_module

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
