"""Integration tests for the statement retry helpers and date-range scoping.

Exercises ``retry_cc_statement_upload`` and ``retry_bank_statement_upload``
against a real SQLite session factory, monkeypatching only the parser adapter
boundary. Covers: wrong-password retry (error set, status preserved), scoped
date range (DB txns outside the statement window ± buffer are not matched),
and retry/reprocess idempotency (re-running does not double-import).
"""

import datetime
import json
from decimal import Decimal

import pytest
from sqlalchemy import select

from financial_dashboard.db import (
    BankStatementUpload,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.statements.shared import (
    retry_bank_statement_upload,
    retry_cc_statement_upload,
)

from . import _helpers as h


async def _seed_cc_upload(maker, acc_id, *, file_path, status="password_required"):
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path=file_path,
            status=status,
            card_number="XXXX XXXX XXXX 1234",
        )
        session.add(upload)
        await session.commit()
        return upload.id


async def _seed_bank_upload(maker, acc_id, *, file_path, status="password_required"):
    async with maker() as session:
        upload = BankStatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="bank.pdf",
            file_path=file_path,
            status=status,
        )
        session.add(upload)
        await session.commit()
        return upload.id


@pytest.mark.anyio
async def test_retry_cc_wrong_password_sets_error_keeps_status(
    maker, statements_dir, monkeypatch, tmp_path
):
    import financial_dashboard.services.statements.cc as cc_module
    from financial_dashboard.services.statements import shared as shared_module

    acc_id = await h.add_cc_account(maker)
    pdf = tmp_path / "cc.pdf"
    pdf.write_bytes(b"%PDF fake")
    upload_id = await _seed_cc_upload(maker, acc_id, file_path=str(pdf))

    def _bad(path, password, bank):
        raise ValueError("The PDF is encrypted and needs a password")

    monkeypatch.setattr(cc_module, "parse_statement", _bad)
    monkeypatch.setattr(shared_module, "parse_statement", _bad)

    ok = await retry_cc_statement_upload(upload_id, "wrongpw")
    assert ok is False

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.status == "password_required"  # preserved (password error)
        assert "encrypted" in (upload.error or "").lower()


@pytest.mark.anyio
async def test_retry_cc_reimports_missing_and_idempotent(
    maker, statements_dir, monkeypatch, tmp_path
):
    import financial_dashboard.services.statements.cc as cc_module
    from financial_dashboard.services.statements import shared as shared_module

    acc_id = await h.add_cc_account(maker)
    pdf = tmp_path / "cc.pdf"
    pdf.write_bytes(b"%PDF fake")
    upload_id = await _seed_cc_upload(maker, acc_id, file_path=str(pdf))

    parsed = h.cc_parsed(
        transactions=[h.cc_txn(date="01/07/2026", amount="1,000.00", narration="X")]
    )
    monkeypatch.setattr(cc_module, "parse_statement", lambda *a, **kw: parsed)
    monkeypatch.setattr(shared_module, "parse_statement", lambda *a, **kw: parsed)

    ok = await retry_cc_statement_upload(upload_id, "secret")
    assert ok is True

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.status == "imported"
        assert upload.imported_count == 1
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 1

    # Re-run retry — re-reconciliation now matches the previously-imported
    # row, so missing is empty and nothing is re-imported. Idempotent: still
    # exactly one Transaction row.
    ok = await retry_cc_statement_upload(upload_id, "secret")
    assert ok is True
    async with maker() as session:
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 1
        upload = await session.get(StatementUpload, upload_id)
        assert upload.status == "imported"


@pytest.mark.anyio
async def test_retry_bank_wrong_password(maker, statements_dir, monkeypatch, tmp_path):
    import financial_dashboard.services.statements.bank as bank_module
    from financial_dashboard.services.statements import shared as shared_module

    acc_id = await h.add_bank_account(maker)
    pdf = tmp_path / "bank.pdf"
    pdf.write_bytes(b"%PDF fake")
    upload_id = await _seed_bank_upload(maker, acc_id, file_path=str(pdf))

    def _bad(path, bank, password):
        raise ValueError("The PDF is encrypted and needs a password")

    monkeypatch.setattr(bank_module, "parse_bank_statement", _bad)
    monkeypatch.setattr(shared_module, "parse_bank_statement", _bad)

    ok = await retry_bank_statement_upload(upload_id, "wrongpw")
    assert ok is False

    async with maker() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        assert upload.status == "password_required"
        assert "encrypted" in (upload.error or "").lower()


@pytest.mark.anyio
async def test_retry_bank_non_password_error_sets_parse_error(
    maker, statements_dir, monkeypatch, tmp_path
):
    """A non-password parse error during retry flips status to parse_error
    (not password_required) so the retry UI stops offering a password form."""
    import financial_dashboard.services.statements.bank as bank_module
    from financial_dashboard.services.statements import shared as shared_module

    acc_id = await h.add_bank_account(maker)
    pdf = tmp_path / "bank.pdf"
    pdf.write_bytes(b"%PDF fake")
    upload_id = await _seed_bank_upload(maker, acc_id, file_path=str(pdf))

    def _bad(path, bank, password):
        raise ValueError("unexpected EOF in PDF")

    monkeypatch.setattr(bank_module, "parse_bank_statement", _bad)
    monkeypatch.setattr(shared_module, "parse_bank_statement", _bad)

    ok = await retry_bank_statement_upload(upload_id, "any")
    assert ok is False

    async with maker() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        assert upload.status == "parse_error"


@pytest.mark.anyio
async def test_retry_bank_scoped_date_range(maker, monkeypatch, tmp_path):
    """``retry_bank_statement_upload`` scopes the DB candidate query to the
    statement period ± buffer. A DB txn far outside that window must NOT be
    matched — it stays a missing/imported row, never an accidental match."""
    import financial_dashboard.services.statements.bank as bank_module
    from financial_dashboard.services.statements import shared as shared_module

    acc_id = await h.add_bank_account(maker)
    pdf = tmp_path / "bank.pdf"
    pdf.write_bytes(b"%PDF fake")
    upload_id = await _seed_bank_upload(maker, acc_id, file_path=str(pdf))

    # A DB txn dated 6 months outside the statement period — same amount/date
    # shape as one stmt row, but outside the scoping window.
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="bank_statement",
                direction="debit",
                amount=Decimal("1000.00"),
                transaction_date=datetime.date(2026, 1, 5),
                reference_number="OLDFAR",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        statement_period_start="01/07/2026",
        statement_period_end="31/07/2026",
        transactions=[
            h.bank_txn(date="05/07/2026", amount="1,000.00", narration="JULY"),
        ],
    )
    monkeypatch.setattr(bank_module, "parse_bank_statement", lambda *a, **kw: parsed)
    monkeypatch.setattr(shared_module, "parse_bank_statement", lambda *a, **kw: parsed)

    ok = await retry_bank_statement_upload(upload_id, "secret")
    assert ok is True

    async with maker() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        recon = json.loads(upload.reconciliation_data)
        # The January DB row was outside the window → stmt row is missing,
        # not matched against it.
        assert len(recon["matched"]) == 0
        assert upload.imported_count == 1


@pytest.mark.anyio
async def test_retry_bank_duplicate_tolerated(
    maker, statements_dir, monkeypatch, tmp_path
):
    """Retry import must be SAVEPOINT-tolerant — one duplicate row must not
    abort the retry's whole import batch."""
    import financial_dashboard.services.statements.bank as bank_module
    from financial_dashboard.services.statements import shared as shared_module

    acc_id = await h.add_bank_account(maker)
    pdf = tmp_path / "bank.pdf"
    pdf.write_bytes(b"%PDF fake")
    upload_id = await _seed_bank_upload(maker, acc_id, file_path=str(pdf))

    # Pre-existing row on the account that the statement will also carry —
    # but via a second same-ref stmt row so reconcile leaves one missing.
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="bank_statement",
                direction="debit",
                amount=Decimal("500.00"),
                transaction_date=datetime.date(2026, 7, 2),
                reference_number="RETRYDUP",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(
                date="02/07/2026",
                amount="500.00",
                reference_number="RETRYDUP",
                narration="matched",
            ),
            h.bank_txn(
                date="03/07/2026",
                amount="500.00",
                reference_number="RETRYDUP",
                narration="dup",
            ),
            h.bank_txn(
                date="04/07/2026",
                amount="900.00",
                reference_number="GOODREF",
                narration="good",
            ),
        ]
    )
    monkeypatch.setattr(bank_module, "parse_bank_statement", lambda *a, **kw: parsed)
    monkeypatch.setattr(shared_module, "parse_bank_statement", lambda *a, **kw: parsed)

    ok = await retry_bank_statement_upload(upload_id, "secret")
    assert ok is True

    async with maker() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        assert upload.imported_count == 1
        assert "1 duplicate" in (upload.error or "")
        txns = (await session.execute(select(Transaction))).scalars().all()
        # Pre-existing + matched (same row) + the good import = 2 rows total.
        assert len(txns) == 2
