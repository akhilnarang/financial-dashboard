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
async def test_retry_cc_wrong_password_then_import_is_idempotent(
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

    def _parse(path, password, bank):
        if password != "secret":
            raise ValueError("The PDF is encrypted and needs a password")
        return parsed

    monkeypatch.setattr(cc_module, "parse_statement", _parse)
    monkeypatch.setattr(shared_module, "parse_statement", _parse)

    assert await retry_cc_statement_upload(upload_id, "wrongpw") is False
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.status == "password_required"
        assert upload.error

    assert await retry_cc_statement_upload(upload_id, "secret") is True
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.status == "imported"
        assert upload.imported_count == 1

    # A second retry matches the imported row and imports nothing.
    assert await retry_cc_statement_upload(upload_id, "secret") is True
    async with maker() as session:
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 1
        upload = await session.get(StatementUpload, upload_id)
        assert upload.status == "imported"


@pytest.mark.anyio
async def test_retry_bank_parse_errors_set_status_by_kind(
    maker, statements_dir, monkeypatch, tmp_path
):
    """A password error keeps password_required. Any other parse error flips
    the status to parse_error, so the retry UI stops offering a password form."""
    import financial_dashboard.services.statements.bank as bank_module
    from financial_dashboard.services.statements import shared as shared_module

    acc_id = await h.add_bank_account(maker)
    pdf = tmp_path / "bank.pdf"
    pdf.write_bytes(b"%PDF fake")
    upload_id = await _seed_bank_upload(maker, acc_id, file_path=str(pdf))

    for message, expected_status in (
        ("The PDF is encrypted and needs a password", "password_required"),
        ("unexpected EOF in PDF", "parse_error"),
    ):

        def _bad(path, bank, password, message=message):
            raise ValueError(message)

        monkeypatch.setattr(bank_module, "parse_bank_statement", _bad)
        monkeypatch.setattr(shared_module, "parse_bank_statement", _bad)

        assert await retry_bank_statement_upload(upload_id, "wrongpw") is False
        async with maker() as session:
            upload = await session.get(BankStatementUpload, upload_id)
            assert upload.status == expected_status
            assert upload.error == message


@pytest.mark.anyio
async def test_retry_bank_scopes_candidates_and_holds_back_contention(
    maker, statements_dir, monkeypatch, tmp_path
):
    """A DB row far outside the statement period is not a candidate, even on
    an exact reference. Two rows that claim one in-window DB row by
    reference are held back."""
    import financial_dashboard.services.statements.bank as bank_module
    from financial_dashboard.services.statements import shared as shared_module

    acc_id = await h.add_bank_account(maker)
    pdf = tmp_path / "bank.pdf"
    pdf.write_bytes(b"%PDF fake")
    upload_id = await _seed_bank_upload(maker, acc_id, file_path=str(pdf))

    async with maker() as session:
        session.add_all(
            [
                Transaction(
                    account_id=acc_id,
                    bank="hdfc",
                    email_type="bank_statement",
                    direction="debit",
                    amount=Decimal("1000.00"),
                    transaction_date=datetime.date(2026, 1, 5),
                    reference_number="FARREF",
                ),
                Transaction(
                    account_id=acc_id,
                    bank="hdfc",
                    email_type="bank_statement",
                    direction="debit",
                    amount=Decimal("500.00"),
                    transaction_date=datetime.date(2026, 7, 2),
                    reference_number="RETRYDUP",
                ),
            ]
        )
        await session.commit()

    # The statement does not tally, so the held rows reach Telegram.
    parsed = h.bank_parsed(
        statement_period_start="01/07/2026",
        statement_period_end="31/07/2026",
        opening_balance="10,000.00",
        closing_balance="5,000.00",
        transactions=[
            h.bank_txn(date="05/07/2026", amount="1,000.00", narration="JULY"),
            h.bank_txn(
                date="10/07/2026",
                amount="1,000.00",
                reference_number="FARREF",
                narration="FAR",
            ),
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
        ],
    )
    monkeypatch.setattr(bank_module, "parse_bank_statement", lambda *a, **kw: parsed)
    monkeypatch.setattr(shared_module, "parse_bank_statement", lambda *a, **kw: parsed)
    notes = h.capture_balance_notes(monkeypatch)

    assert await retry_bank_statement_upload(upload_id, "secret") is True
    assert len(notes) == 1
    assert notes[0].count(": held") == 2

    async with maker() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        assert upload.imported_count == 1
        recon = json.loads(upload.reconciliation_data)
        assert recon["matched"] == []
        ambiguous = [entry for entry in recon["missing"] if entry.get("ambiguous")]
        assert len(ambiguous) == 2
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 3
        # The unique reference index blocks an import. The row must stay
        # unmatched: out of scope, the far row is not a candidate.
        far = next(e for e in recon["missing"] if e["narration"] == "FAR")
        assert far["candidate_transaction_ids"] == []
        assert far["duplicate"] is True
