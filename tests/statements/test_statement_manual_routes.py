"""Integration tests for the manual statement upload + payment routes.

Covers the production hardening: manual CC and bank uploads now use the same
per-row SAVEPOINT / duplicate-tolerant import helper as the email path, so one
bad row cannot abort the whole upload. Also covers mark-paid / mark-unpaid
(partial preservation) and reprocess payment-tracking reset.
"""

import datetime
import io
from decimal import Decimal

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from financial_dashboard.core.deps import get_session
from financial_dashboard.db import (
    BankStatementUpload,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus
from financial_dashboard.web import get_router

from . import _helpers as h


def _build_app(maker):
    app = FastAPI()
    app.include_router(get_router())

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    return app


def _file_bytes(name="statement.pdf"):
    return (name, io.BytesIO(b"%PDF fake"), "application/pdf")


# ---------------------------------------------------------------------------
# Manual CC upload
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_manual_cc_upload_imports_missing(maker, monkeypatch, tmp_path):
    import financial_dashboard.web.statements as cc_routes

    monkeypatch.setattr(cc_routes, "STATEMENTS_DIR", tmp_path)
    acc_id = await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="01/07/2026", amount="1,000.00", narration="AMAZON")
        ]
    )
    monkeypatch.setattr(cc_routes, "parse_statement", lambda *a, **kw: parsed)

    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/statements/upload",
            data={"account_id": acc_id, "password": ""},
            files={"file": _file_bytes()},
        )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/statements/")

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert upload.status == "imported"
        assert upload.imported_count == 1
        txn = (await session.execute(select(Transaction))).scalars().one()
        assert txn.counterparty == "AMAZON"


@pytest.mark.anyio
async def test_manual_cc_upload_duplicate_tolerated(maker, monkeypatch, tmp_path):
    """One row hitting IntegrityError must not abort the manual CC upload."""
    import financial_dashboard.web.statements as cc_routes
    import financial_dashboard.services.statements.cc as cc_module

    monkeypatch.setattr(cc_routes, "STATEMENTS_DIR", tmp_path)
    await h.add_cc_account(maker)
    parsed = h.cc_parsed(
        transactions=[
            h.cc_txn(date="01/07/2026", amount="100.00", narration="OK"),
            h.cc_txn(date="02/07/2026", amount="200.00", narration="BAD"),
        ]
    )
    monkeypatch.setattr(cc_routes, "parse_statement", lambda *a, **kw: parsed)

    real_link = cc_module.link_transaction

    def _flaky(ctx, txn):
        if txn.counterparty == "BAD":
            raise IntegrityError("simulated", {}, Exception("dup"))
        real_link(ctx, txn)

    monkeypatch.setattr(cc_module, "link_transaction", _flaky)

    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/statements/upload",
            data={"account_id": 1, "password": ""},
            files={"file": _file_bytes()},
        )
    assert resp.status_code == 303

    async with maker() as session:
        upload = (await session.execute(select(StatementUpload))).scalars().one()
        assert upload.imported_count == 1
        assert "1 duplicate" in (upload.error or "")
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 1


# ---------------------------------------------------------------------------
# Manual bank upload — the key hardening target
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_manual_bank_upload_imports_missing(maker, monkeypatch, tmp_path):
    import financial_dashboard.web.bank_statements as bank_routes

    monkeypatch.setattr(bank_routes, "STATEMENTS_DIR", tmp_path)
    acc_id = await h.add_bank_account(maker)
    parsed = h.bank_parsed(
        account_number="1234567890",
        opening_balance="10,000.00",
        closing_balance="9,000.00",
        statement_period_start="01/07/2026",
        statement_period_end="31/07/2026",
        debit_total="1,000.00",
        transactions=[
            h.bank_txn(date="05/07/2026", amount="1,000.00", narration="UPI Debit"),
        ],
    )
    monkeypatch.setattr(bank_routes, "parse_bank_statement", lambda *a, **kw: parsed)

    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/statements/upload-bank",
            data={"account_id": acc_id, "password": ""},
            files={"file": _file_bytes()},
        )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/statements/bank/")

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.status == "imported"
        assert upload.imported_count == 1


@pytest.mark.anyio
async def test_manual_bank_upload_duplicate_does_not_abort_batch(
    maker, monkeypatch, tmp_path
):
    """Regression: previously the manual bank upload had no SAVEPOINT around
    its import loop, so one duplicate row aborted the entire upload. Now it
    uses the same per-row tolerant helper as the email path — the good rows
    commit and the duplicate is tagged."""
    import financial_dashboard.web.bank_statements as bank_routes

    monkeypatch.setattr(bank_routes, "STATEMENTS_DIR", tmp_path)
    acc_id = await h.add_bank_account(maker)
    # Pre-existing row that a stmt row will collide with by ref.
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="bank_statement",
                direction="debit",
                amount=Decimal("500.00"),
                transaction_date=datetime.date(2026, 7, 2),
                reference_number="MANUALDUP",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        account_number="1234567890",
        transactions=[
            # First stmt row matches the pre-existing by ref.
            h.bank_txn(
                date="02/07/2026",
                amount="500.00",
                reference_number="MANUALDUP",
                narration="matched",
            ),
            # Second reuses the ref → missing → import collides → duplicate.
            h.bank_txn(
                date="03/07/2026",
                amount="500.00",
                reference_number="MANUALDUP",
                narration="dup",
            ),
            # Third is clean → imports.
            h.bank_txn(
                date="04/07/2026",
                amount="700.00",
                reference_number="CLEANREF",
                narration="clean",
            ),
        ],
    )
    monkeypatch.setattr(bank_routes, "parse_bank_statement", lambda *a, **kw: parsed)

    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/statements/upload-bank",
            data={"account_id": acc_id, "password": ""},
            files={"file": _file_bytes()},
        )
    assert resp.status_code == 303

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        # The clean row imported; the duplicate was skipped, NOT aborted.
        assert upload.imported_count == 1
        assert "1 duplicate" in (upload.error or "")
        # Pre-existing + clean import = 2 rows (duplicate not inserted).
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert len(txns) == 2


@pytest.mark.anyio
async def test_manual_bank_upload_generic_error_tolerated(maker, monkeypatch, tmp_path):
    import financial_dashboard.web.bank_statements as bank_routes
    import financial_dashboard.services.statements.bank as bank_module

    monkeypatch.setattr(bank_routes, "STATEMENTS_DIR", tmp_path)
    await h.add_bank_account(maker)
    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(date="01/07/2026", amount="100.00", narration="GOOD"),
            h.bank_txn(date="02/07/2026", amount="200.00", narration="BOOM"),
        ]
    )
    monkeypatch.setattr(bank_routes, "parse_bank_statement", lambda *a, **kw: parsed)

    real_link = bank_module.link_transaction

    def _flaky(ctx, txn):
        if txn.counterparty == "BOOM":
            raise RuntimeError("kaboom")
        real_link(ctx, txn)

    monkeypatch.setattr(bank_module, "link_transaction", _flaky)

    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/statements/upload-bank",
            data={"account_id": 1, "password": ""},
            files={"file": _file_bytes()},
        )
    assert resp.status_code == 303

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.imported_count == 1
        assert "1 unexpected error" in (upload.error or "")


# ---------------------------------------------------------------------------
# Mark paid / mark unpaid
# ---------------------------------------------------------------------------


async def _seed_cc_upload_with_status(
    maker, *, payment_status, paid_amount=Decimal("0"), total="5,000.00"
):
    acc_id = await h.add_cc_account(maker)
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path="/tmp/cc.pdf",
            status="imported",
            due_date="15/08/2026",
            total_amount_due=total,
            payment_status=payment_status,
            payment_paid_amount=paid_amount,
            payment_paid_at=(
                datetime.datetime.now(datetime.UTC)
                if payment_status == PaymentStatus.PAID
                else None
            ),
        )
        session.add(upload)
        await session.commit()
        return upload.id, acc_id


@pytest.mark.anyio
async def test_mark_paid_sets_status_and_amount(maker):
    upload_id, _ = await _seed_cc_upload_with_status(
        maker, payment_status=PaymentStatus.UNPAID
    )
    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/statements/{upload_id}/payment", data={"action": "mark_paid"}
        )
    assert resp.status_code == 303

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.PAID
        assert upload.payment_paid_amount == Decimal("5000.00")
        assert upload.payment_paid_at is not None


@pytest.mark.anyio
async def test_mark_unpaid_from_full_clears(maker):
    upload_id, _ = await _seed_cc_upload_with_status(
        maker,
        payment_status=PaymentStatus.PAID,
        paid_amount=Decimal("5000.00"),
    )
    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/statements/{upload_id}/payment", data={"action": "mark_unpaid"}
        )
    assert resp.status_code == 303

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.UNPAID
        assert upload.payment_paid_amount == Decimal("0")
        assert upload.payment_paid_at is None
        assert upload.payment_sent_offsets == "[]"


@pytest.mark.anyio
async def test_mark_unpaid_preserves_partial(maker):
    """Marking unpaid from PARTIALLY_PAID must keep the real partial amount
    (from bank auto-detection) so history isn't lost; only the manual full-pay
    marker is cleared, and the status stays PARTIALLY_PAID."""
    upload_id, _ = await _seed_cc_upload_with_status(
        maker,
        payment_status=PaymentStatus.PARTIALLY_PAID,
        paid_amount=Decimal("2000.00"),
    )
    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/statements/{upload_id}/payment", data={"action": "mark_unpaid"}
        )
    assert resp.status_code == 303

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status == PaymentStatus.PARTIALLY_PAID
        assert upload.payment_paid_amount == Decimal("2000.00")
        assert upload.payment_paid_at is None


@pytest.mark.anyio
async def test_mark_paid_is_noop_when_already_paid(maker):
    """Re-marking an already-PAID statement must not stamp a new paid_at."""
    upload_id, _ = await _seed_cc_upload_with_status(
        maker,
        payment_status=PaymentStatus.PAID,
        paid_amount=Decimal("5000.00"),
    )
    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        first_paid_at = upload.payment_paid_at
    assert first_paid_at is not None

    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            f"/statements/{upload_id}/payment", data={"action": "mark_paid"}
        )

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_paid_at == first_paid_at  # unchanged


@pytest.mark.anyio
async def test_reprocess_resets_tracking_when_due_changes(maker, monkeypatch, tmp_path):
    """Reprocess must reset payment_status/paid_amount/offsets when the
    statement's due date or total changes (new statement cycle)."""
    import financial_dashboard.web.statements as cc_routes

    monkeypatch.setattr(cc_routes, "STATEMENTS_DIR", tmp_path)
    acc_id = await h.add_cc_account(maker)
    pdf_path = tmp_path / "cc.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path=str(pdf_path),
            status="imported",
            card_number="XXXX XXXX XXXX 1234",
            due_date="15/07/2026",
            total_amount_due="5,000.00",
            payment_status=PaymentStatus.PARTIALLY_PAID,
            payment_paid_amount=Decimal("2000.00"),
            payment_sent_offsets='["7"]',
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    # Reparse yields a NEW due date → triggers tracking reset.
    parsed = h.cc_parsed(
        card_number="XXXX XXXX XXXX 1234",
        due_date="15/08/2026",
        total_due="6,000.00",
        transactions=[
            h.cc_txn(date="01/08/2026", amount="500.00", narration="NEW"),
        ],
    )
    monkeypatch.setattr(cc_routes, "parse_statement", lambda *a, **kw: parsed)

    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(f"/statements/{upload_id}/reprocess")
    assert resp.status_code == 303

    async with maker() as session:
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status is None
        assert upload.payment_paid_amount == Decimal("0")
        assert upload.payment_paid_at is None
        assert upload.payment_sent_offsets == "[]"
        assert upload.due_date == "15/08/2026"
        assert upload.total_amount_due == "6,000.00"


@pytest.mark.anyio
async def test_reprocess_idempotent_no_double_import(maker, monkeypatch, tmp_path):
    """Reprocessing twice must not duplicate already-imported transactions."""
    import financial_dashboard.web.statements as cc_routes

    monkeypatch.setattr(cc_routes, "STATEMENTS_DIR", tmp_path)
    acc_id = await h.add_cc_account(maker)
    pdf_path = tmp_path / "cc.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    async with maker() as session:
        upload = StatementUpload(
            account_id=acc_id,
            bank="hdfc",
            filename="cc.pdf",
            file_path=str(pdf_path),
            status="parsed",
            card_number="XXXX XXXX XXXX 1234",
        )
        session.add(upload)
        await session.commit()
        upload_id = upload.id

    parsed = h.cc_parsed(
        card_number="XXXX XXXX XXXX 1234",
        transactions=[h.cc_txn(date="01/07/2026", amount="500.00", narration="X")],
    )
    monkeypatch.setattr(cc_routes, "parse_statement", lambda *a, **kw: parsed)

    app = _build_app(maker)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(f"/statements/{upload_id}/reprocess")
        await client.post(f"/statements/{upload_id}/reprocess")

    async with maker() as session:
        txns = (await session.execute(select(Transaction))).scalars().all()
        # Second reprocess re-reconciles: the previously-imported row now
        # MATCHES, so it's not re-imported. Still one row.
        assert len(txns) == 1
