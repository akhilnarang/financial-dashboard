"""Integration tests for the manual statement upload + payment routes.

Covers the production hardening: manual CC and bank uploads now use the same
per-row SAVEPOINT / duplicate-tolerant import helper as the email path, so one
bad row cannot abort the whole upload. Also covers mark-paid / mark-unpaid
(partial preservation) and reprocess payment-tracking reset.
"""

import datetime
import io
import json
from decimal import Decimal

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from financial_dashboard.core.deps import get_session
from financial_dashboard.db import (
    BankStatementUpload,
    StatementUpload,
    Transaction,
)
from financial_dashboard.db.enums import PaymentStatus
from financial_dashboard.api import get_router as get_api_router
from financial_dashboard.web import get_router

from . import _helpers as h


def _build_app(maker):
    app = FastAPI()
    app.include_router(get_router())
    app.include_router(get_api_router(paisa_enabled=False))

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
async def test_manual_cc_upload_imports_missing(maker, monkeypatch, statements_dir):
    import financial_dashboard.web.statements as cc_routes

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


# ---------------------------------------------------------------------------
# Manual bank upload — the key hardening target
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_manual_bank_upload_imports_clean_rows_only(
    maker, monkeypatch, statements_dir
):
    """Manual upload holds back same-reference contenders and tolerates an
    unexpected per-row error. The clean row still commits."""
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
                reference_number="MANUALDUP",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        account_number="1234567890",
        transactions=[
            h.bank_txn(
                date="02/07/2026",
                amount="500.00",
                reference_number="MANUALDUP",
                narration="matched",
            ),
            h.bank_txn(
                date="03/07/2026",
                amount="500.00",
                reference_number="MANUALDUP",
                narration="dup",
            ),
            h.bank_txn(
                date="04/07/2026",
                amount="700.00",
                reference_number="CLEANREF",
                narration="clean",
            ),
            h.bank_txn(date="05/07/2026", amount="200.00", narration="BOOM"),
        ],
    )
    monkeypatch.setattr(bank_module, "parse_bank_statement", lambda *a, **kw: parsed)

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
            data={"account_id": acc_id, "password": ""},
            files={"file": _file_bytes()},
        )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/statements/bank/")

    async with maker() as session:
        upload = (await session.execute(select(BankStatementUpload))).scalars().one()
        assert upload.imported_count == 1
        assert "1 unexpected error" in (upload.error or "")
        recon = json.loads(upload.reconciliation_data)
        ambiguous = [entry for entry in recon["missing"] if entry.get("ambiguous")]
        assert len(ambiguous) == 2
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert sorted(t.reference_number for t in txns) == ["CLEANREF", "MANUALDUP"]


@pytest.mark.anyio
async def test_api_bank_upload_dry_run_writes_nothing_then_real_run_imports(
    maker, monkeypatch, tmp_path
):
    """A dry run reports the split and writes no file, upload, enrichment or
    import. The same request with dry_run=false does all four."""
    import financial_dashboard.core.uploads as uploads_module
    import financial_dashboard.services.statements.bank as bank_module

    monkeypatch.setattr(uploads_module, "STATEMENTS_DIR", tmp_path)
    acc_id = await h.add_bank_account(maker)
    async with maker() as session:
        session.add(
            Transaction(
                account_id=acc_id,
                bank="hdfc",
                email_type="sms_debit",
                direction="debit",
                amount=Decimal("500.00"),
                transaction_date=datetime.date(2026, 7, 2),
                reference_number="KNOWNREF",
            )
        )
        await session.commit()

    parsed = h.bank_parsed(
        transactions=[
            h.bank_txn(
                date="02/07/2026",
                amount="500.00",
                reference_number="KNOWNREF",
                counterparty="Sample Shop",
            ),
            h.bank_txn(date="04/07/2026", amount="700.00", reference_number="NEWREF"),
        ],
    )
    monkeypatch.setattr(bank_module, "parse_bank_statement", lambda *a, **kw: parsed)

    async def _post(**extra):
        async with AsyncClient(
            transport=ASGITransport(app=_build_app(maker)), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/statements/bank/upload",
                data={"account_id": acc_id, **extra},
                files={"file": _file_bytes()},
            )

    async def _state():
        async with maker() as session:
            uploads = (await session.execute(select(BankStatementUpload))).all()
            txns = (await session.execute(select(Transaction))).scalars().all()
            return len(uploads), {t.reference_number: t.counterparty for t in txns}

    preview = await _post()
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["dry_run"] is True
    assert body["upload_id"] is None
    assert [e["reference_number"] for e in body["matched"]] == ["KNOWNREF"]
    assert [e["reference_number"] for e in body["missing"]] == ["NEWREF"]
    assert body["missing"][0]["imported"] is False
    assert await _state() == (0, {"KNOWNREF": None})
    assert list(tmp_path.iterdir()) == []

    real = await _post(dry_run="false")
    assert real.status_code == 200, real.text
    body = real.json()
    assert body["upload_id"] is not None
    assert body["imported_count"] == 1
    assert body["missing"][0]["imported_transaction_id"] is not None
    assert await _state() == (
        1,
        {"KNOWNREF": "Sample Shop", "NEWREF": "UPI-Debit-MERCHANT"},
    )
    assert len(list(tmp_path.iterdir())) == 1


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
async def test_mark_paid_then_unpaid_round_trip(maker):
    """Mark paid stamps the full amount once; mark unpaid clears it."""
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
            first_paid_at = upload.payment_paid_at
        assert first_paid_at is not None

        # A second mark_paid must not stamp a new paid_at.
        await client.post(
            f"/statements/{upload_id}/payment", data={"action": "mark_paid"}
        )
        async with maker() as session:
            upload = await session.get(StatementUpload, upload_id)
            assert upload.payment_paid_at == first_paid_at

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
async def test_reprocess_resets_tracking_when_due_changes(
    maker, monkeypatch, tmp_path, statements_dir
):
    """Reprocess must reset payment_status/paid_amount/offsets when the
    statement's due date or total changes (new statement cycle). A second
    reprocess must not import the same rows again."""
    import financial_dashboard.web.statements as cc_routes

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
        # The second run matches the imported row and imports nothing.
        await client.post(f"/statements/{upload_id}/reprocess")

    async with maker() as session:
        txns = (await session.execute(select(Transaction))).scalars().all()
        assert [t.counterparty for t in txns] == ["NEW"]
        upload = await session.get(StatementUpload, upload_id)
        assert upload.payment_status is None
        assert upload.payment_paid_amount == Decimal("0")
        assert upload.payment_paid_at is None
        assert upload.payment_sent_offsets == "[]"
        assert upload.due_date == "15/08/2026"
        assert upload.total_amount_due == "6,000.00"
