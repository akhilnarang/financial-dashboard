"""Tests for the (account_id, due_date) dedup guard in
``process_statement_email`` — a re-sent CC statement PDF for a due-date we
already have an upload row for must return the existing row and skip
reconcile / PDF write / import.
"""

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import (
    Account,
    Base,
    Card,
    StatementUpload,
)
from financial_dashboard.services.statements import cc as cc_module


@pytest.fixture
async def session_factory(monkeypatch, tmp_path):
    db_path = tmp_path / "pdf-dedup-test.sqlite"
    sync_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(cc_module, "async_session", maker)
    yield maker
    await engine.dispose()


async def _add_cc_account(maker) -> int:
    async with maker() as session:
        acc = Account(bank="jupiter", label="Jupiter", type="credit_card")
        session.add(acc)
        await session.flush()
        session.add(Card(account_id=acc.id, card_mask="1234", is_primary=True))
        await session.commit()
        return acc.id


def _install_common_monkeypatches(monkeypatch, tmp_path, due_date, import_calls):
    monkeypatch.setattr(
        cc_module,
        "extract_pdf_from_email",
        lambda raw_bytes: [("stmt.pdf", b"%PDF-fake")],
    )
    monkeypatch.setattr(cc_module, "extract_password_hint", lambda *a, **k: None)

    def _fake_parse(pdf_bytes, password, bank):
        return SimpleNamespace(
            bank="jupiter",
            name=None,
            card_number="1234",
            due_date=due_date,
            statement_total_amount_due="1,234.56",
            transactions=[],
        )

    monkeypatch.setattr(cc_module, "_parse_pdf_bytes_sync", _fake_parse)

    async def _record_import(session, upload, parsed, account, recon):
        import_calls.append((upload.id, account.id))
        return []

    monkeypatch.setattr(cc_module, "import_missing_cc_txns", _record_import)

    monkeypatch.setattr(
        cc_module,
        "reconcile_statement",
        lambda parsed, db_txns, account_id, card_masks: {
            "matched": [],
            "missing": [],
        },
    )

    statements_dir = tmp_path / "statements"
    monkeypatch.setattr(
        "financial_dashboard.core.uploads.STATEMENTS_DIR", statements_dir
    )

    async def _noop_snapshot(session, upload):
        return None

    monkeypatch.setattr(cc_module, "emit_cc_snapshot", _noop_snapshot)
    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)

    async def _noop_enrich(recon):
        return 0

    monkeypatch.setattr(cc_module, "enrich_matched_transactions", _noop_enrich)

    import financial_dashboard.services.reminders as reminders_mod

    async def _noop_init(_uid):
        return True

    monkeypatch.setattr(reminders_mod, "init_payment_tracking", _noop_init)

    return statements_dir


@pytest.mark.anyio
async def test_pdf_dedups_against_prior_upload_for_same_due_date(
    session_factory, monkeypatch, tmp_path
):
    acc_id = await _add_cc_account(session_factory)
    async with session_factory() as session:
        existing = StatementUpload(
            account_id=acc_id,
            bank="jupiter",
            filename="",
            file_path="",
            source_kind="email_summary",
            status="parsed",
            due_date="05/05/2026",
            payment_status="pending",
        )
        session.add(existing)
        await session.commit()
        existing_id = existing.id

    import_calls: list = []
    statements_dir = _install_common_monkeypatches(
        monkeypatch, tmp_path, "05/05/2026", import_calls
    )

    result = await cc_module.process_statement_email(
        "jupiter", b"raw", "Your Jupiter Card Statement"
    )

    assert result is not None
    assert result["statement_upload_id"] == existing_id
    assert result["matched"] == 0
    assert result["missing"] == 0
    assert result["imported"] == 0

    assert import_calls == []
    assert not statements_dir.exists() or not any(statements_dir.iterdir())

    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert len(rows) == 1
        assert rows[0].id == existing_id
        assert rows[0].payment_status == "pending"


@pytest.mark.anyio
@pytest.mark.parametrize("due_date", [None, "NO PAYMENT REQUIRED"])
async def test_no_dedup_when_parsed_due_date_is_not_a_date(
    session_factory, monkeypatch, tmp_path, due_date
):
    acc_id = await _add_cc_account(session_factory)
    async with session_factory() as session:
        existing = StatementUpload(
            account_id=acc_id,
            bank="jupiter",
            filename="x",
            file_path="x",
            source_kind="pdf",
            status="parsed",
            due_date=due_date,
            payment_status="pending",
        )
        session.add(existing)
        await session.commit()

    import_calls: list = []
    statements_dir = _install_common_monkeypatches(
        monkeypatch, tmp_path, due_date, import_calls
    )

    # The same attachment twice in one second must not overwrite the first PDF.
    fixed = datetime.datetime(2026, 5, 1, tzinfo=datetime.UTC)
    monkeypatch.setattr(
        "financial_dashboard.core.uploads.datetime",
        SimpleNamespace(
            datetime=SimpleNamespace(now=lambda tz: fixed), UTC=datetime.UTC
        ),
    )
    for _ in range(2):
        result = await cc_module.process_statement_email(
            "jupiter", b"raw", "Your Jupiter Card Statement"
        )
        assert result is not None

    async with session_factory() as session:
        rows = (await session.execute(select(StatementUpload))).scalars().all()
        assert len(rows) == 3
        saved = {Path(r.file_path) for r in rows[1:]}
        assert len(saved) == 2
        assert all(p.parent == statements_dir and p.exists() for p in saved)
