from __future__ import annotations

import datetime as dt
import importlib
import importlib.util
import json
import re
from contextlib import asynccontextmanager
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.settings as settings_service
from financial_dashboard.core.deps import get_session
from financial_dashboard.db import (
    Base,
    Email,
    EmailKind,
    FetchRule,
    Setting,
    SmsMessage,
    Transaction,
)
from financial_dashboard.services.linker import build_link_context
from financial_dashboard.services.sms_pipeline import process_sms_row
from financial_dashboard.web import get_router as get_web_router


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    """Snapshot/restore the module-global settings cache around every test.

    Several tests here exercise the real ``save_settings`` / ``load_all_settings``,
    which mutate ``settings_service._cache`` in place. Without restoring it, a
    leaked ``ledger.backend=paisa`` would contaminate later tests across the whole
    suite (the SMS/email pipelines would route to paisa mode and stop creating
    Transaction rows).
    """
    saved = dict(settings_service._cache)
    try:
        yield
    finally:
        settings_service._cache.clear()
        settings_service._cache.update(saved)


@pytest.fixture
async def session_and_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session, maker
    await engine.dispose()


@pytest.fixture
async def session(session_and_maker):
    session, _ = session_and_maker
    return session


@asynccontextmanager
async def _open_session_and_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session, maker
    await engine.dispose()


def _require_module(name: str):
    spec = importlib.util.find_spec(name)
    assert spec is not None, f"Missing module: {name}"
    return importlib.import_module(name)


def _require_attr(obj, name: str):
    assert hasattr(obj, name), f"Missing attribute: {obj}.{name}"
    return getattr(obj, name)


def _require_paisa_export_model():
    import financial_dashboard.db as db

    assert hasattr(db, "PaisaExport"), (
        "financial_dashboard.db must re-export PaisaExport"
    )
    return db.PaisaExport


def _normalize_counterparty_for_asserts(s: str | None) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _txn_data(**overrides) -> dict:
    data = {
        "bank": "hdfc",
        "email_type": "hdfc_dc_transaction_alert",
        "direction": "debit",
        "amount": Decimal("500.00"),
        "currency": "INR",
        "transaction_date": dt.date(2026, 5, 2),
        "transaction_time": dt.time(14, 23, 0),
        "counterparty": "Zomato",
        "card_mask": "x1234",
        "account_mask": None,
        "reference_number": None,
        "channel": "card",
        "balance": None,
        "raw_description": "Spent Rs.500 at Zomato",
    }
    data.update(overrides)
    return data


def _make_sms_row(**overrides) -> SmsMessage:
    base = {
        "bank": "hdfc",
        "sender": "VK-HDFCBK",
        "body": (
            "Spent Rs.500 From HDFC Bank Card x1234 At Zomato "
            "On 2026-05-02:14:23:00 Bal Rs.1000"
        ),
        "received_at": dt.datetime(2026, 5, 2, 8, 53, tzinfo=dt.UTC),
    }
    base.update(overrides)
    return SmsMessage(**base)


def _make_raw_email() -> bytes:
    return (
        b"From: alerts@example.com\n"
        b"Subject: transaction alert\n"
        b"Date: Sat, 2 May 2026 14:30:00 +0530\n"
        b"\n"
        b"body"
    )


def _build_test_app(maker):
    app = FastAPI()
    app.include_router(get_web_router())

    async def _override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _override
    return app


def _equitas_payment_eml(amount: str, card_last4: str) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Payment received !"
    msg["From"] = "cc-alerts@equitas.bank.in"
    msg["Date"] = "Wed, 6 May 2026 00:28:00 +0530"
    msg.set_content(
        "Dear Mr. Test Customer,\n\n"
        f"We inform you that INR {amount} was received on 06/05/2026 and was "
        f"credited to your Equitas Credit Card XX{card_last4}.\n"
    )
    return msg.as_bytes()


def _set_paisa_cache(monkeypatch, **overrides):
    cache = {
        "ledger.backend": "paisa",
        "paisa.ledger_cli": "ledger",
        "paisa.main_journal_path": "/tmp/journal/main.ledger",
        "paisa.generated_journal_path": "/tmp/journal/imports/financial-dashboard.ledger",
        "paisa.default_expense_account": "Expenses:Uncategorized",
        "paisa.default_income_account": "Income:Uncategorized",
        "paisa.fallback_asset_account": "Assets:Unknown",
        "paisa.fallback_liability_account": "Liabilities:Unknown",
        "paisa.account_map": "{}",
    }
    cache.update(overrides)
    monkeypatch.setattr(settings_service, "_cache", cache.copy(), raising=False)


def _set_valid_paisa_cache(monkeypatch, tmp_path: Path, **overrides) -> dict[str, str]:
    main_dir = tmp_path / "books"
    import_dir = main_dir / "imports"
    main_dir.mkdir(parents=True, exist_ok=True)
    import_dir.mkdir(parents=True, exist_ok=True)

    cache = {
        "ledger.backend": "paisa",
        "paisa.ledger_cli": "ledger",
        "paisa.main_journal_path": str(main_dir / "main.ledger"),
        "paisa.generated_journal_path": str(import_dir / "financial-dashboard.ledger"),
        "paisa.default_expense_account": "Expenses:Uncategorized",
        "paisa.default_income_account": "Income:Uncategorized",
        "paisa.fallback_asset_account": "Assets:Unknown",
        "paisa.fallback_liability_account": "Liabilities:Unknown",
        "paisa.account_map": json.dumps(
            {
                "hdfc:card:1234": "Liabilities:CreditCard:HDFC:1234",
                "hdfc:account:5678": "Assets:Bank:HDFC:5678",
            }
        ),
    }
    cache.update(overrides)
    monkeypatch.setattr(settings_service, "_cache", cache.copy(), raising=False)
    return cache


def test_ledger_backend_setting_defaults_to_local():
    assert "ledger.backend" in settings_service.SETTINGS_REGISTRY
    assert settings_service.SETTINGS_REGISTRY["ledger.backend"].default == "local"

    get_ledger_backend = _require_attr(settings_service, "get_ledger_backend")
    assert get_ledger_backend() == "local"


def test_get_paisa_config_rejects_non_ledger_cli(monkeypatch):
    _set_paisa_cache(monkeypatch, **{"paisa.ledger_cli": "hledger"})
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    with pytest.raises(ValueError, match="ledger"):
        get_paisa_config()


def test_get_paisa_config_requires_non_empty_paths(monkeypatch):
    _set_paisa_cache(monkeypatch, **{"paisa.main_journal_path": ""})
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    with pytest.raises(ValueError, match="main_journal_path"):
        get_paisa_config()

    _set_paisa_cache(monkeypatch, **{"paisa.generated_journal_path": ""})
    with pytest.raises(ValueError, match="generated_journal_path"):
        get_paisa_config()


def test_get_paisa_config_requires_generated_path_inside_main_dir(
    monkeypatch, tmp_path
):
    main_dir = tmp_path / "books"
    main_dir.mkdir(parents=True, exist_ok=True)
    _set_paisa_cache(
        monkeypatch,
        **{
            "paisa.main_journal_path": str(main_dir / "main.ledger"),
            "paisa.generated_journal_path": str(
                tmp_path / "other" / "generated.ledger"
            ),
        },
    )
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    with pytest.raises(ValueError, match="subdir|directory"):
        get_paisa_config()


def test_get_paisa_config_allows_generated_path_in_same_dir_or_subdir(
    monkeypatch, tmp_path
):
    main_dir = tmp_path / "books"
    main_dir.mkdir(parents=True, exist_ok=True)
    _set_paisa_cache(
        monkeypatch,
        **{
            "paisa.main_journal_path": str(main_dir / "main.ledger"),
            "paisa.generated_journal_path": str(
                main_dir / "imports" / "generated.ledger"
            ),
        },
    )
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    config = get_paisa_config()
    assert config.main_journal_path == str((main_dir / "main.ledger").resolve())
    assert config.generated_journal_path == str(
        (main_dir / "imports" / "generated.ledger").resolve()
    )


def test_get_paisa_config_rejects_generated_path_equal_to_main(monkeypatch, tmp_path):
    main_dir = tmp_path / "books"
    main_dir.mkdir(parents=True, exist_ok=True)
    _set_paisa_cache(
        monkeypatch,
        **{
            "paisa.main_journal_path": str(main_dir / "main.ledger"),
            "paisa.generated_journal_path": str(main_dir / "." / "main.ledger"),
        },
    )
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    with pytest.raises(ValueError, match="must not be the main journal file"):
        get_paisa_config()


@pytest.mark.parametrize(
    ("key", "path_value", "match"),
    [
        (
            "paisa.main_journal_path",
            "books/main.ledger",
            "paisa.main_journal_path must be an absolute path",
        ),
        (
            "paisa.generated_journal_path",
            "books/imports/generated.ledger",
            "paisa.generated_journal_path must be an absolute path",
        ),
    ],
)
def test_get_paisa_config_rejects_relative_paths(monkeypatch, key, path_value, match):
    _set_paisa_cache(monkeypatch, **{key: path_value})
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    with pytest.raises(ValueError, match=match):
        get_paisa_config()


def test_get_paisa_config_expands_tilde_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    books_dir = tmp_path / "books"
    _set_paisa_cache(
        monkeypatch,
        **{
            "paisa.main_journal_path": "~/books/main.ledger",
            "paisa.generated_journal_path": "~/books/imports/generated.ledger",
        },
    )
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    config = get_paisa_config()
    assert config.main_journal_path == str((books_dir / "main.ledger").resolve())
    assert config.generated_journal_path == str(
        (books_dir / "imports" / "generated.ledger").resolve()
    )


def test_get_paisa_config_rejects_invalid_account_map_json(monkeypatch):
    _set_paisa_cache(monkeypatch, **{"paisa.account_map": "{not json"})
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    with pytest.raises(ValueError, match="account_map|JSON"):
        get_paisa_config()


@pytest.mark.parametrize(
    "bad_account",
    (
        "Liabilities:Card:1234;Expenses:Injected",
        "Assets:\tBank:HDFC:1234",
        "Assets:\rBank:HDFC:1234",
        "Income:Salary\nExpenses:Other",
    ),
)
def test_get_paisa_config_rejects_unsafe_account_map_values(monkeypatch, bad_account):
    _set_paisa_cache(
        monkeypatch,
        **{
            "paisa.account_map": json.dumps({"hdfc:card:1234": bad_account}),
        },
    )
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    with pytest.raises(ValueError, match="account_map"):
        get_paisa_config()


@pytest.mark.parametrize(
    "key",
    (
        "paisa.default_expense_account",
        "paisa.default_income_account",
        "paisa.fallback_asset_account",
        "paisa.fallback_liability_account",
    ),
)
def test_get_paisa_config_rejects_unsafe_default_and_fallback_account_names(
    monkeypatch, key
):
    _set_paisa_cache(monkeypatch, **{key: "Expenses:Food;Injected"})
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    with pytest.raises(ValueError, match="ledger account"):
        get_paisa_config()


def test_parse_form_updates_round_trips_paisa_strings_and_json():
    raw_map = {
        "hdfc:card:1234": "Liabilities:CreditCard:HDFC:1234",
        "hdfc:account:5678": "Assets:Bank:HDFC:5678",
    }
    form = {
        "ledger.backend": "paisa",
        "paisa.ledger_cli": "ledger",
        "paisa.main_journal_path": "/home/user/books/main.ledger",
        "paisa.generated_journal_path": "/home/user/books/imports/generated.ledger",
        "paisa.default_expense_account": "Expenses:Food",
        "paisa.default_income_account": "Income:Salary",
        "paisa.fallback_asset_account": "Assets:Unknown",
        "paisa.fallback_liability_account": "Liabilities:Unknown",
        "paisa.account_map": json.dumps(raw_map),
    }
    updates, errors = settings_service.parse_form_updates(form)
    assert errors == []
    assert updates["paisa.main_journal_path"] == form["paisa.main_journal_path"]
    assert (
        updates["paisa.generated_journal_path"] == form["paisa.generated_journal_path"]
    )
    assert updates["paisa.default_expense_account"] == "Expenses:Food"
    assert json.loads(updates["paisa.account_map"]) == raw_map


def test_parse_form_updates_rejects_invalid_paisa_account_map_json_text():
    form = {
        "ledger.backend": "paisa",
        "paisa.ledger_cli": "ledger",
        "paisa.main_journal_path": "/home/user/books/main.ledger",
        "paisa.generated_journal_path": "/home/user/books/imports/generated.ledger",
        "paisa.default_expense_account": "Expenses:Food",
        "paisa.default_income_account": "Income:Salary",
        "paisa.fallback_asset_account": "Assets:Unknown",
        "paisa.fallback_liability_account": "Liabilities:Unknown",
        "paisa.account_map": '{"hdfc:card:1234":',
    }
    _, errors = settings_service.parse_form_updates(form)
    assert errors
    assert any("account_map" in err.lower() and "json" in err.lower() for err in errors)


@pytest.mark.anyio
async def test_save_settings_round_trips_valid_paisa_settings(monkeypatch, tmp_path):
    for key in (
        "ledger.backend",
        "paisa.ledger_cli",
        "paisa.main_journal_path",
        "paisa.generated_journal_path",
        "paisa.default_expense_account",
        "paisa.default_income_account",
        "paisa.fallback_asset_account",
        "paisa.fallback_liability_account",
        "paisa.account_map",
    ):
        assert key in settings_service.SETTINGS_REGISTRY

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(settings_service, "async_session", maker)

    main_dir = tmp_path / "books"
    generated_path = main_dir / "imports" / "generated.ledger"
    updates = {
        "ledger.backend": "paisa",
        "paisa.ledger_cli": "ledger",
        "paisa.main_journal_path": str(main_dir / "main.ledger"),
        "paisa.generated_journal_path": str(generated_path),
        "paisa.default_expense_account": "Expenses:Food",
        "paisa.default_income_account": "Income:Salary",
        "paisa.fallback_asset_account": "Assets:Unknown",
        "paisa.fallback_liability_account": "Liabilities:Unknown",
        "paisa.account_map": json.dumps(
            {"hdfc:card:1234": "Liabilities:CreditCard:HDFC:1234"}
        ),
    }

    changed = await settings_service.save_settings(updates)
    assert "paisa.main_journal_path" in changed

    loaded = await settings_service.load_all_settings()
    assert loaded["paisa.main_journal_path"] == updates["paisa.main_journal_path"]
    assert json.loads(loaded["paisa.account_map"]) == {
        "hdfc:card:1234": "Liabilities:CreditCard:HDFC:1234"
    }
    await engine.dispose()


@pytest.mark.anyio
async def test_invalid_paisa_config_is_detectable_at_save_time(monkeypatch):
    assert "ledger.backend" in settings_service.SETTINGS_REGISTRY

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(settings_service, "async_session", maker)
    with pytest.raises(ValueError):
        await settings_service.save_settings(
            {
                "ledger.backend": "paisa",
                "paisa.ledger_cli": "hledger",
                "paisa.main_journal_path": "/tmp/j/main.ledger",
                "paisa.generated_journal_path": "/tmp/j/generated.ledger",
                "paisa.default_expense_account": "Expenses:Uncategorized",
                "paisa.default_income_account": "Income:Uncategorized",
                "paisa.fallback_asset_account": "Assets:Unknown",
                "paisa.fallback_liability_account": "Liabilities:Unknown",
                "paisa.account_map": "{}",
            }
        )
    await engine.dispose()


@pytest.mark.anyio
async def test_invalid_paisa_config_is_detectable_at_startup(monkeypatch):
    assert "ledger.backend" in settings_service.SETTINGS_REGISTRY

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(settings_service, "async_session", maker)

    async with maker() as session:
        session.add_all(
            [
                Setting(key="ledger.backend", value="paisa"),
                Setting(key="paisa.ledger_cli", value="hledger"),
                Setting(key="paisa.main_journal_path", value="/tmp/j/main.ledger"),
                Setting(
                    key="paisa.generated_journal_path", value="/tmp/j/generated.ledger"
                ),
                Setting(
                    key="paisa.default_expense_account", value="Expenses:Uncategorized"
                ),
                Setting(
                    key="paisa.default_income_account", value="Income:Uncategorized"
                ),
                Setting(key="paisa.fallback_asset_account", value="Assets:Unknown"),
                Setting(
                    key="paisa.fallback_liability_account", value="Liabilities:Unknown"
                ),
                Setting(key="paisa.account_map", value="{}"),
            ]
        )
        await session.commit()

    with pytest.raises(ValueError):
        await settings_service.load_all_settings()
    await engine.dispose()


def test_build_paisa_idempotency_key_prefers_reference_tuple_semantics():
    paisa_service = _require_module("financial_dashboard.services.paisa")
    build_key = _require_attr(paisa_service, "build_paisa_idempotency_key")

    first = build_key(
        _txn_data(
            bank="icici",
            direction="credit",
            reference_number=" UTR-00991 ",
            amount=Decimal("500.00"),
            transaction_date=dt.date(2026, 5, 1),
            counterparty="Employer Payroll",
            source="sms",
        )
    )
    second = build_key(
        _txn_data(
            bank="icici",
            direction="credit",
            reference_number="UTR-00991",
            amount=Decimal("499.99"),
            transaction_date=dt.date(2026, 5, 9),
            counterparty="Completely Different",
            source="email",
        )
    )
    different = build_key(
        _txn_data(
            bank="icici",
            direction="debit",
            reference_number="UTR-00991",
            amount=Decimal("499.99"),
            source="email",
        )
    )

    assert first == second
    assert first != different


def test_build_paisa_idempotency_key_preserves_reference_case():
    paisa_service = _require_module("financial_dashboard.services.paisa")
    build_key = _require_attr(paisa_service, "build_paisa_idempotency_key")

    upper = build_key(
        _txn_data(
            bank="icici",
            direction="credit",
            reference_number="AbC-123",
            transaction_date=None,
            transaction_time=None,
        )
    )
    lower = build_key(
        _txn_data(
            bank="icici",
            direction="credit",
            reference_number="abc-123",
            transaction_date=None,
            transaction_time=None,
        )
    )
    assert upper != lower


def test_build_paisa_idempotency_key_fallback_excludes_source_and_normalizes_fields():
    paisa_service = _require_module("financial_dashboard.services.paisa")
    build_key = _require_attr(paisa_service, "build_paisa_idempotency_key")

    assert _normalize_counterparty_for_asserts("PhonePe Pvt. Ltd.") == "phonepepvtltd"

    base_sms = build_key(
        _txn_data(
            source="sms",
            reference_number=None,
            amount=Decimal("500"),
            currency="inr",
            counterparty="PhonePe Pvt. Ltd.",
            card_mask="XX-1234",
            account_mask="a/c *0012",
        )
    )
    equivalent_email = build_key(
        _txn_data(
            source="email",
            reference_number="",
            amount=Decimal("500.000"),
            currency=None,
            counterparty="phonepe pvt ltd",
            card_mask="1234",
            account_mask="0012",
        )
    )
    different_amount = build_key(
        _txn_data(
            source="email",
            reference_number="",
            amount=Decimal("500.01"),
            currency="INR",
            counterparty="phonepe pvt ltd",
            card_mask="1234",
            account_mask="0012",
        )
    )
    different_counterparty = build_key(
        _txn_data(
            source="email",
            reference_number="",
            amount=Decimal("500.00"),
            currency="INR",
            counterparty="Some Other Merchant",
            card_mask="1234",
            account_mask="0012",
        )
    )

    assert base_sms == equivalent_email
    assert base_sms != different_amount
    assert base_sms != different_counterparty


@pytest.mark.anyio
async def test_cross_channel_sms_then_email_merges_single_export(monkeypatch, tmp_path):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    render_entry = _require_attr(paisa_service, "render_paisa_journal_entry")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        sms_row = _make_sms_row()
        email_row = Email(
            provider="gmail",
            message_id="mid-1",
            source_id=None,
            remote_id="rid-1",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 34, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([sms_row, email_row])
        await session.flush()

        sms_outcome = await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                reference_number=None,
                transaction_time=dt.time(14, 23),
                counterparty="PHPE00000",
            ),
            sms_row=sms_row,
        )
        email_outcome = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                reference_number="IMPS:ABC123",
                transaction_time=dt.time(14, 27),
                counterparty="John Doe via PhonePe",
            ),
            email_row=email_row,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.source == "sms+email"
        assert row.sms_message_id == sms_row.id
        assert row.email_id == email_row.id
        assert row.reference_number == "IMPS:ABC123"
        assert row.counterparty == "John Doe via PhonePe"
        assert row.status == "exported"
        assert sms_outcome.status == "parsed"
        assert email_outcome.status == "parsed"
        assert sms_outcome.needs_journal_rewrite is True
        assert email_outcome.needs_journal_rewrite is True
        entry = render_entry(row)
        assert entry.count("\n\n") <= 1


@pytest.mark.anyio
async def test_cross_channel_email_then_sms_merges_single_export(monkeypatch, tmp_path):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        sms_row = _make_sms_row()
        email_row = Email(
            provider="gmail",
            message_id="mid-2",
            source_id=None,
            remote_id="rid-2",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 34, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([sms_row, email_row])
        await session.flush()

        first = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                reference_number="IMPS:XYZ1",
                transaction_time=dt.time(14, 24),
                counterparty="Phone Pe Private Limited",
            ),
            email_row=email_row,
        )
        second = await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                reference_number=None,
                transaction_time=dt.time(14, 28),
                counterparty="PZCREDIT0000",
            ),
            sms_row=sms_row,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.source == "sms+email"
        assert row.sms_message_id == sms_row.id
        assert row.email_id == email_row.id
        assert row.reference_number == "IMPS:XYZ1"
        assert row.counterparty == "Phone Pe Private Limited"
        assert first.needs_journal_rewrite is True
        assert second.needs_journal_rewrite is True


@pytest.mark.anyio
async def test_date_only_match_with_counterparty_agreement_merges(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        sms_row = _make_sms_row()
        email_row = Email(
            provider="gmail",
            message_id="mid-date-yes",
            source_id=None,
            remote_id="rid-date-yes",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 40, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([sms_row, email_row])
        await session.flush()

        await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                transaction_time=None,
                reference_number=None,
                counterparty="PhonePe Transfer",
            ),
            sms_row=sms_row,
        )
        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                transaction_time=None,
                reference_number=None,
                counterparty="John via PhonePe Transfer",
            ),
            email_row=email_row,
        )

        count = await session.scalar(select(func.count()).select_from(PaisaExport))
        assert count == 1


@pytest.mark.anyio
async def test_date_only_match_with_counterparty_mismatch_does_not_merge(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        sms_row = _make_sms_row()
        email_row = Email(
            provider="gmail",
            message_id="mid-date-no",
            source_id=None,
            remote_id="rid-date-no",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 40, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([sms_row, email_row])
        await session.flush()

        await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                transaction_time=None,
                reference_number=None,
                counterparty="PhonePe Transfer",
            ),
            sms_row=sms_row,
        )
        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                transaction_time=None,
                reference_number=None,
                counterparty="Unrelated Merchant",
            ),
            email_row=email_row,
        )

        count = await session.scalar(select(func.count()).select_from(PaisaExport))
        assert count == 2


@pytest.mark.anyio
async def test_time_window_just_inside_merges_single_export(monkeypatch, tmp_path):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        sms_row = _make_sms_row()
        email_row = Email(
            provider="gmail",
            message_id="mid-window-in",
            source_id=None,
            remote_id="rid-window-in",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 40, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([sms_row, email_row])
        await session.flush()

        await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                transaction_time=dt.time(14, 23, 0), reference_number=None
            ),
            sms_row=sms_row,
        )
        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                transaction_time=dt.time(14, 32, 59),
                reference_number=None,
                counterparty="Zomato",
            ),
            email_row=email_row,
        )

        count = await session.scalar(select(func.count()).select_from(PaisaExport))
        assert count == 1


@pytest.mark.anyio
async def test_time_window_just_outside_does_not_merge(monkeypatch, tmp_path):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        sms_row = _make_sms_row()
        email_row = Email(
            provider="gmail",
            message_id="mid-window-out",
            source_id=None,
            remote_id="rid-window-out",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 40, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([sms_row, email_row])
        await session.flush()

        await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                transaction_time=dt.time(14, 23, 0), reference_number=None
            ),
            sms_row=sms_row,
        )
        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                transaction_time=dt.time(14, 33, 1),
                reference_number=None,
                counterparty="Zomato",
            ),
            email_row=email_row,
        )

        count = await session.scalar(select(func.count()).select_from(PaisaExport))
        assert count == 2


@pytest.mark.anyio
async def test_cross_midnight_later_email_does_not_overwrite_earlier_transaction_time(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        email_first = Email(
            provider="gmail",
            message_id="mid-cross-midnight-1",
            source_id=None,
            remote_id="rid-cross-midnight-1",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 18, 40, tzinfo=dt.UTC),
            status="pending",
        )
        email_second = Email(
            provider="gmail",
            message_id="mid-cross-midnight-2",
            source_id=None,
            remote_id="rid-cross-midnight-2",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 18, 41, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([email_first, email_second])
        await session.flush()

        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                transaction_date=dt.date(2026, 5, 2),
                transaction_time=dt.time(23, 59, 0),
                reference_number=None,
                counterparty="Late Night Merchant",
            ),
            email_row=email_first,
        )
        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                transaction_date=dt.date(2026, 5, 3),
                transaction_time=dt.time(0, 1, 0),
                reference_number=None,
                counterparty="Late Night Merchant",
            ),
            email_row=email_second,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.transaction_date == dt.date(2026, 5, 2)
        assert row.transaction_time == dt.time(23, 59, 0)


@pytest.mark.anyio
async def test_am_pm_alias_match_enriches_existing_export_and_corrects_time(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        email_row = Email(
            provider="gmail",
            message_id="mid-3",
            source_id=None,
            remote_id="rid-3",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 22, 40, tzinfo=dt.UTC),
            status="pending",
        )
        sms_row = _make_sms_row(
            received_at=dt.datetime(2026, 5, 2, 22, 45, tzinfo=dt.UTC)
        )
        session.add_all([email_row, sms_row])
        await session.flush()

        seeded = PaisaExport(
            source="email",
            email_id=email_row.id,
            sms_message_id=None,
            idempotency_key="seed-key",
            bank="icici",
            email_type="icici_cc_transaction_alert",
            direction="debit",
            amount=Decimal("500.00"),
            currency="INR",
            transaction_date=dt.date(2026, 5, 2),
            transaction_time=dt.time(10, 33, 0),
            counterparty="TEST HOSPITAL",
            reference_number=None,
            card_mask="0000",
            account_mask=None,
            source_account="Liabilities:CreditCard:ICICI:0000",
            counterparty_account="Expenses:Health",
            missing_account_mapping=False,
            status="exported",
        )
        session.add(seeded)
        await session.flush()

        outcome = await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                bank="icici",
                email_type="icici_cc_transaction_alert",
                amount=Decimal("500.00"),
                transaction_date=dt.date(2026, 5, 2),
                transaction_time=dt.time(22, 33, 0),
                counterparty="TEST HOSPITAL",
                card_mask="0000",
            ),
            sms_row=sms_row,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert len(rows) == 1
        assert rows[0].transaction_time == dt.time(22, 33, 0)
        assert rows[0].source == "sms+email"
        assert outcome.needs_journal_rewrite is True


@pytest.mark.anyio
async def test_am_pm_alias_match_handles_midnight_stored_as_noon(monkeypatch, tmp_path):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        email_row = Email(
            provider="gmail",
            message_id="mid-ampm-12",
            source_id=None,
            remote_id="rid-ampm-12",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 22, 40, tzinfo=dt.UTC),
            status="pending",
        )
        sms_row = _make_sms_row(
            received_at=dt.datetime(2026, 5, 2, 22, 45, tzinfo=dt.UTC)
        )
        session.add_all([email_row, sms_row])
        await session.flush()

        seeded = PaisaExport(
            source="email",
            email_id=email_row.id,
            sms_message_id=None,
            idempotency_key="seed-midnight-alias",
            bank="icici",
            email_type="icici_cc_transaction_alert",
            direction="debit",
            amount=Decimal("500.00"),
            currency="INR",
            transaction_date=dt.date(2026, 5, 2),
            transaction_time=dt.time(12, 7, 0),
            counterparty="TEST HOSPITAL",
            reference_number=None,
            card_mask="0000",
            account_mask=None,
            source_account="Liabilities:CreditCard:ICICI:0000",
            counterparty_account="Expenses:Health",
            missing_account_mapping=False,
            status="exported",
        )
        session.add(seeded)
        await session.flush()

        await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                bank="icici",
                email_type="icici_cc_transaction_alert",
                amount=Decimal("500.00"),
                transaction_date=dt.date(2026, 5, 2),
                transaction_time=dt.time(0, 7, 0),
                counterparty="TEST HOSPITAL",
                card_mask="0000",
            ),
            sms_row=sms_row,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert len(rows) == 1
        assert rows[0].transaction_time == dt.time(0, 7, 0)


@pytest.mark.anyio
async def test_am_pm_alias_does_not_merge_for_non_ambiguous_type_or_counterparty_mismatch(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        email_row = Email(
            provider="gmail",
            message_id="mid-ampm-neg",
            source_id=None,
            remote_id="rid-ampm-neg",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 22, 40, tzinfo=dt.UTC),
            status="pending",
        )
        sms_row = _make_sms_row(
            received_at=dt.datetime(2026, 5, 2, 22, 45, tzinfo=dt.UTC)
        )
        session.add_all([email_row, sms_row])
        await session.flush()

        seeded = PaisaExport(
            source="email",
            email_id=email_row.id,
            sms_message_id=None,
            idempotency_key="seed-non-ambiguous",
            bank="icici",
            email_type="hdfc_dc_transaction_alert",
            direction="debit",
            amount=Decimal("500.00"),
            currency="INR",
            transaction_date=dt.date(2026, 5, 2),
            transaction_time=dt.time(10, 33, 0),
            counterparty="SOMESHOP",
            reference_number=None,
            card_mask="0000",
            account_mask=None,
            source_account="Liabilities:CreditCard:ICICI:0000",
            counterparty_account="Expenses:Health",
            missing_account_mapping=False,
            status="exported",
        )
        session.add(seeded)
        await session.flush()

        await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                bank="icici",
                email_type="hdfc_dc_transaction_alert",
                amount=Decimal("500.00"),
                transaction_date=dt.date(2026, 5, 2),
                transaction_time=dt.time(22, 33, 0),
                counterparty="DIFFERENT MERCHANT",
                card_mask="0000",
            ),
            sms_row=sms_row,
        )

        count = await session.scalar(select(func.count()).select_from(PaisaExport))
        assert count == 2


def test_paisa_export_model_does_not_persist_rendered_journal_cache_column():
    PaisaExport = _require_paisa_export_model()
    assert "journal_entry" not in PaisaExport.__table__.columns

    sample = PaisaExport(
        id=99,
        source="sms",
        email_id=None,
        sms_message_id=11,
        idempotency_key="no-journal-cache",
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500.00"),
        currency="INR",
        transaction_date=dt.date(2026, 5, 2),
        transaction_time=dt.time(14, 23, 0),
        counterparty="Zomato",
        reference_number="IMPS:ABC",
        card_mask="1234",
        account_mask=None,
        source_account="Liabilities:CreditCard:HDFC:1234",
        counterparty_account="Expenses:Food",
        missing_account_mapping=False,
        status="exported",
    )
    assert not hasattr(sample, "journal_entry")


def test_render_paisa_journal_entry_has_deterministic_debit_and_credit_output():
    paisa_service = _require_module("financial_dashboard.services.paisa")
    render_entry = _require_attr(paisa_service, "render_paisa_journal_entry")
    PaisaExport = _require_paisa_export_model()

    debit = PaisaExport(
        id=123,
        source="sms+email",
        email_id=789,
        sms_message_id=456,
        idempotency_key="d-1",
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500.00"),
        currency="INR",
        transaction_date=dt.date(2026, 5, 2),
        transaction_time=dt.time(14, 23, 0),
        counterparty="Zomato",
        reference_number="ABC123",
        card_mask="1234",
        account_mask=None,
        source_account="Liabilities:CreditCard:HDFC:1234",
        counterparty_account="Expenses:Food",
        missing_account_mapping=False,
        status="exported",
    )
    credit = PaisaExport(
        id=124,
        source="email",
        email_id=42,
        sms_message_id=None,
        idempotency_key="c-1",
        bank="hdfc",
        email_type="hdfc_account_credit_alert",
        direction="credit",
        amount=Decimal("500.00"),
        currency="INR",
        transaction_date=dt.date(2026, 5, 3),
        transaction_time=dt.time(9, 5, 0),
        counterparty="Salary",
        reference_number="SAL-42",
        card_mask=None,
        account_mask="5678",
        source_account="Assets:Bank:HDFC:5678",
        counterparty_account="Income:Salary",
        missing_account_mapping=False,
        status="exported",
    )

    debit_expected = (
        "2026/05/02 Zomato\n"
        "    Expenses:Food                       500.00 INR\n"
        "    Liabilities:CreditCard:HDFC:1234   -500.00 INR\n"
        "    ; financial-dashboard:id=123 source=sms+email sms_id=456 email_id=789 ref=ABC123\n"
    )
    credit_expected = (
        "2026/05/03 Salary\n"
        "    Assets:Bank:HDFC:5678               500.00 INR\n"
        "    Income:Salary                      -500.00 INR\n"
        "    ; financial-dashboard:id=124 source=email sms_id= email_id=42 ref=SAL-42\n"
    )
    assert render_entry(debit) == debit_expected
    assert render_entry(credit) == credit_expected


def test_render_paisa_journal_entry_uses_at_least_two_spaces_before_negative_amount():
    paisa_service = _require_module("financial_dashboard.services.paisa")
    render_entry = _require_attr(paisa_service, "render_paisa_journal_entry")
    PaisaExport = _require_paisa_export_model()

    export = PaisaExport(
        id=301,
        source="email",
        email_id=9,
        sms_message_id=None,
        idempotency_key="long-neg-gap",
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500.00"),
        currency="INR",
        transaction_date=dt.date(2026, 5, 2),
        transaction_time=dt.time(14, 23, 0),
        counterparty="Long Account Test",
        reference_number="REF-NEG-1",
        card_mask="1234",
        account_mask=None,
        source_account="Liabilities:CreditCard:HDFC Diners Black Metal CC",
        counterparty_account="Expenses:Food",
        missing_account_mapping=False,
        status="exported",
    )
    rendered = render_entry(export)
    negative_line = next(
        line
        for line in rendered.splitlines()
        if "Liabilities:CreditCard:HDFC Diners Black Metal CC" in line
    )
    assert re.search(r"  -500\.00 INR$", negative_line)
    account = "Liabilities:CreditCard:HDFC Diners Black Metal CC"
    amount_token = "-500.00 INR"
    gap = negative_line[
        negative_line.index(account) + len(account) : negative_line.index(amount_token)
    ]
    assert len(gap) >= 2


def test_render_paisa_journal_entry_emits_missing_map_comment_only_when_flagged():
    paisa_service = _require_module("financial_dashboard.services.paisa")
    render_entry = _require_attr(paisa_service, "render_paisa_journal_entry")
    PaisaExport = _require_paisa_export_model()

    unmapped = PaisaExport(
        id=201,
        source="sms",
        email_id=None,
        sms_message_id=19,
        idempotency_key="m-1",
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500.00"),
        currency="INR",
        transaction_date=dt.date(2026, 5, 2),
        transaction_time=dt.time(14, 23, 0),
        counterparty="Zomato",
        reference_number=None,
        card_mask="9999",
        account_mask=None,
        source_account="Liabilities:Unknown",
        counterparty_account="Expenses:Food",
        missing_account_mapping=True,
        status="exported",
    )
    mapped = PaisaExport(
        id=202,
        source="sms",
        email_id=None,
        sms_message_id=20,
        idempotency_key="m-2",
        bank="hdfc",
        email_type="hdfc_dc_transaction_alert",
        direction="debit",
        amount=Decimal("500.00"),
        currency="INR",
        transaction_date=dt.date(2026, 5, 2),
        transaction_time=dt.time(14, 23, 0),
        counterparty="Zomato",
        reference_number=None,
        card_mask="1234",
        account_mask=None,
        source_account="Liabilities:CreditCard:HDFC:1234",
        counterparty_account="Expenses:Food",
        missing_account_mapping=False,
        status="exported",
    )

    unmapped_entry = render_entry(unmapped)
    mapped_entry = render_entry(mapped)

    assert "missing-map" in unmapped_entry
    assert "missing-map" not in mapped_entry


def test_resolve_paisa_accounts_uses_unknown_fallbacks_for_unmapped_masks(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    resolve_accounts = _require_attr(paisa_service, "resolve_paisa_accounts")
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")

    config = get_paisa_config()
    card_accounts = resolve_accounts(
        _txn_data(card_mask="9999", account_mask=None, channel="card"),
        config,
    )
    bank_accounts = resolve_accounts(
        _txn_data(card_mask=None, account_mask="8888", channel="account"),
        config,
    )

    assert card_accounts.source_account == "Liabilities:Unknown"
    assert bank_accounts.source_account == "Assets:Unknown"
    assert card_accounts.missing_account_mapping is True
    assert "missing-map" in card_accounts.comment
    assert "account-mask" in bank_accounts.comment


def test_resolve_paisa_accounts_prefers_card_map_then_account_map(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    resolve_accounts = _require_attr(paisa_service, "resolve_paisa_accounts")
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")

    config = get_paisa_config()

    card_first = resolve_accounts(
        _txn_data(card_mask="1234", account_mask="5678", channel="card"),
        config,
    )
    account_second = resolve_accounts(
        _txn_data(card_mask=None, account_mask="5678", channel="account"),
        config,
    )

    assert card_first.source_account == "Liabilities:CreditCard:HDFC:1234"
    assert account_second.source_account == "Assets:Bank:HDFC:5678"


@pytest.mark.anyio
async def test_paisa_mode_sms_parse_creates_export_not_transaction(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    PaisaExport = _require_paisa_export_model()
    _require_module("financial_dashboard.services.paisa")

    monkeypatch.setitem(settings_service._cache, "ledger.backend", "paisa")

    async with _open_session_and_maker() as (session, _):
        sms_row = _make_sms_row()
        session.add(sms_row)
        await session.flush()
        link_ctx = await build_link_context(session)

        async with session.begin_nested():
            outcome = await process_sms_row(session, sms_row, link_ctx)

        export_count = await session.scalar(
            select(func.count()).select_from(PaisaExport)
        )
        txn_count = await session.scalar(select(func.count()).select_from(Transaction))
        assert outcome.status == "parsed"
        assert outcome.transaction_id is None
        assert export_count == 1
        assert txn_count == 0
        assert sms_row.status == "parsed"
        assert sms_row.transaction_id is None


@pytest.mark.anyio
async def test_paisa_mode_email_parse_creates_export_not_transaction(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    emails_service = importlib.import_module("financial_dashboard.services.emails")
    PaisaExport = _require_paisa_export_model()

    async def _fake_parse_email_by_kind(**kwargs):
        return None, _txn_data(reference_number="IMPS:EMAIL-ONLY"), None, None

    # Intended contract: callers should use the public paisa seam module.
    paisa_service = _require_module("financial_dashboard.services.paisa")

    async with _open_session_and_maker() as (session, maker):
        monkeypatch.setattr(emails_service, "async_session", maker)
        monkeypatch.setattr(
            emails_service, "parse_email_by_kind", _fake_parse_email_by_kind
        )
        monkeypatch.setattr(
            settings_service, "get_ledger_backend", lambda: "paisa", raising=False
        )
        monkeypatch.setattr(
            paisa_service,
            "process_paisa_transaction",
            _require_attr(paisa_service, "process_paisa_transaction"),
        )

        rule = SimpleNamespace(id=1, bank="hdfc", email_kind=EmailKind.TRANSACTION)
        stats = {"parsed": 0, "failed": 0, "skipped": 0}
        link_ctx = await build_link_context(session)

        await emails_service.handle_polled_email(
            rule=rule,
            provider="gmail",
            source_id=1,
            msg_id="mid-paisa-email-only",
            remote_id="rid-paisa-email-only",
            raw_bytes=_make_raw_email(),
            should_notify=False,
            link_context=link_ctx,
            stats=stats,
        )

        export_count = await session.scalar(
            select(func.count()).select_from(PaisaExport)
        )
        txn_count = await session.scalar(select(func.count()).select_from(Transaction))
        saved_email = await session.scalar(
            select(Email).where(Email.message_id == "mid-paisa-email-only")
        )
        assert saved_email is not None
        assert saved_email.status == "parsed"
        # Email has no transaction_id column; no Transaction row is the real check.
        assert export_count == 1
        assert txn_count == 0


@pytest.mark.anyio
async def test_paisa_duplicate_same_source_creates_single_export(monkeypatch, tmp_path):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    render_entry = _require_attr(paisa_service, "render_paisa_journal_entry")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        sms_row = _make_sms_row()
        session.add(sms_row)
        await session.flush()

        first = await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(reference_number="IMPS:DUP-1"),
            sms_row=sms_row,
        )
        second = await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(reference_number="IMPS:DUP-1"),
            sms_row=sms_row,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.status == "exported"
        assert row.source == "sms"
        rendered = render_entry(row)
        assert rendered.count("2026/05/02") == 1
        assert first.needs_journal_rewrite is True
        assert second.needs_journal_rewrite is True


@pytest.mark.anyio
async def test_rewrite_paisa_journal_writes_ordered_file_and_is_idempotent(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    rewrite_paisa_journal = _require_attr(paisa_service, "rewrite_paisa_journal")
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    PaisaExport = _require_paisa_export_model()

    db_path = tmp_path / "paisa-rewrite.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(paisa_service, "async_session", maker)

    async with maker() as session:
        session.add_all(
            [
                PaisaExport(
                    source="email",
                    idempotency_key="rewrite-1",
                    bank="hdfc",
                    email_type="hdfc_dc_transaction_alert",
                    direction="debit",
                    amount=Decimal("100.00"),
                    currency="INR",
                    transaction_date=dt.date(2026, 5, 3),
                    transaction_time=dt.time(12, 0, 0),
                    counterparty="Groceries Mart",
                    reference_number="R-3",
                    card_mask="1234",
                    account_mask=None,
                    source_account="Liabilities:CreditCard:HDFC:1234",
                    counterparty_account="Expenses:Food",
                    missing_account_mapping=False,
                    status="exported",
                ),
                PaisaExport(
                    source="sms",
                    idempotency_key="rewrite-2",
                    bank="hdfc",
                    email_type="hdfc_dc_transaction_alert",
                    direction="credit",
                    amount=Decimal("250.00"),
                    currency="INR",
                    transaction_date=dt.date(2026, 5, 1),
                    transaction_time=dt.time(9, 0, 0),
                    counterparty="Employer Payroll",
                    reference_number="R-1",
                    card_mask=None,
                    account_mask="5678",
                    source_account="Assets:Bank:HDFC:5678",
                    counterparty_account="Income:Salary",
                    missing_account_mapping=False,
                    status="exported",
                ),
                PaisaExport(
                    source="email",
                    idempotency_key="rewrite-3",
                    bank="hdfc",
                    email_type="hdfc_dc_transaction_alert",
                    direction="debit",
                    amount=Decimal("50.00"),
                    currency="INR",
                    transaction_date=dt.date(2026, 5, 1),
                    transaction_time=dt.time(9, 0, 0),
                    counterparty="Coffee Shop",
                    reference_number="R-2",
                    card_mask="1234",
                    account_mask=None,
                    source_account="Liabilities:CreditCard:HDFC:1234",
                    counterparty_account="Expenses:Coffee",
                    missing_account_mapping=False,
                    status="exported",
                ),
                PaisaExport(
                    source="sms",
                    idempotency_key="rewrite-undated",
                    bank="hdfc",
                    email_type="hdfc_dc_transaction_alert",
                    direction="debit",
                    amount=Decimal("80.00"),
                    currency="INR",
                    transaction_date=None,
                    transaction_time=dt.time(10, 0, 0),
                    counterparty="Undated Export",
                    reference_number="R-0",
                    card_mask="1234",
                    account_mask=None,
                    source_account="Liabilities:CreditCard:HDFC:1234",
                    counterparty_account="Expenses:Misc",
                    missing_account_mapping=False,
                    status="exported",
                ),
                PaisaExport(
                    source="email",
                    idempotency_key="rewrite-ignored",
                    bank="hdfc",
                    email_type="hdfc_dc_transaction_alert",
                    direction="debit",
                    amount=Decimal("10.00"),
                    currency="INR",
                    transaction_date=dt.date(2026, 5, 4),
                    transaction_time=dt.time(11, 0, 0),
                    counterparty="Ignored Row",
                    reference_number="R-4",
                    card_mask="1234",
                    account_mask=None,
                    source_account="Liabilities:CreditCard:HDFC:1234",
                    counterparty_account="Expenses:Misc",
                    missing_account_mapping=False,
                    status="error",
                ),
            ]
        )
        await session.commit()

    config = get_paisa_config()
    output_path = Path(config.generated_journal_path)
    await rewrite_paisa_journal(config)
    first_text = output_path.read_text(encoding="utf-8")

    assert first_text.count("financial-dashboard:id=") == 3
    assert "Ignored Row" not in first_text
    assert "Undated Export" not in first_text

    payroll_pos = first_text.index("2026/05/01 Employer Payroll")
    coffee_pos = first_text.index("2026/05/01 Coffee Shop")
    groceries_pos = first_text.index("2026/05/03 Groceries Mart")
    assert payroll_pos < coffee_pos < groceries_pos

    assert re.search(r"Assets:Bank:HDFC:5678\s+250\.00 INR", first_text)
    assert re.search(r"Income:Salary\s+-250\.00 INR", first_text)
    assert re.search(r"Expenses:Food\s+100\.00 INR", first_text)
    assert re.search(r"Liabilities:CreditCard:HDFC:1234\s+-100\.00 INR", first_text)

    await rewrite_paisa_journal(config)
    second_text = output_path.read_text(encoding="utf-8")
    assert second_text == first_text
    assert second_text.count("financial-dashboard:id=") == 3
    await engine.dispose()


@pytest.mark.anyio
async def test_rewrite_paisa_journal_cleans_temp_file_when_write_fails(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    rewrite_paisa_journal = _require_attr(paisa_service, "rewrite_paisa_journal")
    get_paisa_config = _require_attr(settings_service, "get_paisa_config")
    PaisaExport = _require_paisa_export_model()

    db_path = tmp_path / "paisa-rewrite-fail.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(paisa_service, "async_session", maker)

    temp_path = tmp_path / "books" / "imports" / "rewrite-fail.tmp"

    class _FailingTempFile:
        name = str(temp_path)

        def __enter__(self):
            temp_path.write_text("", encoding="utf-8")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, _text):
            raise OSError("simulated write failure")

        def flush(self):
            raise AssertionError("flush should not run after write failure")

        def fileno(self):
            raise AssertionError("fsync should not run after write failure")

    def _fake_named_tmp_file(**kwargs):
        assert kwargs["delete"] is False
        assert kwargs["dir"] == temp_path.parent
        return _FailingTempFile()

    monkeypatch.setattr(paisa_service, "NamedTemporaryFile", _fake_named_tmp_file)

    config = get_paisa_config()
    with pytest.raises(OSError, match="simulated write failure"):
        await rewrite_paisa_journal(config)

    assert not temp_path.exists()
    async with maker() as verify:
        count = await verify.scalar(select(func.count()).select_from(PaisaExport))
        assert count == 0
    await engine.dispose()


@pytest.mark.anyio
async def test_paisa_bulk_reparse_routes_to_paisa_export_and_rewrites_once(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_mock = AsyncMock(
        return_value=SimpleNamespace(
            status="parsed",
            export_id=1,
            needs_journal_rewrite=True,
            error=None,
        )
    )
    rewrite_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(paisa_service, "process_paisa_transaction", process_mock)
    monkeypatch.setattr(paisa_service, "rewrite_paisa_journal", rewrite_mock)
    monkeypatch.setattr(
        settings_service, "get_ledger_backend", lambda: "paisa", raising=False
    )

    async with _open_session_and_maker() as (session, maker):
        rule = FetchRule(
            provider="gmail",
            sender="alerts@example.com",
            bank="hdfc",
            enabled=True,
            email_kind=EmailKind.TRANSACTION,
        )
        session.add(rule)
        await session.flush()
        session.add_all(
            [
                Email(
                    provider="gmail",
                    message_id="bulk-paisa-1",
                    source_id=None,
                    remote_id="rid-1",
                    sender="alerts@example.com",
                    subject="txn mail",
                    received_at=dt.datetime(2026, 5, 2, 14, 30, tzinfo=dt.UTC),
                    status="failed",
                    error="old",
                    rule_id=rule.id,
                ),
                Email(
                    provider="gmail",
                    message_id="bulk-paisa-2",
                    source_id=None,
                    remote_id="rid-2",
                    sender="alerts@example.com",
                    subject="statement mail",
                    received_at=dt.datetime(2026, 5, 2, 14, 31, tzinfo=dt.UTC),
                    status="failed",
                    error="old",
                    rule_id=rule.id,
                ),
            ]
        )
        await session.commit()

        raw = _equitas_payment_eml("123.00", "9999")

        async def _fake_parse_email_by_kind(**kwargs):
            assert kwargs["allow_statement_routing"] is False
            if kwargs["subject"] == "statement mail":
                return (
                    "Statement emails are not imported when ledger.backend=paisa",
                    None,
                    None,
                    None,
                )
            return None, _txn_data(reference_number="BULK-PAISA-1"), None, None

        with (
            patch(
                "financial_dashboard.web.emails.load_or_fetch_raw_email",
                new=AsyncMock(return_value=(raw, None)),
            ),
            patch(
                "financial_dashboard.web.emails.parse_email_by_kind",
                new=AsyncMock(side_effect=_fake_parse_email_by_kind),
            ),
        ):
            app = _build_test_app(maker)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/emails/reparse-all-failed")
                assert resp.status_code == 200, resp.text
                assert resp.json() == {"succeeded": 1, "skipped": 1, "failed": 0}

        async with maker() as verify:
            txns = (await verify.execute(select(Transaction))).scalars().all()
            rows = (
                (
                    await verify.execute(
                        select(Email).where(
                            Email.message_id.in_(("bulk-paisa-1", "bulk-paisa-2"))
                        )
                    )
                )
                .scalars()
                .all()
            )
            by_mid = {row.message_id: row for row in rows}

        assert len(txns) == 0
        assert by_mid["bulk-paisa-1"].status == "parsed"
        assert by_mid["bulk-paisa-2"].status == "skipped"
        process_mock.assert_awaited_once()
        rewrite_mock.assert_awaited_once()


@pytest.mark.anyio
async def test_paisa_bulk_reparse_in_paisa_skips_declined_without_export(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    PaisaExport = _require_paisa_export_model()
    process_mock = AsyncMock()
    monkeypatch.setattr(paisa_service, "process_paisa_transaction", process_mock)
    monkeypatch.setattr(
        settings_service, "get_ledger_backend", lambda: "paisa", raising=False
    )

    async with _open_session_and_maker() as (session, maker):
        rule = FetchRule(
            provider="gmail",
            sender="alerts@example.com",
            bank="hdfc",
            enabled=True,
            email_kind=EmailKind.TRANSACTION,
        )
        session.add(rule)
        await session.flush()
        session.add(
            Email(
                provider="gmail",
                message_id="bulk-declined-1",
                source_id=None,
                remote_id="rid-bulk-declined-1",
                sender="alerts@example.com",
                subject="declined mail",
                received_at=dt.datetime(2026, 5, 2, 14, 30, tzinfo=dt.UTC),
                status="failed",
                error="old",
                rule_id=rule.id,
            )
        )
        await session.commit()

        with (
            patch(
                "financial_dashboard.web.emails.load_or_fetch_raw_email",
                new=AsyncMock(return_value=(_make_raw_email(), None)),
            ),
            patch(
                "financial_dashboard.web.emails.parse_email_by_kind",
                new=AsyncMock(
                    return_value=(
                        None,
                        _txn_data(direction="declined"),
                        None,
                        None,
                    )
                ),
            ),
        ):
            app = _build_test_app(maker)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/emails/reparse-all-failed")
                assert resp.status_code == 200, resp.text
                assert resp.json() == {"succeeded": 0, "skipped": 1, "failed": 0}

        async with maker() as verify:
            refreshed = await verify.scalar(
                select(Email).where(Email.message_id == "bulk-declined-1")
            )
            export_count = await verify.scalar(
                select(func.count()).select_from(PaisaExport)
            )

        assert refreshed is not None
        assert refreshed.status == "skipped"
        assert export_count == 0
        process_mock.assert_not_called()


@pytest.mark.anyio
async def test_same_email_reparse_refreshes_export_fields_and_journal(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    render_entry = _require_attr(paisa_service, "render_paisa_journal_entry")
    build_key = _require_attr(paisa_service, "build_paisa_idempotency_key")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        email_row = Email(
            provider="gmail",
            message_id="mid-reparse-1",
            source_id=1,
            remote_id="rid-reparse-1",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 34, tzinfo=dt.UTC),
            status="pending",
        )
        session.add(email_row)
        await session.flush()

        first_txn = _txn_data(amount=Decimal("500.00"), reference_number=None)
        second_txn = _txn_data(
            amount=Decimal("750.00"),
            reference_number=None,
            counterparty="Zomato Corrected",
        )

        first = await process_paisa(
            session,
            source="email",
            txn_data=first_txn,
            email_row=email_row,
        )
        second = await process_paisa(
            session,
            source="email",
            txn_data=second_txn,
            email_row=email_row,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert first.status == "parsed"
        assert second.status == "parsed"
        assert row.amount == Decimal("750.00")
        assert row.counterparty == "Zomato Corrected"
        assert row.idempotency_key == build_key(first_txn)
        assert row.idempotency_key != build_key(second_txn)
        rendered = render_entry(row)
        assert "750.00 INR" in rendered
        assert "500.00 INR" not in rendered


@pytest.mark.anyio
async def test_process_paisa_transaction_records_error_row_on_export_failure(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    original_resolve = _require_attr(paisa_service, "resolve_paisa_accounts")
    PaisaExport = _require_paisa_export_model()

    def _boom_for_fail(txn_data, config):
        if txn_data.get("reference_number") == "FAIL-EXPORT":
            raise RuntimeError("forced export failure")
        return original_resolve(txn_data, config)

    monkeypatch.setattr(paisa_service, "resolve_paisa_accounts", _boom_for_fail)

    async with _open_session_and_maker() as (session, _):
        email_row = Email(
            provider="gmail",
            message_id="mid-export-fail",
            source_id=1,
            remote_id="rid-export-fail",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 34, tzinfo=dt.UTC),
            status="pending",
        )
        session.add(email_row)
        await session.flush()

        outcome = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(reference_number="FAIL-EXPORT"),
            email_row=email_row,
        )
        assert outcome.status == "error"
        assert outcome.export_id is not None
        assert outcome.needs_journal_rewrite is False
        assert "forced export failure" in (outcome.error or "")

        export = await session.get(PaisaExport, outcome.export_id)
        assert export is not None
        assert export.status == "error"
        assert "forced export failure" in (export.error or "")


@pytest.mark.anyio
async def test_process_paisa_transaction_handles_get_paisa_config_failure(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    monkeypatch.setattr(
        paisa_service,
        "get_paisa_config",
        lambda: (_ for _ in ()).throw(ValueError("broken paisa config")),
    )

    async with _open_session_and_maker() as (session, _):
        email_row = Email(
            provider="gmail",
            message_id="mid-config-fail",
            source_id=1,
            remote_id="rid-config-fail",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 34, tzinfo=dt.UTC),
            status="pending",
        )
        session.add(email_row)
        await session.flush()

        outcome = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(reference_number="CFG-FAIL"),
            email_row=email_row,
        )
        assert outcome.status == "error"
        assert outcome.export_id is not None

        export = await session.get(PaisaExport, outcome.export_id)
        assert export is not None
        assert export.status == "error"
        assert "broken paisa config" in (export.error or "")


@pytest.mark.anyio
async def test_error_rows_are_excluded_from_fuzzy_match_but_same_source_retry_refreshes(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    original_resolve = _require_attr(paisa_service, "resolve_paisa_accounts")
    PaisaExport = _require_paisa_export_model()

    failures_left = 1

    def _fail_once(txn_data, config):
        nonlocal failures_left
        if txn_data.get("reference_number") == "ERR-A" and failures_left > 0:
            failures_left -= 1
            raise RuntimeError("forced first-pass failure")
        return original_resolve(txn_data, config)

    monkeypatch.setattr(paisa_service, "resolve_paisa_accounts", _fail_once)

    async with _open_session_and_maker() as (session, _):
        email_a = Email(
            provider="gmail",
            message_id="mid-error-a",
            source_id=1,
            remote_id="rid-error-a",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 34, tzinfo=dt.UTC),
            status="pending",
        )
        email_b = Email(
            provider="gmail",
            message_id="mid-error-b",
            source_id=1,
            remote_id="rid-error-b",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 35, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([email_a, email_b])
        await session.flush()

        first = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                reference_number="ERR-A",
                transaction_time=dt.time(14, 23, 0),
                counterparty="PhonePe Transfer",
            ),
            email_row=email_a,
        )
        second = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                reference_number=None,
                transaction_time=dt.time(14, 28, 0),
                counterparty="John via PhonePe Transfer",
            ),
            email_row=email_b,
        )
        retry = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                reference_number="ERR-A",
                transaction_time=dt.time(14, 23, 0),
                counterparty="PhonePe Transfer Retry",
            ),
            email_row=email_a,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert first.status == "error"
        assert second.status == "parsed"
        assert retry.status == "parsed"
        assert len(rows) == 2

        by_email = {row.email_id: row for row in rows}
        assert by_email[email_b.id].status == "exported"
        assert by_email[email_b.id].reference_number is None
        assert by_email[email_a.id].status == "exported"
        assert by_email[email_a.id].id == first.export_id


@pytest.mark.anyio
async def test_paisa_email_poll_continues_after_one_export_failure(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    emails_service = importlib.import_module("financial_dashboard.services.emails")
    paisa_service = _require_module("financial_dashboard.services.paisa")
    PaisaExport = _require_paisa_export_model()
    original_resolve = _require_attr(paisa_service, "resolve_paisa_accounts")

    async def _fake_parse_email_by_kind(**kwargs):
        if kwargs["log_ref"] == "mid-fail":
            return None, _txn_data(reference_number="FAIL-EXPORT"), None, None
        # Distinct event (different amount + counterparty) so the fuzzy matcher
        # does not merge mid-success into mid-fail's error row.
        return (
            None,
            _txn_data(
                reference_number="OK-EXPORT",
                amount=Decimal("999.00"),
                counterparty="Different Merchant",
            ),
            None,
            None,
        )

    def _maybe_fail_resolve(txn_data, config):
        if txn_data.get("reference_number") == "FAIL-EXPORT":
            raise RuntimeError("forced export failure in poll")
        return original_resolve(txn_data, config)

    async with _open_session_and_maker() as (session, maker):
        monkeypatch.setattr(emails_service, "async_session", maker)
        monkeypatch.setattr(
            emails_service, "parse_email_by_kind", _fake_parse_email_by_kind
        )
        monkeypatch.setattr(
            settings_service, "get_ledger_backend", lambda: "paisa", raising=False
        )
        monkeypatch.setattr(
            paisa_service,
            "resolve_paisa_accounts",
            _maybe_fail_resolve,
        )

        rule = SimpleNamespace(id=1, bank="hdfc", email_kind=EmailKind.TRANSACTION)
        stats = {"parsed": 0, "failed": 0, "skipped": 0}
        link_ctx = await build_link_context(session)

        await emails_service.handle_polled_email(
            rule=rule,
            provider="gmail",
            source_id=1,
            msg_id="mid-fail",
            remote_id="rid-fail",
            raw_bytes=_make_raw_email(),
            should_notify=False,
            link_context=link_ctx,
            stats=stats,
        )
        await emails_service.handle_polled_email(
            rule=rule,
            provider="gmail",
            source_id=1,
            msg_id="mid-success",
            remote_id="rid-success",
            raw_bytes=_make_raw_email(),
            should_notify=False,
            link_context=link_ctx,
            stats=stats,
        )

        async with maker() as verify_session:
            rows = (
                (
                    await verify_session.execute(
                        select(Email).where(
                            Email.message_id.in_(("mid-fail", "mid-success"))
                        )
                    )
                )
                .scalars()
                .all()
            )
            by_mid = {row.message_id: row for row in rows}
            fail_export = await verify_session.scalar(
                select(PaisaExport).where(
                    PaisaExport.email_id == by_mid["mid-fail"].id,
                    PaisaExport.status == "error",
                )
            )
            success_export = await verify_session.scalar(
                select(PaisaExport).where(
                    PaisaExport.email_id == by_mid["mid-success"].id,
                    PaisaExport.status == "exported",
                )
            )

        assert by_mid["mid-fail"].status == "failed"
        assert by_mid["mid-success"].status == "parsed"
        assert fail_export is not None
        assert success_export is not None
        assert stats == {"parsed": 1, "failed": 1, "skipped": 0}


@pytest.mark.anyio
async def test_paisa_email_poll_handles_enrichment_flush_failure_and_continues(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    emails_service = importlib.import_module("financial_dashboard.services.emails")
    paisa_service = _require_module("financial_dashboard.services.paisa")
    PaisaExport = _require_paisa_export_model()
    original_apply_accounts = _require_attr(
        paisa_service, "_resolve_and_apply_accounts"
    )

    async def _fake_parse_email_by_kind(**kwargs):
        if kwargs["log_ref"] == "mid-base":
            return (
                None,
                _txn_data(
                    reference_number=None,
                    transaction_time=dt.time(14, 23, 0),
                    counterparty="Base Merchant",
                ),
                None,
                None,
            )
        if kwargs["log_ref"] == "mid-fail":
            return (
                None,
                _txn_data(
                    reference_number=None,
                    transaction_time=dt.time(14, 25, 0),
                    counterparty="Base Merchant Updated",
                ),
                None,
                None,
            )
        return (
            None,
            _txn_data(
                reference_number="OK-AFTER-FLUSH-FAIL",
                amount=Decimal("901.00"),
                counterparty="Different Merchant",
            ),
            None,
            None,
        )

    def _poison_on_enrichment(row, config):
        if row.reference_number is None and (row.counterparty or "").startswith(
            "Base Merchant"
        ):
            row.idempotency_key = "collision-key"
        original_apply_accounts(row, config)

    async with _open_session_and_maker() as (session, maker):
        monkeypatch.setattr(emails_service, "async_session", maker)
        monkeypatch.setattr(
            emails_service, "parse_email_by_kind", _fake_parse_email_by_kind
        )
        monkeypatch.setattr(
            settings_service, "get_ledger_backend", lambda: "paisa", raising=False
        )
        monkeypatch.setattr(
            paisa_service,
            "_resolve_and_apply_accounts",
            _poison_on_enrichment,
        )

        async with maker() as seed:
            seed.add(
                PaisaExport(
                    source="email",
                    idempotency_key="collision-key",
                    bank="hdfc",
                    email_type="hdfc_dc_transaction_alert",
                    direction="debit",
                    amount=Decimal("1.00"),
                    currency="INR",
                    transaction_date=dt.date(2026, 5, 1),
                    transaction_time=dt.time(10, 0, 0),
                    counterparty="Collision Holder",
                    reference_number="C-1",
                    card_mask="0001",
                    account_mask=None,
                    source_account="Liabilities:CreditCard:HDFC:0001",
                    counterparty_account="Expenses:Misc",
                    missing_account_mapping=False,
                    status="exported",
                )
            )
            await seed.commit()

        rule = SimpleNamespace(id=1, bank="hdfc", email_kind=EmailKind.TRANSACTION)
        stats = {"parsed": 0, "failed": 0, "skipped": 0}
        link_ctx = await build_link_context(session)

        await emails_service.handle_polled_email(
            rule=rule,
            provider="gmail",
            source_id=1,
            msg_id="mid-base",
            remote_id="rid-base",
            raw_bytes=_make_raw_email(),
            should_notify=False,
            link_context=link_ctx,
            stats=stats,
        )
        await emails_service.handle_polled_email(
            rule=rule,
            provider="gmail",
            source_id=1,
            msg_id="mid-fail",
            remote_id="rid-fail-enrich",
            raw_bytes=_make_raw_email(),
            should_notify=False,
            link_context=link_ctx,
            stats=stats,
        )
        await emails_service.handle_polled_email(
            rule=rule,
            provider="gmail",
            source_id=1,
            msg_id="mid-success",
            remote_id="rid-success-enrich",
            raw_bytes=_make_raw_email(),
            should_notify=False,
            link_context=link_ctx,
            stats=stats,
        )

        async with maker() as verify_session:
            rows = (
                (
                    await verify_session.execute(
                        select(Email).where(
                            Email.message_id.in_(
                                ("mid-base", "mid-fail", "mid-success")
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )
            by_mid = {row.message_id: row for row in rows}
            fail_export = await verify_session.scalar(
                select(PaisaExport).where(
                    PaisaExport.email_id == by_mid["mid-fail"].id,
                    PaisaExport.status == "error",
                )
            )
            success_export = await verify_session.scalar(
                select(PaisaExport).where(
                    PaisaExport.email_id == by_mid["mid-success"].id,
                    PaisaExport.status == "exported",
                )
            )

        assert by_mid["mid-base"].status == "parsed"
        assert by_mid["mid-fail"].status == "failed"
        assert by_mid["mid-success"].status == "parsed"
        assert fail_export is not None
        assert success_export is not None
        assert stats == {"parsed": 2, "failed": 1, "skipped": 0}


@pytest.mark.anyio
async def test_paisa_declined_email_is_skipped_by_direction_gate(monkeypatch, tmp_path):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    emails_service = importlib.import_module("financial_dashboard.services.emails")
    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_mock = AsyncMock()

    async def _fake_parse_email_by_kind(**kwargs):
        return None, _txn_data(direction="declined"), None, None

    async with _open_session_and_maker() as (session, maker):
        monkeypatch.setattr(emails_service, "async_session", maker)
        monkeypatch.setattr(
            emails_service, "parse_email_by_kind", _fake_parse_email_by_kind
        )
        monkeypatch.setattr(
            settings_service, "get_ledger_backend", lambda: "paisa", raising=False
        )
        monkeypatch.setattr(paisa_service, "process_paisa_transaction", process_mock)

        rule = SimpleNamespace(id=1, bank="hdfc", email_kind=EmailKind.TRANSACTION)
        stats = {"parsed": 0, "failed": 0, "skipped": 0}
        link_ctx = await build_link_context(session)

        await emails_service.handle_polled_email(
            rule=rule,
            provider="gmail",
            source_id=1,
            msg_id="mid-declined-generic",
            remote_id="rid-declined-generic",
            raw_bytes=_make_raw_email(),
            should_notify=False,
            link_context=link_ctx,
            stats=stats,
        )

        saved_email = await session.scalar(
            select(Email).where(Email.message_id == "mid-declined-generic")
        )

        assert saved_email is not None
        assert saved_email.status == "skipped"
        assert stats == {"parsed": 0, "failed": 0, "skipped": 1}
        process_mock.assert_not_called()


@pytest.mark.anyio
async def test_paisa_statement_email_in_poll_is_skipped_not_failed(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    emails_service = importlib.import_module("financial_dashboard.services.emails")

    async def _fake_parse_email_by_kind(**kwargs):
        return (
            "Statement emails are not imported when ledger.backend=paisa",
            None,
            None,
            None,
        )

    async with _open_session_and_maker() as (session, maker):
        monkeypatch.setattr(emails_service, "async_session", maker)
        monkeypatch.setattr(
            emails_service, "parse_email_by_kind", _fake_parse_email_by_kind
        )
        monkeypatch.setattr(
            settings_service, "get_ledger_backend", lambda: "paisa", raising=False
        )

        rule = SimpleNamespace(id=1, bank="hdfc", email_kind=EmailKind.TRANSACTION)
        stats = {"parsed": 0, "failed": 0, "skipped": 0}
        link_ctx = await build_link_context(session)

        await emails_service.handle_polled_email(
            rule=rule,
            provider="gmail",
            source_id=1,
            msg_id="mid-statement-skip",
            remote_id="rid-statement-skip",
            raw_bytes=_make_raw_email(),
            should_notify=False,
            link_context=link_ctx,
            stats=stats,
        )

        saved_email = await session.scalar(
            select(Email).where(Email.message_id == "mid-statement-skip")
        )
        assert saved_email is not None
        assert saved_email.status == "skipped"
        assert stats == {"parsed": 0, "failed": 0, "skipped": 1}


@pytest.mark.anyio
async def test_paisa_cross_channel_enrichment_keeps_currency_and_email_type(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        email_first = Email(
            provider="gmail",
            message_id="mid-enrich-email-1",
            source_id=1,
            remote_id="rid-enrich-email-1",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 34, tzinfo=dt.UTC),
            status="pending",
        )
        email_second = Email(
            provider="gmail",
            message_id="mid-enrich-email-2",
            source_id=1,
            remote_id="rid-enrich-email-2",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 35, tzinfo=dt.UTC),
            status="pending",
        )
        email_third = Email(
            provider="gmail",
            message_id="mid-enrich-email-3",
            source_id=1,
            remote_id="rid-enrich-email-3",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 36, tzinfo=dt.UTC),
            status="pending",
        )
        sms_row = _make_sms_row()
        session.add_all([email_first, email_second, email_third, sms_row])
        await session.flush()

        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                email_type="email_primary_type",
                currency="INR",
                transaction_time=dt.time(14, 23, 0),
                reference_number=None,
            ),
            email_row=email_first,
        )
        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                email_type="email_secondary_type",
                currency="INR",
                transaction_time=dt.time(14, 25, 0),
                reference_number=None,
            ),
            email_row=email_second,
        )
        await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                email_type="email_earlier_type",
                currency="INR",
                transaction_time=dt.time(14, 21, 0),
                reference_number=None,
            ),
            email_row=email_third,
        )
        await process_paisa(
            session,
            source="sms",
            txn_data=_txn_data(
                email_type="sms_earlier_type",
                currency="INR",
                transaction_time=dt.time(14, 19, 0),
                reference_number=None,
            ),
            sms_row=sms_row,
        )

        rows = (await session.execute(select(PaisaExport))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.source == "sms+email"
        assert row.email_type == "email_primary_type"
        assert row.transaction_time == dt.time(14, 21, 0)


@pytest.mark.anyio
async def test_paisa_reparse_with_case_variant_reference_does_not_error(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    process_paisa = _require_attr(paisa_service, "process_paisa_transaction")
    PaisaExport = _require_paisa_export_model()

    async with _open_session_and_maker() as (session, _):
        email_one = Email(
            provider="gmail",
            message_id="mid-refcase-1",
            source_id=1,
            remote_id="rid-refcase-1",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 34, tzinfo=dt.UTC),
            status="pending",
        )
        email_two = Email(
            provider="gmail",
            message_id="mid-refcase-2",
            source_id=1,
            remote_id="rid-refcase-2",
            sender="alerts@example.com",
            subject="alert",
            received_at=dt.datetime(2026, 5, 2, 14, 35, tzinfo=dt.UTC),
            status="pending",
        )
        session.add_all([email_one, email_two])
        await session.flush()

        first = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                reference_number="AbC-Ref-1",
                transaction_date=None,
                transaction_time=None,
            ),
            email_row=email_one,
        )
        second = await process_paisa(
            session,
            source="email",
            txn_data=_txn_data(
                reference_number="abc-ref-1",
                transaction_date=None,
                transaction_time=None,
            ),
            email_row=email_two,
        )

        count = await session.scalar(select(func.count()).select_from(PaisaExport))
        assert first.status == "parsed"
        assert second.status == "parsed"
        assert count == 2


@pytest.mark.anyio
async def test_single_email_reparse_in_paisa_skips_declined_without_export(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    paisa_service = _require_module("financial_dashboard.services.paisa")
    PaisaExport = _require_paisa_export_model()
    process_mock = AsyncMock()
    monkeypatch.setattr(paisa_service, "process_paisa_transaction", process_mock)
    monkeypatch.setattr(
        settings_service, "get_ledger_backend", lambda: "paisa", raising=False
    )

    async with _open_session_and_maker() as (session, maker):
        rule = FetchRule(
            provider="gmail",
            sender="alerts@example.com",
            bank="hdfc",
            enabled=True,
            email_kind=EmailKind.TRANSACTION,
        )
        session.add(rule)
        await session.flush()
        email_row = Email(
            provider="gmail",
            message_id="mid-reparse-declined",
            source_id=1,
            remote_id="rid-reparse-declined",
            sender="alerts@example.com",
            subject="declined",
            received_at=dt.datetime(2026, 5, 2, 14, 31, tzinfo=dt.UTC),
            status="failed",
            error="old",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()

        with (
            patch(
                "financial_dashboard.web.emails.load_or_fetch_raw_email",
                new=AsyncMock(return_value=(_make_raw_email(), None)),
            ),
            patch(
                "financial_dashboard.web.emails.parse_email_by_kind",
                new=AsyncMock(
                    return_value=(
                        None,
                        _txn_data(direction="declined"),
                        None,
                        None,
                    )
                ),
            ),
        ):
            app = _build_test_app(maker)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/emails/{email_row.id}/reparse")
                assert resp.status_code == 200, resp.text
                payload = resp.json()
                assert payload["new_status"] == "skipped"

        async with maker() as verify:
            refreshed = await verify.get(Email, email_row.id)
            assert refreshed is not None
            assert refreshed.status == "skipped"
            export_count = await verify.scalar(
                select(func.count()).select_from(PaisaExport)
            )

        assert export_count == 0
        process_mock.assert_not_called()


@pytest.mark.anyio
async def test_single_email_reparse_in_paisa_treats_statement_as_skipped(
    monkeypatch, tmp_path
):
    _set_valid_paisa_cache(monkeypatch, tmp_path)

    monkeypatch.setattr(
        settings_service, "get_ledger_backend", lambda: "paisa", raising=False
    )

    async with _open_session_and_maker() as (session, maker):
        rule = FetchRule(
            provider="gmail",
            sender="alerts@example.com",
            bank="hdfc",
            enabled=True,
            email_kind=EmailKind.TRANSACTION,
        )
        session.add(rule)
        await session.flush()
        email_row = Email(
            provider="gmail",
            message_id="mid-reparse-statement",
            source_id=1,
            remote_id="rid-reparse-statement",
            sender="alerts@example.com",
            subject="statement",
            received_at=dt.datetime(2026, 5, 2, 14, 31, tzinfo=dt.UTC),
            status="failed",
            error="old",
            rule_id=rule.id,
        )
        session.add(email_row)
        await session.commit()

        with (
            patch(
                "financial_dashboard.web.emails.load_or_fetch_raw_email",
                new=AsyncMock(return_value=(_make_raw_email(), None)),
            ),
            patch(
                "financial_dashboard.web.emails.parse_email_by_kind",
                new=AsyncMock(
                    return_value=(
                        "Statement emails are not imported when ledger.backend=paisa",
                        None,
                        None,
                        None,
                    )
                ),
            ),
        ):
            app = _build_test_app(maker)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/emails/{email_row.id}/reparse")
                assert resp.status_code == 200, resp.text
                assert resp.json()["new_status"] == "skipped"

        async with maker() as verify:
            refreshed = await verify.get(Email, email_row.id)
            assert refreshed is not None
            assert refreshed.status == "skipped"
