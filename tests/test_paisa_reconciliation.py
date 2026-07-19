"""Paisa reconciliation + audit + FX/backend config tests.

Covers:
* Reconciliation is read-only: it never writes a core row (accounts/transactions/
  snapshots unchanged), never writes corrections, and joins native ↔ Paisa only
  by explicit account_mappings (no fuzzy match).
* Reconciliation behaviour across modes (disabled/connect/project), with and
  without upstream availability, and that projected balances come from the
  projection's openings + postings.
* Mapping suggestions are preview-only and deterministic.
* FX rate editor validation: positive decimals only, valid dates, currency
  shape, deterministic serialization to the nested JSON the config reads.
* Ledger backend selector validation (ledger/hledger/beancount).
* Manual generate/sync/probe audit wrapping: a start/complete row is recorded
  with sanitized details and no raw journal/credentials, committed via the
  request session.
"""

import datetime as dt
import types as _types
from decimal import Decimal

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.config as config_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    Base,
    Transaction,
)
from financial_dashboard.integrations.paisa import (
    PaisaAssetsBalanceReport,
    PaisaAssetBreakdown,
    PaisaLiabilitiesReport,
    PaisaLiabilityBreakdown,
)
from financial_dashboard.schemas.extensions import (
    PaisaConfigInput,
    PaisaFxRateRow,
    PaisaGenerateResponse,
    PaisaPublishInfo,
    PaisaSyncResponse,
)
from financial_dashboard.services.paisa import surface
from financial_dashboard.services.paisa.audit import (
    OPERATION_GENERATE,
    OPERATION_PROBE,
    OPERATION_SYNC,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    recent_runs,
)
from financial_dashboard.services.paisa.config import FxRate, PaisaProjectionConfig
from financial_dashboard.services.paisa import reconciliation as recon_mod
from financial_dashboard.services.paisa.reconciliation import (
    _balance_by_posting_account,
    build_reconciliation,
)

pytestmark = pytest.mark.anyio

CUTOVER = dt.date(2026, 1, 1)


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


@pytest.fixture
async def settings_db(monkeypatch):
    """Isolated in-memory settings DB so save_config round-trips for real."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(settings_mod, "async_session", maker)
    monkeypatch.setattr(
        config_mod.settings, "email_source_master_key", Fernet.generate_key().decode()
    )
    monkeypatch.setattr(config_mod, "_fernet_instance", None)
    yield maker
    await engine.dispose()


def _set_config(monkeypatch, cfg):
    """Patch BOTH surface and reconciliation load_config (they import it directly)."""
    monkeypatch.setattr(surface, "load_config", lambda: cfg)
    monkeypatch.setattr(recon_mod, "load_config", lambda: cfg)


def _cfg(mode="project", **overrides) -> PaisaProjectionConfig:
    base = dict(
        mode=mode,
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=(1,),
        cutover_date=CUTOVER,
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
    )
    base.update(overrides)
    return PaisaProjectionConfig(**base)


async def _seed_bank(session, *, id=1):
    session.add(Account(id=id, bank="hdfc", label="Savings", type="bank_account"))
    await session.flush()


async def _seed_txn(session, account_id, *, date=dt.date(2026, 2, 1), amount="10.00"):
    session.add(
        Transaction(
            account_id=account_id,
            bank="hdfc",
            email_type="test_account_transaction",
            direction="debit",
            amount=Decimal(amount),
            transaction_date=date,
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()


def _assets(*groups) -> PaisaAssetsBalanceReport:
    return PaisaAssetsBalanceReport(
        breakdowns=tuple(
            PaisaAssetBreakdown(group=g[0], market_amount=g[1]) for g in groups
        )
    )


def _liabs(*groups) -> PaisaLiabilitiesReport:
    return PaisaLiabilitiesReport(
        breakdowns=tuple(
            PaisaLiabilityBreakdown(
                group=g[0],
                drawn_amount="0",
                repaid_amount="0",
                interest_amount="0",
                balance_amount=g[1],
                apr=None,
            )
            for g in groups
        )
    )


# --------------------------------------------------------------------------- #
# Reconciliation: read-only, no fuzzy matching
# --------------------------------------------------------------------------- #


async def test_reconciliation_uses_injected_config_snapshot(session, monkeypatch):
    config = _cfg(mode="disabled", selected_account_ids=())

    def unexpected_global_load():
        raise AssertionError("reconciliation reloaded global config")

    monkeypatch.setattr(recon_mod, "load_config", unexpected_global_load)

    response = await build_reconciliation(
        session,
        config=config,
        asset_report=None,
        liability_report=None,
    )

    assert response.ok is True
    assert response.mode == "disabled"


async def test_reconciliation_writes_no_core_rows(session, monkeypatch):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(monkeypatch, _cfg(selected_account_ids=(1,)))

    txns_before = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    accts_before = [
        a.id for a in (await session.execute(select(Account))).scalars().all()
    ]

    resp = await build_reconciliation(
        session,
        asset_report=_assets(("Assets:Bank:HDFC:Savings", "100.00")),
        liability_report=_liabs(),
        upstream_available=True,
    )
    assert resp.ok is True

    txns_after = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    accts_after = [
        a.id for a in (await session.execute(select(Account))).scalars().all()
    ]
    assert txns_before == txns_after
    assert accts_before == accts_after


async def test_reconciliation_projects_ending_balance(session, monkeypatch):
    await _seed_bank(session)
    # An opening via a pre-cutover snapshot + a post-cutover spend.
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    await _seed_txn(session, 1, amount="10.00")
    _set_config(monkeypatch, _cfg(selected_account_ids=(1,)))

    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    assert resp.projection is not None
    assert resp.projection.emitted_count == 1
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.projected_available is True
    assert row.projected_balance is not None


async def test_reconciliation_joins_paisa_only_by_explicit_mapping(
    session, monkeypatch
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    # Account 1 has NO explicit mapping → Paisa cell must be unavailable.
    _set_config(monkeypatch, _cfg(selected_account_ids=(1,), account_mappings={}))
    resp = await build_reconciliation(
        session,
        asset_report=_assets(("Assets:Bank:HDFC:Savings", "100.00")),
        liability_report=_liabs(),
        upstream_available=True,
    )
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.paisa_available is False
    assert row.paisa_balance is None
    assert row.delta is None  # no delta without an explicit mapping join

    # WITH an explicit mapping → the join resolves and a delta is computed.
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:HDFC:Savings"},
        ),
    )
    resp2 = await build_reconciliation(
        session,
        asset_report=_assets(("Assets:Bank:HDFC:Savings", "100.00")),
        liability_report=_liabs(),
        upstream_available=True,
    )
    row2 = next(r for r in resp2.accounts if r.account_id == 1)
    assert row2.paisa_available is True
    assert row2.paisa_balance == "100.00"
    assert row2.mapped_to == "Assets:Bank:HDFC:Savings"


async def test_reconciliation_never_fuzzy_matches(session, monkeypatch):
    """A Paisa group that merely resembles (but is not equal to / child of) the
    mapped name must NOT be joined — no fuzzy/text-similarity matching."""
    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:HDFC:Savings"},
        ),
    )
    resp = await build_reconciliation(
        session,
        # Near-miss names that a fuzzy matcher might accept:
        asset_report=_assets(
            ("Assets:Bank:HDFC", "50.00"),
            ("Assets:Bank:HDFC:Savingsx", "60.00"),
            ("Assets:Bank:ICICI:Savings", "70.00"),
        ),
        liability_report=_liabs(),
        upstream_available=True,
    )
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.paisa_available is False  # exact + direct-child only; no fuzzy


async def test_reconciliation_mapping_suggestions_are_preview_only(
    session, monkeypatch
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(monkeypatch, _cfg(selected_account_ids=(1,), account_mappings={}))
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    assert any(s.account_id == 1 for s in resp.suggestions)
    sug = next(s for s in resp.suggestions if s.account_id == 1)
    assert sug.suggested_mapping == "Assets:Bank:Hdfc:Savings"
    # And it must NOT have been written as a mapping.
    assert recon_mod.load_config().account_mappings == {}
    # The build never persists anything.
    assert len((await session.execute(select(Account))).scalars().all()) == 1


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("ledger", "Assets:Bank:Hdfc:Savings Account"),
        ("hledger", "Assets:Bank:Hdfc:Savings Account"),
        ("beancount", "Assets:Bank:Hdfc:SavingsAccount"),
    ],
)
async def test_reconciliation_resolves_bank_default_without_opening(
    session, monkeypatch, backend, expected
):
    session.add(
        Account(
            id=1,
            bank="hdfc",
            label="savings_account",
            type="bank_account",
        )
    )
    await session.flush()
    await _seed_txn(session, 1)
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(1,), ledger_cli=backend, account_mappings={}),
    )

    resp = await build_reconciliation(session, asset_report=None, liability_report=None)

    row = resp.accounts[0]
    assert row.mapped_to == expected
    assert row.opening_available is False
    assert row.projected_available is True
    assert row.projected_balance == "-10.00"
    assert [suggestion.suggested_mapping for suggestion in resp.suggestions] == [
        expected
    ]


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("ledger", "Liabilities:Card:Hdfc:Rewards Card"),
        ("hledger", "Liabilities:Card:Hdfc:Rewards Card"),
        ("beancount", "Liabilities:Card:Hdfc:RewardsCard"),
    ],
)
async def test_reconciliation_resolves_liability_default_without_opening(
    session, monkeypatch, backend, expected
):
    session.add(
        Account(
            id=2,
            bank="hdfc",
            label="rewards_card",
            type="credit_card",
        )
    )
    await session.flush()
    session.add(
        Transaction(
            account_id=2,
            bank="hdfc",
            email_type="cc_debit_purchase",
            direction="debit",
            amount=Decimal("25.00"),
            transaction_date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(2,), ledger_cli=backend, account_mappings={}),
    )

    resp = await build_reconciliation(session, asset_report=None, liability_report=None)

    row = resp.accounts[0]
    assert row.mapped_to == expected
    assert row.opening_available is False
    assert row.projected_available is True
    assert row.projected_balance == "25.00"
    assert [suggestion.suggested_mapping for suggestion in resp.suggestions] == [
        expected
    ]


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("ledger", "Assets:Bank:Hdfc Bank:Salary Plus Cash Account"),
        ("hledger", "Assets:Bank:Hdfc Bank:Salary Plus Cash Account"),
        ("beancount", "Assets:Bank:HdfcBank:SalaryPlusCashAccount"),
    ],
)
async def test_reconciliation_suggestion_sanitizes_invalid_label_characters(
    session, monkeypatch, backend, expected
):
    session.add(
        Account(
            id=1,
            bank="hdfc;:{bank}",
            label="salary_plus;\n{cash}:account",
            type="bank_account",
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(1,), ledger_cli=backend, account_mappings={}),
    )

    resp = await build_reconciliation(session, asset_report=None, liability_report=None)

    assert resp.accounts[0].mapped_to == expected
    assert resp.suggestions[0].suggested_mapping == expected
    assert all(char not in expected for char in ";\n{}")


async def test_reconciliation_explicit_mapping_wins_and_only_unmapped_is_suggested(
    session, monkeypatch
):
    await _seed_bank(session, id=1)
    await _seed_card(session, id=2)
    explicit = "Assets:Operator:Selected"
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(2, 1),
            account_mappings={"1": explicit},
            ledger_cli="beancount",
        ),
    )

    resp = await build_reconciliation(session, asset_report=None, liability_report=None)

    assert [(row.account_id, row.mapped_to) for row in resp.accounts] == [
        (1, explicit),
        (2, "Liabilities:Card:Hdfc:Card"),
    ]
    assert [suggestion.account_id for suggestion in resp.suggestions] == [2]
    assert resp.suggestions[0].suggested_mapping == "Liabilities:Card:Hdfc:Card"


async def test_reconciliation_explicit_parent_mapping_rolls_up_direct_children_only(
    session, monkeypatch
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    parent = "Assets:Bank:Hdfc"
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(1,), account_mappings={"1": parent}),
    )

    resp = await build_reconciliation(
        session,
        asset_report=_assets(
            (f"{parent}:Checking", "30.00"),
            (f"{parent}:Savings", "70.00"),
            (f"{parent}:Savings:Reserve", "900.00"),
            ("Assets:Bank:Hdfcx:Savings", "800.00"),
        ),
        liability_report=_liabs(),
        upstream_available=True,
    )

    row = resp.accounts[0]
    assert row.mapped_to == parent
    assert row.paisa_available is True
    assert row.paisa_balance == "100.00"
    assert row.delta == "-110.00"
    assert resp.suggestions == []


async def test_reconciliation_account_resolution_output_is_deterministic(
    session, monkeypatch
):
    await _seed_bank(session, id=1)
    await _seed_card(session, id=2)
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(2, 1),
            account_mappings={},
            ledger_cli="beancount",
        ),
    )

    first = await build_reconciliation(
        session, asset_report=None, liability_report=None
    )
    second = await build_reconciliation(
        session, asset_report=None, liability_report=None
    )

    assert first.model_dump() == second.model_dump()
    assert [suggestion.account_id for suggestion in first.suggestions] == [1, 2]
    assert [suggestion.suggested_mapping for suggestion in first.suggestions] == [
        "Assets:Bank:Hdfc:Savings",
        "Liabilities:Card:Hdfc:Card",
    ]


async def test_reconciliation_disabled_mode_labels_reason(session, monkeypatch):
    _set_config(monkeypatch, _cfg(mode="disabled"))
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    assert resp.can_connect is False
    assert resp.reason == "disabled"
    assert resp.projection is None


async def test_reconciliation_native_snapshot_cell(session, monkeypatch):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=dt.date(2026, 3, 1),
            value=Decimal("500.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    _set_config(monkeypatch, _cfg(selected_account_ids=(1,)))
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.native_balance == "500.00"
    assert row.native_as_of == "2026-03-01"


# --------------------------------------------------------------------------- #
# Multi-commodity projected balance: foreign legs excluded, clearly noted
# --------------------------------------------------------------------------- #


def _fake_report(*entries):
    """Minimal stand-in for a ProjectionReport: each entry is a tuple of
    ``(account, amount, commodity)`` postings."""
    return _types.SimpleNamespace(
        entries=[
            _types.SimpleNamespace(
                postings=[
                    _types.SimpleNamespace(
                        account=a, amount=Decimal(amt), commodity=ccy
                    )
                    for (a, amt, ccy) in entry
                ]
            )
            for entry in entries
        ]
    )


def test_balance_by_posting_account_foreign_only_contributes_zero_inr():
    totals = _balance_by_posting_account(
        _fake_report([("Assets:Bank:Fx", "-7", "USD")])
    )

    fx = totals["Assets:Bank:Fx"]
    assert fx.amount == Decimal("0")
    assert fx.has_foreign_commodity is True
    assert fx.foreign_posting_counts == (("USD", 1),)


def test_balance_by_posting_account_multiple_foreign_is_order_independent():
    account = "Assets:Bank:Fx"
    forward = _balance_by_posting_account(
        _fake_report(
            [(account, "-7", "USD")],
            [(account, "-5", "EUR")],
            [(account, "-3", "USD")],
        )
    )[account]
    reverse = _balance_by_posting_account(
        _fake_report(
            [(account, "-3", "USD")],
            [(account, "-5", "EUR")],
            [(account, "-7", "USD")],
        )
    )[account]

    assert forward == reverse
    assert forward.amount == Decimal("0")
    assert forward.foreign_posting_counts == (("EUR", 1), ("USD", 2))


def test_balance_by_posting_account_mixed_uses_inr_and_blank_as_inr():
    report = _fake_report(
        [("Assets:Bank:Inr", "-10", "INR")],
        [
            ("Assets:Bank:Mix", "-10", "INR"),
            ("Assets:Bank:Mix", "2", None),
            ("Assets:Bank:Mix", "3", ""),
            ("Assets:Bank:Mix", "-5", "USD"),
        ],
    )
    totals = _balance_by_posting_account(report)
    inr = totals["Assets:Bank:Inr"]
    assert inr.amount == Decimal("-10")
    assert inr.has_foreign_commodity is False
    assert inr.foreign_posting_counts == ()
    # Mixed account: INR + legacy blank commodities only; USD is excluded.
    mix = totals["Assets:Bank:Mix"]
    assert mix.amount == Decimal("-5")
    assert mix.has_foreign_commodity is True
    assert mix.foreign_posting_counts == (("USD", 1),)


async def test_reconciliation_notes_foreign_commodity_excluded(session, monkeypatch):
    """A mapped projected account with both INR and foreign-commodity legs shows
    an INR-only projected balance plus a clear note that the foreign legs are
    excluded — never silently implied complete, never FX-converted."""
    await _seed_bank(session)
    # INR opening + an INR spend + a USD spend (priced policy, rate configured).
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    await _seed_txn(session, 1, amount="10.00")  # INR debit → -10 INR
    session.add(
        Transaction(
            account_id=1,
            bank="hdfc",
            email_type="test_account_transaction",
            direction="debit",
            amount=Decimal("10.00"),
            currency="USD",
            transaction_date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:Hdfc:Savings"},
            non_inr_policy="priced",
            fx_rates={"USD": (FxRate(CUTOVER, Decimal("83.0000")),)},
        ),
    )
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.projected_available is True
    # INR leg only (opening 1000 − INR spend 10); the USD leg is excluded.
    assert row.projected_balance == "990.00"
    assert row.note == (
        "projected balance shown in INR only; foreign-commodity postings "
        "excluded: USD=1; no FX conversion performed"
    )


async def test_reconciliation_usd_only_leaves_inr_opening_and_delta_unchanged(
    session, monkeypatch
):
    await _seed_bank(session)
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    session.add(
        Transaction(
            account_id=1,
            bank="hdfc",
            email_type="test_account_transaction",
            direction="debit",
            amount=Decimal("10.00"),
            currency="USD",
            transaction_date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()
    mapped_to = "Assets:Bank:Hdfc:Savings"
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": mapped_to},
            non_inr_policy="priced",
            fx_rates={"USD": (FxRate(CUTOVER, Decimal("83.0000")),)},
        ),
    )

    resp = await build_reconciliation(
        session,
        asset_report=_assets((mapped_to, "1000.00")),
        liability_report=_liabs(),
        upstream_available=True,
    )
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.projected_balance == "1000.00"
    assert row.delta == "0.00"


@pytest.mark.parametrize(
    "currencies",
    [("USD", "EUR", "USD"), ("USD", "USD", "EUR")],
)
async def test_reconciliation_foreign_note_is_sorted_and_order_independent(
    session, monkeypatch, currencies
):
    await _seed_bank(session)
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    for day, currency in enumerate(currencies, start=1):
        session.add(
            Transaction(
                account_id=1,
                bank="hdfc",
                email_type="test_account_transaction",
                direction="debit",
                amount=Decimal("10.00"),
                currency=currency,
                transaction_date=dt.date(2026, 2, day),
                category="groceries",
                counterparty="Store",
            )
        )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:Hdfc:Savings"},
            non_inr_policy="priced",
            fx_rates={
                "EUR": (FxRate(CUTOVER, Decimal("90.0000")),),
                "USD": (FxRate(CUTOVER, Decimal("83.0000")),),
            },
        ),
    )

    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.projected_balance == "1000.00"
    assert row.note == (
        "projected balance shown in INR only; foreign-commodity postings "
        "excluded: EUR=1, USD=2; no FX conversion performed"
    )


async def test_reconciliation_foreign_only_liability_preserves_sign(
    session, monkeypatch
):
    await _seed_card(session)
    session.add(
        BalanceSnapshot(
            account_id=2,
            kind="liability",
            category="credit_card_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    session.add(
        Transaction(
            account_id=2,
            bank="hdfc",
            email_type="cc_debit_purchase",
            direction="debit",
            amount=Decimal("10.00"),
            currency="USD",
            transaction_date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()
    mapped_to = "Liabilities:Card:Hdfc:Card"
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(2,),
            account_mappings={"2": mapped_to},
            non_inr_policy="priced",
            fx_rates={"USD": (FxRate(CUTOVER, Decimal("83.0000")),)},
        ),
    )

    resp = await build_reconciliation(
        session,
        asset_report=_assets(),
        liability_report=_liabs((mapped_to, "1000.00")),
        upstream_available=True,
    )
    row = next(r for r in resp.accounts if r.account_id == 2)
    assert row.projected_balance == "1000.00"
    assert row.delta == "0.00"
    assert "excluded: USD=1; no FX conversion performed" in (row.note or "")


async def test_reconciliation_foreign_only_without_opening_starts_at_zero(
    session, monkeypatch
):
    await _seed_bank(session)
    session.add(
        Transaction(
            account_id=1,
            bank="hdfc",
            email_type="test_account_transaction",
            direction="debit",
            amount=Decimal("10.00"),
            currency="USD",
            transaction_date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:Hdfc:Savings"},
            non_inr_policy="priced",
            fx_rates={"USD": (FxRate(CUTOVER, Decimal("83.0000")),)},
        ),
    )

    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.opening_available is False
    assert row.projected_balance == "0.00"
    assert row.note == (
        "no reliable pre-cutover snapshot or running balance; projected balance "
        "starts from zero (opening not invented); projected balance shown in INR "
        "only; foreign-commodity postings excluded: USD=1; no FX conversion "
        "performed"
    )


async def test_reconciliation_no_foreign_note_for_inr_only(session, monkeypatch):
    """An INR-only projected account carries no foreign-commodity note."""
    await _seed_bank(session)
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    await _seed_txn(session, 1, amount="10.00")
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:Hdfc:Savings"},
        ),
    )
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.projected_balance == "990.00"
    assert row.note is None


# --------------------------------------------------------------------------- #
# FX rate + backend config validation
# --------------------------------------------------------------------------- #


def _input(**overrides) -> PaisaConfigInput:
    base = dict(
        mode="connect",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=[],
        project_since="",
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
        ledger_cli="ledger",
        fx_rates=[],
    )
    base.update(overrides)
    return PaisaConfigInput(**base)


async def test_fx_serializes_to_nested_json_and_round_trips(session, settings_db):
    rows = [
        PaisaFxRateRow(currency="usd", date="2026-01-15", rate="83"),
        PaisaFxRateRow(currency="USD", date="2026-02-01", rate="83.5"),
    ]
    result = await surface.save_config(session, _input(fx_rates=rows))
    assert result.ok is True
    stored = settings_mod._cache["paisa.fx_rates"]
    assert '"USD"' in stored
    # Deterministic: quantized to 4dp, sorted by date.
    assert "83.0000" in stored
    assert "83.5000" in stored
    # Surfaces back as flat rows.
    back = result.config.fx_rates
    assert {r.currency for r in back} == {"USD"}
    assert {r.date for r in back} == {"2026-01-15", "2026-02-01"}


async def test_fx_rejects_non_positive_rate(session):
    result = await surface.save_config(
        session,
        _input(fx_rates=[PaisaFxRateRow(currency="USD", date="2026-01-01", rate="-5")]),
    )
    assert result.ok is False
    assert any("FX Rates" in e and "positive" in e for e in result.errors)


async def test_fx_rejects_bad_date(session):
    result = await surface.save_config(
        session,
        _input(fx_rates=[PaisaFxRateRow(currency="USD", date="not-a-date", rate="5")]),
    )
    assert result.ok is False
    assert any("FX Rates" in e for e in result.errors)


@pytest.mark.parametrize(
    "rate",
    ["NaN", "sNaN", "Infinity", "-Infinity", "1e999999", "1e-999999"],
)
async def test_fx_rejects_non_finite_and_extreme_rate_as_validation_error(
    session, rate
):
    result = await surface.save_config(
        session,
        _input(fx_rates=[PaisaFxRateRow(currency="USD", date="2026-01-01", rate=rate)]),
    )
    assert result.ok is False
    assert any("FX Rates" in error for error in result.errors)


async def test_fx_drops_empty_rows(session, settings_db):
    result = await surface.save_config(
        session,
        _input(
            fx_rates=[
                PaisaFxRateRow(currency="", date="", rate=""),
                PaisaFxRateRow(currency="USD", date="2026-01-01", rate="83"),
            ]
        ),
    )
    assert result.ok is True
    assert settings_mod._cache["paisa.fx_rates"].count("USD") >= 1


async def test_backend_selector_rejects_unknown(session):
    result = await surface.save_config(session, _input(ledger_cli="quicken"))
    assert result.ok is False
    assert any("Ledger CLI Backend" in e for e in result.errors)


async def test_backend_selector_accepts_each_supported(session, settings_db):
    for b in ("ledger", "hledger", "beancount"):
        result = await surface.save_config(session, _input(ledger_cli=b))
        assert result.ok is True, b
        assert result.config.ledger_cli == b


async def test_non_inr_policy_accepts_priced_and_rejects_unknown(session, settings_db):
    assert (
        await surface.save_config(session, _input(non_inr_policy="priced"))
    ).ok is True
    bad = await surface.save_config(session, _input(non_inr_policy="include"))
    assert bad.ok is False
    assert any("Non-INR Policy" in e for e in bad.errors)


async def test_project_investments_round_trips_through_save(session, settings_db):
    """The user-facing project_investments bool persists and surfaces on the
    redacted config DTO (no secret content); saving is allowed in any mode and
    only takes effect in project mode."""
    result = await surface.save_config(session, _input(project_investments=True))
    assert result.ok is True
    assert settings_mod._cache["paisa.project_investments"] == "true"
    assert result.config.project_investments is True


async def test_project_investments_defaults_false_when_unchecked(session, settings_db):
    result = await surface.save_config(session, _input())
    assert result.ok is True
    assert settings_mod._cache["paisa.project_investments"] == "false"
    assert result.config.project_investments is False


# --------------------------------------------------------------------------- #
# Liability sign: projected credit-normal balance shown as positive amount-owed
# --------------------------------------------------------------------------- #


async def _seed_card(session, *, id=2):
    session.add(Account(id=id, bank="hdfc", label="Card", type="credit_card"))
    await session.flush()


async def test_reconciliation_liability_projected_balance_is_positive_owed(
    session, monkeypatch
):
    """A credit-card (liability) projects under the ledger credit-normal
    convention (negative = owed); the reconciliation shows it as a positive
    amount-owed so it compares like-for-like with the positive native/Paisa
    balances and the delta is meaningful (not sign-flipped)."""
    await _seed_card(session)
    # Opening: owed 1000 (stored positive), struck just before the cutover.
    session.add(
        BalanceSnapshot(
            account_id=2,
            kind="liability",
            category="credit_card_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    # A post-cutover card purchase (debit) increases what is owed.
    session.add(
        Transaction(
            account_id=2,
            bank="hdfc",
            email_type="cc_debit_purchase",
            direction="debit",
            amount=Decimal("100.00"),
            transaction_date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(2,),
            account_mappings={"2": "Liabilities:Card:Hdfc:Card"},
        ),
    )
    resp = await build_reconciliation(
        session,
        asset_report=_assets(),
        liability_report=_liabs(("Liabilities:Card:Hdfc:Card", "1100.00")),
        upstream_available=True,
    )
    row = next(r for r in resp.accounts if r.account_id == 2)
    assert row.projected_available is True
    # Positive amount-owed (1000 opening + 100 purchase), NOT negative.
    assert row.projected_balance == "1100.00"
    assert row.paisa_balance == "1100.00"
    # Delta is correct: projected_owed - paisa_owed = 0 (not a sign flip).
    assert row.delta == "0.00"
    assert row.opening_available is True
    assert row.opening_source == "snapshot"


async def test_reconciliation_liability_delta_sign_correct_on_mismatch(
    session, monkeypatch
):
    """When projected and Paisa liability balances disagree, the delta carries
    the correct sign (projected_owed - paisa_owed), proving the conversion is
    not just abs()'d or sign-flipped."""
    await _seed_card(session)
    session.add(
        BalanceSnapshot(
            account_id=2,
            kind="liability",
            category="credit_card_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    session.add(
        Transaction(
            account_id=2,
            bank="hdfc",
            email_type="cc_debit_purchase",
            direction="debit",
            amount=Decimal("100.00"),
            transaction_date=dt.date(2026, 2, 1),
            category="groceries",
            counterparty="Store",
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(2,),
            account_mappings={"2": "Liabilities:Card:Hdfc:Card"},
        ),
    )
    # Projected owed = 1100; Paisa owed = 1000 -> delta = +100.
    resp = await build_reconciliation(
        session,
        asset_report=_assets(),
        liability_report=_liabs(("Liabilities:Card:Hdfc:Card", "1000.00")),
        upstream_available=True,
    )
    row = next(r for r in resp.accounts if r.account_id == 2)
    assert row.projected_balance == "1100.00"
    assert row.paisa_balance == "1000.00"
    assert row.delta == "100.00"


# --------------------------------------------------------------------------- #
# Opening-data diagnostic: missing / stale-gap opening is surfaced, never invented
# --------------------------------------------------------------------------- #


async def test_reconciliation_flags_missing_opening_without_inventing(
    session, monkeypatch
):
    """An account with no reliable pre-cutover snapshot or running balance
    surfaces an opening-data diagnostic; the projected balance starts from zero
    and no opening is invented."""
    await _seed_bank(session)
    # Only a post-cutover spend; no snapshot, no pre-cutover running balance.
    await _seed_txn(session, 1, amount="10.00")
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:Hdfc:Savings"},
        ),
    )
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.opening_available is False
    assert row.opening_source is None
    assert row.note is not None
    assert "opening not invented" in row.note.lower()
    # Projected balance starts from zero (no opening) — honest, not fabricated.
    assert row.projected_balance == "-10.00"


async def test_reconciliation_flags_opening_gap_before_cutover(session, monkeypatch):
    """An opening struck far before the cutover leaves an unprojected gap; the
    limitation is surfaced in the note (the opening itself is still used)."""
    await _seed_bank(session)
    # Opening 92 days before the cutover -> beyond OPENING_GAP_DAYS (45).
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=dt.date(2025, 10, 1),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    await _seed_txn(session, 1, amount="10.00")
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:Hdfc:Savings"},
        ),
    )
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.opening_available is True
    assert row.opening_source == "snapshot"
    assert row.opening_as_of == "2025-10-01"
    assert row.note is not None
    assert "gap" in row.note.lower()


async def test_reconciliation_no_opening_note_when_opening_is_recent(
    session, monkeypatch
):
    """A recent opening (within the gap window) carries no opening-gap note."""
    await _seed_bank(session)
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=dt.date(2025, 12, 31),  # 1 day before the cutover
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    await session.flush()
    await _seed_txn(session, 1, amount="10.00")
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            account_mappings={"1": "Assets:Bank:Hdfc:Savings"},
        ),
    )
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    row = next(r for r in resp.accounts if r.account_id == 1)
    assert row.opening_available is True
    assert row.note is None


# --------------------------------------------------------------------------- #
# Investment-lot diagnostics surface in the reconciliation projection diag
# --------------------------------------------------------------------------- #


async def test_reconciliation_surfaces_investment_disposal_diagnostic(
    session, monkeypatch
):
    """When investment-lot projection is on and an instrument has an unresolvable
    disposal, the reconciliation's projection diag surfaces the suppression
    (lot count zero, the ``disposal_history_unresolved`` label, and a nonzero
    unresolved count) so an operator sees it in the reconciliation view."""
    import json

    from financial_dashboard.db.models import CasUpload, InvestmentLot

    await _seed_bank(session)
    # An acquisition lot for an instrument that also has a redemption in the
    # preserved CAS facts -> suppressed.
    session.add(
        InvestmentLot(
            cas_upload_id=1,
            instrument_id="INE000A01018",
            instrument_name="Example Fund",
            quantity=Decimal("1000"),
            unit_cost=Decimal("50"),
            cost_basis=Decimal("50000"),
            currency="INR",
            acquired_on=dt.date(2026, 1, 15),
            source_ref="mf/1",
            transaction_type="purchase",
            reference="TXN001",
        )
    )
    session.add(
        CasUpload(
            portfolio_key="PAN",
            depository_source="cdsl",
            statement_date=dt.date(2026, 4, 30),
            grand_total=Decimal("0"),
            raw_holdings_json=json.dumps(
                {
                    "transactions": [
                        {
                            "scope": "mf",
                            "source_ref": "mf/1",
                            "date": "2026-01-15",
                            "description": "Example Fund",
                            "isin": "INE000A01018",
                            "transaction_type": "purchase",
                            "units": "1000",
                            "nav": "50.00",
                            "amount": "50000.00",
                            "reference": "TXN001",
                        },
                        {
                            "scope": "mf",
                            "source_ref": "mf/2",
                            "date": "2026-02-01",
                            "description": "Example Fund",
                            "isin": "INE000A01018",
                            "transaction_type": "redemption",
                            "units": "-50",
                            "nav": "51.00",
                            "amount": "-2550.00",
                            "reference": "RED001",
                        },
                    ]
                }
            ),
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            project_investments=True,
        ),
    )
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    assert resp.projection is not None
    # The lot was suppressed -> zero emitted, the disposal label surfaced, and a
    # nonzero unresolved count.
    assert resp.projection.investment_lot_count == 0
    assert "disposal_history_unresolved" in resp.projection.investment_excluded
    assert resp.projection.investment_disposal_unresolved_count == 1


# --------------------------------------------------------------------------- #
# Audit wrapping of manual operations
# --------------------------------------------------------------------------- #


async def test_generate_audited_records_start_and_complete(
    session, monkeypatch, tmp_path
):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(1,), generated_path=str(tmp_path / "g.journal")),
    )
    result = await surface.generate_now_audited(session, trigger="web")
    assert result.ok is True

    runs = await recent_runs(
        session, extension_id="paisa", operation=OPERATION_GENERATE
    )
    assert len(runs) == 1
    run = runs[0]
    assert run.status == STATUS_SUCCESS
    assert run.trigger == "web"
    assert run.completed_at is not None
    assert run.emitted_count == 1
    # No credentials or raw journal in details.
    import json

    details = json.loads(run.details) if run.details else {}
    assert "journal" not in details
    assert "auth_password" not in details


async def test_sync_audited_classifies_guard_as_skipped(session, monkeypatch):
    """A connect_only sync guard is STATUS_SKIPPED (not failure) with no error,
    mirroring the automatic runtime's classification of the same condition."""
    await _seed_bank(session)
    await _seed_txn(session, 1)
    # Force a non-fatal sync refusal (connect_only) by overriding mode.
    _set_config(monkeypatch, _cfg(mode="connect"))
    result = await surface.sync_now_audited(session, trigger="api")
    assert result.ok is False
    runs = await recent_runs(session, extension_id="paisa", operation=OPERATION_SYNC)
    assert len(runs) == 1
    assert runs[0].status == STATUS_SKIPPED
    assert runs[0].outcome == "connect_only"
    assert runs[0].error is None


async def test_generate_audited_disabled_guard_is_skipped(session, monkeypatch):
    """A disabled-mode generate guard is STATUS_SKIPPED with no error."""
    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(monkeypatch, _cfg(mode="disabled"))
    result = await surface.generate_now_audited(session, trigger="api")
    assert result.ok is False
    runs = await recent_runs(
        session, extension_id="paisa", operation=OPERATION_GENERATE
    )
    assert len(runs) == 1
    assert runs[0].status == STATUS_SKIPPED
    assert runs[0].outcome == "disabled"
    assert runs[0].error is None


async def test_skipped_guard_excluded_from_last_error(session, monkeypatch):
    """A skipped guard never surfaces in audit_view's last_error."""
    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(monkeypatch, _cfg(mode="disabled"))
    await surface.generate_now_audited(session, trigger="api")
    view = await surface.audit_view(session)
    assert view.last_error is None
    assert view.runs[0].status == STATUS_SKIPPED
    assert view.runs[0].outcome == "disabled"


async def test_probe_audited_records_even_on_exception(session, monkeypatch):
    async def boom():
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(surface, "probe_status", boom)
    _set_config(monkeypatch, _cfg(mode="connect"))
    with pytest.raises(RuntimeError):
        await surface.probe_status_audited(session, trigger="api")
    runs = await recent_runs(session, extension_id="paisa", operation=OPERATION_PROBE)
    assert len(runs) == 1
    assert runs[0].status == "failure"
    assert runs[0].outcome == "error"
    assert runs[0].error is not None and "blew up" in runs[0].error


async def test_audit_view_never_surfaces_credentials(session, monkeypatch, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(1,), generated_path=str(tmp_path / "g.journal")),
    )
    await surface.generate_now_audited(session, trigger="api")
    view = await surface.audit_view(session)
    assert view.last_success is not None
    blob = view.model_dump_json()
    assert "auth_password" not in blob


# --------------------------------------------------------------------------- #
# Closed-population reconciliation: projection underlying the recon is closed,
# diagnostics agree with journal tags, and the view is read-only.
# --------------------------------------------------------------------------- #


def _ids_in(report) -> tuple[set[int], set[int]]:
    """Split a projection report into (emitted_ids, skipped_ids)."""
    emitted: set[int] = set()
    for entry in report.entries:
        emitted.update(entry.txn_ids)
    skipped = {s.txn_id for s in report.skipped if s.txn_id is not None}
    return emitted, skipped


async def _seed_card_account(session, *, id=2):
    session.add(Account(id=id, bank="icici", label="Platinum", type="credit_card"))
    await session.flush()


async def _seed_card_row(session, *, id=10, account_id=2, mask="1234"):
    from financial_dashboard.db.models import Card

    session.add(Card(id=id, account_id=account_id, card_mask=mask))
    await session.flush()


def _txn_kwargs(
    *,
    account_id,
    direction,
    amount,
    date,
    category="groceries",
    id=None,
    counterparty="Store",
    reference_number=None,
    card_id=None,
    card_mask=None,
    currency="INR",
):
    kw = dict(
        account_id=account_id,
        bank="hdfc",
        email_type="test_txn",
        direction=direction,
        amount=Decimal(amount),
        currency=currency,
        transaction_date=date,
        category=category,
        counterparty=counterparty,
        reference_number=reference_number,
        card_id=card_id,
        card_mask=card_mask,
        source="email",
        channel="email",
    )
    if id is not None:
        kw["id"] = id
    return kw


async def test_reconciliation_projection_is_closed_population(session, monkeypatch):
    """The projection underlying the reconciliation is closed: every eligible
    selected post-cutover transaction id is either emitted or skipped, never
    both, never neither. The reconciliation's projection diag agrees with the
    projection report's emitted count."""
    from financial_dashboard.services.paisa.projection import project

    await _seed_bank(session)
    await _seed_card_account(session, id=2)
    await _seed_card_row(session, id=10, account_id=2, mask="1234")
    # Eligible emitted: expense, income, resolved card payment (bank leg).
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="10.00",
                date=dt.date(2026, 2, 1),
                category="groceries",
                id=1,
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="credit",
                amount="5000.00",
                date=dt.date(2026, 2, 2),
                category="salary",
                id=2,
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="1000.00",
                date=dt.date(2026, 2, 3),
                category="credit_card_payment",
                counterparty="Card Bill",
                card_id=10,
                id=3,
            )
        )
    )
    # Eligible skipped: card-side payment (emitted via bank leg only).
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=2,
                direction="credit",
                amount="1000.00",
                date=dt.date(2026, 2, 3),
                category="credit_card_payment",
                counterparty="Payment",
                id=4,
            )
        )
    )
    # Eligible emitted: a self-transfer pair (one entry, two ids).
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="200.00",
                date=dt.date(2026, 2, 4),
                category="self_transfer",
                reference_number="IMPS1",
                id=5,
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=2,
                direction="credit",
                amount="200.00",
                date=dt.date(2026, 2, 4),
                category="self_transfer",
                reference_number="IMPS1",
                id=6,
            )
        )
    )
    # NOT eligible: pre-cutover (never loaded).
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="10.00",
                date=dt.date(2025, 12, 15),
                category="groceries",
                id=99,
            )
        )
    )
    await session.flush()

    cfg = _cfg(selected_account_ids=(1, 2))
    _set_config(monkeypatch, cfg)
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    report = await project(session, cfg)

    emitted, skipped = _ids_in(report)
    assert emitted.isdisjoint(skipped)
    union = emitted | skipped
    # Every post-cutover selected txn is accounted for; pre-cutover is not.
    assert union == {1, 2, 3, 4, 5, 6}
    assert 99 not in union
    # The reconciliation diag agrees with the projection report's emitted count.
    assert resp.projection is not None
    assert resp.projection.emitted_count == report.emitted_count


async def test_reconciliation_txn_ids_signs_amounts_trace_source(session, monkeypatch):
    """Every emitted entry's txn ids, posting signs and amounts trace back to a
    source transaction. A debit decreases the bank (negative posting); a credit
    increases it; amounts match the source row."""
    from financial_dashboard.services.paisa.projection import project

    await _seed_bank(session)
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="123.45",
                date=dt.date(2026, 2, 1),
                category="groceries",
                id=1,
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="credit",
                amount="50000.00",
                date=dt.date(2026, 2, 2),
                category="salary",
                id=2,
            )
        )
    )
    await session.flush()

    cfg = _cfg(selected_account_ids=(1,))
    _set_config(monkeypatch, cfg)
    await build_reconciliation(session, asset_report=None, liability_report=None)
    report = await project(session, cfg)

    by_id = {tid: e for e in report.entries for tid in e.txn_ids}
    debit = by_id[1]
    credit = by_id[2]
    # Debit: bank posting negative; contra positive; |amount| traces source.
    assert debit.postings[0].amount == Decimal("-123.45")
    assert debit.postings[1].amount == Decimal("123.45")
    # Credit: bank posting positive; contra negative.
    assert credit.postings[0].amount == Decimal("50000.00")
    assert credit.postings[1].amount == Decimal("-50000.00")
    # Every entry's posting magnitude equals its source txn amount.
    for entry in report.entries:
        src_amounts = {
            Decimal("123.45") if 1 in entry.txn_ids else None,
            Decimal("50000.00") if 2 in entry.txn_ids else None,
        }
        for p in entry.postings:
            assert abs(p.amount) in src_amounts, (entry.txn_ids, p.amount)


async def test_reconciliation_diagnostics_agree_with_journal_tags(session, monkeypatch):
    """The reconciliation projection diag (kind_counts, card resolved/unresolved,
    imprecise, foreign counts) agrees with the dashboard tags rendered in the
    journal — the diag is a true summary of what the file actually carries."""
    from financial_dashboard.services.paisa.projection import project

    await _seed_bank(session)
    # imprecise (emi_loan) + expense + foreign (USD, priced).
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="5000.00",
                date=dt.date(2026, 2, 1),
                category="emi_loan",
                id=1,
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="10.00",
                date=dt.date(2026, 2, 2),
                category="groceries",
                id=2,
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="10.00",
                date=dt.date(2026, 2, 3),
                category="dining",
                currency="USD",
                id=3,
            )
        )
    )
    await session.flush()

    cfg = _cfg(
        selected_account_ids=(1,),
        non_inr_policy="priced",
        fx_rates={"USD": (FxRate(CUTOVER, Decimal("83.0000")),)},
    )
    _set_config(monkeypatch, cfg)
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    report = await project(session, cfg)

    diag = resp.projection
    assert diag is not None
    # Diag agrees with the report's own computed diagnostics.
    assert diag.kind_counts == report.kind_counts
    assert diag.imprecise_count == report.imprecise_count == 1
    assert diag.card_payments_resolved == report.card_payments_resolved
    assert diag.card_payments_unresolved == report.card_payments_unresolved
    assert diag.projected_foreign_count == report.projected_foreign_count == 1
    assert diag.source_currencies == list(report.source_currencies) == ["USD"]
    assert diag.missing_fx_rate_count == report.missing_fx_rate_count == 0

    # The journal carries the matching dashboard_kind tags.
    journal = report.journal
    for kind, count in diag.kind_counts.items():
        assert journal.count(f"dashboard_kind: {kind}") == count, (
            f"kind {kind}: diag={count} journal={journal.count('dashboard_kind: ' + kind)}"
        )
    # A USD price directive is present (foreign entry emitted under priced policy).
    assert "price USD" in journal or "P " in journal


async def test_reconciliation_funding_diagnostics_agree_with_journal(
    session, monkeypatch
):
    """When an investment lot is suppressed for funding ambiguity, the
    reconciliation diag surfaces the funding-unresolved label and the journal
    carries neither the lot nor an orphan price for it."""
    import json

    from financial_dashboard.db.models import CasUpload, InvestmentLot
    from financial_dashboard.services.paisa.projection import project

    await _seed_bank(session)
    for isin in ("INE000A01018", "INE000A01019"):
        session.add(
            InvestmentLot(
                cas_upload_id=1,
                instrument_id=isin,
                instrument_name=f"Fund {isin}",
                quantity=Decimal("1000"),
                unit_cost=Decimal("50"),
                cost_basis=Decimal("50000"),
                currency="INR",
                acquired_on=dt.date(2026, 2, 15),
                source_ref=f"mf/{isin}",
                transaction_type="purchase",
                reference=f"TXN_{isin}",
            )
        )
    session.add(
        CasUpload(
            portfolio_key="PAN",
            depository_source="cdsl",
            statement_date=dt.date(2026, 4, 30),
            grand_total=Decimal("0"),
            raw_holdings_json=json.dumps(
                {
                    "transactions": [
                        {
                            "scope": "mf",
                            "source_ref": f"mf/{isin}",
                            "date": "2026-02-15",
                            "description": f"Fund {isin}",
                            "isin": isin,
                            "transaction_type": "purchase",
                            "units": "1000",
                            "nav": "50.00",
                            "amount": "50000.00",
                            "reference": f"TXN_{isin}",
                        }
                        for isin in ("INE000A01018", "INE000A01019")
                    ]
                }
            ),
        )
    )
    # Bank investment txn matching date+amount of both lots (ambiguous).
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="50000.00",
                date=dt.date(2026, 2, 15),
                category="investment",
                counterparty="MF Purchase",
                reference_number="SHARED_REF",
                id=1,
            )
        )
    )
    await session.flush()

    cfg = _cfg(selected_account_ids=(1,), project_investments=True)
    _set_config(monkeypatch, cfg)
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    report = await project(session, cfg)

    diag = resp.projection
    assert diag is not None
    # Both instruments funding-unresolved; the label surfaces in the diag.
    assert diag.investment_funding_unresolved == [
        "INE000A01018",
        "INE000A01019",
    ]
    assert "investment_funding_unresolved" in diag.investment_excluded
    # The journal carries no asset account / price for either suppressed lot.
    for isin in ("INE000A01018", "INE000A01019"):
        assert isin not in report.journal
    assert diag.investment_lot_count == report.investment_lot_count == 0


async def test_reconciliation_is_read_only_closed_population(session, monkeypatch):
    """build_reconciliation never writes a core row even with a rich projection
    (read-only), and the projection is closed-population across a mixed set."""
    from financial_dashboard.services.paisa.projection import project

    await _seed_bank(session)
    await _seed_card_account(session, id=2)
    await _seed_card_row(session, id=10, account_id=2, mask="1234")
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="10.00",
                date=dt.date(2026, 2, 1),
                category="groceries",
                id=1,
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="5000.00",
                date=dt.date(2026, 2, 3),
                category="credit_card_payment",
                counterparty="Card Bill",
                card_mask="9999",
                id=2,  # unresolved
            )
        )
    )
    await session.flush()

    cfg = _cfg(selected_account_ids=(1, 2))
    _set_config(monkeypatch, cfg)

    txns_before = len((await session.execute(select(Transaction))).scalars().all())
    accts_before = len((await session.execute(select(Account))).scalars().all())

    resp = await build_reconciliation(
        session,
        asset_report=_assets(("Assets:Bank:Hdfc:Savings", "100.00")),
        liability_report=_liabs(("Liabilities:Card:Icici:Platinum", "50.00")),
        upstream_available=True,
    )
    assert resp.ok is True

    txns_after = len((await session.execute(select(Transaction))).scalars().all())
    accts_after = len((await session.execute(select(Account))).scalars().all())
    assert txns_before == txns_after
    assert accts_before == accts_after

    # Closed-population: the unresolved card payment is emitted (bank leg); the
    # diag surfaces the unresolved count.
    report = await project(session, cfg)
    emitted, skipped = _ids_in(report)
    assert emitted.isdisjoint(skipped)
    assert (emitted | skipped) == {1, 2}
    assert resp.projection.card_payments_unresolved == 1
    assert resp.projection.card_payments == 1


async def test_reconciliation_notes_unresolved_card_clearing_cannot_attribute(
    session, monkeypatch
):
    """A specific card's projected balance does not include unresolved card
    payments (they post to the generic clearing), and the reconciliation surfaces
    an explanatory note — never a silent or 'corrected' number."""
    await _seed_bank(session, id=1)
    await _seed_card_account(session, id=2)
    await _seed_card_row(session, id=10, account_id=2, mask="1234")
    # An opening for the card so it has a projected balance.
    session.add(
        BalanceSnapshot(
            account_id=2,
            kind="liability",
            category="credit_card_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    # Bank-side card payment with a NON-matching mask → unresolved → generic.
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="5000.00",
                date=dt.date(2026, 2, 1),
                category="credit_card_payment",
                counterparty="Card Bill",
                card_mask="9999",
                id=1,
            )
        )
    )
    await session.flush()

    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1, 2),
            account_mappings={"2": "Liabilities:Card:Icici:Platinum"},
        ),
    )
    resp = await build_reconciliation(
        session,
        asset_report=_assets(),
        liability_report=_liabs(("Liabilities:Card:Icici:Platinum", "1000.00")),
        upstream_available=True,
    )
    card_row = next(r for r in resp.accounts if r.account_id == 2)
    assert card_row.projected_available is True
    # The card's projected balance starts from the opening (1000) and is
    # unaffected by the unresolved payment (which posted to the generic clearing).
    assert card_row.projected_balance == "1000.00"
    # An explanatory note surfaces the unattributed payment.
    assert card_row.note is not None
    assert "unresolved card payment" in card_row.note.lower()
    assert "Liabilities:Credit Card" in card_row.note


async def test_reconciliation_no_clearing_note_when_card_maps_to_clearing(
    session, monkeypatch
):
    """When a card is itself mapped to the generic clearing account, unresolved
    payments DO affect it — so the explanatory note must NOT fire (it would be
    misleading)."""
    await _seed_bank(session, id=1)
    await _seed_card_account(session, id=2)
    session.add(
        BalanceSnapshot(
            account_id=2,
            kind="liability",
            category="credit_card_balance",
            as_of_date=dt.date(2025, 12, 31),
            value=Decimal("1000.00"),
            source="statement",
            currency="INR",
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="5000.00",
                date=dt.date(2026, 2, 1),
                category="credit_card_payment",
                counterparty="Card Bill",
                card_mask="9999",
                id=1,
            )
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1, 2),
            account_mappings={"2": "Liabilities:Credit Card"},
        ),
    )
    resp = await build_reconciliation(session, asset_report=None, liability_report=None)
    card_row = next(r for r in resp.accounts if r.account_id == 2)
    # The unresolved payment now lands on this account's mapped name → no note.
    assert "unresolved card payment" not in (card_row.note or "").lower()


async def test_projection_summary_surfaces_computed_diagnostics(session, monkeypatch):
    """The projection summary DTO (preview/generate/sync) surfaces the computed
    diagnostics (imprecise, card resolved/unresolved, funding, kind_counts,
    projected foreign, source currencies, missing rates) — additive fields that
    never reshape the existing summary contract."""
    from financial_dashboard.services.paisa import surface as surface_mod

    await _seed_bank(session)
    await _seed_card_account(session, id=2)
    await _seed_card_row(session, id=10, account_id=2, mask="1234")
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="5000.00",
                date=dt.date(2026, 2, 1),
                category="emi_loan",
                id=1,
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="1000.00",
                date=dt.date(2026, 2, 2),
                category="credit_card_payment",
                counterparty="Card Bill",
                card_mask="9999",
                id=2,  # unresolved
            )
        )
    )
    session.add(
        Transaction(
            **_txn_kwargs(
                account_id=1,
                direction="debit",
                amount="10.00",
                date=dt.date(2026, 2, 3),
                category="dining",
                currency="USD",
                id=3,
            )
        )
    )
    await session.flush()
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1, 2),
            non_inr_policy="priced",
            fx_rates={"USD": (FxRate(CUTOVER, Decimal("83.0000")),)},
        ),
    )
    dto = await surface_mod.preview_projection(session)
    assert dto.ok is True
    assert dto.summary is not None
    s = dto.summary
    # The additive computed fields are populated and agree with the journal.
    assert s.imprecise_count == 1
    assert s.card_payments == 1
    assert s.card_payments_unresolved == 1
    assert s.card_payments_resolved == 0
    # emi_loan (imprecise, still an expense) + dining expense = 2 expenses.
    assert s.kind_counts.get("expense") == 2
    assert s.kind_counts.get("card_payment") == 1
    assert s.projected_foreign_count == 1
    assert s.source_currencies == ["USD"]
    assert s.missing_fx_rate_count == 0
    assert s.investment_funding_remapped == 0
    # Existing contract fields are unchanged.
    assert s.emitted_count == 3
    assert "txn:1" in dto.journal


# --------------------------------------------------------------------------- #
# Diagnosis classification: audit/DTO surfacing + no credentials/raw journal
# --------------------------------------------------------------------------- #


async def test_sync_audited_surfaces_diagnosis_counts(session, monkeypatch):
    """A successful sync whose dangers were all expected contra-expense
    ``Debit Entry`` issues records the expected/accepted/fatal counts in the
    audit details and surfaces them on the DTO."""
    import json

    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            generated_path="/tmp/paisa-audit-test.journal",
        ),
    )

    async def fake_sync(_session, *, client=None):
        return PaisaSyncResponse(
            ok=True,
            mode="project",
            outcome="synced",
            diagnosis_ok=True,
            diagnosis_expected=3,
            diagnosis_accepted=3,
            diagnosis_fatal=0,
            reason=None,
        )

    monkeypatch.setattr(surface, "sync_now", fake_sync)
    result = await surface.sync_now_audited(session, trigger="api")
    assert result.ok is True
    assert result.diagnosis_expected == 3
    assert result.diagnosis_accepted == 3
    assert result.diagnosis_fatal == 0

    runs = await recent_runs(session, extension_id="paisa", operation=OPERATION_SYNC)
    run = runs[0]
    assert run.status == STATUS_SUCCESS
    details = json.loads(run.details) if run.details else {}
    assert details["diagnosis_expected"] == 3
    assert details["diagnosis_accepted"] == 3
    assert details["diagnosis_fatal"] == 0
    # No credentials and no raw journal text in the audit details.
    blob = json.dumps(details)
    assert "auth_password" not in blob
    assert "journal" not in details  # no raw journal body key


async def test_sync_audited_records_fatal_diagnosis_counts_on_failure(
    session, monkeypatch
):
    """A sync that fails with a fatal (unmatched) danger records the counts and
    a sanitized error; the audit error carries no raw journal text."""
    import json

    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(1,), generated_path="/tmp/paisa-audit.journal"),
    )

    async def fake_sync(_session, *, client=None):
        return PaisaSyncResponse(
            ok=False,
            mode="project",
            outcome="diagnosis_failed",
            diagnosis_ok=False,
            diagnosis_expected=1,
            diagnosis_accepted=1,
            diagnosis_fatal=1,
            reason="Negative Balance: Assets:Bank went negative",
        )

    monkeypatch.setattr(surface, "sync_now", fake_sync)
    result = await surface.sync_now_audited(session, trigger="api")
    assert result.ok is False
    assert result.diagnosis_fatal == 1

    runs = await recent_runs(session, extension_id="paisa", operation=OPERATION_SYNC)
    run = runs[0]
    assert run.status == "failure"
    details = json.loads(run.details) if run.details else {}
    assert details["diagnosis_fatal"] == 1
    assert details["diagnosis_accepted"] == 1
    # The error is the sanitized sync reason; no credentials.
    assert run.error is not None
    assert "Negative Balance" in run.error
    blob = json.dumps(details) + (run.error or "")
    assert "auth_password" not in blob


async def test_sync_dto_exposes_diagnosis_counts_through_api(
    client, session, monkeypatch
):
    """The /api/extensions/paisa/sync JSON response exposes the classified
    diagnosis counts (additive fields, never reshaping the existing contract)."""
    from financial_dashboard.services.paisa import surface as surface_mod

    async def fake_sync(_session, *, client=None):
        return PaisaSyncResponse(
            ok=True,
            mode="project",
            outcome="synced",
            diagnosis_ok=True,
            diagnosis_expected=2,
            diagnosis_accepted=2,
            diagnosis_fatal=0,
            reason=None,
        )

    monkeypatch.setattr(surface_mod, "sync_now", fake_sync)
    _set_config(
        monkeypatch,
        _cfg(
            selected_account_ids=(1,),
            generated_path="/tmp/paisa-api-sync.journal",
        ),
    )
    await _seed_bank(session)
    await _seed_txn(session, 1)
    r = await client.post("/api/extensions/paisa/sync")
    body = r.json()
    assert body["ok"] is True
    assert body["outcome"] == "synced"
    assert body["diagnosis_expected"] == 2
    assert body["diagnosis_accepted"] == 2
    assert body["diagnosis_fatal"] == 0


# --------------------------------------------------------------------------- #
# Manual lease heartbeat + state-write failure surfacing (stress/race)
# --------------------------------------------------------------------------- #


async def test_sync_audited_surfaces_stale_lease_state_write_failure(
    session, monkeypatch
):
    """When the manual sync's token-guarded state writes fail (stale lease),
    the audit details must record ``state_recorded: false`` rather than
    swallowing the failure — so the audit never claims a coordinated sync
    while state remains stale."""
    import json

    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(1,), generated_path="/tmp/paisa-stale.journal"),
    )

    async def fake_sync(_session, *, client=None):
        return PaisaSyncResponse(
            ok=True,
            mode="project",
            outcome="synced",
            summary=None,
            publish=PaisaPublishInfo(
                published=True,
                skipped=False,
                path="/tmp/paisa-stale.journal",
                version="3",
                body_hash="stalebody",
                bytes_written=10,
            ),
            diagnosis_ok=True,
            reason=None,
        )

    monkeypatch.setattr(surface, "sync_now", fake_sync)

    # Force the state-write to raise LeaseStaleError.
    from financial_dashboard.services.paisa.sync_state import LeaseStaleError

    async def boom_state(*_a, **_k):
        raise LeaseStaleError("paisa", token="stale", reason="stale_token")

    monkeypatch.setattr(surface, "_apply_manual_sync_state", boom_state)

    result = await surface.sync_now_audited(session, trigger="api")
    assert result.ok is True  # the POST was accepted (remote perspective)

    runs = await recent_runs(session, extension_id="paisa", operation=OPERATION_SYNC)
    run = runs[0]
    details = json.loads(run.details) if run.details else {}
    assert details["state_recorded"] is False
    assert details["state_error"] == "lease_stale_or_write_failed"


async def test_generate_audited_surfaces_stale_lease_hash_write_failure(
    session, monkeypatch, tmp_path
):
    """Same property for manual generate: a stale-lease hash write is surfaced
    in details, not swallowed."""
    import json

    await _seed_bank(session)
    await _seed_txn(session, 1)
    _set_config(
        monkeypatch,
        _cfg(selected_account_ids=(1,), generated_path=str(tmp_path / "g.journal")),
    )

    async def fake_generate(_session):
        return PaisaGenerateResponse(
            ok=True,
            mode="project",
            summary=None,
            publish=PaisaPublishInfo(
                published=True,
                skipped=False,
                path=str(tmp_path / "g.journal"),
                version="3",
                body_hash="genbody",
                bytes_written=10,
            ),
            reason=None,
        )

    monkeypatch.setattr(surface, "generate_now", fake_generate)

    from financial_dashboard.services.paisa.sync_state import LeaseStaleError

    async def boom_hash(*_a, **_k):
        raise LeaseStaleError("paisa", token="stale", reason="stale_token")

    monkeypatch.setattr(surface, "record_published_hash", boom_hash)

    await surface.generate_now_audited(session, trigger="api")
    runs = await recent_runs(
        session, extension_id="paisa", operation=OPERATION_GENERATE
    )
    details = json.loads(runs[0].details) if runs[0].details else {}
    assert details["state_recorded"] is False


async def test_manual_heartbeat_extends_lease_during_long_sync(tmp_path):
    """The heartbeat loop extends a held lease; a stale token is a no-op.

    Verified directly against the heartbeat primitive: after one beat the
    lease expiry is pushed out by the TTL. This is the mechanism that keeps a
    long manual operation (large journal + slow POST) from losing its lease to
    the automatic coordinator. Uses a file-based WAL DB so the heartbeat's
    fresh session observes the caller's committed lease."""
    import asyncio

    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from financial_dashboard.db.models import Base
    from financial_dashboard.services.paisa import surface as surf
    from financial_dashboard.services.paisa.sync_state import (
        claim_lease,
        ensure_sync_state,
        read_sync_state,
    )

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'hb.db'}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        async with factory() as s:
            await ensure_sync_state(s)
            claim = await claim_lease(s, owner="manual-test")
            await s.commit()
        assert claim.claimed
        token = claim.token

        async with factory() as s:
            snap_before = await read_sync_state(s)
        assert snap_before.lease_expires_at is not None
        original_expiry = snap_before.lease_expires_at

        # Run one heartbeat iteration with a tiny sleep; cancel after it fires.
        beat_done = asyncio.Event()

        async def tiny_sleep(_s):
            beat_done.set()
            await asyncio.sleep(10)  # block so we cancel mid-wait

        task = asyncio.create_task(
            surf._manual_heartbeat_loop(engine, token, interval=0.001, sleep=tiny_sleep)
        )
        await asyncio.wait_for(beat_done.wait(), timeout=2.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        async with factory() as s:
            snap_after = await read_sync_state(s)
        assert snap_after.lease_expires_at is not None
        # The heartbeat pushed the expiry out (it renewed with now + ttl).
        assert snap_after.lease_expires_at >= original_expiry
        # The token is unchanged (heartbeat extends, never re-mints).
        assert snap_after.lease_token == token
    finally:
        await engine.dispose()


async def test_manual_lease_polling_commits_running_audit_row(tmp_path, monkeypatch):
    """claim_manual_lease commits the session, so the 'running' audit row
    started by _audited is committed (observable) before the operation
    finishes. This is by design: a crash leaves a 'running' row, not a silent
    loss. Uses a file-based DB so a fresh observer session sees the committed
    row."""

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from financial_dashboard.db.models import Base, ExtensionRun
    from financial_dashboard.services.paisa import surface as surf
    from financial_dashboard.services.paisa.config import PaisaProjectionConfig

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'preempt.db'}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        cfg = PaisaProjectionConfig(
            mode="project",
            base_url="http://127.0.0.1:7500",
            external_url="",
            allow_remote=False,
            auth_username="",
            auth_password="",
            generated_path="/tmp/paisa-preempt.journal",
            selected_account_ids=(1,),
            cutover_date=dt.date(2026, 1, 1),
            account_mappings={},
            category_mappings={},
            non_inr_policy="skip",
            request_timeout_seconds=15,
        )
        monkeypatch.setattr(surf, "load_config", lambda: cfg)

        captured = {"running_committed": False}

        async def fake_generate(_session):
            # While the generate is "running", check if there's a committed
            # 'running' row visible from a *fresh* session (proving it was
            # committed by claim_manual_lease, not just pending).
            async with factory() as fresh:
                rows = (
                    (
                        await fresh.execute(
                            select(ExtensionRun).where(
                                ExtensionRun.extension_id == "paisa",
                                ExtensionRun.status == "running",
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                captured["running_committed"] = len(rows) > 0
            return PaisaGenerateResponse(
                ok=True,
                mode="project",
                summary=None,
                publish=PaisaPublishInfo(
                    published=True,
                    skipped=False,
                    path="/tmp/paisa-preempt.journal",
                    version="3",
                    body_hash="preempt",
                    bytes_written=10,
                ),
                reason=None,
            )

        monkeypatch.setattr(surf, "generate_now", fake_generate)

        async with factory() as session:
            result = await surf.generate_now_audited(session, trigger="api")
            assert result.ok is True

        assert captured["running_committed"] is True

        # The final audit row is completed and committed.
        async with factory() as s:
            from financial_dashboard.services.paisa.audit import recent_runs

            runs = await recent_runs(
                s, extension_id="paisa", operation=OPERATION_GENERATE
            )
            assert len(runs) == 1
            assert runs[0].status == STATUS_SUCCESS
    finally:
        await engine.dispose()
