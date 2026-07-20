"""Truthful 20,000-event Paisa coordinator stress/correctness gate.

This is deliberately a SQLite transaction-layer benchmark, **not** a claim of
20,000 FastAPI/HTTP requests per second.  It applies exactly 20,000 row mutation
events (19,800 committed and 200 intentionally rolled back) through the real
SQLite triggers and reports measured committed mutation events/second.  There
is no wall-clock assertion: correctness and final ledger parity are the gate.

The expensive integration test is opt-in so the default suite remains quick::

    PAISA_COORDINATOR_STRESS=1 \
      uv run pytest -q tests/test_paisa_coordinator_stress.py -s

The fake Paisa boundary is below the production projection/publisher and above
the network client.  Projection, atomic publication, revision triggers, lease,
retry, audit, and renderer behavior are therefore real; external I/O is not.
"""

import asyncio
import datetime
import hashlib
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
from sqlalchemy import bindparam, event, func, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.services.paisa.coordinator as coordinator_module
from financial_dashboard.db.init_db import init_db
from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    Card,
    CasUpload,
    ExtensionRun,
    InvestmentLot,
    Setting,
    Transaction,
)
from financial_dashboard.services.paisa.config import FxRate, PaisaProjectionConfig
from financial_dashboard.services.paisa.coordinator import PaisaCoordinator
from financial_dashboard.services.paisa.orchestrator import (
    PreflightReport,
    RemoteSyncReport,
    generate,
)
from financial_dashboard.services.paisa.projection import ProjectionReport, project
from financial_dashboard.services.paisa.sync_state import read_sync_state

STRESS_ENV = "PAISA_COORDINATOR_STRESS"
SEED = 0x20_000_5A
CUTOVER = datetime.date(2026, 1, 1)
LOAD_START_ID = 10_000
SELECTED_ACCOUNT_IDS = (1, 2, 3)
CARD_ID = 101

# One event means one row reached a trigger-eligible INSERT/UPDATE/DELETE.  The
# two rollback groups are included in the exactly-20,000 attempted population,
# but correctly excluded from COMMITTED_EVENTS and the revision delta.
WORKLOAD_DISTRIBUTION = {
    "transaction_insert_statement": 12_000,
    "transaction_insert_rapid": 1_800,
    "transaction_update_historical": 4_000,
    "transaction_delete": 1_500,
    "account_update": 80,
    "card_update": 80,
    "balance_snapshot_upsert": 120,
    "paisa_setting_update": 120,
    "cas_insert": 15,
    "cas_update": 15,
    "cas_delete": 10,
    "investment_lot_insert": 20,
    "investment_lot_update": 20,
    "investment_lot_delete": 20,
    "rollback_outer_insert": 100,
    "rollback_savepoint_insert": 100,
}
ATTEMPTED_EVENTS = sum(WORKLOAD_DISTRIBUTION.values())
ROLLED_BACK_EVENTS = (
    WORKLOAD_DISTRIBUTION["rollback_outer_insert"]
    + WORKLOAD_DISTRIBUTION["rollback_savepoint_insert"]
)
COMMITTED_EVENTS = ATTEMPTED_EVENTS - ROLLED_BACK_EVENTS

FLAVORS = (
    "debit_groceries",
    "credit_salary",
    "credit_refund",
    "credit_cashback",
    "credit_fee_reversal",
    "debit_fee",
    "card_payment",
    "fx_usd_debit",
    "fx_eur_credit",
    "self_transfer_debit",
    "self_transfer_credit",
    "debit_investment",
)


class FakeClock:
    def __init__(self, value: datetime.datetime) -> None:
        self.value = value.astimezone(datetime.UTC)

    def __call__(self) -> datetime.datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += datetime.timedelta(**kwargs)


class WorkloadPlan(NamedTuple):
    statement_inserts: list[dict]
    rapid_inserts: list[dict]
    updates: list[dict]
    delete_ids: list[int]
    expected_transactions: dict[int, dict]
    deleted_ids: set[int]
    rolled_back_ids: set[int]
    flavor_counts: Counter
    historical_id: int
    card_payment_id: int
    fx_id: int
    fee_reversal_id: int
    self_transfer_ids: tuple[int, int]


class ParsedLedgerEntry(NamedTuple):
    date: datetime.date
    header: str
    meta: dict[str, str]
    postings: tuple[tuple[str, Decimal, str], ...]


class FakeClient:
    async def aclose(self) -> None:
        return None


class FakeRemote:
    """Deterministic remote: success, blocked success, failure, retry success."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.blocked_attempt_entered = asyncio.Event()
        self.release_blocked_attempt = asyncio.Event()

    async def __call__(
        self,
        report: ProjectionReport,
        _config: PaisaProjectionConfig,
        *,
        client=None,
    ) -> RemoteSyncReport:
        del client
        body_hash = hashlib.sha256(report.journal.encode()).hexdigest()
        call_number = len(self.calls) + 1
        self.calls.append((body_hash, report.journal))
        if call_number == 2:
            self.blocked_attempt_entered.set()
            await self.release_blocked_attempt.wait()
        if call_number == 3:
            return RemoteSyncReport(
                ok=False,
                outcome="unreachable",
                post_accepted=False,
                diagnosis_ok=None,
                reason="simulated remote unavailable after local publish",
            )
        return RemoteSyncReport(
            ok=True,
            outcome="synced",
            post_accepted=True,
            diagnosis_ok=True,
            reason=None,
            diagnosis_expected=0,
            diagnosis_accepted=0,
            diagnosis_fatal=0,
        )


def test_20k_distribution_contract_is_explicit_and_truthful():
    """Cheap default-suite guard for the opt-in benchmark's claimed scale."""
    assert ATTEMPTED_EVENTS == 20_000
    assert COMMITTED_EVENTS == 19_800
    assert ROLLED_BACK_EVENTS == 200
    assert set(WORKLOAD_DISTRIBUTION) >= {
        "transaction_insert_statement",
        "transaction_insert_rapid",
        "transaction_update_historical",
        "transaction_delete",
        "account_update",
        "card_update",
        "balance_snapshot_upsert",
        "paisa_setting_update",
        "cas_insert",
        "cas_update",
        "cas_delete",
        "investment_lot_insert",
        "investment_lot_update",
        "investment_lot_delete",
        "rollback_outer_insert",
        "rollback_savepoint_insert",
    }


def _config(generated_path: Path, *, backend: str = "ledger") -> PaisaProjectionConfig:
    return PaisaProjectionConfig(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path=str(generated_path),
        selected_account_ids=SELECTED_ACCOUNT_IDS,
        cutover_date=CUTOVER,
        account_mappings={
            "1": "Assets:Bank:HDFC:Salary",
            "2": "Assets:Bank:Axis:Savings",
            "3": "Liabilities:CreditCard:ICICI",
        },
        category_mappings={},
        non_inr_policy="priced",
        request_timeout_seconds=15,
        ledger_cli=backend,
        fx_rates={
            "USD": (FxRate(CUTOVER, Decimal("83.2500")),),
            "EUR": (FxRate(CUTOVER, Decimal("91.5000")),),
        },
        project_investments=False,
    )


def _flavor_row(
    *, txn_id: int, flavor: str, group: int, rng: random.Random, source: str
) -> dict:
    amount = Decimal(rng.randrange(100, 500_000)) / 100
    date = CUTOVER + datetime.timedelta(days=1 + rng.randrange(240))
    direction = "debit"
    category = "groceries"
    currency = "INR"
    account_id = 1 if group % 2 == 0 else 2
    card_id = None
    card_mask = None
    counterparty = f"Merchant {rng.randrange(400):03d}"
    reference = f"load-{txn_id}"

    if flavor == "credit_salary":
        direction, category, counterparty = "credit", "salary", "Employer"
    elif flavor == "credit_refund":
        direction, category, counterparty = "credit", "refund", "Refund Desk"
    elif flavor == "credit_cashback":
        direction, category, counterparty = (
            "credit",
            "cashback_rewards",
            "Card Rewards",
        )
    elif flavor == "credit_fee_reversal":
        direction, category, counterparty = (
            "credit",
            "fees_charges",
            "Fee Reversal",
        )
    elif flavor == "debit_fee":
        category, counterparty = "fees_charges", "Monthly Fee"
    elif flavor == "card_payment":
        category, counterparty = "credit_card_payment", "ICICI Card Payment"
        account_id, card_id, card_mask = 1, CARD_ID, "****1111"
    elif flavor == "fx_usd_debit":
        category, currency, counterparty = "travel", "USD", "US Hotel"
    elif flavor == "fx_eur_credit":
        direction, category, currency, counterparty = (
            "credit",
            "refund",
            "EUR",
            "EU Refund",
        )
    elif flavor in ("self_transfer_debit", "self_transfer_credit"):
        category = "self_transfer"
        direction = "debit" if flavor.endswith("debit") else "credit"
        account_id = 1 if direction == "debit" else 2
        amount = Decimal(10_000 + group) / 100
        date = CUTOVER + datetime.timedelta(days=1 + group % 240)
        counterparty = "Own Account Transfer"
        reference = f"self-{group}"
    elif flavor == "debit_investment":
        category, counterparty = "investment", "Index Fund"

    return {
        "id": txn_id,
        "account_id": account_id,
        "card_id": card_id,
        "bank": "stress-bank",
        "email_type": f"stress_{flavor}",
        "direction": direction,
        "amount": amount.quantize(Decimal("0.01")),
        "currency": currency,
        "transaction_date": date,
        "counterparty": counterparty,
        "card_mask": card_mask,
        "reference_number": reference,
        "channel": "statement" if source == "statement" else "email",
        "category": category,
        "source": source,
    }


def _initial_transaction_rows() -> list[dict]:
    rng = random.Random(SEED - 1)
    return [
        _flavor_row(
            txn_id=index + 1,
            flavor=flavor,
            group=-1,
            rng=rng,
            source="seed",
        )
        for index, flavor in enumerate(FLAVORS)
    ]


def _build_workload(initial_rows: list[dict]) -> WorkloadPlan:
    rng = random.Random(SEED)
    load_rows: list[dict] = []
    flavors: list[str] = []
    total_inserted = (
        WORKLOAD_DISTRIBUTION["transaction_insert_statement"]
        + WORKLOAD_DISTRIBUTION["transaction_insert_rapid"]
    )
    assert total_inserted % len(FLAVORS) == 0
    for offset in range(total_inserted):
        flavor = FLAVORS[offset % len(FLAVORS)]
        group = offset // len(FLAVORS)
        source = "statement" if offset < 12_000 else "email"
        load_rows.append(
            _flavor_row(
                txn_id=LOAD_START_ID + offset,
                flavor=flavor,
                group=group,
                rng=rng,
                source=source,
            )
        )
        flavors.append(flavor)

    expected = {row["id"]: dict(row) for row in (*initial_rows, *load_rows)}
    card_payment_id = LOAD_START_ID + FLAVORS.index("card_payment")
    fx_id = LOAD_START_ID + FLAVORS.index("fx_usd_debit")
    fee_reversal_id = LOAD_START_ID + FLAVORS.index("credit_fee_reversal")
    self_debit = LOAD_START_ID + FLAVORS.index("self_transfer_debit")
    self_credit = LOAD_START_ID + FLAVORS.index("self_transfer_credit")
    protected = {card_payment_id, fx_id, fee_reversal_id, self_debit, self_credit}

    delete_candidates = [
        row["id"]
        for row in load_rows
        if row["id"] not in protected and row["category"] != "self_transfer"
    ]
    delete_ids = [2, *delete_candidates[:1_499]]
    deleted = set(delete_ids)

    update_candidates = [
        row["id"]
        for row in load_rows
        if row["id"] not in protected
        and row["id"] not in deleted
        and row["category"] != "self_transfer"
    ]
    update_ids = [1, *update_candidates[:3_999]]
    categories = (
        "groceries",
        "refund",
        "cashback_rewards",
        "fees_charges",
        "investment",
        "repayment",
    )
    currencies = ("INR", "USD", "EUR", "JPY")
    updates: list[dict] = []
    for index, txn_id in enumerate(update_ids):
        if txn_id == 1:
            values = {
                "p_id": txn_id,
                "p_account_id": 2,
                "p_card_id": CARD_ID,
                "p_direction": "credit",
                "p_amount": Decimal("321.09"),
                "p_currency": "USD",
                "p_date": CUTOVER + datetime.timedelta(days=3),
                "p_category": "refund",
                "p_reference": "edited-historical-1",
                "p_counterparty": "Historical Edited Refund",
            }
        else:
            values = {
                "p_id": txn_id,
                "p_account_id": SELECTED_ACCOUNT_IDS[index % 3],
                "p_card_id": CARD_ID if index % 3 == 0 else None,
                "p_direction": "credit" if index % 2 else "debit",
                "p_amount": (Decimal(rng.randrange(100, 900_000)) / 100).quantize(
                    Decimal("0.01")
                ),
                "p_currency": currencies[index % len(currencies)],
                "p_date": CUTOVER + datetime.timedelta(days=2 + index % 90),
                "p_category": categories[index % len(categories)],
                "p_reference": f"edit-{SEED}-{txn_id}",
                "p_counterparty": f"Edited Counterparty {rng.randrange(1000):03d}",
            }
        updates.append(values)
        expected[txn_id].update(
            {
                "account_id": values["p_account_id"],
                "card_id": values["p_card_id"],
                "direction": values["p_direction"],
                "amount": values["p_amount"],
                "currency": values["p_currency"],
                "transaction_date": values["p_date"],
                "category": values["p_category"],
                "reference_number": values["p_reference"],
                "counterparty": values["p_counterparty"],
            }
        )
    for txn_id in delete_ids:
        expected.pop(txn_id)

    rolled_back_ids = set(range(800_000, 800_200))
    return WorkloadPlan(
        statement_inserts=load_rows[:12_000],
        rapid_inserts=load_rows[12_000:],
        updates=updates,
        delete_ids=delete_ids,
        expected_transactions=expected,
        deleted_ids=deleted,
        rolled_back_ids=rolled_back_ids,
        flavor_counts=Counter(flavors),
        historical_id=1,
        card_payment_id=card_payment_id,
        fx_id=fx_id,
        fee_reversal_id=fee_reversal_id,
        self_transfer_ids=(self_debit, self_credit),
    )


async def _preflight_ok(_config, *, client=None) -> PreflightReport:
    del client
    return PreflightReport(
        ok=True,
        outcome=None,
        capabilities=SimpleNamespace(ledger_cli="ledger", readonly=False),
        reason=None,
    )


async def _revision(factory: async_sessionmaker) -> int:
    async with factory() as session:
        value = await session.scalar(
            text(
                "SELECT desired_revision FROM extension_sync_state "
                "WHERE extension_id = 'paisa'"
            )
        )
        await session.rollback()
    assert value is not None
    return int(value)


async def _state(factory: async_sessionmaker):
    async with factory() as session:
        result = await read_sync_state(session)
        await session.rollback()
    assert result is not None
    return result


async def _execute_batches(engine, statement, rows: list[dict], size: int) -> int:
    commits = 0
    for start in range(0, len(rows), size):
        async with engine.begin() as connection:
            await connection.execute(statement, rows[start : start + size])
        commits += 1
    return commits


def _rollback_rows(start_id: int, count: int) -> list[dict]:
    return [
        {
            "id": start_id + index,
            "account_id": 1,
            "bank": "rollback-bank",
            "email_type": "rolled_back",
            "direction": "debit",
            "amount": Decimal("1.00"),
            "currency": "INR",
            "transaction_date": CUTOVER + datetime.timedelta(days=10),
            "counterparty": "Must Never Project",
            "reference_number": f"rollback-{start_id + index}",
            "category": "groceries",
            "source": "stress-rollback",
        }
        for index in range(count)
    ]


async def _exercise_rollbacks(engine, factory, base_revision: int) -> None:
    outer_rows = _rollback_rows(800_000, 100)
    async with engine.connect() as connection:
        outer = await connection.begin()
        await connection.execute(Transaction.__table__.insert(), outer_rows)
        own_revision = await connection.scalar(
            text(
                "SELECT desired_revision FROM extension_sync_state "
                "WHERE extension_id = 'paisa'"
            )
        )
        assert own_revision == base_revision + 100
        assert await _revision(factory) == base_revision
        await outer.rollback()
    assert await _revision(factory) == base_revision

    savepoint_rows = _rollback_rows(800_100, 100)
    async with engine.connect() as connection:
        outer = await connection.begin()
        savepoint = await connection.begin_nested()
        await connection.execute(Transaction.__table__.insert(), savepoint_rows)
        own_revision = await connection.scalar(
            text(
                "SELECT desired_revision FROM extension_sync_state "
                "WHERE extension_id = 'paisa'"
            )
        )
        assert own_revision == base_revision + 100
        await savepoint.rollback()
        own_after_rollback = await connection.scalar(
            text(
                "SELECT desired_revision FROM extension_sync_state "
                "WHERE extension_id = 'paisa'"
            )
        )
        assert own_after_rollback == base_revision
        await outer.commit()
    assert await _revision(factory) == base_revision


async def _rapid_insert_producer(engine, rows: list[dict]) -> int:
    return await _execute_batches(engine, Transaction.__table__.insert(), rows, 100)


async def _update_producer(engine, rows: list[dict]) -> int:
    table = Transaction.__table__
    statement = (
        table.update()
        .where(table.c.id == bindparam("p_id"))
        .values(
            account_id=bindparam("p_account_id"),
            card_id=bindparam("p_card_id"),
            direction=bindparam("p_direction"),
            amount=bindparam("p_amount"),
            currency=bindparam("p_currency"),
            transaction_date=bindparam("p_date"),
            category=bindparam("p_category"),
            reference_number=bindparam("p_reference"),
            counterparty=bindparam("p_counterparty"),
        )
    )
    return await _execute_batches(engine, statement, rows, 200)


async def _delete_producer(engine, ids: list[int]) -> int:
    statement = Transaction.__table__.delete().where(
        Transaction.id == bindparam("p_id")
    )
    rows = [{"p_id": txn_id} for txn_id in ids]
    return await _execute_batches(engine, statement, rows, 100)


async def _misc_producer(engine) -> int:
    commits = 0
    account_rows = [
        {"p_id": index % 4 + 1, "p_label": f"stress-account-{index:03d}"}
        for index in range(80)
    ]
    account_update = (
        Account.__table__.update()
        .where(Account.id == bindparam("p_id"))
        .values(label=bindparam("p_label"))
    )
    commits += await _execute_batches(engine, account_update, account_rows, 20)

    card_rows = [
        {
            "p_id": CARD_ID + index % 2,
            "p_label": f"stress-card-{index:03d}",
            "p_mask": f"****S{index:04d}",
            "p_active": bool(index % 2),
        }
        for index in range(80)
    ]
    card_update = (
        Card.__table__.update()
        .where(Card.id == bindparam("p_id"))
        .values(
            label=bindparam("p_label"),
            card_mask=bindparam("p_mask"),
            active=bindparam("p_active"),
        )
    )
    commits += await _execute_batches(engine, card_update, card_rows, 20)

    snapshot_insert = sqlite_insert(BalanceSnapshot.__table__)
    snapshot_upsert = snapshot_insert.on_conflict_do_update(
        index_elements=["account_id", "category", "as_of_date"],
        index_where=BalanceSnapshot.account_id.is_not(None),
        set_={
            "value": snapshot_insert.excluded.value,
            "source": snapshot_insert.excluded.source,
        },
    )
    snapshot_rows = []
    for index in range(120):
        account_id = index % 3 + 1
        snapshot_rows.append(
            {
                "account_id": account_id,
                "kind": "liability" if account_id == 3 else "asset",
                "category": ("cc_outstanding" if account_id == 3 else "bank_balance"),
                "as_of_date": CUTOVER - datetime.timedelta(days=1),
                "value": Decimal(50_000 + index),
                "source": "stress_upsert",
                "currency": "INR",
            }
        )
    commits += await _execute_batches(engine, snapshot_upsert, snapshot_rows, 20)

    setting_update = (
        Setting.__table__.update()
        .where(Setting.key == bindparam("p_key"))
        .values(value=bindparam("p_value"))
    )
    setting_rows = [
        {
            "p_key": "paisa.auto_sync_min_interval_minutes",
            "p_value": str(1 + index % 2),
        }
        for index in range(120)
    ]
    commits += await _execute_batches(engine, setting_update, setting_rows, 20)

    cas_rows = [
        {
            "id": 90_000 + index,
            "portfolio_key": f"stress-pf-{index}",
            "depository_source": "nsdl",
            "statement_date": CUTOVER + datetime.timedelta(days=index),
            "grand_total": Decimal(10_000 + index),
            "portfolio_ok": True,
            "raw_holdings_json": "{}",
        }
        for index in range(15)
    ]
    async with engine.begin() as connection:
        await connection.execute(CasUpload.__table__.insert(), cas_rows)
    commits += 1
    cas_update = (
        CasUpload.__table__.update()
        .where(CasUpload.id == bindparam("p_id"))
        .values(grand_total=bindparam("p_total"))
    )
    async with engine.begin() as connection:
        await connection.execute(
            cas_update,
            [
                {"p_id": 90_000 + index, "p_total": Decimal(20_000 + index)}
                for index in range(15)
            ],
        )
    commits += 1

    lot_rows = [
        {
            "id": 91_000 + index,
            "cas_upload_id": 90_014,
            "instrument_id": f"INE{index:09d}",
            "instrument_name": f"Stress Fund {index}",
            "quantity": Decimal("10.000000"),
            "unit_cost": Decimal("100.000000"),
            "cost_basis": Decimal("1000.0000"),
            "currency": "INR",
            "acquired_on": CUTOVER + datetime.timedelta(days=index),
            "source_ref": f"stress-lot-{index}",
            "reference": f"stress-lot-ref-{index}",
        }
        for index in range(20)
    ]
    async with engine.begin() as connection:
        await connection.execute(InvestmentLot.__table__.insert(), lot_rows)
    commits += 1
    lot_update = (
        InvestmentLot.__table__.update()
        .where(InvestmentLot.id == bindparam("p_id"))
        .values(instrument_name=bindparam("p_name"))
    )
    async with engine.begin() as connection:
        await connection.execute(
            lot_update,
            [
                {"p_id": 91_000 + index, "p_name": f"Edited Stress Fund {index}"}
                for index in range(20)
            ],
        )
    commits += 1
    async with engine.begin() as connection:
        await connection.execute(
            InvestmentLot.__table__.delete().where(
                InvestmentLot.id.in_(range(91_000, 91_020))
            )
        )
    commits += 1
    async with engine.begin() as connection:
        await connection.execute(
            CasUpload.__table__.delete().where(CasUpload.id.in_(range(90_000, 90_010)))
        )
    commits += 1
    return commits


def _body_from_published(path: Path) -> str:
    _header, separator, body = path.read_text().partition("\n\n")
    assert separator == "\n\n"
    return body


def _parse_ledger(journal: str) -> tuple[list[ParsedLedgerEntry], list[tuple]]:
    entries: list[ParsedLedgerEntry] = []
    prices: list[tuple] = []
    lines = journal.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("P "):
            parts = line.replace('"', "").split()
            prices.append(
                (
                    datetime.date.fromisoformat(parts[1]),
                    parts[2],
                    Decimal(parts[3]),
                    parts[4],
                )
            )
            index += 1
            continue
        match = re.match(r"^(\d{4}-\d{2}-\d{2}) \* (.*)$", line)
        if match is None:
            index += 1
            continue
        date = datetime.date.fromisoformat(match.group(1))
        header = match.group(2)
        meta: dict[str, str] = {}
        postings: list[tuple[str, Decimal, str]] = []
        index += 1
        while index < len(lines) and lines[index]:
            child = lines[index]
            stripped = child.strip()
            if stripped.startswith(";"):
                tag = stripped[1:].strip()
                if ": " in tag:
                    key, value = tag.split(": ", 1)
                    meta[key] = value
            elif child.startswith("    "):
                posting_parts = re.split(r"\s{2,}", stripped, maxsplit=1)
                assert len(posting_parts) == 2, child
                amount_parts = posting_parts[1].replace('"', "").split()
                postings.append(
                    (
                        posting_parts[0],
                        Decimal(amount_parts[0]),
                        amount_parts[1],
                    )
                )
            index += 1
        entries.append(ParsedLedgerEntry(date, header, meta, tuple(postings)))
        index += 1
    return entries, prices


def _entry_txn_ids(entry: ParsedLedgerEntry) -> tuple[int, ...]:
    raw = entry.meta.get("dashboard_txn_ids", "")
    return tuple(int(token.removeprefix("txn-")) for token in raw.split("|") if token)


def _assert_sign_and_metadata_contract(
    entries: list[ParsedLedgerEntry], prices: list[tuple], plan: WorkloadPlan
) -> None:
    by_id = {txn_id: entry for entry in entries for txn_id in _entry_txn_ids(entry)}
    historical = by_id[plan.historical_id]
    assert historical.date == CUTOVER + datetime.timedelta(days=3)
    assert historical.meta["dashboard_kind"] == "contra_expense"
    assert historical.meta["dashboard_category"] == "refund"
    assert historical.meta["dashboard_reference"] == "edited-historical-1"
    assert historical.meta["dashboard_account_ids"] == "2"
    assert historical.meta["dashboard_card_ids"] == str(CARD_ID)
    assert "Historical Edited Refund" in historical.header
    assert (
        "Assets:Bank:Axis:Savings",
        Decimal("321.09"),
        "USD",
    ) in historical.postings
    assert any(
        account.startswith("Expenses:Refund")
        and amount == Decimal("-321.09")
        and commodity == "USD"
        for account, amount, commodity in historical.postings
    )

    card_payment = by_id[plan.card_payment_id]
    assert card_payment.meta["dashboard_kind"] == "card_payment"
    assert card_payment.meta["dashboard_card_resolution"] == "resolved"
    assert card_payment.meta["dashboard_card_ids"] == str(CARD_ID)
    assert card_payment.meta["dashboard_account_ids"] == "1|3"
    assert any(
        account == "Liabilities:CreditCard:ICICI" and amount > 0
        for account, amount, _commodity in card_payment.postings
    )
    assert any(
        account == "Assets:Bank:HDFC:Salary" and amount < 0
        for account, amount, _commodity in card_payment.postings
    )

    fx = by_id[plan.fx_id]
    assert fx.meta["dashboard_category"] == "travel"
    assert {commodity for _account, _amount, commodity in fx.postings} == {"USD"}
    assert any(
        currency == "USD" and rate == Decimal("83.2500") and unit == "INR"
        for _date, currency, rate, unit in prices
    )

    fee_reversal = by_id[plan.fee_reversal_id]
    assert fee_reversal.meta["dashboard_kind"] == "expense"
    assert fee_reversal.meta["dashboard_category"] == "fees_charges"
    assert any(
        account.startswith("Expenses:Fees Charges") and amount < 0
        for account, amount, _commodity in fee_reversal.postings
    )

    transfer = by_id[plan.self_transfer_ids[0]]
    assert _entry_txn_ids(transfer) == tuple(sorted(plan.self_transfer_ids))
    assert transfer.meta["dashboard_kind"] == "self_transfer"
    assert any(
        account == "Assets:Bank:HDFC:Salary" and amount < 0
        for account, amount, _commodity in transfer.postings
    )
    assert any(
        account == "Assets:Bank:Axis:Savings" and amount > 0
        for account, amount, _commodity in transfer.postings
    )


def _transaction_identity(row: Transaction) -> tuple:
    return (
        row.account_id,
        row.card_id,
        row.bank,
        row.email_type,
        row.direction,
        Decimal(row.amount),
        row.currency,
        row.transaction_date,
        row.counterparty,
        row.card_mask,
        row.reference_number,
        row.channel,
        row.category,
        row.source,
    )


def _expected_transaction_identity(row: dict) -> tuple:
    return (
        row.get("account_id"),
        row.get("card_id"),
        row["bank"],
        row["email_type"],
        row["direction"],
        Decimal(row["amount"]),
        row.get("currency"),
        row.get("transaction_date"),
        row.get("counterparty"),
        row.get("card_mask"),
        row.get("reference_number"),
        row.get("channel"),
        row.get("category"),
        row.get("source"),
    )


async def _assert_expected_database(factory, plan: WorkloadPlan) -> None:
    async with factory() as session:
        transaction_rows = (
            (await session.execute(select(Transaction).order_by(Transaction.id)))
            .scalars()
            .all()
        )
        actual = {row.id: _transaction_identity(row) for row in transaction_rows}
        expected = {
            txn_id: _expected_transaction_identity(row)
            for txn_id, row in plan.expected_transactions.items()
        }
        assert actual == expected

        accounts = {
            row.id: row.label
            for row in (await session.execute(select(Account))).scalars().all()
        }
        assert accounts == {
            1: "stress-account-076",
            2: "stress-account-077",
            3: "stress-account-078",
            4: "stress-account-079",
        }
        cards = {
            row.id: (row.label, row.card_mask, row.active)
            for row in (await session.execute(select(Card))).scalars().all()
        }
        assert cards == {
            101: ("stress-card-078", "****S0078", False),
            102: ("stress-card-079", "****S0079", True),
        }
        snapshots = (
            (
                await session.execute(
                    select(BalanceSnapshot).order_by(BalanceSnapshot.account_id)
                )
            )
            .scalars()
            .all()
        )
        assert [
            (row.account_id, Decimal(row.value), row.source) for row in snapshots
        ] == [
            (1, Decimal("50117.00"), "stress_upsert"),
            (2, Decimal("50118.00"), "stress_upsert"),
            (3, Decimal("50119.00"), "stress_upsert"),
        ]
        assert (
            await session.scalar(
                select(Setting.value).where(
                    Setting.key == "paisa.auto_sync_min_interval_minutes"
                )
            )
            == "2"
        )
        cas_ids = set((await session.scalars(select(CasUpload.id))).all())
        assert cas_ids == set(range(90_010, 90_015))
        assert (
            await session.scalar(select(func.count()).select_from(InvestmentLot)) == 0
        )
        await session.rollback()


def _validate_ledger_cli(path: Path) -> str:
    ledger = shutil.which("ledger")
    if ledger is None:
        return "not-installed"
    completed = subprocess.run(
        [ledger, "-f", str(path), "balanced"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return "passed"


async def _validate_beancount_identity(factory, config, ledger_ids: Counter) -> None:
    from beancount import loader
    from beancount.core.data import Transaction as BeanTransaction

    async with factory() as session:
        report = await project(session, config)
        await session.rollback()
    entries, errors, _options = loader.load_string(report.journal)
    assert errors == [], errors
    bean_ids: Counter = Counter()
    for entry in entries:
        if not isinstance(entry, BeanTransaction):
            continue
        raw = entry.meta.get("dashboard_txn_ids")
        if not isinstance(raw, str):
            continue
        for token in raw.split("|"):
            bean_ids[int(token.removeprefix("txn-"))] += 1
    assert bean_ids == ledger_ids


@pytest.mark.anyio
@pytest.mark.skipif(
    os.environ.get(STRESS_ENV) != "1",
    reason=f"set {STRESS_ENV}=1 to run the 20,000-event coordinator stress gate",
)
async def test_20k_mixed_mutations_converge_to_exact_final_ledger(
    tmp_path, monkeypatch
):
    from financial_dashboard.services import settings as settings_module
    from financial_dashboard.services.categorization import merchant_rules

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(settings_module, "load_all_settings", noop)
    monkeypatch.setattr(merchant_rules, "load_merchant_rules", noop)

    db_path = tmp_path / "paisa-coordinator-stress.db"
    generated_path = tmp_path / "paisa-generated.ledger"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    @event.listens_for(engine.sync_engine, "connect")
    def configure_sqlite(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        await init_db(engine)
        initial_rows = _initial_transaction_rows()
        plan = _build_workload(initial_rows)
        assert len(plan.statement_inserts) == 12_000
        assert len(plan.rapid_inserts) == 1_800
        assert len(plan.updates) == 4_000
        assert len(plan.delete_ids) == 1_500
        assert plan.flavor_counts == Counter({flavor: 1_150 for flavor in FLAVORS})

        async with engine.begin() as connection:
            await connection.execute(
                Account.__table__.insert(),
                [
                    {
                        "id": 1,
                        "bank": "hdfc",
                        "label": "Salary",
                        "type": "bank_account",
                    },
                    {
                        "id": 2,
                        "bank": "axis",
                        "label": "Savings",
                        "type": "bank_account",
                    },
                    {
                        "id": 3,
                        "bank": "icici",
                        "label": "Credit Card",
                        "type": "credit_card",
                    },
                    {
                        "id": 4,
                        "bank": "sbi",
                        "label": "Unselected",
                        "type": "bank_account",
                    },
                ],
            )
            await connection.execute(
                Card.__table__.insert(),
                [
                    {
                        "id": CARD_ID,
                        "account_id": 3,
                        "card_mask": "****1111",
                        "label": "Primary",
                        "active": True,
                    },
                    {
                        "id": CARD_ID + 1,
                        "account_id": 3,
                        "card_mask": "****2222",
                        "label": "Backup",
                        "active": True,
                    },
                ],
            )
            await connection.execute(
                BalanceSnapshot.__table__.insert(),
                [
                    {
                        "id": 201,
                        "account_id": 1,
                        "kind": "asset",
                        "category": "bank_balance",
                        "as_of_date": CUTOVER - datetime.timedelta(days=1),
                        "value": Decimal("100000.00"),
                        "source": "bank_statement",
                        "currency": "INR",
                    },
                    {
                        "id": 202,
                        "account_id": 2,
                        "kind": "asset",
                        "category": "bank_balance",
                        "as_of_date": CUTOVER - datetime.timedelta(days=1),
                        "value": Decimal("50000.00"),
                        "source": "bank_statement",
                        "currency": "INR",
                    },
                    {
                        "id": 203,
                        "account_id": 3,
                        "kind": "liability",
                        "category": "cc_outstanding",
                        "as_of_date": CUTOVER - datetime.timedelta(days=1),
                        "value": Decimal("12500.00"),
                        "source": "cc_statement",
                        "currency": "INR",
                    },
                ],
            )
            await connection.execute(
                Setting.__table__.insert(),
                {"key": "paisa.auto_sync_min_interval_minutes", "value": "0"},
            )
            await connection.execute(Transaction.__table__.insert(), initial_rows)

        config = _config(generated_path)
        monkeypatch.setattr(coordinator_module, "load_config", lambda: config)

        def get_setting_bool(key: str, default: bool = False) -> bool:
            if key == "paisa.auto_sync_enabled":
                return True
            if key == "paisa.notify_sync_failures":
                return False
            return default

        monkeypatch.setattr(coordinator_module, "get_setting_bool", get_setting_bool)
        clock = FakeClock(datetime.datetime.now(datetime.UTC))
        remote = FakeRemote()
        generated_results = []

        async def counted_generate(session, cfg):
            result = await generate(session, cfg)
            generated_results.append(result)
            return result

        def make_coordinator(owner: str) -> PaisaCoordinator:
            return PaisaCoordinator(
                session_factory=factory,
                owner=owner,
                now=clock,
                sleep=asyncio.sleep,
                client_factory=lambda _cfg: FakeClient(),
                preflight_fn=_preflight_ok,
                generate_fn=counted_generate,
                sync_remote_fn=remote,
                min_interval_minutes_fn=lambda: 1,
                heartbeat_interval=3_600,
                poll_interval=0.01,
            )

        coordinator = make_coordinator("stress-initial")
        await coordinator._tick()
        assert len(remote.calls) == 1
        initial_state = await _state(factory)
        initial_hash = hashlib.sha256(remote.calls[0][1].encode()).hexdigest()
        assert initial_state.applied_revision == initial_state.desired_revision
        assert initial_state.last_remote_hash == initial_hash
        assert initial_state.last_healthy_hash == initial_hash
        assert generated_path.exists()
        base_revision = initial_state.desired_revision

        # Shape A: sixty 200-row statement chunks in ONE outer transaction.
        burst_started = time.perf_counter()
        async with engine.connect() as writer:
            transaction = await writer.begin()
            for start in range(0, 12_000, 200):
                await writer.execute(
                    Transaction.__table__.insert(),
                    plan.statement_inserts[start : start + 200],
                )
            own_revision = await writer.scalar(
                text(
                    "SELECT desired_revision FROM extension_sync_state "
                    "WHERE extension_id = 'paisa'"
                )
            )
            assert own_revision == base_revision + 12_000
            assert await _revision(factory) == base_revision
            await coordinator._tick()
            assert len(remote.calls) == 1  # no pre-commit coordinator visibility
            await transaction.commit()
        burst_seconds = time.perf_counter() - burst_started
        assert await _revision(factory) == base_revision + 12_000

        # Start a reconcile and block it at the fake remote boundary.  Its target
        # and published journal cover the statement burst, not the commits below.
        clock.advance(seconds=61)
        in_flight = asyncio.create_task(coordinator._tick())
        await asyncio.wait_for(remote.blocked_attempt_entered.wait(), timeout=60)
        assert len(remote.calls) == 2

        # Both rollback shapes execute real triggers locally but publish no bump.
        producer_started = time.perf_counter()
        await _exercise_rollbacks(engine, factory, base_revision + 12_000)

        # Shape B: four concurrent producers, 79 rapid commits while the first
        # remote attempt is in flight.  SQLite serializes writers truthfully;
        # concurrency here tests coordination/visibility, not parallel writes.
        producer_commits = await asyncio.gather(
            _rapid_insert_producer(engine, plan.rapid_inserts),
            _update_producer(engine, plan.updates),
            _delete_producer(engine, plan.delete_ids),
            _misc_producer(engine),
        )
        producer_seconds = time.perf_counter() - producer_started
        assert sum(producer_commits) >= 70
        final_desired = base_revision + COMMITTED_EVENTS
        assert await _revision(factory) == final_desired
        mutation_seconds = burst_seconds + producer_seconds
        committed_events_per_second = COMMITTED_EVENTS / mutation_seconds

        remote.release_blocked_attempt.set()
        await asyncio.wait_for(in_flight, timeout=60)
        after_in_flight = await _state(factory)
        assert after_in_flight.desired_revision == final_desired
        assert after_in_flight.applied_revision == base_revision + 12_000
        assert after_in_flight.desired_revision > after_in_flight.applied_revision
        assert after_in_flight.lease_token is None

        # The follow-up publishes final local bytes but fails before remote
        # acceptance.  last_remote_hash must stay at the in-flight success hash.
        clock.advance(seconds=61)
        await coordinator._tick()
        assert len(remote.calls) == 3
        failed_hash = remote.calls[2][0]
        failed_state = await _state(factory)
        assert failed_state.desired_revision == final_desired
        assert failed_state.applied_revision < failed_state.desired_revision
        assert failed_state.last_published_hash == failed_hash
        assert failed_state.last_remote_hash == remote.calls[1][0]
        assert failed_state.last_remote_hash != failed_hash
        assert failed_state.failure_count == 1
        assert failed_state.next_attempt_at is not None
        failed_body = _body_from_published(generated_path)

        # Simulate process restart.  The regenerated file is unchanged and the
        # publisher skips its write, but retry MUST POST because comparison is
        # against last_remote_hash rather than last_published_hash/local mtime.
        restarted = make_coordinator("stress-restarted")
        clock.advance(seconds=61)
        await restarted._tick()
        assert len(remote.calls) == 4
        assert remote.calls[3][0] == failed_hash
        assert remote.calls[3][1] == failed_body
        assert generated_results[3].publish is not None
        assert generated_results[3].publish.published is False

        final_state = await _state(factory)
        assert final_state.desired_revision == final_desired
        assert final_state.applied_revision == final_desired
        assert final_state.lease_owner is None
        assert final_state.lease_token is None
        assert final_state.lease_expires_at is None
        assert final_state.failure_count == 0
        assert final_state.next_attempt_at is None
        assert final_state.diagnosis_state == "healthy"

        # Independent clean projection proves final file/remote parity by bytes
        # and SHA-256, not merely by coordinator state fields.
        async with factory() as session:
            clean_report = await project(session, config)
            await session.rollback()
        published_body = _body_from_published(generated_path)
        final_hash = hashlib.sha256(clean_report.journal.encode()).hexdigest()
        assert published_body == clean_report.journal == remote.calls[-1][1]
        assert hashlib.sha256(published_body.encode()).hexdigest() == final_hash
        assert final_state.last_published_hash == final_hash
        assert final_state.last_remote_hash == final_hash
        assert final_state.last_healthy_hash == final_hash

        parsed_entries, parsed_prices = _parse_ledger(published_body)
        parsed_id_counts: Counter = Counter(
            txn_id for entry in parsed_entries for txn_id in _entry_txn_ids(entry)
        )
        skipped = {
            row.txn_id: row.reason
            for row in clean_report.skipped
            if row.txn_id is not None
        }
        relevant_ids = {
            txn_id
            for txn_id, row in plan.expected_transactions.items()
            if row.get("account_id") in SELECTED_ACCOUNT_IDS
            and row.get("transaction_date") is not None
            and row["transaction_date"] > CUTOVER
        }
        assert set(parsed_id_counts).isdisjoint(skipped)
        assert set(parsed_id_counts) | set(skipped) == relevant_ids
        assert all(count == 1 for count in parsed_id_counts.values())
        assert not (plan.deleted_ids & (set(parsed_id_counts) | set(skipped)))
        assert not (plan.rolled_back_ids & (set(parsed_id_counts) | set(skipped)))
        _assert_sign_and_metadata_contract(parsed_entries, parsed_prices, plan)
        await _assert_expected_database(factory, plan)

        # One singleton and four audits/sync attempts (initial + in-flight +
        # failed follow-up + restart retry), never one job/outbox row per event.
        async with factory() as session:
            sync_state_rows = await session.scalar(
                text("SELECT count(*) FROM extension_sync_state")
            )
            audit_rows = await session.scalar(
                select(func.count())
                .select_from(ExtensionRun)
                .where(
                    ExtensionRun.extension_id == "paisa",
                    ExtensionRun.operation == "automatic",
                )
            )
            await session.rollback()
        assert sync_state_rows == 1
        assert audit_rows == 4
        assert len(generated_results) == 4
        assert len(remote.calls) == 4

        ledger_validation = _validate_ledger_cli(generated_path)
        await _validate_beancount_identity(
            factory,
            _config(generated_path, backend="beancount"),
            parsed_id_counts,
        )

        print(
            "\n[paisa-coordinator-stress] "
            f"seed={SEED} attempted_events={ATTEMPTED_EVENTS} "
            f"committed_events={COMMITTED_EVENTS} rolled_back={ROLLED_BACK_EVENTS} "
            f"mutation_seconds={mutation_seconds:.3f} "
            f"committed_events_per_second={committed_events_per_second:.1f} "
            "http_requests=0 http_rps=not-measured "
            f"producer_commits={sum(producer_commits) + 1} "
            f"sync_attempts={len(remote.calls)} emitted={clean_report.emitted_count} "
            f"skipped={len(clean_report.skipped)} final_revision={final_desired} "
            f"final_hash={final_hash} ledger_cli={ledger_validation}",
            flush=True,
        )
    finally:
        await engine.dispose()
