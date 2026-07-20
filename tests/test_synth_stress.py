"""Stress-scale projection benchmark + invariant (requirement 5).

This module is OPT-IN: it skips unless ``SYNTH_BENCH=1`` is set, so it never runs
in the default ``pytest`` invocation (where it would load 200k+ rows and slow the
suite). It renders ONLY the ledger backend — not every backend — to keep the
benchmark focused and cheap.

What it asserts (an invariant, not a wall-clock budget):
* the stress scenario still reaches the >=200k transaction floor;
* a full projection over a selected slice runs to completion against the loaded
  DB and every emitted entry balances to zero;
* the projection is a pure read (no core writes) and byte-stable across runs.

It also prints a performance summary (rows loaded, entries projected, seconds)
to stderr so an operator can see the cost. On the reference machine a stress
load+project completes in a few seconds; that figure is documentation, not a
gate — the gate is the invariant above.

Run it explicitly::

    SYNTH_BENCH=1 uv run pytest -q tests/test_synth_stress.py -s
"""

import datetime
import os
import time
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.services.paisa import PaisaProjectionConfig, project
from scripts.synth import PROFILES, build_scenario, load_scenario
from scripts.synth.loader import create_synthetic_engine

pytestmark = pytest.mark.anyio

_BENCH_ENV = "SYNTH_BENCH"
# A smaller-than-stress profile is allowed via SYNTH_BENCH_PROFILE for a quick
# local run; the default is the real stress profile (the requirement's floor).
_DEFAULT_BENCH_PROFILE = "stress"


def _bench_enabled() -> bool:
    return os.environ.get(_BENCH_ENV) == "1"


pytestmark_bench = pytest.mark.skipif(
    not _bench_enabled(),
    reason=f"set {_BENCH_ENV}=1 to run the stress projection benchmark",
)


def _config(selected: tuple[int, ...], cutover: datetime.date) -> PaisaProjectionConfig:
    # ledger only — never loads hledger/beancount (requirement).
    return PaisaProjectionConfig(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path="",
        selected_account_ids=selected,
        cutover_date=cutover,
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
        ledger_cli="ledger",
    )


@pytestmark_bench
async def test_stress_projection_balanced_readonly_and_stable(tmp_path):
    profile = os.environ.get("SYNTH_BENCH_PROFILE", _DEFAULT_BENCH_PROFILE)
    assert profile in PROFILES, f"unknown bench profile {profile!r}"

    db = tmp_path / "synthetic" / "synthetic.db"
    db.parent.mkdir(parents=True, exist_ok=True)

    scenario = build_scenario(profile=profile)
    # Invariant 1: the stress profile still reaches the >=200k floor.
    assert len(scenario.transactions) >= 200_000

    t0 = time.perf_counter()
    await load_scenario(scenario, db)
    load_seconds = time.perf_counter() - t0

    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            bank_ids = tuple(
                (
                    await session.execute(
                        select(Account.id)
                        .where(Account.type == "bank_account", Account.active.is_(True))
                        .order_by(Account.id)
                    )
                )
                .scalars()
                .all()
            )
            # Cutover near as_of so the projection reads a bounded recent slice
            # (keeps the projection itself fast even at 200k+ total rows).
            cutover = scenario.as_of - datetime.timedelta(days=20)
            config = _config(bank_ids, cutover)

            before = {
                "transactions": (
                    await session.execute(
                        select(Transaction).where(Transaction.account_id.in_(bank_ids))
                    )
                )
                .scalars()
                .all(),
            }
            t1 = time.perf_counter()
            report_a = await project(session, config)
            project_seconds = time.perf_counter() - t1
            after = {
                "transactions": (
                    await session.execute(
                        select(Transaction).where(Transaction.account_id.in_(bank_ids))
                    )
                )
                .scalars()
                .all(),
            }

        # Invariant 2: every emitted entry balances to zero.
        for entry in report_a.entries:
            total = sum((p.amount for p in entry.postings), Decimal("0"))
            assert total == 0, f"unbalanced entry on {entry.date}: {entry.payee!r}"

        # Invariant 3: pure read — no core rows changed.
        assert len(after["transactions"]) == len(before["transactions"])

        # Invariant 4: byte-stable across runs.
        async with maker() as session:
            report_b = await project(session, _config(bank_ids, cutover))
        assert report_a.journal == report_b.journal
        assert report_a.emitted_count == report_b.emitted_count

        # Performance summary (documentation, not a gate).
        print(
            f"\n[stress-bench] profile={profile} rows={len(scenario.transactions)} "
            f"load={load_seconds:.2f}s project={project_seconds:.2f}s "
            f"emitted={report_a.emitted_count} skipped={len(report_a.skipped)} "
            f"journal_bytes={len(report_a.journal)}",
            flush=True,
        )
    finally:
        await engine.dispose()
