"""CAS canonicalization + offline statement reconciliation fidelity.

Exercises the production CAS ingestion guard (NSDL-embeds-CDSL conflict),
same-PAN-different-date canonicalization, multiple-PAN counting, and the real
``reconcile_statement`` / ``reconcile_bank_statement`` paths over scenario-
derived rows — all offline (no PDF parse, no network).

These complement ``test_synth_coverage`` (the branch-registry contract) and
``test_synth_read_parity`` (the read services) by pinning the CAS + statement
behaviours the canonical scenario is shaped to exercise.
"""

import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db.models import CasUpload
from financial_dashboard.services.cas_ingestion import (
    CasIngestError,
    ingest_cas_payload,
)
from scripts.synth import build_scenario, load_scenario
from scripts.synth.loader import create_synthetic_engine
from scripts.synth.reconcile import reconcile_bank_offline, reconcile_cc_offline

pytestmark = pytest.mark.anyio


def _synthetic_db(tmp_path, name="synthetic.db"):
    p = tmp_path / "synthetic" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _cas_payload(*, depository: str, pan: str, period_end, grand_total="50000.00"):
    return {
        "file": "synthetic.pdf",
        "meta": {
            "source": depository,
            "investor_name": "Synthetic Investor",
            "pan": pan,
            "statement_period_start": "2026-04-01",
            "statement_period_end": period_end.isoformat(),
            "generated_on": period_end.isoformat(),
        },
        "accounts": [
            {
                "depository": depository.upper(),
                "dp_id": "12088700",
                "client_id": "00000001",
                "dp_name": f"Synthetic {depository.upper()} DP",
                "total_value": grand_total,
                "holdings": [
                    {
                        "name": "Synthetic Equity",
                        "isin": "INE000A01020",
                        "asset_class": "equity",
                        "quantity": "100",
                        "price": grand_total,
                        "value": grand_total,
                        "flags": [],
                        "notes": None,
                    }
                ],
            }
        ],
        "folios": [],
        "transactions": [],
        "summary": {
            "asset_class_totals": {"Equity": grand_total},
            "grand_total": grand_total,
        },
        "reconciliation": {
            "portfolio_ok": True,
            "portfolio_delta": "0.00",
            "holdings": [],
            "warnings": [],
        },
    }


# ---------------------------------------------------------------------------
# CAS: multiple PANs, same-PAN different date, NSDL/CDSL conflict
# ---------------------------------------------------------------------------


def test_scenario_seeds_multiple_pans_and_depositories():
    s = build_scenario(profile="smoke")
    pans = {cas.portfolio_key for cas in s.cas_uploads}
    depos = {cas.depository_source for cas in s.cas_uploads}
    assert len(pans) >= 2, f"expected multiple PANs, got {sorted(pans)}"
    assert {"nsdl", "cdsl"} <= depos


async def test_same_pan_different_date_creates_distinct_uploads(tmp_path):
    db = _synthetic_db(tmp_path)
    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            d1 = datetime.date(2026, 5, 31)
            d2 = datetime.date(2026, 6, 30)
            await ingest_cas_payload(
                session,
                _cas_payload(depository="nsdl", pan="CONFLICT1F", period_end=d1),
            )
            await ingest_cas_payload(
                session,
                _cas_payload(depository="nsdl", pan="CONFLICT1F", period_end=d2),
            )
            await session.commit()
        async with maker() as session:
            uploads = (
                (
                    await session.execute(
                        select(CasUpload).where(CasUpload.portfolio_key == "CONFLICT1F")
                    )
                )
                .scalars()
                .all()
            )
            assert len(uploads) == 2
            dates = sorted(u.statement_date for u in uploads)
            assert dates == [d1, d2]
    finally:
        await engine.dispose()


async def test_nsdl_then_cdsl_same_pan_same_date_conflict_is_refused(tmp_path):
    """The canonical NSDL-embeds-CDSL guard: a CDSL upload for a PAN+date that
    already carries an NSDL upload is refused (not silently overwritten)."""
    db = _synthetic_db(tmp_path)
    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        d = datetime.date(2026, 6, 30)
        async with maker() as session:
            await ingest_cas_payload(
                session, _cas_payload(depository="nsdl", pan="CONFLICT2F", period_end=d)
            )
            await session.commit()
        async with maker() as session:
            with pytest.raises(CasIngestError, match="NSDL"):
                await ingest_cas_payload(
                    session,
                    _cas_payload(depository="cdsl", pan="CONFLICT2F", period_end=d),
                )
    finally:
        await engine.dispose()


async def test_force_replace_overrides_conflict_guard(tmp_path):
    db = _synthetic_db(tmp_path)
    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        d = datetime.date(2026, 6, 30)
        async with maker() as session:
            await ingest_cas_payload(
                session, _cas_payload(depository="nsdl", pan="CONFLICT3F", period_end=d)
            )
            await session.commit()
        async with maker() as session:
            # force_replace bypasses the guard and replaces the upload.
            await ingest_cas_payload(
                session,
                _cas_payload(depository="cdsl", pan="CONFLICT3F", period_end=d),
                force_replace=True,
            )
            await session.commit()
        async with maker() as session:
            uploads = (
                (
                    await session.execute(
                        select(CasUpload).where(CasUpload.portfolio_key == "CONFLICT3F")
                    )
                )
                .scalars()
                .all()
            )
            assert len(uploads) == 1
            assert uploads[0].depository_source == "cdsl"
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Offline statement reconciliation fidelity (real reconcile service)
# ---------------------------------------------------------------------------


def test_cc_reconciliation_produced_by_real_service():
    """The scenario's CC statement ``reconciliation_data`` is produced by the
    real ``reconcile_statement`` over scenario-derived rows, not hand-invented.
    The in-window CC purchase exact-matches its DB stand-in."""
    s = build_scenario(profile="smoke")
    cc_stmt = next(
        st for st in s.statement_uploads if st.card_number and st.reconciliation_data
    )
    recon = cc_stmt.reconciliation_data
    assert recon is not None
    assert isinstance(recon["matched"], list)
    assert isinstance(recon["missing"], list)
    # The in-window CC purchase exact-matches → at least one matched row.
    assert len(recon["matched"]) >= 1


def test_bank_reconciliation_produced_by_real_service():
    """The bank statement ``reconciliation_data`` is produced by the real
    ``reconcile_bank_statement``; reference-bearing rows match on reference."""
    s = build_scenario(profile="smoke")
    bank_stmt = next(
        st
        for st in s.statement_uploads
        if not st.card_number and st.reconciliation_data
    )
    recon = bank_stmt.reconciliation_data
    assert recon is not None
    assert "matched" in recon and "missing" in recon
    # The bank statement links >=1 row in window (the _eligible pass).
    assert bank_stmt.matched_count >= 1


def test_reconcile_offline_handles_no_in_window_rows():
    """The fidelity-boundary helper is robust to an account with no in-window
    transactions: it returns a real (empty-match) reconciliation dict rather
    than raising — generation never breaks on an empty window."""
    s = build_scenario(profile="smoke")
    # An account with debits, but a window far in the future → no rows match.
    cc_acct = next(a for a in s.accounts if a.label == "HDFC Millennia CC")
    future_window = (
        s.as_of + datetime.timedelta(days=365),
        s.as_of + datetime.timedelta(days=395),
    )
    result = reconcile_cc_offline(
        s.transactions,
        account_pk=cc_acct.pk,
        window=future_window,
        period_end=future_window[1],
    )
    assert result is not None
    assert result["matched"] == []
    bank_acct = next(a for a in s.accounts if a.label == "HDFC Savings")
    result_b = reconcile_bank_offline(
        s.transactions,
        account_pk=bank_acct.pk,
        window=future_window,
        closing_balance="0.00",
    )
    assert result_b is not None
    assert result_b["matched"] == []


async def test_loaded_scenario_carries_reconciled_and_unreconciled_cas(tmp_path):
    db = _synthetic_db(tmp_path)
    s = build_scenario(profile="smoke")
    await load_scenario(s, db)
    engine = await create_synthetic_engine(db)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as session:
            all_cas = (await session.execute(select(CasUpload))).scalars().all()
            ok_flags = {c.portfolio_ok for c in all_cas}
            # Both a reconciled and an unreconciled portfolio are present.
            assert True in ok_flags
            assert False in ok_flags
    finally:
        await engine.dispose()
