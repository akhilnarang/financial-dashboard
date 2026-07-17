"""Orchestration: probe / preview / generate / manual-sync wiring, mode gating,
readonly/backend rejection, sync-flow contracts, and the no-core-writes
guarantee."""

import datetime as dt
from decimal import Decimal

import httpx
import pytest
from sqlalchemy import select

from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.integrations.paisa import PaisaClient
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.orchestrator import (
    GenerateResult,
    ProbeReport,
    SyncReport,
    generate,
    manual_sync,
    preview,
    probe,
)

pytestmark = pytest.mark.anyio

CUTOVER = dt.date(2026, 1, 1)
UNUSED_PATH = "/tmp/paisa-test-unused.journal"


def _config(**overrides) -> PaisaProjectionConfig:
    base = dict(
        mode="project",
        base_url="http://127.0.0.1:7500",
        external_url="",
        allow_remote=False,
        auth_username="",
        auth_password="",
        generated_path=UNUSED_PATH,
        selected_account_ids=(1,),
        cutover_date=CUTOVER,
        account_mappings={},
        category_mappings={},
        non_inr_policy="skip",
        request_timeout_seconds=15,
        ledger_cli="ledger",
        fx_rates={},
    )
    base.update(overrides)
    return PaisaProjectionConfig(**base)


def _mock_client(handler) -> PaisaClient:
    return PaisaClient(
        base_url="http://127.0.0.1:7500",
        transport=httpx.MockTransport(handler),
    )


async def _seed_bank(session, *, id=1):
    session.add(Account(id=id, bank="hdfc", label="Savings", type="bank_account"))
    await session.flush()


async def _seed_txn(
    session,
    account_id,
    *,
    direction="debit",
    amount="10.00",
    date=dt.date(2026, 2, 1),
    category="groceries",
):
    session.add(
        Transaction(
            account_id=account_id,
            bank="hdfc",
            email_type="test_account_transaction",
            direction=direction,
            amount=Decimal(amount),
            transaction_date=date,
            category=category,
            counterparty="Store",
        )
    )
    await session.flush()


# ---------------------------------------------------------------------------
# Mode gating
# ---------------------------------------------------------------------------


async def test_disabled_mode_blocks_everything(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    cfg = _config(mode="disabled", generated_path=str(tmp_path / "gen.journal"))

    prev = await preview(session, cfg)
    assert prev.ok is False
    assert prev.reason == "disabled"

    gen = await generate(session, cfg)
    assert gen.ok is False
    assert gen.reason == "disabled"
    assert not (tmp_path / "gen.journal").exists()

    sync = await manual_sync(session, cfg)
    assert sync.ok is False
    assert sync.outcome == "disabled"


async def test_connect_mode_blocks_writes_but_allows_probe(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    cfg = _config(mode="connect", generated_path=str(tmp_path / "gen.journal"))

    # connect may probe but MUST NOT preview/generate/sync.
    prev = await preview(session, cfg)
    assert prev.ok is False
    assert prev.reason == "connect_only"

    gen = await generate(session, cfg)
    assert gen.ok is False
    assert gen.reason == "connect_only"
    assert not (tmp_path / "gen.journal").exists()

    sync = await manual_sync(session, cfg)
    assert sync.ok is False
    assert sync.outcome == "connect_only"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    report = await probe(cfg, client=_mock_client(handler))
    assert isinstance(report, ProbeReport)
    assert report.ok is True
    assert report.capabilities.ledger_cli == "ledger"


async def test_disabled_mode_blocks_probe():
    cfg = _config(mode="disabled")
    report = await probe(cfg)
    assert report.ok is False
    assert report.reason == "disabled"
    assert report.reachable is False


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


async def test_probe_reports_unreachable():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=req)

    report = await probe(_config(mode="connect"), client=_mock_client(handler))
    assert report.ok is False
    assert report.reachable is False


async def test_probe_returns_capabilities_and_diagnosis():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(
                200,
                json={"config": {"ledger_cli": "ledger", "readonly": False}},
            )
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    report = await probe(_config(mode="project"), client=_mock_client(handler))
    assert report.ok is True
    assert report.capabilities.readonly is False
    assert report.diagnosis.ok is True


# ---------------------------------------------------------------------------
# preview / generate (project mode)
# ---------------------------------------------------------------------------


async def test_preview_not_configured_without_cutover(session):
    await _seed_bank(session)
    report = await preview(session, _config(cutover_date=None))
    assert report.ok is False
    assert report.reason == "not_configured"


async def test_preview_not_configured_without_accounts(session):
    await _seed_bank(session)
    report = await preview(session, _config(selected_account_ids=()))
    assert report.ok is False
    assert report.reason == "not_configured"


async def test_preview_returns_projection(session):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    report = await preview(session, _config())
    assert report.ok is True
    assert report.report.emitted_count == 1


async def test_generate_writes_file(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"
    result = await generate(session, _config(generated_path=str(target)))
    assert isinstance(result, GenerateResult)
    assert result.ok is True
    assert target.exists()
    assert result.publish.published is True
    assert "; txn:" in target.read_text()


async def test_generate_without_path_reports_reason(session):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    result = await generate(session, _config(generated_path=""))
    assert result.ok is False
    assert result.reason == "generated_path not configured"


async def test_generate_skips_rewrite_on_identical_content(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"
    cfg = _config(generated_path=str(target))
    first = await generate(session, cfg)
    second = await generate(session, cfg)
    assert first.publish.published is True
    assert second.publish.published is False  # bytes unchanged


# ---------------------------------------------------------------------------
# manual_sync: rejection paths
# ---------------------------------------------------------------------------


async def test_sync_readonly_rejected_before_post(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"

    posts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"readonly": True}})
        if req.url.path == "/api/sync":
            posts["n"] += 1
            return httpx.Response(200, json={"success": True})
        return httpx.Response(200, json={})

    report = await manual_sync(
        session, _config(generated_path=str(target)), client=_mock_client(handler)
    )
    assert isinstance(report, SyncReport)
    assert report.ok is False
    assert report.outcome == "readonly"
    # Critical: a readonly upstream would fake-sync, so we must NOT have POSTed.
    assert posts["n"] == 0
    # And we must NOT have written the file either (probe-first contract).
    assert not target.exists()


async def test_sync_unsupported_backend_rejected(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "hledger"}})
        return httpx.Response(200, json={})

    report = await manual_sync(
        session, _config(generated_path=str(target)), client=_mock_client(handler)
    )
    assert report.ok is False
    assert report.outcome == "unsupported_backend"
    assert not target.exists()


async def test_sync_unreachable(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=req)

    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(handler),
    )
    assert report.ok is False
    assert report.outcome == "unreachable"


async def test_sync_not_configured(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    report = await manual_sync(
        session,
        _config(cutover_date=None, generated_path=str(tmp_path / "gen.journal")),
    )
    assert report.ok is False
    assert report.outcome == "not_configured"


# ---------------------------------------------------------------------------
# manual_sync: happy path + failure isolation
# ---------------------------------------------------------------------------


async def test_sync_happy_path_writes_and_posts(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"

    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    report = await manual_sync(
        session, _config(generated_path=str(target)), client=_mock_client(handler)
    )
    assert report.ok is True
    assert report.outcome == "synced"
    assert report.diagnosis_ok is True
    # Order: config probe, then sync, then diagnosis.
    assert seen == ["/api/config", "/api/sync", "/api/diagnosis"]
    assert target.exists()


async def test_sync_rejects_http200_success_false(session, tmp_path):
    # Paisa returns HTTP 200 with {success: false, message} on a journal-reload
    # failure; that must be reported as a rejected sync, not success.
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(
                200, json={"success": False, "message": "parse error at line 3"}
            )
        return httpx.Response(200, json={"issues": []})

    report = await manual_sync(
        session, _config(generated_path=str(target)), client=_mock_client(handler)
    )
    assert report.ok is False
    assert report.outcome == "sync_rejected"
    # The upstream reason is preserved (sanitized).
    assert "parse error at line 3" in (report.reason or "")
    # The file was written; Paisa just refused the reload.
    assert target.exists()


async def test_sync_diagnosis_danger_fails(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(
                200,
                json={
                    "issues": [
                        {
                            "level": "danger",
                            "summary": "Negative Balance",
                            "details": "Assets:Bank went negative",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(handler),
    )
    assert report.ok is False
    assert report.outcome == "diagnosis_failed"
    assert report.diagnosis_ok is False
    assert "Negative Balance" in (report.reason or "")


async def test_sync_diagnosis_warning_does_not_fail(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(
                200,
                json={
                    "issues": [
                        {"level": "warning", "summary": "Price Mismatch", "details": ""}
                    ]
                },
            )
        return httpx.Response(404)

    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(handler),
    )
    assert report.ok is True
    assert report.outcome == "synced"


# ---------------------------------------------------------------------------
# Diagnosis classification: contra-expense Debit Entry dangers the projection
# provably generated are accepted; everything else stays fatal.
# ---------------------------------------------------------------------------


def _debit_details(amount: str, account: str, date) -> str:
    """Paisa v0.7.4 ``ruleNonDebitAccount`` details template, exact."""
    amt = Decimal(amount).quantize(Decimal("0.0001"))
    return (
        f"<b>{amt:.4f}</b> got debited from <b>{account}</b> on "
        f"{date.strftime('%d %b %Y')}"
    )


async def _seed_contra_expense(session):
    """Seed a bank + snapshot + one refund/cashback/fee-reversal each so the
    projection generates three negative Expenses postings."""
    from financial_dashboard.db.models import BalanceSnapshot

    await _seed_bank(session)
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category="bank_balance",
            as_of_date=CUTOVER,
            value=Decimal("100000.00"),
            source="statement",
            currency="INR",
        )
    )
    # Refund → Expenses:Refund (contra_expense), negative contra.
    await _seed_txn(
        session,
        1,
        direction="credit",
        amount="50.00",
        date=dt.date(2026, 2, 1),
        category="refund",
    )
    # Cashback → Expenses:Cashback Rewards (contra_expense), negative contra.
    await _seed_txn(
        session,
        1,
        direction="credit",
        amount="2916.11",
        date=dt.date(2026, 2, 2),
        category="cashback_rewards",
    )
    # Fee reversal → Expenses:Fees Charges (expense reversal), negative contra.
    await _seed_txn(
        session,
        1,
        direction="credit",
        amount="236.00",
        date=dt.date(2026, 2, 3),
        category="fees_charges",
    )
    await session.flush()


def _diag_handler_with_issues(issues):
    """Mock handler whose /api/diagnosis returns the given issues list."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": issues})
        return httpx.Response(404)

    return handler


async def test_sync_accepts_expected_contra_expense_debit_entries(session, tmp_path):
    """The live-sync failure: Paisa reports ~hundreds of ``Debit Entry`` dangers
    for our legitimate negative Expenses postings (refund/cashback/fee
    reversal). They must be classified as expected (account/date/amount derived
    from the projection) and the sync must succeed with the counts surfaced."""
    await _seed_contra_expense(session)
    issues = [
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details("-50.00", "Expenses:Refund", dt.date(2026, 2, 1)),
        },
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details(
                "-2916.11", "Expenses:Cashback Rewards", dt.date(2026, 2, 2)
            ),
        },
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details(
                "-236.00", "Expenses:Fees Charges", dt.date(2026, 2, 3)
            ),
        },
    ]
    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(_diag_handler_with_issues(issues)),
    )
    # All three dangers matched expected fingerprints → sync succeeds.
    assert report.ok is True
    assert report.outcome == "synced"
    assert report.diagnosis_ok is True
    assert report.diagnosis_expected == 3
    assert report.diagnosis_accepted == 3
    assert report.diagnosis_fatal == 0
    # The file was still written.
    assert (tmp_path / "gen.journal").exists()


async def test_sync_fails_when_debit_entry_exceeds_expected_multiplicity(
    session, tmp_path
):
    """A fourth ``Debit Entry`` for a fingerprint expected once is unmatched and
    fatal — the matching is multiplicity-aware, not set-membership."""
    await _seed_contra_expense(session)
    # Duplicate the refund danger: one expected, two reported → one fatal.
    issues = [
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details("-50.00", "Expenses:Refund", dt.date(2026, 2, 1)),
        },
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details("-50.00", "Expenses:Refund", dt.date(2026, 2, 1)),
        },
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details(
                "-2916.11", "Expenses:Cashback Rewards", dt.date(2026, 2, 2)
            ),
        },
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details(
                "-236.00", "Expenses:Fees Charges", dt.date(2026, 2, 3)
            ),
        },
    ]
    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(_diag_handler_with_issues(issues)),
    )
    assert report.ok is False
    assert report.outcome == "diagnosis_failed"
    assert report.diagnosis_fatal == 1
    assert report.diagnosis_accepted == 3
    assert "Debit Entry" in (report.reason or "")


async def test_sync_fails_on_unmatched_operator_journal_debit_entry(session, tmp_path):
    """A ``Debit Entry`` whose fingerprint is NOT in our projection (an
    operator-authored negative Expenses posting in their own include) stays
    fatal even when our refund is correctly accepted."""
    await _seed_contra_expense(session)
    issues = [
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details("-50.00", "Expenses:Refund", dt.date(2026, 2, 1)),
        },
        {
            "level": "danger",
            "summary": "Debit Entry",
            # Operator's own content — not generated by us.
            "details": _debit_details(
                "-77.00", "Expenses:Hobbies", dt.date(2026, 3, 1)
            ),
        },
    ]
    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(_diag_handler_with_issues(issues)),
    )
    assert report.ok is False
    assert report.outcome == "diagnosis_failed"
    assert report.diagnosis_accepted == 1
    assert report.diagnosis_fatal == 1


async def test_sync_fails_on_negative_balance_alongside_accepted_refund(
    session, tmp_path
):
    """A ``Negative Balance`` danger is never classifiable — it stays fatal
    even when the refund's ``Debit Entry`` is correctly accepted."""
    await _seed_contra_expense(session)
    issues = [
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details("-50.00", "Expenses:Refund", dt.date(2026, 2, 1)),
        },
        {
            "level": "danger",
            "summary": "Negative Balance",
            "details": (
                "<b>Assets:Bank:Hdfc:Savings</b> account went negative "
                "(-100.00) on 01 Feb 2026"
            ),
        },
    ]
    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(_diag_handler_with_issues(issues)),
    )
    assert report.ok is False
    assert report.outcome == "diagnosis_failed"
    assert report.diagnosis_accepted == 1
    assert report.diagnosis_fatal == 1
    assert "Negative Balance" in (report.reason or "")


async def test_probe_still_surfaces_raw_diagnosis_dangers(session, tmp_path):
    """The probe path is never touched by classification: it surfaces the raw
    upstream diagnosis (dangers and all) so the operator can see everything
    Paisa reports, including contra-expense ``Debit Entry`` dangers."""
    await _seed_contra_expense(session)
    raw_issues = [
        {
            "level": "danger",
            "summary": "Debit Entry",
            "details": _debit_details("-50.00", "Expenses:Refund", dt.date(2026, 2, 1)),
        },
        {
            "level": "danger",
            "summary": "Negative Balance",
            "details": "<b>Assets:Bank</b> account went negative",
        },
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": raw_issues})
        return httpx.Response(404)

    report = await probe(_config(), client=_mock_client(handler))
    assert report.ok is True
    assert report.diagnosis is not None
    # Raw dangers are surfaced verbatim — NOT classified/accepted.
    assert report.diagnosis.danger_count == 2
    summaries = {i.summary for i in report.diagnosis.issues}
    assert summaries == {"Debit Entry", "Negative Balance"}


async def test_sync_upstream_5xx_rejected(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"issues": []})

    report = await manual_sync(
        session, _config(generated_path=str(target)), client=_mock_client(handler)
    )
    assert report.ok is False
    assert report.outcome == "sync_rejected"
    assert target.exists()


# ---------------------------------------------------------------------------
# No core writes — the load-bearing guarantee
# ---------------------------------------------------------------------------


async def test_sync_never_mutates_core_rows(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    txn_ids_before = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    acct_ids_before = [
        a.id for a in (await session.execute(select(Account))).scalars().all()
    ]

    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(handler),
    )
    assert report.ok is True

    txn_ids_after = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    acct_ids_after = [
        a.id for a in (await session.execute(select(Account))).scalars().all()
    ]
    assert txn_ids_before == txn_ids_after
    assert acct_ids_before == acct_ids_after
    assert list(session.sync_session.new) == []
    assert list(session.sync_session.dirty) == []


async def test_failed_sync_never_mutates_core_rows(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=req)

    txn_ids_before = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]

    report = await manual_sync(
        session,
        _config(generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(handler),
    )
    assert report.ok is False

    txn_ids_after = [
        t.id for t in (await session.execute(select(Transaction))).scalars().all()
    ]
    assert txn_ids_before == txn_ids_after
    assert list(session.sync_session.new) == []


async def test_manual_sync_closes_owned_client(session, tmp_path, monkeypatch):
    await _seed_bank(session)
    await _seed_txn(session, 1)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"readonly": True}})
        return httpx.Response(200, json={})

    # Inject a MockTransport-backed client via the builder so the owned-client
    # path exercises real plumbing without a socket, and confirm aclose ran.
    built = {"client": None}

    def fake_build_client(cfg):
        client = PaisaClient(
            base_url=cfg.base_url, transport=httpx.MockTransport(handler)
        )
        built["client"] = client
        return client

    monkeypatch.setattr(
        "financial_dashboard.services.paisa.orchestrator._build_client",
        fake_build_client,
    )

    report = await manual_sync(
        session, _config(generated_path=str(tmp_path / "gen.journal"))
    )
    assert report.outcome == "readonly"
    assert built["client"] is not None
    assert built["client"]._client.is_closed


async def test_connect_mode_sync_blocked_even_with_healthy_client(session, tmp_path):
    await _seed_bank(session)
    await _seed_txn(session, 1)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": "ledger"}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    report = await manual_sync(
        session,
        _config(mode="connect", generated_path=str(tmp_path / "gen.journal")),
        client=_mock_client(handler),
    )
    assert report.ok is False
    assert report.outcome == "connect_only"
    assert not (tmp_path / "gen.journal").exists()


# ---------------------------------------------------------------------------
# Multi-backend: configured backend must equal the probed upstream backend
# ---------------------------------------------------------------------------


def _config_handler(backend: str) -> object:
    """A mock handler whose /api/config advertises ``backend``."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {"ledger_cli": backend}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    return handler


@pytest.mark.parametrize("backend", ["ledger", "hledger", "beancount"])
async def test_sync_succeeds_when_probed_backend_matches_configured(
    session, tmp_path, backend
):
    # Every supported backend syncs when the upstream Paisa advertises the same
    # backend as the configured ``paisa.ledger_cli``.
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"

    report = await manual_sync(
        session,
        _config(ledger_cli=backend, generated_path=str(target)),
        client=_mock_client(_config_handler(backend)),
    )
    assert report.ok is True
    assert report.outcome == "synced"
    assert report.diagnosis_ok is True
    assert target.exists()


@pytest.mark.parametrize(
    "configured, probed",
    [
        ("ledger", "hledger"),
        ("hledger", "ledger"),
        ("ledger", "beancount"),
        ("beancount", "hledger"),
    ],
)
async def test_sync_rejects_backend_mismatch(session, tmp_path, configured, probed):
    # A probed upstream backend that differs from the configured one is a hard
    # mismatch: ledger output does not parse as hledger, etc. The file must not
    # be written and no POST issued.
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"

    posts = {"n": 0}
    probed_handler = _config_handler(probed)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/sync":
            posts["n"] += 1
        return probed_handler(req)  # type: ignore[operator]

    report = await manual_sync(
        session,
        _config(ledger_cli=configured, generated_path=str(target)),
        client=_mock_client(handler),
    )
    assert report.ok is False
    assert report.outcome == "unsupported_backend"
    assert "does not match" in (report.reason or "")
    # Probe-first contract: nothing written, nothing POSTed.
    assert not target.exists()
    assert posts["n"] == 0


async def test_sync_allows_missing_upstream_backend(session, tmp_path):
    # An older Paisa that does not report ``ledger_cli`` is tolerated (we cannot
    # verify a field that is absent); the sync proceeds against the configured
    # backend. This keeps connectivity working for upstreams that predate the
    # field without weakening the mismatch check for upstreams that do report.
    await _seed_bank(session)
    await _seed_txn(session, 1)
    target = tmp_path / "gen.journal"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/config":
            return httpx.Response(200, json={"config": {}})
        if req.url.path == "/api/sync":
            return httpx.Response(200, json={"success": True})
        if req.url.path == "/api/diagnosis":
            return httpx.Response(200, json={"issues": []})
        return httpx.Response(404)

    report = await manual_sync(
        session,
        _config(ledger_cli="ledger", generated_path=str(target)),
        client=_mock_client(handler),
    )
    assert report.ok is True
    assert report.outcome == "synced"
