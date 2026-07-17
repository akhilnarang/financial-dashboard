"""Dashboard taxonomy semantics, canonical metadata schema, card-payment
traceability, investment-funding double-count prevention, and strict
closed-population tests for the Paisa projection.

Covers:
* (1) Category+direction+account-kind semantics: income always Income, expense
  always Expenses (reversals net), refund/cashback contra-expense, investment
  asset movement, repayment equity clearing, self_transfer / card_payment
  special-cased. Every combination tested including fee reversal.
* (2) Canonical ``dashboard_*`` metadata on every emitted entry across
  Ledger/hledger/Beancount. Tags validated structurally and via real
  beancount meta identity (not substring).
* (3) Card payment traceability: explicit card_id / exact mask → specific
  liability; otherwise generic clearing with ``dashboard_card_resolution=
  unresolved``. Never fuzzy.
* (4) Investment funding double-count prevention: provable link remaps the
  bank leg; unprovable suppresses the lot.
* (5) Closed-population: emitted txn-id multiset disjoint from skipped, union
  equals eligible selected post-cutover transactions, no duplicates, kind
  cardinalities, signs trace source rows.
"""

import datetime as dt
from collections import Counter
from decimal import Decimal

import pytest

from financial_dashboard.db.models import (
    Account,
    BalanceSnapshot,
    Card,
    InvestmentLot,
    Transaction,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig
from financial_dashboard.services.paisa.projection import (
    DASHBOARD_KINDS,
    KIND_CONTRA_EXPENSE,
    KIND_EXPENSE,
    KIND_INCOME,
    KIND_INVESTMENT,
    KIND_REPAYMENT,
    KIND_SELF_TRANSFER,
    KIND_UNKNOWN,
    project,
)
from financial_dashboard.services.paisa.renderers import (
    SUPPORTED_BACKENDS,
)

pytestmark = pytest.mark.anyio

CUTOVER = dt.date(2026, 1, 1)


def _config(**overrides) -> PaisaProjectionConfig:
    base = dict(
        mode="project",
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
        ledger_cli="ledger",
        fx_rates={},
    )
    base.update(overrides)
    return PaisaProjectionConfig(**base)


async def _bank(session, *, id=1, bank="hdfc", label="Savings"):
    acct = Account(id=id, bank=bank, label=label, type="bank_account", active=True)
    session.add(acct)
    await session.flush()
    return acct


async def _card(session, *, id=2, bank="icici", label="Platinum"):
    acct = Account(id=id, bank=bank, label=label, type="credit_card", active=True)
    session.add(acct)
    await session.flush()
    return acct


async def _txn(
    session,
    *,
    account_id,
    direction,
    amount,
    date,
    category=None,
    id=None,
    bank="hdfc",
    counterparty=None,
    reference_number=None,
    card_id=None,
    card_mask=None,
    source="email",
    channel="email",
    email_type="test_txn",
    currency="INR",
):
    kwargs = dict(
        account_id=account_id,
        bank=bank,
        email_type=email_type,
        direction=direction,
        amount=Decimal(amount),
        currency=currency,
        transaction_date=date,
        category=category,
        counterparty=counterparty,
        reference_number=reference_number,
        card_id=card_id,
        card_mask=card_mask,
        source=source,
        channel=channel,
    )
    if id is not None:
        kwargs["id"] = id
    session.add(Transaction(**kwargs))
    await session.flush()


async def _snapshot(session, account_id, date, value):
    from financial_dashboard.db.enums import SnapshotCategory, SnapshotSource

    session.add(
        BalanceSnapshot(
            account_id=account_id,
            kind="asset",
            category=SnapshotCategory.bank_balance.value,
            as_of_date=date,
            value=Decimal(value),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await session.flush()


# ===========================================================================
# (1) Dashboard taxonomy semantics: category × direction × account-kind
# ===========================================================================


SEMANTICS_CASES = [
    # (category, direction, account_kind, expected_contra_root, expected_kind)
    # Income slugs → always Income root
    ("salary", "credit", "asset", "Income:", KIND_INCOME),
    ("interest", "credit", "asset", "Income:", KIND_INCOME),
    ("other_income", "credit", "asset", "Income:", KIND_INCOME),
    # Expense slugs → always Expenses root (even on credit = reversal)
    ("groceries", "debit", "asset", "Expenses:", KIND_EXPENSE),
    ("groceries", "credit", "asset", "Expenses:", KIND_EXPENSE),
    ("dining", "debit", "asset", "Expenses:", KIND_EXPENSE),
    ("dining", "credit", "asset", "Expenses:", KIND_EXPENSE),
    # Fee reversal: fees_charges on credit → Expenses (nets, not Income)
    ("fees_charges", "debit", "asset", "Expenses:", KIND_EXPENSE),
    ("fees_charges", "credit", "asset", "Expenses:", KIND_EXPENSE),
    # Refund/cashback → contra-expense (Expenses root, not Income)
    ("refund", "credit", "asset", "Expenses:", KIND_CONTRA_EXPENSE),
    ("cashback_rewards", "credit", "asset", "Expenses:", KIND_CONTRA_EXPENSE),
    # Investment → asset movement (not expense/income)
    ("investment", "debit", "asset", "Assets:Investments:Unallocated", KIND_INVESTMENT),
    (
        "investment_redemption",
        "credit",
        "asset",
        "Assets:Investments:Unallocated",
        KIND_INVESTMENT,
    ),
    # Repayment → equity clearing (not income)
    ("repayment", "credit", "asset", "Equity:Transfers In", KIND_REPAYMENT),
    # Card spend (expense on a liability account)
    ("groceries", "debit", "liability", "Expenses:", KIND_EXPENSE),
    # Unknown/blank → suspense Expenses
    ("unknown", "debit", "asset", "Expenses:Unknown", KIND_UNKNOWN),
    (None, "debit", "asset", "Expenses:Unknown", KIND_UNKNOWN),
]


@pytest.mark.parametrize(
    "category,direction,acct_kind,contra_root,kind", SEMANTICS_CASES
)
async def test_category_direction_semantics(
    session, category, direction, acct_kind, contra_root, kind
):
    """Every category+direction combination roots to the correct contra and
    carries the correct ``dashboard_kind``. Direction affects sign only — a
    credit on an expense slug is a reversal that nets, not a relabel to Income."""
    if acct_kind == "liability":
        await _card(session, id=2)
        aid = 2
        selected = (2,)
    else:
        await _bank(session, id=1)
        aid = 1
        selected = (1,)
    await _txn(
        session,
        account_id=aid,
        direction=direction,
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category=category,
        counterparty="Test",
        id=1,
    )
    report = await project(session, _config(selected_account_ids=selected))
    assert report.emitted_count == 1
    entry = report.entries[0]
    assert entry.kind == kind
    # The contra account matches the expected root.
    contra_posting = entry.postings[1]  # second posting is the contra
    assert (
        contra_posting.account.startswith(contra_root)
        or contra_root in contra_posting.account
    ), f"contra {contra_posting.account!r} does not start with {contra_root!r}"


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_fee_reversal_nets_against_expense(session, backend):
    """A fee debit followed by a fee reversal (credit) must net to zero in the
    Expenses:Fees Charges account — both post to the same root."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="500.00",
        date=dt.date(2026, 2, 1),
        category="fees_charges",
        counterparty="Fee Charge",
        id=1,
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="500.00",
        date=dt.date(2026, 2, 10),
        category="fees_charges",
        counterparty="Fee Reversal",
        id=2,
    )
    report = await project(session, _config(ledger_cli=backend))
    assert report.emitted_count == 2
    # Both entries post to Expenses:Fees Charges (not Income for the reversal).
    for entry in report.entries:
        contra = entry.postings[1].account
        assert "Fees" in contra and "Charge" in contra, contra
        assert contra.startswith("Expenses:"), contra
    # The contra postings net to zero (500 debit + (-500) credit).
    contra_total = sum(
        e.postings[1].amount for e in report.entries if e.kind == KIND_EXPENSE
    )
    assert contra_total == Decimal("0"), contra_total


async def test_expense_on_credit_does_not_become_income(session):
    """The headline semantics change: an expense slug on credit roots to
    Expenses (reversal), NOT to Income. This is what makes reversals net."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="shopping",
        counterparty="Refund Store",
    )
    report = await project(session, _config())
    contra = report.entries[0].postings[1].account
    assert contra.startswith("Expenses:"), f"expense credit rooted to {contra!r}"
    assert "Income:" not in contra


async def test_refund_is_contra_expense_not_income(session):
    """A refund posts to Expenses (contra-expense), never to Income."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="50.00",
        date=dt.date(2026, 2, 1),
        category="refund",
        counterparty="Store Refund",
    )
    report = await project(session, _config())
    entry = report.entries[0]
    assert entry.kind == KIND_CONTRA_EXPENSE
    contra = entry.postings[1].account
    assert "Expenses:" in contra
    assert "Income:" not in contra


async def test_investment_posts_to_asset_not_expense(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="investment",
        counterparty="MF Purchase",
    )
    report = await project(session, _config())
    entry = report.entries[0]
    assert entry.kind == KIND_INVESTMENT
    contra = entry.postings[1].account
    assert contra == "Assets:Investments:Unallocated"
    assert "Expenses:" not in contra
    assert "Income:" not in contra


async def test_repayment_posts_to_equity_not_income(session):
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="repayment",
        counterparty="Friend",
    )
    report = await project(session, _config())
    entry = report.entries[0]
    assert entry.kind == KIND_REPAYMENT
    contra = entry.postings[1].account
    assert contra == "Equity:Transfers In"
    assert "Income:" not in contra


async def test_imprecise_categories_counted(session):
    """emi_loan and cash_withdrawal are imprecise: posted to a conservative
    Expenses clearing and counted in ``imprecise_count`` rather than
    fabricating a principal/cash account."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="emi_loan",
        counterparty="Loan EMI",
        id=1,
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="2000.00",
        date=dt.date(2026, 2, 2),
        category="cash_withdrawal",
        counterparty="ATM",
        id=2,
    )
    report = await project(session, _config())
    assert report.imprecise_count == 2
    assert report.emitted_count == 2
    # Both post to Expenses (conservative clearing, no fabricated loan/cash).
    for entry in report.entries:
        contra = entry.postings[1].account
        assert contra.startswith("Expenses:")


async def test_operator_category_mapping_overrides_semantics(session):
    """Operator ``category_mappings`` overrides win over the taxonomy defaults,
    even for special categories like investment/repayment."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="investment",
        counterparty="Custom",
    )
    report = await project(
        session,
        _config(category_mappings={"investment": "Assets:Custom:Fund"}),
    )
    contra = report.entries[0].postings[1].account
    assert contra == "Assets:Custom:Fund"


# ===========================================================================
# (2) Canonical metadata schema
# ===========================================================================


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_every_entry_carries_dashboard_metadata(session, backend):
    """Every emitted standard entry carries the closed canonical ``dashboard_*``
    metadata schema. For beancount this is validated via ``loader.load_string``
    meta identity (not substring)."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Store",
        source="email",
        channel="email",
        email_type="debit_purchase",
        reference_number="UTR12345",
        id=42,
    )
    report = await project(session, _config(ledger_cli=backend))
    assert report.emitted_count == 1
    entry = report.entries[0]
    meta_keys = {k for k, _ in entry.meta}
    required = {
        "dashboard_txn_ids",
        "dashboard_kind",
        "dashboard_category",
        "dashboard_source",
        "dashboard_channel",
        "dashboard_email_type",
        "dashboard_account_ids",
        "dashboard_card_ids",
        "dashboard_reference",
    }
    assert required <= meta_keys, f"missing: {required - meta_keys}"
    meta_dict = dict(entry.meta)
    assert meta_dict["dashboard_txn_ids"] == "txn-42"
    assert meta_dict["dashboard_kind"] == KIND_EXPENSE
    assert meta_dict["dashboard_category"] == "groceries"
    assert meta_dict["dashboard_reference"] == "UTR12345"


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_beancount_meta_identity_round_trip(session, backend):
    """For beancount: render, parse with ``loader.load_string``, and assert the
    parsed transaction META carries the exact dashboard_* key-value pairs — not
    a substring check but structural identity."""
    if backend != "beancount":
        pytest.skip("beancount-only identity check")
    from beancount import loader

    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Store",
        source="email",
        channel="email",
        email_type="debit_purchase",
        reference_number="UTR99",
        id=7,
    )
    report = await project(session, _config(ledger_cli="beancount"))
    entries, errors, _opts = loader.load_string(report.journal)
    assert errors == [], errors
    # Find the non-opening transaction (the grocery spend).
    txn = next(
        e
        for e in entries
        if hasattr(e, "meta") and e.meta.get("dashboard_kind") not in (None, "opening")
    )
    assert txn.meta["dashboard_txn_ids"] == "txn-7"
    assert txn.meta["dashboard_kind"] == "expense"
    assert txn.meta["dashboard_category"] == "groceries"
    assert txn.meta["dashboard_source"] == "email"
    assert txn.meta["dashboard_channel"] == "email"
    assert txn.meta["dashboard_email_type"] == "debit_purchase"
    assert txn.meta["dashboard_account_ids"] == "1"
    assert txn.meta["dashboard_card_ids"] == "none"
    assert txn.meta["dashboard_reference"] == "UTR99"
    # The backward-compatible txn tag is also present as beancount meta.
    assert txn.meta["txn"] == "7"


# ---------------------------------------------------------------------------
# Beancount meta identity: opening, lot, resolved/unresolved card, FX
# (real loader.load_string meta identity — not substring checks)
# ---------------------------------------------------------------------------

#: Metadata keys that ONLY a transaction-derived entry may carry. Source-less
#: entries (opening/lot) have no dashboard Transaction, so they must never
#: carry these.
_TXN_ONLY_META_KEYS = (
    "dashboard_txn_ids",
    "dashboard_category",
    "dashboard_channel",
    "dashboard_email_type",
    "dashboard_card_ids",
)


def _load_beancount(journal: str):
    """Parse a beancount journal, asserting a clean parse."""
    from beancount import loader

    entries, errors, _opts = loader.load_string(journal)
    assert errors == [], errors
    return entries


async def test_beancount_opening_meta_is_reduced_schema(session):
    """An opening-balance transaction carries the REDUCED posting-level schema
    (account_ids/source/as_of) — never the transaction-derived fields (txn_ids/
    category/channel/email_type/card_ids). Parsed via real beancount meta
    identity, not substring."""
    bank = await _bank(session)
    await _snapshot(session, bank.id, dt.date(2025, 12, 15), "100000.00")
    report = await project(session, _config(ledger_cli="beancount"))
    entries = _load_beancount(report.journal)
    from beancount.core.data import Open, Transaction

    opening = next(
        e
        for e in entries
        if isinstance(e, Transaction) and e.meta.get("dashboard_kind") == "opening"
    )
    # Entry-level: only dashboard_kind, never the transaction-only fields.
    assert opening.meta["dashboard_kind"] == "opening"
    for key in _TXN_ONLY_META_KEYS:
        assert key not in opening.meta, f"opening entry carries {key}"
    # The bank posting carries the reduced posting-level schema; equity does not.
    bank_post = next(p for p in opening.postings if "Assets:" in p.account)
    assert bank_post.meta["dashboard_account_ids"] == "1"
    assert bank_post.meta["dashboard_source"] == "snapshot"
    assert bank_post.meta["dashboard_as_of"] == "2025-12-15"
    for key in _TXN_ONLY_META_KEYS:
        assert key not in bank_post.meta, f"opening posting carries {key}"
    # The Open directives parse cleanly alongside (no errors above).
    assert any(isinstance(e, Open) for e in entries)


async def test_beancount_lot_meta_is_reduced_schema(session):
    """An investment-lot transaction carries the REDUCED entry-level schema
    (kind/instrument/acquired_on + CAS provenance) — never the transaction-only
    fields. Parsed via real beancount meta identity."""
    import json

    from financial_dashboard.db.models import CasUpload

    await _seed_bank_and_snapshot(session)
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
                        }
                    ]
                }
            ),
        )
    )
    await session.flush()
    report = await project(
        session, _config(project_investments=True, ledger_cli="beancount")
    )
    entries = _load_beancount(report.journal)
    from beancount.core.data import Transaction

    lot = next(
        e
        for e in entries
        if isinstance(e, Transaction)
        and e.meta.get("dashboard_kind") == "investment_lot"
    )
    assert lot.meta["dashboard_kind"] == "investment_lot"
    assert lot.meta["dashboard_instrument"] == "INE000A01018"
    assert lot.meta["dashboard_acquired_on"] == "2026-01-15"
    # CAS provenance is carried, transaction-only fields are not.
    assert lot.meta["dashboard_source_ref"] == "mf/1"
    assert lot.meta["dashboard_reference"] == "TXN001"
    for key in _TXN_ONLY_META_KEYS:
        assert key not in lot.meta, f"lot entry carries {key}"


async def test_beancount_card_payment_resolved_meta_identity(session):
    """A resolved card-payment entry carries the full schema + the
    ``dashboard_card_resolution=resolved`` tag, and posts to the specific card
    liability. Parsed via real beancount meta identity."""
    bank = await _bank(session, id=1)
    await _card(session, id=2, bank="icici", label="Platinum")
    session.add(Card(id=10, account_id=2, card_mask="1234"))
    await session.flush()
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Card Bill",
        card_id=10,
        id=1,
    )
    report = await project(
        session, _config(selected_account_ids=(1, 2), ledger_cli="beancount")
    )
    entries = _load_beancount(report.journal)
    from beancount.core.data import Transaction

    txn = next(
        e
        for e in entries
        if isinstance(e, Transaction) and e.meta.get("dashboard_kind") == "card_payment"
    )
    assert txn.meta["dashboard_card_resolution"] == "resolved"
    assert txn.meta["dashboard_txn_ids"] == "txn-1"
    # The card_id (10) and the resolved card account id (2) both trace.
    assert "10" in txn.meta["dashboard_card_ids"].split("|")
    # The specific card liability is the first posting.
    liability = txn.postings[0].account
    assert liability.startswith("Liabilities:Card:"), liability


async def test_beancount_card_payment_unresolved_meta_identity(session):
    """An unresolved card-payment entry carries ``dashboard_card_resolution=
    unresolved`` and posts to the generic clearing liability. Parsed via real
    beancount meta identity."""
    bank = await _bank(session, id=1)
    await _card(session, id=2, bank="icici", label="Platinum")
    session.add(Card(id=10, account_id=2, card_mask="4567"))
    await session.flush()
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Card Bill",
        card_mask="9999",  # non-matching
        id=1,
    )
    report = await project(
        session, _config(selected_account_ids=(1, 2), ledger_cli="beancount")
    )
    entries = _load_beancount(report.journal)
    from beancount.core.data import Transaction

    txn = next(
        e
        for e in entries
        if isinstance(e, Transaction) and e.meta.get("dashboard_kind") == "card_payment"
    )
    assert txn.meta["dashboard_card_resolution"] == "unresolved"
    liability = txn.postings[0].account
    assert liability == "Liabilities:CreditCard", liability


async def test_beancount_fx_entry_meta_identity(session):
    """A priced foreign-currency entry carries the full transaction-derived
    schema, posts both legs in the foreign commodity, and a price directive lets
    beancount value it in INR. Parsed via real beancount meta identity."""
    from financial_dashboard.services.paisa.config import FxRate

    bank = await _bank(session)
    await _snapshot(session, bank.id, dt.date(2025, 12, 31), "1000.00")
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Store",
        currency="USD",
        reference_number="UTRFX1",
        id=5,
    )
    report = await project(
        session,
        _config(
            ledger_cli="beancount",
            non_inr_policy="priced",
            fx_rates={"USD": (FxRate(CUTOVER, Decimal("83.0000")),)},
        ),
    )
    entries = _load_beancount(report.journal)
    from beancount.core.data import Price, Transaction

    txn = next(
        e
        for e in entries
        if isinstance(e, Transaction) and e.meta.get("dashboard_txn_ids") == "txn-5"
    )
    # Full transaction-derived schema is present.
    assert txn.meta["dashboard_kind"] == "expense"
    assert txn.meta["dashboard_category"] == "groceries"
    assert txn.meta["dashboard_reference"] == "UTRFX1"
    # Both legs are denominated in the foreign commodity (no relabelling).
    commodities = {p.units.currency for p in txn.postings}
    assert commodities == {"USD"}, commodities
    # A price directive for USD on/before the txn date is present.
    prices = [e for e in entries if isinstance(e, Price)]
    assert any(p.currency == "USD" for p in prices)


# ---------------------------------------------------------------------------
# Ledger / hledger tag queries (only when the binary is available)
# ---------------------------------------------------------------------------


def _has_cli(binary: str) -> bool:
    import shutil

    return shutil.which(binary) is not None


@pytest.mark.parametrize("binary", ["ledger", "hledger"], ids=["ledger", "hledger"])
async def test_ledger_family_tag_queries_when_binary_available(
    session, binary, tmp_path
):
    """Render a journal with the full metadata schema and query a real
    ledger/hledger binary for the dashboard tags by id. Skipped when the binary
    is absent (no network installs in CI); runs wherever the binary exists."""
    if not _has_cli(binary):
        pytest.skip(f"{binary!r} binary not available")
    import subprocess

    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Store",
        reference_number="UTRTAG1",
        id=42,
    )
    report = await project(session, _config(ledger_cli=binary))
    journal_file = tmp_path / "proj.journal"
    journal_file.write_text(report.journal)

    def _run(args: list[str]) -> str:
        return subprocess.run(
            [binary, "-f", str(journal_file), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    # The dashboard_kind tag is queryable and resolves to the expense entry.
    out = _run(["--format", '%(tag("dashboard_kind"))\n', "reg", "Expenses"])
    assert "expense" in out
    # The dashboard_txn_ids tag traces the entry back to its dashboard row.
    txn_out = _run(["--format", '%(tag("dashboard_txn_ids"))\n', "reg"])
    assert "txn-42" in txn_out


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_metadata_values_sanitized_no_secrets(session, backend):
    """A reference_number carrying semicolons, newlines, or braces is sanitized
    so it cannot inject a directive or corrupt a tag/meta line."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        counterparty="Store",
        reference_number="UTR;evil\n{inject}",
        id=1,
    )
    report = await project(session, _config(ledger_cli=backend))
    journal = report.journal
    # No raw control chars, semicolons-in-values, or braces survive.
    assert "\n{" not in journal
    # The reference value was sanitized (semicolons, braces stripped).
    entry = report.entries[0]
    ref_val = dict(entry.meta).get("dashboard_reference", "")
    assert ";" not in ref_val
    assert "{" not in ref_val
    assert "\n" not in ref_val


async def test_backward_compatible_txn_tag_preserved(session):
    """The ``txn: <id>`` tag is queryable in every backend for existing
    drill-through tooling."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        id=99,
    )
    for backend in SUPPORTED_BACKENDS:
        report = await project(session, _config(ledger_cli=backend))
        if backend == "beancount":
            assert 'txn: "99"' in report.journal
        else:
            assert "; txn:99" in report.journal


async def test_opening_postings_carry_metadata(session):
    """Opening balance postings carry account-id/source/as-of posting-level
    metadata."""
    bank = await _bank(session)
    await _snapshot(session, bank.id, dt.date(2025, 12, 15), "100000.00")
    report = await project(session, _config(ledger_cli="ledger"))
    opening = report.openings[0]
    meta_dict = dict(opening.meta)
    assert meta_dict["dashboard_account_ids"] == "1"
    assert meta_dict["dashboard_source"] == "snapshot"
    assert meta_dict["dashboard_as_of"] == "2025-12-15"
    # The rendered journal carries the posting-level tag.
    assert "; dashboard_account_ids: 1" in report.journal
    assert "; dashboard_source: snapshot" in report.journal
    assert "; dashboard_as_of: 2025-12-15" in report.journal


async def test_self_transfer_carries_both_account_ids(session):
    """A self-transfer entry's metadata carries both account ids."""
    a = await _bank(session, id=1, bank="hdfc", label="A")
    b = await _bank(session, id=3, bank="axis", label="B")
    ref = "IMPS999"
    await _txn(
        session,
        account_id=a.id,
        direction="debit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        reference_number=ref,
        id=1,
    )
    await _txn(
        session,
        account_id=b.id,
        direction="credit",
        amount="1000.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        reference_number=ref,
        id=2,
    )
    report = await project(session, _config(selected_account_ids=(1, 3)))
    entry = report.entries[0]
    assert entry.kind == KIND_SELF_TRANSFER
    meta_dict = dict(entry.meta)
    assert meta_dict["dashboard_account_ids"] == "1|3"
    assert meta_dict["dashboard_txn_ids"] == "txn-1|txn-2"


async def test_kind_counts_reported(session):
    """The report's ``kind_counts`` tracks per-kind cardinality among emitted
    entries."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        id=1,
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="50000.00",
        date=dt.date(2026, 2, 2),
        category="salary",
        id=2,
    )
    report = await project(session, _config())
    assert report.kind_counts.get(KIND_EXPENSE) == 1
    assert report.kind_counts.get(KIND_INCOME) == 1


async def test_dashboard_kind_is_closed_taxonomy(session):
    """Every emitted entry's kind is in the closed ``DASHBOARD_KINDS`` set."""
    bank = await _bank(session)
    for i, (cat, d) in enumerate(
        [
            ("groceries", "debit"),
            ("salary", "credit"),
            ("refund", "credit"),
            ("investment", "debit"),
            ("repayment", "credit"),
            ("fees_charges", "credit"),
        ],
        start=1,
    ):
        await _txn(
            session,
            account_id=bank.id,
            direction=d,
            amount="10.00",
            date=dt.date(2026, 2, i),
            category=cat,
            id=i,
        )
    report = await project(session, _config())
    for entry in report.entries:
        assert entry.kind in DASHBOARD_KINDS, entry.kind


# ===========================================================================
# (3) Card payment traceability
# ===========================================================================


async def test_card_payment_resolved_by_card_id(session):
    """A bank-side card payment with an explicit linked ``card_id`` whose Card
    belongs to a selected card account posts to that specific liability."""
    bank = await _bank(session, id=1)
    await _card(session, id=2, bank="icici", label="Platinum")
    session.add(Card(id=10, account_id=2, card_mask="1234"))
    await session.flush()
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Card Bill",
        card_id=10,
        id=1,
    )
    report = await project(session, _config(selected_account_ids=(1, 2)))
    assert report.card_payments == 1
    assert report.card_payments_resolved == 1
    assert report.card_payments_unresolved == 0
    # The entry posts to the specific card liability, not generic clearing.
    entry = report.entries[0]
    liability = entry.postings[0].account
    assert liability == "Liabilities:Card:Icici:Platinum"
    assert "dashboard_card_resolution: resolved" in report.journal


async def test_card_payment_resolved_by_exact_mask(session):
    """A bank-side card payment with an exact ``card_mask`` matching a selected
    card account posts to that specific liability."""
    bank = await _bank(session, id=1)
    await _card(session, id=2, bank="icici", label="Platinum")
    session.add(Card(id=10, account_id=2, card_mask="4567"))
    await session.flush()
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Card Bill",
        card_mask="4567",
        id=1,
    )
    report = await project(session, _config(selected_account_ids=(1, 2)))
    assert report.card_payments_resolved == 1
    entry = report.entries[0]
    liability = entry.postings[0].account
    assert liability == "Liabilities:Card:Icici:Platinum"


async def test_card_payment_unresolved_uses_generic_clearing(session):
    """A card payment with no resolvable card_id or exact mask posts to the
    generic clearing with ``dashboard_card_resolution=unresolved``."""
    bank = await _bank(session, id=1)
    await _card(session, id=2, bank="icici", label="Platinum")
    session.add(Card(id=10, account_id=2, card_mask="4567"))
    await session.flush()
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Card Bill",
        # No card_id; a non-matching mask.
        card_mask="9999",
        id=1,
    )
    report = await project(session, _config(selected_account_ids=(1, 2)))
    assert report.card_payments_unresolved == 1
    assert report.card_payments_resolved == 0
    entry = report.entries[0]
    liability = entry.postings[0].account
    assert liability == "Liabilities:Credit Card"
    meta_dict = dict(entry.meta)
    assert meta_dict["dashboard_card_resolution"] == "unresolved"


async def test_card_payment_never_fuzzy_matches(session):
    """A near-miss mask (``456`` vs ``4567``) does NOT resolve — no fuzzy
    matching."""
    bank = await _bank(session, id=1)
    await _card(session, id=2, bank="icici", label="Platinum")
    session.add(Card(id=10, account_id=2, card_mask="4567"))
    await session.flush()
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Card Bill",
        card_mask="456",  # partial — must not fuzzy-match
        id=1,
    )
    report = await project(session, _config(selected_account_ids=(1, 2)))
    assert report.card_payments_unresolved == 1


async def test_card_side_payment_remains_explicit_skip(session):
    """A card-side credit_card_payment is explicitly skipped (card_side_payment)
    and not emitted — the bank leg is the authoritative event."""
    card_acct = await _card(session, id=2)
    await _txn(
        session,
        account_id=card_acct.id,
        direction="credit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        counterparty="Payment",
        id=1,
    )
    report = await project(session, _config(selected_account_ids=(2,)))
    assert report.card_side_payments == 1
    assert report.emitted_count == 0
    assert any(s.reason == "card_side_payment" for s in report.skipped)


# ===========================================================================
# (4) Investment funding double-count prevention
# ===========================================================================


async def _seed_bank_and_snapshot(session):
    from financial_dashboard.db.enums import SnapshotCategory, SnapshotSource

    session.add(Account(id=1, bank="hdfc", label="Savings", type="bank_account"))
    session.add(
        BalanceSnapshot(
            account_id=1,
            kind="asset",
            category=SnapshotCategory.bank_balance.value,
            as_of_date=CUTOVER,
            value=Decimal("100000.00"),
            source=SnapshotSource.bank_statement.value,
        )
    )
    await session.flush()


async def test_investment_funding_remapped_when_provable(session):
    """When a bank investment transaction provably funds an emitted lot (exact
    date+amount, deterministic), the bank leg's contra is remapped to the
    investment equity (not Assets:Investments:Unallocated) so the investment
    asset is counted once."""
    import json

    from financial_dashboard.db.models import CasUpload

    await _seed_bank_and_snapshot(session)
    # An investment lot: 50000 cost, acquired 2026-02-15.
    session.add(
        InvestmentLot(
            cas_upload_id=1,
            instrument_id="INE000A01018",
            instrument_name="Example Fund",
            quantity=Decimal("1000"),
            unit_cost=Decimal("50"),
            cost_basis=Decimal("50000"),
            currency="INR",
            acquired_on=dt.date(2026, 2, 15),
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
                            "date": "2026-02-15",
                            "description": "Example Fund",
                            "isin": "INE000A01018",
                            "transaction_type": "purchase",
                            "units": "1000",
                            "nav": "50.00",
                            "amount": "50000.00",
                            "reference": "TXN001",
                        }
                    ]
                }
            ),
        )
    )
    # A bank investment transaction matching the lot exactly.
    await _txn(
        session,
        account_id=1,
        direction="debit",
        amount="50000.00",
        date=dt.date(2026, 2, 15),
        category="investment",
        counterparty="MF Purchase",
        reference_number="TXN001",
        id=1,
    )
    await session.flush()
    report = await project(session, _config(project_investments=True))
    assert report.investment_lot_count == 1
    assert report.investment_funding_remapped == 1
    # The bank investment entry's contra is the investment equity, not
    # Assets:Investments:Unallocated.
    investment_entry = next(
        e for e in report.entries if e.kind == KIND_INVESTMENT and e.txn_ids
    )
    contra = investment_entry.postings[1].account
    assert contra == "Equity:Opening Balances:Investment", contra


async def test_investment_funding_unresolved_suppresses_lot(session):
    """When a bank investment transaction shares a date or amount with a lot
    but the link is not provably deterministic (multiple lots on same date),
    the lot is suppressed conservatively."""
    import json

    from financial_dashboard.db.models import CasUpload

    await _seed_bank_and_snapshot(session)
    # Two lots on the same date with the same amount — ambiguous.
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
    # A bank investment transaction matching date+amount but ambiguous (2 lots).
    await _txn(
        session,
        account_id=1,
        direction="debit",
        amount="50000.00",
        date=dt.date(2026, 2, 15),
        category="investment",
        counterparty="MF Purchase",
        reference_number="DIFFERENT_REF",
        id=1,
    )
    await session.flush()
    report = await project(session, _config(project_investments=True))
    # Both lots suppressed (ambiguous funding).
    assert report.investment_lot_count == 0
    assert len(report.investment_funding_unresolved) == 2
    assert "investment_funding_unresolved" in report.investment_excluded
    # No orphan price directive for the suppressed instruments.
    price_currencies = {p.currency for p in report.document.price_directives}
    assert "INE000A01018" not in price_currencies
    assert "INE000A01019" not in price_currencies


def _mf_fact(*, isin, date, reference="TXN001", source_ref=None, amount="50000.00"):
    """One complete MF acquisition CAS fact."""
    return {
        "scope": "mf",
        "source_ref": source_ref or f"mf/{isin}",
        "date": date,
        "description": f"Fund {isin}",
        "isin": isin,
        "transaction_type": "purchase",
        "units": "1000",
        "nav": "50.00",
        "amount": amount,
        "reference": reference,
    }


def _cas_with_facts(session, facts):
    """Seed a CasUpload carrying the given CAS facts (no redemptions)."""
    import json

    from financial_dashboard.db.models import CasUpload

    session.add(
        CasUpload(
            portfolio_key="PAN",
            depository_source="cdsl",
            statement_date=dt.date(2026, 4, 30),
            grand_total=Decimal("0"),
            raw_holdings_json=json.dumps({"transactions": facts}),
        )
    )


async def _seed_lot_full(
    session,
    *,
    isin,
    acquired_on,
    cost="50000",
    reference="TXN001",
    source_ref=None,
):
    session.add(
        InvestmentLot(
            cas_upload_id=1,
            instrument_id=isin,
            instrument_name=f"Fund {isin}",
            quantity=Decimal("1000"),
            unit_cost=Decimal("50"),
            cost_basis=Decimal(cost),
            currency="INR",
            acquired_on=acquired_on,
            source_ref=source_ref or f"mf/{isin}",
            transaction_type="purchase",
            reference=reference,
        )
    )
    await session.flush()


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_funding_exact_reference_remaps_across_backends(session, backend):
    """An exact reference that maps to a single instrument remaps the bank leg
    even when date+amount do NOT match — the reference is the sole deterministic
    signal. Verified across Ledger/hledger/Beancount: bank leg contra is the
    investment equity and the lot is emitted (asset counted once)."""
    await _seed_bank_and_snapshot(session)
    await _seed_lot_full(
        session,
        isin="INE000A01018",
        acquired_on=dt.date(2026, 2, 15),
        reference="LOT_REF",
    )
    _cas_with_facts(
        session,
        [_mf_fact(isin="INE000A01018", date="2026-02-15", reference="LOT_REF")],
    )
    # Same reference, but a DIFFERENT date and a different amount — so only the
    # exact reference disambiguates (date+amount cannot).
    await _txn(
        session,
        account_id=1,
        direction="debit",
        amount="49999.00",
        date=dt.date(2026, 3, 1),
        category="investment",
        counterparty="MF Purchase",
        reference_number="LOT_REF",
        id=1,
    )
    await session.flush()
    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )
    assert report.investment_funding_remapped == 1
    assert report.investment_lot_count == 1
    assert report.investment_funding_unresolved == ()
    investment_entry = next(
        e for e in report.entries if e.kind == KIND_INVESTMENT and e.txn_ids
    )
    contra = investment_entry.postings[1].account
    assert "Equity:Opening Balances:Investment" in contra or "Investment" in contra
    assert "Unallocated" not in contra


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_funding_shared_ref_deterministic_date_amount_remaps(session, backend):
    """A reference shared by multiple instruments must NOT early-return into a
    double-count window. The projection falls through to the deterministic
    exact date+amount check; when that singles out one instrument, the bank leg
    is remapped (asset counted once) and the other instrument's lot is emitted
    independently. No lot is suppressed."""
    await _seed_bank_and_snapshot(session)
    # Two instruments SHARE a reference, but have distinct acquisition dates so
    # the bank txn's exact date+amount matches only instrument A.
    await _seed_lot_full(
        session,
        isin="INE000A01018",
        acquired_on=dt.date(2026, 2, 15),
        reference="SHARED_REF",
    )
    await _seed_lot_full(
        session,
        isin="INE000A01019",
        acquired_on=dt.date(2026, 2, 20),
        reference="SHARED_REF",
    )
    _cas_with_facts(
        session,
        [
            _mf_fact(isin="INE000A01018", date="2026-02-15", reference="SHARED_REF"),
            _mf_fact(isin="INE000A01019", date="2026-02-20", reference="SHARED_REF"),
        ],
    )
    # Bank leg carries the shared reference AND matches A's exact date+amount.
    await _txn(
        session,
        account_id=1,
        direction="debit",
        amount="50000.00",
        date=dt.date(2026, 2, 15),
        category="investment",
        counterparty="MF Purchase",
        reference_number="SHARED_REF",
        id=1,
    )
    await session.flush()
    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )
    # The bank leg is remapped to the investment equity (funds A provably).
    assert report.investment_funding_remapped == 1
    # Both lots are emitted (A funded by the bank leg; B independent).
    assert report.investment_lot_count == 2
    assert report.investment_funding_unresolved == ()
    investment_entry = next(
        e for e in report.entries if e.kind == KIND_INVESTMENT and e.txn_ids
    )
    contra = investment_entry.postings[1].account
    assert "Equity:Opening Balances:Investment" in contra or "Investment" in contra
    assert "Unallocated" not in contra  # not the generic investment contra


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_funding_shared_ref_non_deterministic_suppresses_all(session, backend):
    """When a shared reference cannot be disambiguated by date+amount (both
    instruments share the same date+amount), every instrument sharing the ref is
    suppressed conservatively. No bank-leg remap, no orphan price, and neither
    suppressed instrument's asset account appears in the journal."""
    await _seed_bank_and_snapshot(session)
    # Both instruments share the reference AND the same date+amount.
    for isin in ("INE000A01018", "INE000A01019"):
        await _seed_lot_full(
            session,
            isin=isin,
            acquired_on=dt.date(2026, 2, 15),
            reference="SHARED_REF",
        )
    _cas_with_facts(
        session,
        [
            _mf_fact(isin=isin, date="2026-02-15", reference="SHARED_REF")
            for isin in ("INE000A01018", "INE000A01019")
        ],
    )
    await _txn(
        session,
        account_id=1,
        direction="debit",
        amount="50000.00",
        date=dt.date(2026, 2, 15),
        category="investment",
        counterparty="MF Purchase",
        reference_number="SHARED_REF",
        id=1,
    )
    await session.flush()
    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )
    assert report.investment_lot_count == 0
    assert set(report.investment_funding_unresolved) == {
        "INE000A01018",
        "INE000A01019",
    }
    assert report.investment_funding_remapped == 0
    # No orphan price directive, no asset-account line for either instrument.
    price_currencies = {p.currency for p in report.document.price_directives}
    for isin in ("INE000A01018", "INE000A01019"):
        assert isin not in price_currencies
        assert isin not in report.journal
    # The bank leg was emitted as an ordinary investment (Unallocated contra),
    # so the bank decrease is still captured.
    investment_entry = next(
        e for e in report.entries if e.kind == KIND_INVESTMENT and e.txn_ids
    )
    assert "Unallocated" in investment_entry.postings[1].account


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_funding_date_only_collision_suppresses(session, backend):
    """A bank investment txn sharing only the date (different amount, no shared
    reference) with a lot is a potential-but-not-provable link: the lot is
    suppressed conservatively rather than risk a double count."""
    await _seed_bank_and_snapshot(session)
    await _seed_lot_full(
        session,
        isin="INE000A01018",
        acquired_on=dt.date(2026, 2, 15),
        reference="LOT_REF",
    )
    _cas_with_facts(
        session,
        [_mf_fact(isin="INE000A01018", date="2026-02-15", reference="LOT_REF")],
    )
    # Same date, DIFFERENT amount, no shared reference.
    await _txn(
        session,
        account_id=1,
        direction="debit",
        amount="49999.00",  # != 50000
        date=dt.date(2026, 2, 15),
        category="investment",
        counterparty="MF Purchase",
        reference_number="BANK_REF",
        id=1,
    )
    await session.flush()
    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )
    assert report.investment_lot_count == 0
    assert "INE000A01018" in report.investment_funding_unresolved
    price_currencies = {p.currency for p in report.document.price_directives}
    assert "INE000A01018" not in price_currencies


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
async def test_funding_amount_only_collision_suppresses(session, backend):
    """A bank investment txn sharing only the amount (different date, no shared
    reference) with a lot is a potential-but-not-provable link: the lot is
    suppressed conservatively rather than risk a double count."""
    await _seed_bank_and_snapshot(session)
    await _seed_lot_full(
        session,
        isin="INE000A01018",
        acquired_on=dt.date(2026, 2, 15),
        reference="LOT_REF",
    )
    _cas_with_facts(
        session,
        [_mf_fact(isin="INE000A01018", date="2026-02-15", reference="LOT_REF")],
    )
    # Same amount, DIFFERENT date, no shared reference.
    await _txn(
        session,
        account_id=1,
        direction="debit",
        amount="50000.00",
        date=dt.date(2026, 3, 1),  # != 2026-02-15
        category="investment",
        counterparty="MF Purchase",
        reference_number="BANK_REF",
        id=1,
    )
    await session.flush()
    report = await project(
        session, _config(project_investments=True, ledger_cli=backend)
    )
    assert report.investment_lot_count == 0
    assert "INE000A01018" in report.investment_funding_unresolved
    price_currencies = {p.currency for p in report.document.price_directives}
    assert "INE000A01018" not in price_currencies


# ===========================================================================
# (5) Closed-population tests
# ===========================================================================


async def test_emitted_txn_ids_disjoint_from_skipped(session):
    """No txn id appears in both an emitted entry and a skipped row."""
    bank = await _bank(session)
    card_acct = await _card(session, id=2)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        id=1,
    )
    await _txn(
        session,
        account_id=card_acct.id,
        direction="credit",
        amount="5000.00",
        date=dt.date(2026, 2, 1),
        category="credit_card_payment",
        id=2,
    )
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2026, 2, 1),
        category="self_transfer",
        reference_number="LONE",
        id=3,
    )
    report = await project(session, _config(selected_account_ids=(1, 2)))
    emitted_ids: set[int] = set()
    for entry in report.entries:
        emitted_ids.update(entry.txn_ids)
    skipped_ids = {s.txn_id for s in report.skipped if s.txn_id is not None}
    assert emitted_ids.isdisjoint(skipped_ids), f"overlap: {emitted_ids & skipped_ids}"


async def test_emitted_union_skipped_equals_eligible(session):
    """The union of emitted and skipped txn ids equals every eligible selected
    post-cutover transaction id."""
    bank = await _bank(session)
    # Eligible: post-cutover, selected account, INR.
    for i in range(1, 6):
        await _txn(
            session,
            account_id=bank.id,
            direction="debit",
            amount="10.00",
            date=dt.date(2026, 2, i),
            category="groceries",
            id=i,
        )
    # Pre-cutover: NOT eligible.
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="10.00",
        date=dt.date(2025, 12, 15),
        category="groceries",
        id=99,
    )
    report = await project(session, _config())
    emitted_ids: set[int] = set()
    for entry in report.entries:
        emitted_ids.update(entry.txn_ids)
    skipped_ids = {s.txn_id for s in report.skipped if s.txn_id is not None}
    union = emitted_ids | skipped_ids
    assert union == {1, 2, 3, 4, 5}, f"unexpected union: {union}"
    assert 99 not in union  # pre-cutover never loaded


async def test_no_duplicate_emitted_txn_ids(session):
    """No txn id appears more than once across all emitted entries."""
    bank = await _bank(session)
    for i in range(1, 11):
        await _txn(
            session,
            account_id=bank.id,
            direction="debit",
            amount="10.00",
            date=dt.date(2026, 2, i),
            category="groceries",
            id=i,
        )
    report = await project(session, _config())
    all_ids: list[int] = []
    for entry in report.entries:
        all_ids.extend(entry.txn_ids)
    counts = Counter(all_ids)
    dups = {tid: c for tid, c in counts.items() if c > 1}
    assert not dups, f"duplicate emitted txn ids: {dups}"


async def test_amounts_trace_source_rows(session):
    """Every posting amount in an emitted entry traces to its source
    transaction's amount (up to sign)."""
    bank = await _bank(session)
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="123.45",
        date=dt.date(2026, 2, 1),
        category="groceries",
        id=1,
    )
    report = await project(session, _config())
    entry = report.entries[0]
    amounts = [abs(p.amount) for p in entry.postings]
    assert all(a == Decimal("123.45") for a in amounts), amounts


async def test_signs_follow_direction(session):
    """A debit decreases the account (negative posting); a credit increases it
    (positive posting). The contra takes the opposite sign."""
    bank = await _bank(session)
    # Debit: account ↓, contra ↑
    await _txn(
        session,
        account_id=bank.id,
        direction="debit",
        amount="100.00",
        date=dt.date(2026, 2, 1),
        category="groceries",
        id=1,
    )
    # Credit: account ↑, contra ↓
    await _txn(
        session,
        account_id=bank.id,
        direction="credit",
        amount="50000.00",
        date=dt.date(2026, 2, 2),
        category="salary",
        id=2,
    )
    report = await project(session, _config())
    debit_entry = next(e for e in report.entries if 1 in e.txn_ids)
    credit_entry = next(e for e in report.entries if 2 in e.txn_ids)
    # Debit: bank posting is negative (money out).
    bank_posting = debit_entry.postings[0]
    assert bank_posting.amount == Decimal("-100.00")
    # Credit: bank posting is positive (money in).
    bank_posting = credit_entry.postings[0]
    assert bank_posting.amount == Decimal("50000.00")
