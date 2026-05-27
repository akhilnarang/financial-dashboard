"""Tests for amount-based CC payment disambiguation.

Covers ``find_cc_account_by_total_due`` and ``is_cc_payment_received_email``
from financial_dashboard.services.cc_disambiguation.
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import (
    Account,
    Base,
    PaymentStatus,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.cc_disambiguation import (
    find_cc_account_by_total_due,
    is_cc_payment_received_email,
    resolve_cc_payment_account,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


def _stmt(
    *,
    account_id: int,
    total: str,
    due_date: str = "20/05/2026",
    status: PaymentStatus | None = PaymentStatus.UNPAID,
) -> StatementUpload:
    return StatementUpload(
        account_id=account_id,
        bank="indusind",
        filename="x.pdf",
        file_path="/tmp/x.pdf",
        status="imported",
        due_date=due_date,
        total_amount_due=total,
        payment_status=status,
    )


async def _seed_three_indusind_ccs(session: AsyncSession) -> tuple[int, int, int]:
    a1 = Account(bank="indusind", type="credit_card", label="IndusInd A", active=True)
    a2 = Account(bank="indusind", type="credit_card", label="IndusInd B", active=True)
    a3 = Account(bank="indusind", type="credit_card", label="IndusInd C", active=True)
    session.add_all([a1, a2, a3])
    await session.flush()
    return a1.id, a2.id, a3.id


# ---------- is_cc_payment_received_email ----------


def test_is_cc_payment_received_email_recognises_payment_alert():
    assert is_cc_payment_received_email("indusind_cc_payment_alert") is True
    assert is_cc_payment_received_email("slice_cc_payment_alert") is True
    assert is_cc_payment_received_email("icici_cc_payment_alert") is True


def test_is_cc_payment_received_email_recognises_upi_payment_alert():
    assert is_cc_payment_received_email("icici_cc_upi_payment_alert") is True


def test_is_cc_payment_received_email_recognises_sms_and_slice_shapes():
    # SMS shapes emitted by bank-sms-parser parsers.
    assert is_cc_payment_received_email("axis_cc_payment_received_alert") is True
    assert is_cc_payment_received_email("indusind_cc_payment_received_alert") is True
    # Slice's bill-autopay SMS.
    assert is_cc_payment_received_email("slice_cc_bill_paid_alert") is True


def test_is_cc_payment_received_email_recognises_remaining_bill_payment_shapes():
    """Bill-payment shapes that don't share a suffix with the above
    must also be recognized so the email-path payment-check gate
    doesn't silently skip them."""
    # "Payment received on CC" via email — credit-alert wording.
    assert is_cc_payment_received_email("hsbc_cc_credit_alert") is True
    assert is_cc_payment_received_email("idfc_cc_credit_alert") is True
    # Bare suffixes (no `_alert` tail).
    assert is_cc_payment_received_email("kotak_cc_payment") is True
    assert is_cc_payment_received_email("kotak_cc_bill_paid") is True
    # SBI's payment acknowledgement is matched by full-string literal
    # because the BillDesk template doesn't share any `_cc_*` shape.
    assert is_cc_payment_received_email("sbi_payment_ack") is True


def test_is_cc_payment_received_email_rejects_refund_and_reversal_shapes():
    """Refund/reversal credits are NOT bill payments — they must not
    match the predicate, or the email-path gate would treat a merchant
    refund as a statement payment and corrupt payment_status."""
    assert is_cc_payment_received_email("icici_cc_reversal") is False
    assert is_cc_payment_received_email("hdfc_cc_refund_alert") is False


# ---------- should_auto_reconcile_statement ----------


def _bare_txn(**kwargs):
    """Build a Transaction with the minimum field set the gate inspects.
    Constructed in memory; no DB session needed."""
    from financial_dashboard.db import Transaction

    return Transaction(
        bank=kwargs.pop("bank", "indusind"),
        email_type=kwargs.pop("email_type", "indusind_cc_payment_alert"),
        direction=kwargs.pop("direction", "credit"),
        amount=kwargs.pop("amount", Decimal("133")),
        currency=kwargs.pop("currency", "INR"),
        account_id=kwargs.pop("account_id", 7),
        **kwargs,
    )


def test_should_auto_reconcile_true_for_bill_payment_credit_with_account():
    from financial_dashboard.services.cc_disambiguation import (
        should_auto_reconcile_statement,
    )

    assert should_auto_reconcile_statement(_bare_txn()) is True


def test_should_auto_reconcile_false_for_debit():
    from financial_dashboard.services.cc_disambiguation import (
        should_auto_reconcile_statement,
    )

    assert should_auto_reconcile_statement(_bare_txn(direction="debit")) is False


def test_should_auto_reconcile_false_without_account_id():
    from financial_dashboard.services.cc_disambiguation import (
        should_auto_reconcile_statement,
    )

    assert should_auto_reconcile_statement(_bare_txn(account_id=None)) is False


def test_should_auto_reconcile_false_for_refund_shapes():
    """A credit-direction merchant refund must not auto-reconcile —
    payment_paid_amount is cumulative and would corrupt the statement."""
    from financial_dashboard.services.cc_disambiguation import (
        should_auto_reconcile_statement,
    )

    assert (
        should_auto_reconcile_statement(_bare_txn(email_type="icici_cc_reversal"))
        is False
    )
    assert (
        should_auto_reconcile_statement(_bare_txn(email_type="hdfc_cc_refund_alert"))
        is False
    )


def test_is_cc_payment_received_email_rejects_unrelated_types():
    assert is_cc_payment_received_email("icici_cc_transaction_alert") is False
    assert is_cc_payment_received_email("hdfc_bank_transfer_alert") is False
    assert is_cc_payment_received_email(None) is False
    assert is_cc_payment_received_email("") is False


# ---------- find_cc_account_by_total_due ----------


@pytest.mark.anyio
async def test_returns_none_when_bank_has_no_cc_accounts(session):
    """No CC accounts at all → nothing to disambiguate."""
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out is None


@pytest.mark.anyio
async def test_returns_none_when_bank_has_single_cc_account(session):
    """One CC account: the linker's bank-only fallback already handles
    this case, so the matcher refuses to act."""
    only = Account(bank="indusind", type="credit_card", label="solo", active=True)
    session.add(only)
    await session.flush()
    session.add(_stmt(account_id=only.id, total="133.00"))
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out is None


@pytest.mark.anyio
async def test_exact_total_match_picks_single_account(session):
    """Three CCs, only one has an open statement of the matching total."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="1,616.00"),
            _stmt(account_id=b, total="133.00"),  # ← the match
            _stmt(account_id=c, total="4,661.00"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out == b


@pytest.mark.anyio
async def test_no_match_returns_none(session):
    """Three CCs, no open statement total equals the payment amount."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="1,616.00"),
            _stmt(account_id=b, total="500.00"),
            _stmt(account_id=c, total="4,661.00"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out is None


@pytest.mark.anyio
async def test_multiple_matches_returns_none(session):
    """Two open statements share the same total — refuse to guess."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="133.00"),
            _stmt(account_id=b, total="133.00"),
            _stmt(account_id=c, total="500.00"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out is None


@pytest.mark.anyio
async def test_already_paid_statements_are_excluded(session):
    """A statement marked PAID is no longer a candidate even if its
    total matches the incoming payment amount."""
    a, b, _ = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="133.00", status=PaymentStatus.PAID),
            _stmt(account_id=b, total="133.00", status=PaymentStatus.UNPAID),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out == b


@pytest.mark.anyio
async def test_statements_without_due_date_are_excluded(session):
    """A statement with no due_date is ineligible for the latest-cycle
    pick — match must come from an upload with a parseable due date."""
    a, b, _ = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="133.00", due_date=None),
            _stmt(account_id=b, total="133.00"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out == b


@pytest.mark.anyio
async def test_only_latest_cycle_per_account_is_considered(session):
    """An older cycle whose total matches must NOT win when a newer
    cycle on the same account is already on file with a different
    total — matches the rule in check_payment_received."""
    a, b, _ = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            # Account a: older cycle matches, but newer cycle is different.
            _stmt(account_id=a, total="133.00", due_date="20/04/2026"),
            _stmt(account_id=a, total="999.00", due_date="20/05/2026"),
            # Account b: latest cycle matches.
            _stmt(account_id=b, total="133.00", due_date="20/05/2026"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out == b


@pytest.mark.anyio
async def test_unparseable_total_amount_due_is_skipped_and_logged(session, caplog):
    """A stored total_amount_due that parse_cc_amount can't read (e.g.
    a stray currency symbol) is skipped — and the skip is logged so
    silent fall-throughs are diagnosable."""
    import logging

    a, b, _ = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="₹133.00"),  # currency-prefixed, unparseable
            _stmt(account_id=b, total="133.00"),
        ]
    )
    await session.flush()

    with caplog.at_level(
        logging.WARNING, logger="financial_dashboard.services.cc_disambiguation"
    ):
        out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))

    assert out == b
    assert any("not parseable" in rec.message for rec in caplog.records), (
        f"expected a parseability warning, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.anyio
async def test_other_banks_are_not_considered(session):
    """A statement on a different-bank CC with a matching total is not
    a candidate."""
    a, b, _ = await _seed_three_indusind_ccs(session)
    other = Account(bank="hdfc", type="credit_card", label="HDFC", active=True)
    session.add(other)
    await session.flush()
    session.add_all(
        [
            _stmt(account_id=other.id, total="133.00"),
            _stmt(account_id=a, total="999.00"),
            _stmt(account_id=b, total="999.00"),
        ]
    )
    await session.flush()
    out = await find_cc_account_by_total_due(session, "indusind", Decimal("133"))
    assert out is None


# ---------- resolve_cc_payment_account ----------


async def _txn(
    session: AsyncSession,
    *,
    bank: str = "indusind",
    email_type: str = "indusind_cc_payment_alert",
    direction: str = "credit",
    amount: Decimal | None = Decimal("133"),
    account_id: int | None = None,
) -> Transaction:
    t = Transaction(
        bank=bank,
        email_type=email_type,
        direction=direction,
        amount=amount,
        currency="INR",
        account_id=account_id,
    )
    session.add(t)
    await session.flush()
    return t


@pytest.mark.anyio
async def test_resolver_skips_when_already_linked(session):
    """A txn that already has an account_id is left untouched."""
    t = await _txn(session, account_id=42)
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id == 42


@pytest.mark.anyio
async def test_resolver_skips_when_not_credit(session):
    """A debit txn never disambiguates as a CC bill-payment received."""
    await _seed_three_indusind_ccs(session)
    t = await _txn(session, direction="debit")
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id is None


@pytest.mark.anyio
async def test_resolver_skips_when_email_type_unrecognised(session):
    """A non-CC-payment email_type (e.g. transaction_alert) is ignored."""
    await _seed_three_indusind_ccs(session)
    t = await _txn(session, email_type="icici_cc_transaction_alert")
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id is None


@pytest.mark.anyio
async def test_resolver_skips_when_amount_is_none(session):
    """Defense in depth: a None amount short-circuits before any DB read.

    Schema enforces NOT NULL on Transaction.amount, so we construct the
    row in-memory (no flush) — the guard fires before any query runs."""
    await _seed_three_indusind_ccs(session)
    t = Transaction(
        bank="indusind",
        email_type="indusind_cc_payment_alert",
        direction="credit",
        amount=None,
        currency="INR",
    )
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id is None


@pytest.mark.anyio
async def test_resolver_returns_none_when_no_cc_candidates(session):
    """No CC accounts on the bank → silent no-op, account_id stays None."""
    t = await _txn(session)
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id is None


@pytest.mark.anyio
async def test_resolver_auto_resolves_when_single_candidate(session):
    """A single CC for the bank → auto-resolve account_id, return None."""
    only = Account(bank="indusind", type="credit_card", label="solo", active=True)
    session.add(only)
    await session.flush()
    t = await _txn(session)
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id == only.id


@pytest.mark.anyio
async def test_resolver_amount_match_auto_resolves(session):
    """Multi-CC bank, one matching open statement → auto-resolve to it."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="1,616.00"),
            _stmt(account_id=b, total="133.00"),  # match
            _stmt(account_id=c, total="4,661.00"),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=Decimal("133"))
    out = await resolve_cc_payment_account(session, t)
    assert out is None
    assert t.account_id == b


@pytest.mark.anyio
async def test_resolver_returns_prompt_payload_when_no_amount_match(session):
    """Multi-CC, no amount match → prompt payload, account_id stays None."""
    a, b, c = await _seed_three_indusind_ccs(session)
    session.add_all(
        [
            _stmt(account_id=a, total="1,616.00"),
            _stmt(account_id=b, total="500.00"),
            _stmt(account_id=c, total="4,661.00"),
        ]
    )
    await session.flush()
    t = await _txn(session, amount=Decimal("133"))
    out = await resolve_cc_payment_account(session, t)
    assert t.account_id is None
    assert out is not None
    assert out["txn_id"] == t.id
    assert out["bank"] == "indusind"
    assert out["amount"] == Decimal("133")
    assert set(out["candidate_account_ids"]) == {a, b, c}
    assert set(out["candidate_labels"].keys()) == {a, b, c}
