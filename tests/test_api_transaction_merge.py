"""Explicit merge of a duplicate transaction. All values are synthetic."""

import datetime as dt
import json
from decimal import Decimal

import pytest
from bank_statement_parser.models import BankTransaction, ParsedBankStatement
from sqlalchemy import select

from financial_dashboard.db import (
    Account,
    AuditAction,
    BankStatementUpload,
    PaymentStatus,
    SmsMessage,
    StatementUpload,
    Transaction,
)
from financial_dashboard.services.reminders import recompute_cc_payment_state
from financial_dashboard.services.statements.bank import reconcile_bank_statement

pytestmark = pytest.mark.anyio

DAY = dt.date(2030, 3, 4)


async def _seed_pair(
    session,
    *,
    alert_ref: str = "123456",
    statement_ref: str = "UTR000123456",
    email_ids: tuple[int | None, int | None] = (None, None),
    statement_balance: Decimal | None = None,
    statement_date: dt.date = DAY + dt.timedelta(days=1),
    alert_on_statement: bool = False,
):
    """Seed an alert row and a statement row for one synthetic debit."""
    account = Account(bank="testbank", label="Test savings", type="bank_account")
    session.add(account)
    await session.flush()
    sms, stray_sms = (
        SmsMessage(
            bank="testbank",
            sender="TESTBK",
            body=f"synthetic {kind} {alert_ref}",
            received_at=dt.datetime(2030, 3, 4, 9, 30),
            status="parsed",
        )
        for kind in ("alert", "repeat")
    )
    upload = BankStatementUpload(
        account_id=account.id,
        bank="testbank",
        filename="stmt.pdf",
        file_path="/synthetic/stmt.pdf",
    )
    session.add_all([sms, stray_sms, upload])
    await session.flush()
    alert = Transaction(
        account_id=account.id,
        sms_message_id=sms.id,
        bank="testbank",
        email_type="testbank_account_debit_alert",
        direction="debit",
        amount=Decimal("1234.00"),
        currency="INR",
        transaction_date=DAY,
        transaction_time=dt.time(9, 30),
        balance=Decimal("5000.00"),
        reference_number=alert_ref,
        email_id=email_ids[0],
        source="sms",
        category="shopping",
        category_method="llm",
    )
    statement = Transaction(
        account_id=account.id,
        bank_statement_upload_id=upload.id,
        bank="testbank",
        email_type="bank_statement",
        direction="debit",
        amount=Decimal("1234.00"),
        currency="INR",
        transaction_date=statement_date,
        counterparty="SYNTHETIC SHOP",
        raw_description="UPI/SYNTHETIC SHOP/UTR000123456",
        reference_number=statement_ref,
        balance=statement_balance,
        email_id=email_ids[1],
        category="groceries",
        category_method="manual",
        note="weekly shop",
    )
    session.add_all([alert, statement])
    await session.flush()
    sms.transaction_id = alert.id
    stray_sms.transaction_id = statement.id
    if alert_on_statement:
        alert.bank_statement_upload_id = upload.id
    session.add(
        AuditAction(
            transaction_id=statement.id,
            action_type="set_category",
            target_type="transaction",
            target_id=statement.id,
        )
    )
    upload.reconciliation_data = json.dumps(
        {"matched": [], "missing": [{"stmt_idx": 0, "imported_txn_id": statement.id}]}
    )
    await session.commit()
    return alert.id, statement.id, upload.id


def _batch(*pairs: tuple[int, int], dry_run: bool = False) -> dict:
    return {
        "pairs": [{"keep_id": keep, "duplicate_id": dup} for keep, dup in pairs],
        "dry_run": dry_run,
    }


async def test_dry_run_writes_nothing(client, session):
    alert_id, statement_id, upload_id = await _seed_pair(session)

    request = _batch((alert_id, statement_id))
    del request["dry_run"]
    response = await client.post("/api/transactions/merge-batch", json=request)

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    merge = body["merges"][0]
    assert merge["keeper_before"]["category"] == "shopping"
    assert merge["keeper_after"]["category"] == "groceries"
    session.expire_all()
    assert await session.get(Transaction, statement_id) is not None
    keeper = await session.get(Transaction, alert_id)
    assert keeper is not None and keeper.category == "shopping"
    upload = await session.get(BankStatementUpload, upload_id)
    assert upload is not None and str(statement_id) in (
        upload.reconciliation_data or ""
    )


async def test_batch_merges_each_pair_once(client, session):
    alert_id, statement_id, upload_id = await _seed_pair(session)
    other_alert, other_statement, _ = await _seed_pair(
        session, alert_ref="UTR000777888", statement_ref="777888"
    )
    keeper_row = await session.get(Transaction, other_alert)
    assert keeper_row is not None
    email_copy = Transaction(
        account_id=keeper_row.account_id,
        bank="testbank",
        email_type="testbank_account_debit_alert",
        direction="debit",
        amount=Decimal("1234.00"),
        currency="INR",
        transaction_date=DAY,
        reference_number="XUTR000777888",
        source="email",
    )
    session.add(email_copy)
    await session.commit()
    email_copy_id = email_copy.id
    request = _batch(
        (alert_id, statement_id),
        (other_alert, other_statement),
        (other_alert, email_copy_id),
    )

    response = await client.post("/api/transactions/merge-batch", json=request)
    repeat = await client.post("/api/transactions/merge-batch", json=request)

    assert response.status_code == 200
    assert repeat.status_code == 409
    report = response.json()["merges"][0]
    assert report["moved_references"] == {
        "sms_messages": 1,
        "audit_actions": 1,
        "bank_statement_uploads.reconciliation_data": 1,
    }
    session.expire_all()
    assert await session.get(Transaction, statement_id) is None
    assert await session.get(Transaction, other_statement) is None
    assert await session.get(Transaction, email_copy_id) is None
    keeper = await session.get(Transaction, alert_id)
    assert keeper is not None
    assert keeper.bank_statement_upload_id == upload_id
    assert keeper.raw_description == "UPI/SYNTHETIC SHOP/UTR000123456"
    assert keeper.transaction_date == DAY
    assert keeper.balance == Decimal("5000.00")
    assert (keeper.category, keeper.category_method) == ("groceries", "manual")
    assert keeper.note == "weekly shop"
    stray = await session.scalar(
        select(SmsMessage).where(SmsMessage.body == "synthetic repeat 123456")
    )
    assert stray is not None and stray.transaction_id == alert_id
    action = await session.scalar(select(AuditAction))
    assert action is not None
    assert (action.transaction_id, action.target_id) == (alert_id, alert_id)
    upload = await session.get(BankStatementUpload, upload_id)
    assert upload is not None
    recon = json.loads(upload.reconciliation_data or "{}")
    assert recon["missing"][0]["imported_txn_id"] == alert_id

    for keeper_id, statement_ref in (
        (alert_id, "UTR000123456"),
        (other_alert, "777888"),
    ):
        merged = await session.get(Transaction, keeper_id)
        assert merged is not None
        statement_row = BankTransaction(
            date="05/03/2030",
            narration="UPI/SYNTHETIC SHOP",
            amount="1234.00",
            transaction_type="debit",
            reference_number=statement_ref,
        )
        reprocess = reconcile_bank_statement(
            ParsedBankStatement(
                file="stmt.pdf", bank="testbank", transactions=[statement_row]
            ),
            [merged],
            account_id=merged.account_id,
        )
        assert reprocess["missing"] == []
        assert reprocess["matched"][0]["db_txn_id"] == keeper_id


@pytest.mark.parametrize(
    ("seed", "reason"),
    [
        ({"statement_ref": "654321"}, "references prove distinct events"),
        ({"email_ids": (11, 12)}, "both rows own a different email"),
        ({"statement_balance": Decimal("9999.00")}, "balances differ"),
        (
            {"statement_date": DAY + dt.timedelta(days=2)},
            "dates are more than 1 day apart",
        ),
        (
            {"alert_on_statement": True},
            "both statement rows carry a different reference",
        ),
    ],
)
async def test_refused_pair_refuses_the_whole_batch(client, session, seed, reason):
    alert_id, statement_id, _ = await _seed_pair(session, **seed)
    other_alert, other_statement, _ = await _seed_pair(
        session, alert_ref="777888", statement_ref="UTR000777888"
    )

    response = await client.post(
        "/api/transactions/merge-batch",
        json=_batch((other_alert, other_statement), (alert_id, statement_id)),
    )

    assert response.status_code == 409
    assert response.json()["detail"]["refused"] == [
        {
            "keep_id": alert_id,
            "duplicate_id": statement_id,
            "reasons": [reason],
        }
    ]
    session.expire_all()
    for txn_id in (alert_id, statement_id, other_alert, other_statement):
        assert await session.get(Transaction, txn_id) is not None


@pytest.mark.parametrize(
    ("case", "conflict"),
    [
        ("recompute", None),
        ("paid_by_duplicate", "the duplicate is a payment in a paid card cycle"),
        ("no_total", "has no total due; not recomputed"),
        ("older_cycle", "does not hold the duplicate; not recomputed"),
    ],
)
async def test_cc_payment_merge_matches_a_fresh_recompute(
    client, session, case, conflict
):
    account = Account(bank="testbank", label="Test card", type="credit_card")
    session.add(account)
    await session.flush()
    upload = StatementUpload(
        account_id=account.id,
        bank="testbank",
        filename="cc.pdf",
        file_path="/synthetic/cc.pdf",
        due_date="25/03/2030",
        total_amount_due="3000.00",
        payment_status=PaymentStatus.UNPAID,
        created_at=dt.datetime(2030, 3, 4, tzinfo=dt.UTC),
    )
    session.add(upload)
    rows = [
        Transaction(
            account_id=account.id,
            bank="testbank",
            email_type="testbank_cc_payment_alert",
            direction="credit",
            amount=Decimal("1000.00"),
            transaction_date=date,
            source=source,
        )
        for source, date in (
            ("sms", DAY),
            ("email", DAY - dt.timedelta(days=1 if case == "older_cycle" else 0)),
        )
    ]
    session.add_all(rows)
    await session.flush()
    if case == "paid_by_duplicate":
        upload.total_amount_due = "2000.00"
    await recompute_cc_payment_state(session, upload)
    if case == "no_total":
        upload.total_amount_due = None
    await session.commit()
    keep_id, dup_id, upload_id = rows[0].id, rows[1].id, upload.id
    paid_before = upload.payment_paid_amount

    response = await client.post(
        "/api/transactions/merge-batch", json=_batch((keep_id, dup_id))
    )

    session.expire_all()
    upload = await session.get(StatementUpload, upload_id)
    assert upload is not None
    if case == "paid_by_duplicate":
        assert response.status_code == 409
        assert response.json()["detail"]["refused"][0]["reasons"] == [conflict]
        assert upload.payment_status == PaymentStatus.PAID
        assert upload.payment_paid_amount == paid_before
        return
    assert response.status_code == 200
    report = response.json()["merges"][0]
    if conflict is not None:
        assert report["cc_payment_state"] is None
        assert report["conflicts"] == [f"cc cycle {upload_id} {conflict}"]
        assert upload.payment_paid_amount == paid_before
        return
    state = report["cc_payment_state"]
    assert (state["paid_before"], state["paid_after"]) == ("2000.00", "1000.00")
    stored = upload.payment_paid_amount
    assert stored == await recompute_cc_payment_state(session, upload)
    assert stored == Decimal("1000.00")
