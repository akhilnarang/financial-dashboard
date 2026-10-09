"""Explicit merge of a duplicate transaction. All values are synthetic."""

import datetime as dt
import json
from decimal import Decimal

import pytest
from bank_statement_parser.models import BankTransaction, ParsedBankStatement
from sqlalchemy import func, select

from financial_dashboard.db import (
    Account,
    AuditAction,
    BankStatementUpload,
    PaymentStatus,
    SmsMessage,
    StatementUpload,
    TelegramMessageContext,
    Transaction,
)
from financial_dashboard.services.reminders import recompute_cc_payment_state
from financial_dashboard.services.statements.bank import reconcile_bank_statement

pytestmark = pytest.mark.anyio

DAY = dt.date(2030, 3, 4)


async def _seed_pair(
    session,
    *,
    alert_ref: str | None = "123456",
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
        card_holder="Addon Holder",
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


def _batch(*pairs: tuple[int, int], dry_run: bool = False, **fields) -> dict:
    return {
        "pairs": [
            {"keep_id": keep, "duplicate_id": dup} | fields for keep, dup in pairs
        ],
        "dry_run": dry_run,
    }


async def test_dry_run_writes_nothing(client, session):
    alert_id, statement_id, upload_id = await _seed_pair(session)

    request = _batch((alert_id, statement_id))
    del request["dry_run"]
    unexplained = await client.post(
        "/api/transactions/merge-batch",
        json=_batch((alert_id, statement_id), override=["dates"], reason=" "),
    )
    response = await client.post("/api/transactions/merge-batch", json=request)

    assert unexplained.status_code == 422
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
    assert keeper.card_holder == "Addon Holder"
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
    ("seed", "reason", "override"),
    [
        ({"statement_ref": "654321"}, "references prove distinct events", "references"),
        ({"email_ids": (11, 12)}, "both rows own a different email", None),
        ({"statement_balance": Decimal("9999.00")}, "balances differ", None),
        (
            {"statement_date": DAY + dt.timedelta(days=2)},
            "dates are more than 1 day apart",
            "dates",
        ),
        (
            {"alert_on_statement": True},
            "both statement rows carry a different reference",
            None,
        ),
        ({"statement_date": None}, "a row has no date", None),
    ],
)
async def test_refused_pair_refuses_the_whole_batch(
    client, session, seed, reason, override
):
    alert_id, statement_id, _ = await _seed_pair(session, **seed)
    other_alert, other_statement, _ = await _seed_pair(
        session, alert_ref="777888", statement_ref="UTR000777888"
    )
    pairs = ((other_alert, other_statement), (alert_id, statement_id))
    # Each override lifts only its own refusal.
    wrong = [kind for kind in ("references", "dates") if kind != override]

    for request in (
        _batch(*pairs),
        _batch(*pairs, override=wrong, reason="checked by hand"),
    ):
        response = await client.post("/api/transactions/merge-batch", json=request)

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
    ("override", "keep_statement"),
    [("references", False), ("dates", False), ("dates", True)],
)
async def test_override_merges_only_what_todays_parse_backs(
    client, session, monkeypatch, tmp_path, override, keep_statement
):
    if override == "references":
        seeded = await _seed_pair(session, statement_ref="654321")
    else:
        seeded = await _seed_pair(
            session, statement_date=DAY + dt.timedelta(days=2), alert_ref=None
        )
    alert_id, statement_id, upload_id = seeded
    # The statement posts the purchase days after the alert. The statement row
    # stays, and the alert folds into it.
    keep_id, dup_id = (
        (statement_id, alert_id) if keep_statement else (alert_id, statement_id)
    )
    other_alert, other_statement, _ = await _seed_pair(
        session, alert_ref="777888", statement_ref="UTR000777888"
    )
    pairs = ((other_alert, other_statement), (keep_id, dup_id))
    upload = await session.get(BankStatementUpload, upload_id)
    assert upload is not None
    pdf = tmp_path / "stmt.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload.file_path = str(pdf)
    upload.parsed_txn_count = 2
    await session.commit()
    # The old parse printed the duplicate's line. Today's parse prints the
    # true date and the keeper's reference.
    old = BankTransaction(
        date="06/03/2030" if override == "dates" else "10/03/2030",
        narration="UPI/SYNTHETIC SHOP",
        amount="1234.00",
        transaction_type="debit",
        reference_number="654321" if override == "references" else None,
    )
    today = old.model_copy(
        update={
            "date": "04/03/2030",
            "reference_number": "123456" if override == "references" else "UTR000555",
        }
    )
    if keep_statement:
        old, today = (
            today.model_copy(update={"reference_number": None}),
            old.model_copy(update={"reference_number": "UTR000123456"}),
        )
    parsed = [today]
    period = ["01/03/2030"]
    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_bank_statement",
        lambda path, _bank, _password: ParsedBankStatement(
            file=path.name,
            bank="testbank",
            statement_period_start=period[0],
            statement_period_end="31/03/2030",
            transactions=parsed,
        ),
    )
    request = _batch(*pairs, override=[override], reason="checked by hand")
    name = f"today's parse of bank statement {upload_id}"

    short = await client.post("/api/transactions/merge-batch", json=request)
    upload = await session.get(BankStatementUpload, upload_id)
    assert upload is not None
    upload.parsed_txn_count = 1
    await session.commit()
    # A reparse of a later period cannot see the keeper.
    period[0] = "20/03/2030"
    far = await client.post("/api/transactions/merge-batch", json=request)
    period[0] = "01/03/2030"
    parsed[:] = [old]
    stale = await client.post("/api/transactions/merge-batch", json=request)
    parsed[:] = [today, old]
    doubled = await client.post("/api/transactions/merge-batch", json=request)
    # Only a lookup by reference finds a row out of the reparse's date range.
    session.add(
        Transaction(
            account_id=upload.account_id,
            bank="otherbank",
            email_type="bank_statement",
            direction="debit",
            amount=Decimal("1234.00"),
            transaction_date=dt.date(2030, 1, 5),
            reference_number="999999",
        )
    )
    await session.commit()
    parsed[:] = [today, today.model_copy(update={"reference_number": "999999"})]
    distant = await client.post("/api/transactions/merge-batch", json=request)
    parsed[:] = [today]
    if override == "dates" and not keep_statement:
        # A later pair in the batch fills the keeper's empty reference.
        copy = Transaction(
            account_id=upload.account_id,
            bank="testbank",
            email_type="testbank_account_debit_alert",
            direction="debit",
            amount=Decimal("1234.00"),
            currency="INR",
            transaction_date=DAY,
            reference_number="999999",
            source="email",
        )
        session.add(copy)
        await session.commit()
        shared = await client.post(
            "/api/transactions/merge-batch",
            json=request
            | {"pairs": [*request["pairs"], _batch((alert_id, copy.id))["pairs"][0]]},
        )
        assert shared.json()["detail"]["refused"][0]["reasons"] == [
            f"{name} matches no line to the keeper"
        ]
    if keep_statement:
        # A statement already matched the alert. The merge would drop that link.
        alert = await session.get(Transaction, alert_id)
        assert alert is not None
        alert.bank_statement_upload_id = upload_id
        await session.commit()
        linked = await client.post("/api/transactions/merge-batch", json=request)
        assert linked.json()["detail"]["refused"][0]["reasons"] == [
            "an override needs a statement row as the duplicate, "
            "or as the keeper of a row no statement matched"
        ]
        alert.bank_statement_upload_id = None
        await session.commit()
    response = await client.post("/api/transactions/merge-batch", json=request)
    preview = await client.post(f"/api/statements/bank/{upload_id}/reconcile-preview")

    refused = (short, far, stale, doubled, distant)
    assert [r.json()["detail"]["refused"][0]["reasons"] for r in refused] == [
        [
            f"bank statement {upload_id} cannot be checked: "
            "today's parse found 1 lines; the stored parse found 2"
        ],
        [f"{name} matches no line to the keeper"],
        [f"{name} matches no line to the keeper"],
        [f"{name} leaves the row unmatched; a reparse would import it again"],
        [f"{name} leaves the row unmatched; a reparse would import it again"],
    ]
    assert response.status_code == 200
    assert [m["overrides"] for m in response.json()["merges"]] == [[], [override]]
    assert preview.json()["missing_count"] == 0
    assert preview.json()["matched"][0]["matched_transaction_id"] == keep_id
    session.expire_all()
    assert await session.get(Transaction, dup_id) is None
    keeper = await session.get(Transaction, keep_id)
    assert keeper is not None
    # The keeper keeps its own reference, even an empty one.
    assert (keeper.transaction_date, keeper.reference_number) == (
        (DAY + dt.timedelta(days=2), "UTR000123456")
        if keep_statement
        else (DAY, "123456" if override == "references" else None)
    )
    assert keeper.bank_statement_upload_id == upload_id
    assert keeper.balance == Decimal("5000.00")
    alert_sms = await session.scalar(
        select(SmsMessage).where(SmsMessage.body.startswith("synthetic alert"))
    )
    assert alert_sms is not None
    assert (keeper.sms_message_id, alert_sms.transaction_id) == (alert_sms.id, keep_id)
    record = await session.scalar(
        select(AuditAction).where(AuditAction.action_type == "merge_transaction")
    )
    assert record is not None and record.target_id == dup_id
    assert json.loads(record.arguments_json or "{}") == {
        "keep_id": keep_id,
        "override": [override],
        "reason": "checked by hand",
    }
    assert json.loads(record.before_json or "{}")["reference_number"] == (
        None
        if keep_statement
        else {"references": "654321", "dates": "UTR000123456"}[override]
    )


async def test_currency_override_folds_a_foreign_alert_into_the_statement_row(
    client, session, monkeypatch, tmp_path
):
    alert_id, statement_id, upload_id = await _seed_pair(session, alert_ref=None)
    alert = await session.get(Transaction, alert_id)
    upload = await session.get(BankStatementUpload, upload_id)
    assert alert is not None and upload is not None
    alert.currency, alert.amount = "EUR", Decimal("14.00")
    pdf = tmp_path / "stmt.pdf"
    pdf.write_bytes(b"synthetic PDF bytes")
    upload.file_path, upload.parsed_txn_count = str(pdf), 1
    await session.commit()
    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_bank_statement",
        lambda path, _bank, _password: ParsedBankStatement(
            file=path.name,
            bank="testbank",
            statement_period_start="01/03/2030",
            statement_period_end="31/03/2030",
            transactions=[
                BankTransaction(
                    date="05/03/2030",
                    narration="UPI/SYNTHETIC SHOP",
                    amount="1234.00",
                    transaction_type="debit",
                    reference_number="UTR000123456",
                )
            ],
        ),
    )
    checked = {"override": ["currency"], "reason": "checked by hand"}

    plain = await client.post(
        "/api/transactions/merge-batch", json=_batch((statement_id, alert_id))
    )
    foreign_keeper = await client.post(
        "/api/transactions/merge-batch",
        json=_batch((alert_id, statement_id), **checked),
    )
    response = await client.post(
        "/api/transactions/merge-batch",
        json=_batch((statement_id, alert_id), **checked),
    )

    assert plain.json()["detail"]["refused"][0]["reasons"] == ["currency differs"]
    assert foreign_keeper.json()["detail"]["refused"][0]["reasons"] == [
        "a foreign-currency row folds only into a rupee statement row"
    ]
    assert response.status_code == 200
    session.expire_all()
    assert await session.get(Transaction, alert_id) is None
    keeper = await session.get(Transaction, statement_id)
    assert keeper is not None
    assert (keeper.amount, keeper.currency) == (Decimal("1234.00"), "INR")
    assert keeper.sms_message_id is not None


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


async def test_delete_batch_removes_only_phantom_statement_rows(
    client, session, monkeypatch, tmp_path
):
    account = Account(bank="testbank", label="Test savings", type="bank_account")
    session.add(account)
    await session.flush()
    # stmt: today's parse drops the phantom line. other: the stored parse saw
    # nine lines, today's sees one. gone: no PDF. empty: today's parse is empty.
    uploads = {}
    for name, parsed_count in (("stmt", 2), ("other", 9), ("gone", 1), ("empty", 1)):
        pdf = tmp_path / f"{name}.pdf"
        if name != "gone":
            pdf.write_bytes(b"synthetic PDF bytes")
        uploads[name] = BankStatementUpload(
            account_id=account.id,
            bank="testbank",
            filename=pdf.name,
            file_path=str(pdf),
            status="imported",
            parsed_txn_count=parsed_count,
        )
    session.add_all(uploads.values())
    await session.flush()
    rows = {
        name: Transaction(
            account_id=account.id,
            bank_statement_upload_id=uploads[upload].id,
            bank="testbank",
            email_type=email_type,
            direction="debit",
            amount=Decimal(amount),
            currency="INR",
            transaction_date=DAY,
            counterparty=f"SYNTHETIC {name.upper()}",
        )
        for name, upload, email_type, amount in (
            ("real", "stmt", "bank_statement", "1234.00"),
            ("alert_owned", "stmt", "bank_statement", "55.00"),
            ("phantom", "stmt", "bank_statement", "9000.00"),
            ("named_elsewhere", "stmt", "bank_statement", "66.00"),
            ("alert", "stmt", "testbank_account_debit_alert", "77.00"),
            ("ref_candidate", "stmt", "bank_statement", "100.00"),
            ("candidate_elsewhere", "stmt", "bank_statement", "44.00"),
            ("unresolved_elsewhere", "stmt", "bank_statement", "33.00"),
            ("short_parse", "other", "bank_statement", "88.00"),
            ("no_pdf", "gone", "bank_statement", "99.00"),
            ("empty_parse", "empty", "bank_statement", "11.00"),
        )
    }
    rows["named_elsewhere"].note = "synthetic note"
    rows["ref_candidate"].reference_number = "SYNREF123"
    session.add_all(rows.values())
    await session.flush()
    session.add_all(
        [
            SmsMessage(
                bank="testbank",
                sender="TESTBK",
                body="synthetic cafe alert",
                received_at=dt.datetime(2030, 3, 4, 9, 30),
                status="parsed",
                transaction_id=rows["alert_owned"].id,
            ),
            AuditAction(
                transaction_id=rows["phantom"].id,
                action_type="set_category",
                target_type="transaction",
                target_id=rows["phantom"].id,
            ),
            TelegramMessageContext(
                chat_id=1,
                message_id=2,
                context_kind="transaction_alert",
                transaction_id=rows["phantom"].id,
            ),
        ]
    )
    real_id, phantom_id = rows["real"].id, rows["phantom"].id
    uploads["stmt"].reconciliation_data = json.dumps(
        {
            "matched": [
                {
                    "stmt_idx": 0,
                    "db_txn_id": real_id,
                    "candidate_transaction_ids": [real_id, phantom_id],
                }
            ],
            "missing": [{"stmt_idx": 1, "imported_txn_id": phantom_id}],
        }
    )
    uploads["other"].reconciliation_data = json.dumps(
        {
            "matched": [{"stmt_idx": 0, "db_txn_id": rows["named_elsewhere"].id}],
            "missing": [
                {
                    "stmt_idx": 1,
                    "date": "04/03/2030",
                    "amount": "45.00",
                    "direction": "debit",
                    "ambiguous": True,
                    "candidate_transaction_ids": [rows["candidate_elsewhere"].id],
                },
                {
                    "stmt_idx": 2,
                    "date": "05/03/2030",
                    "amount": "33.00",
                    "direction": "debit",
                    "import_error": "duplicate transaction",
                },
            ],
        }
    )
    await session.commit()
    ids = {name: row.id for name, row in rows.items()}
    upload_ids = {name: upload.id for name, upload in uploads.items()}

    def parse(path, _bank, _password):
        """Today's parse holds the real row and a same-reference line."""
        lines = [
            BankTransaction(
                date="04/03/2030",
                narration="UPI/SYNTHETIC SHOP",
                amount="1234.00",
                transaction_type="debit",
            ),
            BankTransaction(
                date="04/03/2030",
                narration="NEFT/SYNTHETIC/SYNREF123",
                amount="101.00",
                transaction_type="debit",
                reference_number="SYNREF123",
            ),
        ]
        return ParsedBankStatement(
            file=path.name,
            bank="testbank",
            statement_period_start="01/03/2030",
            statement_period_end="31/03/2030",
            transactions=[] if path.name == "empty.pdf" else lines,
        )

    monkeypatch.setattr(
        "financial_dashboard.services.statement_previews.parse_bank_statement", parse
    )

    refused = await client.post(
        "/api/transactions/delete-batch",
        json={
            "ids": list(ids.values()),
            "reason": "linked deposit line",
            "dry_run": False,
        },
    )
    assert refused.status_code == 409
    stmt, other = upload_ids["stmt"], upload_ids["other"]
    assert refused.json()["detail"]["refused"] == [
        {
            "id": ids["real"],
            "reasons": [f"today's parse of bank statement {stmt} still holds the row"],
        },
        {
            "id": ids["alert_owned"],
            "reasons": [
                "the row owns an sms or email; a reparse would create it again"
            ],
        },
        {
            "id": ids["named_elsewhere"],
            "reasons": [
                "the row carries an attachment or a note; check it by hand",
                f"bank statement {other} could claim the row; "
                "its reparse would import it again",
            ],
        },
        {"id": ids["alert"], "reasons": ["the row is not a statement import"]},
        {
            "id": ids["ref_candidate"],
            "reasons": [f"today's parse of bank statement {stmt} still holds the row"],
        },
        {
            "id": ids["candidate_elsewhere"],
            "reasons": [
                f"bank statement {other} could claim the row; "
                "its reparse would import it again"
            ],
        },
        {
            "id": ids["unresolved_elsewhere"],
            "reasons": [
                f"bank statement {other} could claim the row; "
                "its reparse would import it again"
            ],
        },
        {
            "id": ids["short_parse"],
            "reasons": [
                f"bank statement {other} cannot be checked: "
                "today's parse found 2 lines; the stored parse found 9"
            ],
        },
        {
            "id": ids["no_pdf"],
            "reasons": [
                f"bank statement {upload_ids['gone']} cannot be checked: "
                "Statement PDF is unavailable"
            ],
        },
        {
            "id": ids["empty_parse"],
            "reasons": [
                f"bank statement {upload_ids['empty']} cannot be checked: "
                "today's parse found no lines"
            ],
        },
    ]

    request = {"ids": [phantom_id], "reason": "linked deposit line"}
    dry = await client.post("/api/transactions/delete-batch", json=request)
    assert dry.status_code == 200
    assert dry.json()["dry_run"] is True
    session.expire_all()
    assert await session.get(Transaction, phantom_id) is not None
    assert await session.scalar(select(func.count(AuditAction.id))) == 1

    response = await client.post(
        "/api/transactions/delete-batch", json=request | {"dry_run": False}
    )

    assert response.status_code == 200
    report = response.json()["deletions"][0]
    assert report["transaction"]["counterparty"] == "SYNTHETIC PHANTOM"
    assert report["cleaned_references"] == {
        "audit_actions": 1,
        "bank_statement_uploads.reconciliation_data": 1,
    }
    session.expire_all()
    remaining = await session.scalars(select(Transaction.id))
    assert sorted(remaining) == sorted(set(ids.values()) - {phantom_id})
    assert await session.scalar(select(TelegramMessageContext.transaction_id)) is None
    record = await session.scalar(
        select(AuditAction).where(AuditAction.action_type == "delete_transaction")
    )
    assert record is not None and record.target_id == phantom_id
    assert json.loads(record.arguments_json or "{}") == {
        "reason": "linked deposit line"
    }
    assert json.loads(record.before_json or "{}")["amount"] == "9000.00"
    stored = await session.get(BankStatementUpload, stmt)
    assert stored is not None
    assert json.loads(stored.reconciliation_data or "{}") == {
        "matched": [
            {
                "stmt_idx": 0,
                "db_txn_id": real_id,
                "candidate_transaction_ids": [real_id],
            }
        ],
        "missing": [],
    }
