import datetime
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Account, BankStatementUpload, Transaction

pytestmark = pytest.mark.anyio

RANGE = {"date_from": "2030-01-01", "date_to": "2030-03-31"}


async def _accounts(session: AsyncSession) -> tuple[Account, Account]:
    first = Account(bank="synth", label="First", type="bank_account")
    second = Account(bank="synth", label="Second", type="bank_account")
    session.add_all([first, second])
    await session.flush()
    return first, second


def _txn(
    account: Account, day: int, direction: str, amount: str, month: int = 1, **fields
) -> Transaction:
    return Transaction(
        account_id=account.id,
        bank="synth",
        email_type=fields.pop("email_type", "synth_alert"),
        direction=direction,
        amount=Decimal(amount),
        transaction_date=datetime.date(2030, month, day),
        **fields,
    )


async def test_balance_gap_flagged_and_matching_month_zero(client, session):
    first, _ = await _accounts(session)
    session.add_all(
        [
            BankStatementUpload(
                account_id=first.id,
                bank="synth",
                filename="jan.pdf",
                file_path="/synthetic/jan.pdf",
                opening_balance="1,00,000.00",
                closing_balance="1,00,500.00",
                statement_period_start="01/01/2030",
                statement_period_end="31/01/2030",
            ),
            BankStatementUpload(
                account_id=first.id,
                bank="synth",
                filename="feb.pdf",
                file_path="/synthetic/feb.pdf",
                opening_balance="1,00,500.00",
                closing_balance="1,00,300.00",
                # A statement that starts mid-month still covers its own days.
                statement_period_start="05/02/2030",
                statement_period_end="28/02/2030",
            ),
            _txn(first, 10, "credit", "500"),
            _txn(first, 15, "credit", "77", currency="USD"),
            _txn(first, 2, "debit", "9", month=2),
            _txn(first, 10, "debit", "100", month=2),
            _txn(first, 5, "debit", "40", month=3),
        ]
    )
    await session.commit()

    body = (await client.get("/api/reconcile", params=RANGE)).json()

    gaps = {s["period_start"]: Decimal(s["gap"]) for s in body["statements"]}
    assert gaps == {"2030-01-01": Decimal(0), "2030-02-05": Decimal(100)}
    assert body["missing_statements"] == [
        {"account_id": first.id, "months": ["2030-02", "2030-03"]}
    ]


async def test_unpaired_self_transfer_flagged_and_pair_not(client, session):
    first, second = await _accounts(session)
    paired = _txn(first, 5, "debit", "1000", category="self_transfer")
    partner = _txn(second, 7, "credit", "999.50", category="self_transfer")
    orphan = _txn(first, 10, "debit", "2000", category="self_transfer")
    # A leg on the same account is not the other side of a transfer.
    same_account = _txn(first, 11, "credit", "2000", category="self_transfer")
    # A foreign-currency leg does not pair and stays out of the INR net.
    foreign = _txn(
        second, 10, "credit", "2000", category="self_transfer", currency="USD"
    )
    # Two debit legs claim one credit. The leg with the shared reference wins it.
    twin = _txn(first, 20, "debit", "3000", category="self_transfer")
    referenced = _txn(
        first, 20, "debit", "3000", category="self_transfer", reference_number="R1"
    )
    credit = _txn(
        second, 20, "credit", "3000", category="self_transfer", reference_number="R1"
    )
    session.add_all(
        [paired, partner, orphan, same_account, foreign, twin, referenced, credit]
    )
    await session.commit()

    # The leg dated before the range still pairs with its partner inside it.
    params = {**RANGE, "date_from": "2030-01-06"}
    body = (await client.get("/api/reconcile", params=params)).json()
    unpaired = body["unpaired_self_transfers"]

    assert [i["id"] for i in unpaired["items"]] == [
        orphan.id,
        foreign.id,
        same_account.id,
        twin.id,
    ]
    assert Decimal(unpaired["net"]) == Decimal("-3000")


async def test_duplicate_candidate_flagged_and_distinct_refs_not(client, session):
    first, _ = await _accounts(session)
    statement = BankStatementUpload(
        account_id=first.id,
        bank="synth",
        filename="jan.pdf",
        file_path="/synthetic/jan.pdf",
    )
    session.add(statement)
    await session.flush()
    from_statement = {
        "email_type": "bank_statement",
        "bank_statement_upload_id": statement.id,
    }
    stmt_dup = _txn(
        first, 12, "debit", "700", reference_number="ABCD123456789", **from_statement
    )
    # An alert can carry a shortened form of the statement reference.
    alert_dup = _txn(
        first, 11, "debit", "700", reference_number="ABCD1234", source="sms"
    )
    stmt_other = _txn(
        first, 15, "debit", "800", reference_number="AAAA1111", **from_statement
    )
    alert_other = _txn(
        first, 15, "debit", "800", reference_number="BBBB2222", source="sms"
    )
    foreign = _txn(first, 12, "debit", "700", currency="USD", source="email")
    short_ref = _txn(first, 13, "debit", "700", reference_number="ABC-", source="sms")
    session.add_all([stmt_dup, alert_dup, short_ref, stmt_other, alert_other, foreign])
    await session.commit()

    body = (await client.get("/api/reconcile", params=RANGE)).json()

    assert [
        (p["statement_txn"]["id"], p["alert_txn"]["id"])
        for p in body["duplicate_candidates"]
    ] == [(stmt_dup.id, alert_dup.id), (stmt_dup.id, short_ref.id)]
