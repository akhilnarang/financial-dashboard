"""Shared builders for statement integration tests.

These tests exercise the real ``process_*_statement_email`` / retry / manual
route pipelines against an in-memory SQLite session factory, monkeypatching
only the parser adapter boundary (``_parse_pdf_bytes_sync``) and the
password-hint extractor — never the reconciliation or import services.

Fixtures live in ``conftest.py``; this module holds the parser-model,
email, and account builders so the test modules stay tiny and offline.
"""

from email.message import EmailMessage

from financial_dashboard.db.models import Account, Card

# cc_parser exposes its Pydantic models under ``parsers.models``.
from cc_parser.parsers.models import (
    AdjustmentPair,
    ParsedStatement as CcParsedStatement,
    Reconciliation as CcReconciliation,
    Transaction as CcParserTxn,
)
from bank_statement_parser.models import BankTransaction, ParsedBankStatement


# ---------------------------------------------------------------------------
# Parser-model builders
# ---------------------------------------------------------------------------


def _cc_recon():
    return CcReconciliation(
        parsed_debit_total="0",
        parsed_credit_total="0",
        parsed_net_due_estimate="0",
        header_previous_balance="0",
        header_purchases_debit="0",
        header_finance_charges="0",
        header_payments_credits_received="0",
        header_computed_due_estimate="0",
        smart_expected_total="0",
        smart_delta="0",
        delta_statement_vs_parsed_debit="0",
        delta_statement_vs_parsed_net="0",
        delta_statement_vs_header_estimate="0",
    )


def cc_txn(
    *,
    date="01/07/2026",
    amount="1,000.00",
    narration="AMAZON INDIA",
    card_number=None,
    person=None,
    transaction_type="debit",
):
    return CcParserTxn(
        date=date,
        narration=narration,
        amount=amount,
        card_number=card_number,
        person=person,
        transaction_type=transaction_type,
    )


def cc_parsed(
    *,
    bank="hdfc",
    card_number="XXXX XXXX XXXX 1234",
    transactions=None,
    payments_refunds=None,
    due_date="15/08/2026",
    total_due="5,000.00",
    minimum_due="250.00",
    adjustment_pairs=None,
    card_summaries=None,
):
    return CcParsedStatement(
        file="test.pdf",
        bank=bank,
        card_number=card_number,
        due_date=due_date,
        statement_total_amount_due=total_due,
        card_summaries=card_summaries or [],
        overall_total="0",
        person_groups=[],
        payments_refunds=payments_refunds or [],
        payments_refunds_total="0",
        overall_reward_points="0",
        transactions=transactions or [],
        reconciliation=_cc_recon(),
        possible_adjustment_pairs=adjustment_pairs or [],
    )


def cc_adjustment_pair(
    *,
    pair_id="p1",
    confidence="high",
    kind="exact_refund",
    debit=None,
    credit=None,
):
    return AdjustmentPair(
        pair_id=pair_id,
        debit_transaction_id="d1",
        credit_transaction_id="c1",
        debit=debit,
        credit=credit,
        score=100,
        confidence=confidence,
        kind=kind,
        amount_delta="0.00",
    )


def bank_txn(
    *,
    date="01/07/2026",
    amount="1,000.00",
    narration="UPI-Debit-MERCHANT",
    transaction_type="debit",
    reference_number=None,
    channel=None,
    counterparty=None,
    balance=None,
):
    return BankTransaction(
        date=date,
        narration=narration,
        amount=amount,
        transaction_type=transaction_type,
        reference_number=reference_number,
        channel=channel,
        counterparty=counterparty,
        balance=balance,
    )


def bank_parsed(
    *,
    bank="hdfc",
    account_number="1234567890",
    transactions=None,
    opening_balance=None,
    closing_balance=None,
    statement_period_start=None,
    statement_period_end=None,
    account_holder_name=None,
    debit_total="0.00",
    credit_total="0.00",
):
    return ParsedBankStatement(
        file="test.pdf",
        bank=bank,
        account_number=account_number,
        transactions=transactions or [],
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        statement_period_start=statement_period_start,
        statement_period_end=statement_period_end,
        account_holder_name=account_holder_name,
        debit_total=debit_total,
        credit_total=credit_total,
    )


# ---------------------------------------------------------------------------
# Email + account builders
# ---------------------------------------------------------------------------


def email_with_pdf(
    *,
    subject="Your credit card statement is ready",
    body="Please find attached statement",
    filename="statement.pdf",
    pdf_bytes=b"%PDF-1.4 fake content",
    from_addr="statements@hdfcbank.com",
):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["Date"] = "Thu, 17 Jul 2026 10:00:00 +0530"
    msg.set_content(body)
    msg.add_attachment(
        pdf_bytes, maintype="application", subtype="pdf", filename=filename
    )
    return msg.as_bytes()


async def add_cc_account(
    maker,
    *,
    bank="hdfc",
    label="HDFC CC",
    account_number="XXXX XXXX XXXX 1234",
    active=True,
    cards=None,
    statement_password=None,
    password_hint=None,
):
    async with maker() as session:
        acc = Account(
            bank=bank,
            label=label,
            type="credit_card",
            account_number=account_number,
            active=active,
        )
        if statement_password:
            acc.statement_password = statement_password
        if password_hint:
            acc.statement_password_hint = password_hint
        session.add(acc)
        await session.flush()
        for mask in cards or []:
            session.add(
                Card(account_id=acc.id, card_mask=mask, is_primary=True, active=True)
            )
        await session.commit()
        return acc.id


async def add_bank_account(
    maker,
    *,
    bank="hdfc",
    label="HDFC Savings",
    account_number="1234567890",
    active=True,
    statement_password=None,
    password_hint=None,
):
    async with maker() as session:
        acc = Account(
            bank=bank,
            label=label,
            type="bank_account",
            account_number=account_number,
            active=active,
        )
        if statement_password:
            acc.statement_password = statement_password
        if password_hint:
            acc.statement_password_hint = password_hint
        session.add(acc)
        await session.commit()
        return acc.id


def encrypt_password(plaintext: str) -> str:
    from financial_dashboard.config import get_fernet

    return get_fernet().encrypt(plaintext.encode()).decode()


def make_cc_parser(parsed, *, password_required=False, correct_password=None):
    """Return a fake ``_parse_pdf_bytes_sync`` for the CC pipeline.

    - No encryption: ignores args, returns ``parsed``.
    - Encrypted: raises ValueError (password) when password is None / wrong,
      returns ``parsed`` when password == correct_password.
    """

    def _fake(pdf_bytes, password=None, bank="auto"):
        if password_required and password != correct_password:
            raise ValueError("The PDF is encrypted and needs a password")
        return parsed

    return _fake


def make_bank_parser(parsed, *, password_required=False, correct_password=None):
    """Return a fake ``_parse_pdf_bytes_sync`` for the bank pipeline."""

    def _fake(pdf_bytes, bank, password=None):
        if password_required and password != correct_password:
            raise ValueError("The PDF is encrypted and needs a password")
        return parsed

    return _fake
