"""Response shapes for the read-only bank reconciliation report."""

import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from financial_dashboard.schemas.cashflow import MonthKey
from financial_dashboard.schemas.transactions import TransactionRead


class StatementBalance(BaseModel):
    """One bank statement checked against the transactions the DB holds."""

    statement_id: int
    account_id: int
    period_start: datetime.date
    period_end: datetime.date
    opening_balance: Decimal | None
    closing_balance: Decimal | None
    statement_delta: Decimal | None = Field(
        description="closing_balance - opening_balance. Null when a balance is unreadable.",
    )
    db_net: Decimal = Field(
        description=(
            "Credits minus debits of the INR rows on the account, dated inside "
            "the period."
        ),
    )
    gap: Decimal | None = Field(
        description=(
            "db_net - statement_delta. Zero when the DB agrees with the statement. "
            "Positive means the DB holds more net credit than the bank shows."
        ),
    )


class MissingStatementMonths(BaseModel):
    """A bank account with transactions on dates that no statement covers."""

    account_id: int
    months: list[MonthKey] = Field(
        description="Months with at least one row on a date no statement covers.",
    )


class UnpairedSelfTransfers(BaseModel):
    """Self-transfers with no opposite leg on another bank account."""

    net: Decimal = Field(
        description="Signed net of the INR orphans; credits positive.",
    )
    items: list[TransactionRead]


class DuplicateCandidate(BaseModel):
    """A statement row and an alert row that may be one event the merge missed."""

    statement_txn: TransactionRead
    alert_txn: TransactionRead


class ReconcileReport(BaseModel):
    """Bank-account reconciliation for one inclusive date range."""

    date_from: datetime.date
    date_to: datetime.date
    accounts: dict[int, str] = Field(
        description="Label of every bank account, keyed by account id.",
    )
    statements: list[StatementBalance] = Field(
        description="Statements whose whole period falls inside the range.",
    )
    missing_statements: list[MissingStatementMonths]
    unpaired_self_transfers: UnpairedSelfTransfers
    duplicate_candidates: list[DuplicateCandidate]
