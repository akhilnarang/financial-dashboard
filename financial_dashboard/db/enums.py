"""Database enum definitions."""

from enum import StrEnum


class PaymentStatus(StrEnum):
    UNPAID = "unpaid"
    PARTIALLY_PAID = "partially_paid"
    PAID = "paid"
    LATE = "late"


class EmailKind(StrEnum):
    TRANSACTION = "transaction"
    CC_STATEMENT = "cc_statement"
    BANK_STATEMENT = "bank_statement"
    # Legacy: kept so older fetch_rules rows don't break. Treated as
    # "try both statement pipelines" in dispatch.
    STATEMENT = "statement"
