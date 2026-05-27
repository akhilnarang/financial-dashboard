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
    CAS_STATEMENT = "cas_statement"
    # Legacy: kept so older fetch_rules rows don't break. Treated as
    # "try both statement pipelines" in dispatch.
    STATEMENT = "statement"


class SnapshotKind(StrEnum):
    asset = "asset"
    liability = "liability"


class SnapshotCategory(StrEnum):
    bank_balance = "bank_balance"
    cc_outstanding = "cc_outstanding"
    investment = "investment"
    manual_asset = "manual_asset"
    manual_liability = "manual_liability"


class SnapshotSource(StrEnum):
    bank_statement = "bank_statement"
    cc_statement = "cc_statement"
    cas = "cas"
    manual = "manual"


class DepositorySource(StrEnum):
    nsdl = "nsdl"
    cdsl = "cdsl"


class ManualKind(StrEnum):
    asset = "asset"
    liability = "liability"


class ManualCategory(StrEnum):
    property = "property"
    epf_ppf = "epf_ppf"
    gold = "gold"
    cash = "cash"
    loan = "loan"
    other = "other"
