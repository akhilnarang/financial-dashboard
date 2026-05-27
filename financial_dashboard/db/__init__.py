"""Database package with compatibility exports."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.config import settings
from financial_dashboard.db.enums import (
    DepositorySource,
    EmailKind,
    ManualCategory,
    ManualKind,
    PaymentStatus,
    SnapshotCategory,
    SnapshotKind,
    SnapshotSource,
)
from financial_dashboard.db.init_db import init_db as _init_db
from financial_dashboard.db.models import (
    Account,
    BankStatementUpload,
    Base,
    BalanceSnapshot,
    Card,
    CasUpload,
    Email,
    EmailSource,
    FetchRule,
    ManualItem,
    Setting,
    SmsMessage,
    SnapshotHolding,
    StatementUpload,
    Transaction,
)

engine = create_async_engine(settings.db_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    await _init_db(engine)


__all__ = [
    "Account",
    "AsyncSession",
    "BankStatementUpload",
    "Base",
    "BalanceSnapshot",
    "Card",
    "CasUpload",
    "DepositorySource",
    "Email",
    "EmailKind",
    "EmailSource",
    "FetchRule",
    "ManualCategory",
    "ManualItem",
    "ManualKind",
    "PaymentStatus",
    "Setting",
    "SmsMessage",
    "SnapshotCategory",
    "SnapshotHolding",
    "SnapshotKind",
    "SnapshotSource",
    "StatementUpload",
    "Transaction",
    "async_session",
    "engine",
    "init_db",
]
