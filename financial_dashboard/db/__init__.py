"""Database package with compatibility exports."""

from sqlalchemy import event
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
    MerchantRule,
    Setting,
    SmsMessage,
    SnapshotHolding,
    StatementUpload,
    Transaction,
)

engine = create_async_engine(settings.db_url, echo=False)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record) -> None:
    """Apply WAL journaling + a busy timeout on each raw SQLite connection.

    WAL lets readers and a writer proceed concurrently; busy_timeout makes a
    contended write wait (up to 5s) instead of failing immediately with
    'database is locked'. No-op on non-sqlite dialects. Foreign-key enforcement
    is intentionally NOT set here.
    """
    if engine.dialect.name != "sqlite":
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


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
    "MerchantRule",
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
