"""Database package with compatibility exports."""

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session as _Session

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
    ExtensionRun,
    ExtensionSyncState,
    FetchRule,
    InvestmentLot,
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


# Commit-aware wake hook for in-process coordinators (Paisa). This is a pure
# latency optimization: a registered wake signal fires on every committed
# session so a coordinator reacts to a committed change sooner than its 2s
# poll. It never gates the coordinator and never fails a commit. Registered on
# the sync ``Session`` class — the class that backs ``AsyncSession`` — so it
# fires for both sync and async commits app-wide. The import is deferred to the
# handler body to avoid a db <-> services.paisa import cycle at module-eval
# time (at commit time the package graph is fully initialized). When no
# coordinator is registered the fire iterates an empty list — a no-op.
@event.listens_for(_Session, "after_commit")
def _signal_commit_wake(session) -> None:  # noqa: ANN001
    from financial_dashboard.services.paisa.wakeup import _fire_commit_wake

    _fire_commit_wake()


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
    "ExtensionRun",
    "ExtensionSyncState",
    "FetchRule",
    "InvestmentLot",
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
