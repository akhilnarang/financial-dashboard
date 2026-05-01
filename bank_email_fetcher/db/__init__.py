"""Database package with compatibility exports."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bank_email_fetcher.config import settings
from bank_email_fetcher.db.enums import EmailKind, PaymentStatus
from bank_email_fetcher.db.init_db import init_db as _init_db
from bank_email_fetcher.db.models import (
    Account,
    BankStatementUpload,
    Base,
    Card,
    Email,
    EmailSource,
    FetchRule,
    Setting,
    SmsMessage,
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
    "Card",
    "Email",
    "EmailKind",
    "EmailSource",
    "FetchRule",
    "PaymentStatus",
    "Setting",
    "SmsMessage",
    "StatementUpload",
    "Transaction",
    "async_session",
    "engine",
    "init_db",
]
