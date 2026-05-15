"""SQLAlchemy ORM models."""

import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Time,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from financial_dashboard.db.enums import PaymentStatus


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class EmailSource(Base):
    __tablename__ = "email_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    account_identifier: Mapped[str | None] = mapped_column(String)
    credentials: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool | None] = mapped_column(Boolean, default=True)
    sync_cursor: Mapped[str | None] = mapped_column(String)
    last_synced_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    last_error: Mapped[str | None] = mapped_column(String)


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bank: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    account_number: Mapped[str | None] = mapped_column(String)
    statement_password: Mapped[str | None] = mapped_column(String)
    statement_password_hint: Mapped[str | None] = mapped_column(String)
    active: Mapped[bool | None] = mapped_column(Boolean, default=True)

    cards: Mapped[list["Card"]] = relationship(
        lazy="selectin", order_by="Card.is_primary.desc(), Card.id"
    )


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=False
    )
    card_mask: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str | None] = mapped_column(String)
    is_primary: Mapped[bool | None] = mapped_column(Boolean, default=False)
    active: Mapped[bool | None] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("account_id", "card_mask", name="uq_card_account_mask"),
        Index("ix_cards_card_mask", "card_mask"),
    )


class FetchRule(Base):
    __tablename__ = "fetch_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("email_sources.id"), nullable=True
    )
    sender: Mapped[str | None] = mapped_column(String)
    subject: Mapped[str | None] = mapped_column(String)
    bank: Mapped[str] = mapped_column(String, nullable=False)
    folder: Mapped[str | None] = mapped_column(String)
    email_kind: Mapped[str | None] = mapped_column(String)
    enabled: Mapped[bool | None] = mapped_column(Boolean, default=True)
    initial_backfill_done_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)

    source: Mapped["EmailSource | None"] = relationship(lazy="joined")


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    message_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    source_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("email_sources.id"), nullable=True
    )
    remote_id: Mapped[str | None] = mapped_column(String)
    sender: Mapped[str | None] = mapped_column(String)
    subject: Mapped[str | None] = mapped_column(String)
    received_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    fetched_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )
    status: Mapped[str | None] = mapped_column(String, default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    rule_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("fetch_rules.id"))

    __table_args__ = (
        Index("ix_emails_fetched_at", "fetched_at"),
        UniqueConstraint("source_id", "remote_id", name="uq_email_source_remote"),
    )


class StatementUpload(Base):
    __tablename__ = "statement_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=False
    )
    email_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("emails.id"), nullable=True
    )
    bank: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    source_kind: Mapped[str] = mapped_column(
        String, nullable=False, server_default="pdf", default="pdf"
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="parsed")
    card_number: Mapped[str | None] = mapped_column(String)
    statement_name: Mapped[str | None] = mapped_column(String)
    due_date: Mapped[str | None] = mapped_column(String)
    total_amount_due: Mapped[str | None] = mapped_column(String)
    minimum_amount_due: Mapped[str | None] = mapped_column(String)
    parsed_txn_count: Mapped[int | None] = mapped_column(Integer, default=0)
    matched_count: Mapped[int | None] = mapped_column(Integer, default=0)
    missing_count: Mapped[int | None] = mapped_column(Integer, default=0)
    imported_count: Mapped[int | None] = mapped_column(Integer, default=0)
    reconciliation_data: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )
    payment_status: Mapped[PaymentStatus | None] = mapped_column(String)
    payment_sent_offsets: Mapped[str | None] = mapped_column(Text, default="[]")
    payment_last_reminded_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    payment_paid_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    payment_paid_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=12, scale=2), default=0
    )

    account: Mapped["Account"] = relationship(lazy="joined")


class BankStatementUpload(Base):
    __tablename__ = "bank_statement_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=False
    )
    email_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("emails.id"), nullable=True
    )
    bank: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="parsed")
    account_number: Mapped[str | None] = mapped_column(String)
    account_holder_name: Mapped[str | None] = mapped_column(String)
    opening_balance: Mapped[str | None] = mapped_column(String)
    closing_balance: Mapped[str | None] = mapped_column(String)
    statement_period_start: Mapped[str | None] = mapped_column(String)
    statement_period_end: Mapped[str | None] = mapped_column(String)
    parsed_txn_count: Mapped[int | None] = mapped_column(Integer, default=0)
    matched_count: Mapped[int | None] = mapped_column(Integer, default=0)
    missing_count: Mapped[int | None] = mapped_column(Integer, default=0)
    imported_count: Mapped[int | None] = mapped_column(Integer, default=0)
    reconciliation_data: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )

    account: Mapped["Account"] = relationship(lazy="joined")


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("emails.id"))
    account_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=True
    )
    card_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cards.id"), nullable=True
    )
    statement_upload_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("statement_uploads.id"), nullable=True
    )
    bank_statement_upload_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("bank_statement_uploads.id"), nullable=True
    )

    account: Mapped["Account | None"] = relationship(lazy="joined")
    card: Mapped["Card | None"] = relationship(lazy="joined")

    bank: Mapped[str] = mapped_column(String, nullable=False)
    email_type: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=12, scale=2), nullable=False
    )
    currency: Mapped[str | None] = mapped_column(String, default="INR")
    transaction_date: Mapped[datetime.date | None] = mapped_column(Date)
    transaction_time: Mapped[datetime.time | None] = mapped_column(Time)
    counterparty: Mapped[str | None] = mapped_column(String)
    card_mask: Mapped[str | None] = mapped_column(String)
    account_mask: Mapped[str | None] = mapped_column(String)
    reference_number: Mapped[str | None] = mapped_column(String)
    channel: Mapped[str | None] = mapped_column(String)
    balance: Mapped[Decimal | None] = mapped_column(Numeric(precision=12, scale=2))
    raw_description: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, default=utc_now
    )

    __table_args__ = (
        Index("ix_transactions_transaction_date", "transaction_date"),
        Index("ix_transactions_bank", "bank"),
        Index(
            "uq_transactions_ref",
            "bank",
            "reference_number",
            "direction",
            unique=True,
            sqlite_where=text("reference_number IS NOT NULL"),
        ),
    )


class SmsMessage(Base):
    __tablename__ = "sms_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bank: Mapped[str] = mapped_column(String, nullable=False)
    sender: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=utc_now
    )

    __table_args__ = (
        UniqueConstraint(
            "sender", "received_at", "body", name="uq_sms_sender_received_body"
        ),
    )
