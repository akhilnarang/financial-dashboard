"""Populate the database from .eml files stored under data/<bank>/.

Walks all .eml files under data/, determines the bank from the parent folder,
parses each email, and inserts Email + Transaction rows into the DB.

Usage:
    uv run python populate.py
"""

import asyncio
import datetime
import email as email_lib
import email.utils
import logging
from email.header import decode_header
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from bank_email_parser.api import parse_email
from bank_email_parser.exceptions import ParseError, UnsupportedEmailTypeError

from financial_dashboard.db import Base, Email, Transaction, async_session, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "financial_dashboard.db"

KNOWN_BANKS = {
    "axis",
    "equitas",
    "hdfc",
    "hsbc",
    "icici",
    "idfc",
    "indusind",
    "kotak",
    "onecard",
    "sbi",
    "slice",
    "uboi",
}


# ---------------------------------------------------------------------------
# Helpers (reused from fetcher.py / main.py patterns)
# ---------------------------------------------------------------------------


def _decode_header_value(raw: str | None) -> str:
    """Decode an RFC 2047 encoded header value."""
    if not raw:
        return ""
    parts = decode_header(raw)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return " ".join(decoded)


def _extract_html_body(raw_bytes: bytes) -> str | None:
    """Extract HTML (or plain-text fallback) body from raw email bytes."""
    msg = email_lib.message_from_bytes(raw_bytes)

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
    else:
        ct = msg.get_content_type()
        if ct in ("text/html", "text/plain"):
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                charset = msg.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
    return None


def _parse_email_date(msg: email_lib.message.Message) -> datetime.datetime | None:
    """Parse the Date header from an email.message.Message."""
    date_str = msg.get("Date")
    if not date_str:
        return None
    try:
        return email.utils.parsedate_to_datetime(date_str)
    except ValueError, TypeError:
        return None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _process_eml(bank: str, raw_bytes: bytes) -> tuple[str | None, dict | None]:
    """Parse a single .eml file. Returns (error, transaction_dict)."""
    html = _extract_html_body(raw_bytes)
    if not html:
        return "No HTML/text body found in email", None

    try:
        parsed = parse_email(bank, html)
    except (ParseError, UnsupportedEmailTypeError) as e:
        return str(e), None

    txn = parsed.transaction
    if txn is None:
        return None, None
    return None, {
        "bank": parsed.bank,
        "email_type": parsed.email_type,
        "direction": txn.direction,
        "amount": float(txn.amount.amount),
        "currency": txn.amount.currency,
        "transaction_date": txn.transaction_date,
        "counterparty": txn.counterparty,
        "card_mask": txn.card_mask,
        "account_mask": txn.account_mask,
        "reference_number": txn.reference_number,
        "channel": txn.channel,
        "balance": float(txn.balance.amount) if txn.balance else None,
        "raw_description": txn.raw_description,
    }


async def populate() -> None:
    """Walk .eml files, parse, and populate the DB."""

    # Delete existing DB for a fresh start
    if DB_PATH.exists():
        logger.info("Removing existing database: %s", DB_PATH)
        DB_PATH.unlink()

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema created")

    # Collect all .eml files (excluding the failed/ spool)
    eml_files: list[tuple[str, Path]] = []
    for bank_dir in sorted(DATA_DIR.iterdir()):
        if not bank_dir.is_dir():
            continue
        bank_name = bank_dir.name.lower()
        if bank_name not in KNOWN_BANKS:
            continue
        for eml_path in sorted(bank_dir.rglob("*.eml")):
            eml_files.append((bank_name, eml_path))

    logger.info(
        "Found %d .eml files across %d banks",
        len(eml_files),
        len({b for b, _ in eml_files}),
    )

    stats = {"processed": 0, "transactions": 0, "failures": 0, "skipped_dup": 0}
    known_message_ids: set[str] = set()

    for bank, eml_path in eml_files:
        raw_bytes = eml_path.read_bytes()
        msg = email_lib.message_from_bytes(raw_bytes)

        # Extract metadata
        message_id = msg.get("Message-ID", "").strip()
        if not message_id:
            # Fallback: use filename as a pseudo message-id
            message_id = f"file:{eml_path.name}"

        sender = _decode_header_value(msg.get("From", ""))
        subject = _decode_header_value(msg.get("Subject", ""))
        received_at = _parse_email_date(msg)

        # Skip duplicates by message_id
        if message_id in known_message_ids:
            logger.debug("Skipping duplicate: %s", eml_path.name)
            stats["skipped_dup"] += 1
            continue
        known_message_ids.add(message_id)

        # Parse the email content
        error, txn_data = _process_eml(bank, raw_bytes)

        # Insert into DB
        async with async_session() as session:
            async with session.begin():
                # Check DB-level duplicate (in case of re-runs without fresh DB)
                existing = await session.execute(
                    select(Email.id).where(Email.message_id == message_id)
                )
                if existing.scalar_one_or_none() is not None:
                    stats["skipped_dup"] += 1
                    continue

                email_row = Email(
                    provider="file",
                    message_id=message_id,
                    sender=sender,
                    subject=subject,
                    received_at=received_at,
                    status="parsed" if txn_data else "failed",
                    error=error,
                )
                session.add(email_row)
                await session.flush()
                email_row_id = email_row.id

            if txn_data:
                try:
                    async with session.begin():
                        txn_row = Transaction(email_id=email_row_id, **txn_data)
                        session.add(txn_row)
                    stats["transactions"] += 1
                    logger.info(
                        "OK  %-8s %-50s %s %s %s",
                        bank,
                        subject[:50],
                        txn_data["direction"],
                        txn_data["amount"],
                        txn_data.get("counterparty", ""),
                    )
                except IntegrityError:
                    stats["skipped_dup"] += 1
                    logger.info(
                        "DUP  %-8s %-50s (duplicate transaction)",
                        bank,
                        subject[:50],
                    )
            else:
                stats["failures"] += 1
                logger.warning(
                    "FAIL %-8s %-50s %s",
                    bank,
                    subject[:50],
                    error,
                )

        stats["processed"] += 1

    # Summary
    print()
    print("=" * 60)
    print(f"  Emails processed:      {stats['processed']}")
    print(f"  Transactions created:  {stats['transactions']}")
    print(f"  Failures:              {stats['failures']}")
    print(f"  Skipped (duplicates):  {stats['skipped_dup']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(populate())
