"""Email parsing helpers."""

from __future__ import annotations

import datetime
import email as email_lib
import email.utils
from email.header import decode_header


def _parse_email_date(raw_bytes: bytes) -> datetime.datetime | None:
    """Extract and parse the Date header from raw email bytes."""
    msg = email_lib.message_from_bytes(raw_bytes)
    date_str = msg.get("Date")
    if not date_str:
        return None
    try:
        return email.utils.parsedate_to_datetime(date_str)
    except ValueError, TypeError:
        return None


def _decode_header_value(raw: str | None) -> str:
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


def _extract_message_metadata(raw_bytes: bytes) -> dict:
    """Extract sender, subject, date from raw email bytes."""
    msg = email_lib.message_from_bytes(raw_bytes)
    return {
        "sender": _decode_header_value(msg.get("From", "")),
        "subject": _decode_header_value(msg.get("Subject", "")),
        "date": _decode_header_value(msg.get("Date", "")),
    }
