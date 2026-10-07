"""Email parsing helpers."""

import datetime
import email as email_lib
import email.utils
from email.header import Header, decode_header


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
    # Header keeps the RFC 2047 spacing. A plain " ".join doubles the space
    # next to an encoded word.
    header = Header()
    for part, charset in decode_header(raw):
        header.append(part, charset, errors="replace")
    return str(header)


def _extract_message_metadata(raw_bytes: bytes) -> dict:
    """Extract sender, subject, date from raw email bytes."""
    msg = email_lib.message_from_bytes(raw_bytes)
    return {
        "sender": _decode_header_value(msg.get("From", "")),
        "subject": _decode_header_value(msg.get("Subject", "")),
        "date": _decode_header_value(msg.get("Date", "")),
    }
