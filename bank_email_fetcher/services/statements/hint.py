"""Password hint extraction helper shared by the CC and bank statement pipelines."""

import email as email_lib

from bank_email_fetcher.integrations.parsers import (
    ParseError,
    UnsupportedEmailTypeError,
    parse_transaction_email,
)


def _extract_html_from_email(raw_bytes: bytes) -> str | None:
    msg = email_lib.message_from_bytes(raw_bytes)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    return payload.decode("utf-8", errors="replace")
    elif msg.get_content_type() == "text/html":
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
    return None


def extract_password_hint(raw_bytes: bytes, bank: str) -> str | None:
    """Fallback hint extraction when the hint wasn't threaded from the parse step."""
    if not (html := _extract_html_from_email(raw_bytes)):
        return None
    try:
        return parse_transaction_email(bank, html).password_hint
    except ParseError, UnsupportedEmailTypeError:
        return None
