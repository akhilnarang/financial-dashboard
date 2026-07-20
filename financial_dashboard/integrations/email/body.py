"""Email body and spool helpers."""

import asyncio
import email as email_lib
import logging
import re
import time
from pathlib import Path
from typing import Literal, NamedTuple

from financial_dashboard.core.crypto import decrypt_credentials
from financial_dashboard.db import EmailSource, async_session
from financial_dashboard.integrations.email.base import (
    FAILED_SPOOL_DIR,
    FAILED_SPOOL_MAX_AGE_DAYS,
)
from financial_dashboard.integrations.email.imap_gmail import _fetch_gmail_single_sync
from financial_dashboard.integrations.email.jmap_fastmail import (
    _fetch_fastmail_single_sync,
)

logger = logging.getLogger(__name__)


class RawEmailResult(NamedTuple):
    """Raw email bytes plus a safe account of where they were loaded."""

    raw_bytes: bytes | None
    error: str | None
    provenance: Literal["spool", "provider"] | None


def _save_failed_email(provider: str, message_id: str, raw_bytes: bytes) -> None:
    """Save raw .eml to the failed spool directory for debugging."""
    FAILED_SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitize message_id for use as filename
    safe_id = re.sub(r"[^\w\-.]", "_", message_id)
    path = FAILED_SPOOL_DIR / f"{provider}_{safe_id}.eml"
    path.write_bytes(raw_bytes)
    logger.info("Saved failed email to %s", path)


def _spool_path_for(provider: str, message_id: str) -> Path:
    """Location on disk where this email's .eml would be (if spooled)."""
    safe_id = re.sub(r"[^\w\-.]", "_", message_id)
    return FAILED_SPOOL_DIR / f"{provider}_{safe_id}.eml"


async def load_or_fetch_raw_email(email_row) -> RawEmailResult:
    """Load raw bytes and report only safe spool/provider provenance.

    The failed spool is a short-lived debugging cache, not a permanent archive;
    callers must therefore tolerate provider fallback whenever cleanup has
    removed the local copy.
    """
    spool_path = _spool_path_for(email_row.provider, email_row.message_id)
    if spool_path.exists():
        return RawEmailResult(spool_path.read_bytes(), None, "spool")

    if not email_row.source_id or not email_row.remote_id:
        return RawEmailResult(
            None,
            f"Spool file missing ({spool_path.name}) and no source/remote ID to re-fetch",
            None,
        )

    async with async_session() as session:
        source = await session.get(EmailSource, email_row.source_id)
    if not source:
        return RawEmailResult(
            None, f"Email source {email_row.source_id} not found for re-fetch", None
        )

    try:
        creds = decrypt_credentials(source.credentials)
    except Exception as e:
        return RawEmailResult(None, f"Credential decryption failed: {e}", None)

    if source.provider == "gmail":
        raw = await asyncio.to_thread(
            _fetch_gmail_single_sync,
            creds["user"],
            creds["app_password"],
            email_row.remote_id,
        )
    elif source.provider == "fastmail":
        raw = await asyncio.to_thread(
            _fetch_fastmail_single_sync, creds["token"], email_row.remote_id
        )
    else:
        return RawEmailResult(None, f"Unknown provider {source.provider!r}", None)

    if not raw:
        return RawEmailResult(
            None, "Provider returned no data (email may have been deleted)", None
        )

    logger.info(
        "Re-fetched email %s from %s (spool was missing)",
        email_row.message_id,
        source.provider,
    )
    return RawEmailResult(raw, None, "provider")


def _cleanup_failed_spool() -> None:
    """Delete .eml files in the failed spool older than FAILED_SPOOL_MAX_AGE_DAYS."""
    if not FAILED_SPOOL_DIR.exists():
        return
    cutoff = time.time() - (FAILED_SPOOL_MAX_AGE_DAYS * 86400)
    for path in FAILED_SPOOL_DIR.glob("*.eml"):
        if path.stat().st_mtime < cutoff:
            path.unlink()
            logger.debug("Cleaned up old failed email: %s", path.name)


def _extract_body_by_type(raw_bytes: bytes, content_type: str) -> str | None:
    """Extract a body of ``content_type`` from raw email bytes.

    ``Message.get_payload(decode=True)`` returns bytes for leaf parts
    per the documented behavior, but stub annotations widen it to a
    union (Message | bytes | Any). Guard the decode with an
    ``isinstance`` so the union is narrowed before ``.decode()``.
    """
    msg = email_lib.message_from_bytes(raw_bytes)

    def _decode(part) -> str | None:
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes) or not payload:
            return None
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == content_type:
                decoded = _decode(part)
                if decoded is not None:
                    return decoded
    elif msg.get_content_type() == content_type:
        return _decode(msg)
    return None


def _extract_html_body(raw_bytes: bytes) -> str | None:
    """Extract the HTML body from raw email bytes."""
    return _extract_body_by_type(raw_bytes, "text/html")


def _extract_text_body(raw_bytes: bytes) -> str | None:
    """Extract the plain-text body from raw email bytes."""
    return _extract_body_by_type(raw_bytes, "text/plain")
