"""Email body and spool helpers."""

import asyncio
import email as email_lib
import logging
import re
import time
from pathlib import Path
from typing import NamedTuple

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


class RawEmailLoadResult(NamedTuple):
    raw_bytes: bytes | None
    error: str | None


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


async def load_or_fetch_raw_email(email_row) -> RawEmailLoadResult:
    """Return the raw .eml for an ``Email`` row, preferring the local spool
    and falling back to a live provider fetch when the spool has expired.

    The failed spool is not a permanent archive — ``_cleanup_failed_spool``
    deletes anything older than FAILED_SPOOL_MAX_AGE_DAYS — so every retry
    path needs to tolerate a missing file. Returns a ``RawEmailLoadResult``
    (NamedTuple) with ``(raw_bytes, error)`` set on success/failure
    respectively; positional unpacking still works. Does not mutate
    ``email_row``.
    """
    spool_path = _spool_path_for(email_row.provider, email_row.message_id)
    if spool_path.exists():
        return RawEmailLoadResult(spool_path.read_bytes(), None)

    if not email_row.source_id or not email_row.remote_id:
        return RawEmailLoadResult(
            None,
            f"Spool file missing ({spool_path.name}) and no source/remote ID to re-fetch",
        )

    async with async_session() as session:
        source = await session.get(EmailSource, email_row.source_id)
    if not source:
        return RawEmailLoadResult(
            None, f"Email source {email_row.source_id} not found for re-fetch"
        )

    try:
        creds = decrypt_credentials(source.credentials)
    except Exception as e:
        return RawEmailLoadResult(None, f"Credential decryption failed: {e}")

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
        return RawEmailLoadResult(None, f"Unknown provider {source.provider!r}")

    if not raw:
        return RawEmailLoadResult(
            None, "Provider returned no data (email may have been deleted)"
        )

    logger.info(
        "Re-fetched email %s from %s (spool was missing)",
        email_row.message_id,
        source.provider,
    )
    return RawEmailLoadResult(raw, None)


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
