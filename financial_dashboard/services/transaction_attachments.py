"""Bounded storage and retrieval helpers for transaction receipt files."""

import os
import json
import logging
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.config import settings
from financial_dashboard.db.models import AuditAction, AuditInteraction, Transaction

logger = logging.getLogger(__name__)

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
_SNIFF_BYTES = 16


class AttachmentError(ValueError):
    """A receipt file failed a deterministic storage validation."""


class StoredAttachment(NamedTuple):
    relative_path: str
    absolute_path: Path
    media_type: str


class AttachmentType(NamedTuple):
    """Validated media type and server-owned suffix."""

    media_type: str
    suffix: str


class AttachmentMutation(NamedTuple):
    """Metadata needed by the caller to commit and clean up safely."""

    transaction_id: int
    new_path: str
    old_path: str | None
    audit_action_id: int


def attachment_root() -> Path:
    """Return the configured, resolved receipt root and create it if needed."""
    root = Path(settings.transaction_attachment_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_attachment_path(relative_path: str) -> Path:
    """Resolve a persisted relative path while rejecting path traversal."""
    if not relative_path or Path(relative_path).is_absolute():
        raise AttachmentError("Invalid attachment path")
    root = attachment_root()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AttachmentError("Invalid attachment path") from exc
    return target


def detect_attachment_type(header: bytes) -> AttachmentType:
    """Identify the small set of file signatures accepted from Telegram."""
    if header.startswith(b"%PDF-"):
        return AttachmentType("application/pdf", ".pdf")
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return AttachmentType("image/png", ".png")
    if header.startswith(b"\xff\xd8\xff"):
        return AttachmentType("image/jpeg", ".jpg")
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return AttachmentType("image/webp", ".webp")
    raise AttachmentError("Only PDF, JPEG, PNG, and WebP receipts are supported")


async def download_attachment(
    url: str,
    *,
    transaction_id: int,
    declared_size: int | None,
    client: httpx.AsyncClient | None = None,
) -> StoredAttachment:
    """Stream one Telegram file to a temporary file, then atomically publish it.

    Both Telegram's declared size and the bytes actually received are bounded.
    The returned file is not yet attached to a database row; callers remove it
    if their subsequent transaction fails.
    """
    if declared_size is not None and declared_size > MAX_ATTACHMENT_BYTES:
        raise AttachmentError("Receipt exceeds the 20 MB limit")

    root = attachment_root()
    token = uuid4().hex
    temporary = root / f".txn-{transaction_id}-{token}.part"
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=60.0, follow_redirects=False)
    received = 0
    header = bytearray()
    try:
        async with http.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    parsed_content_length = int(content_length)
                except ValueError as exc:
                    raise AttachmentError("Invalid receipt content length") from exc
                if parsed_content_length > MAX_ATTACHMENT_BYTES:
                    raise AttachmentError("Receipt exceeds the 20 MB limit")
            with temporary.open("xb") as output:
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > MAX_ATTACHMENT_BYTES:
                        raise AttachmentError("Receipt exceeds the 20 MB limit")
                    if len(header) < _SNIFF_BYTES:
                        header.extend(chunk[: _SNIFF_BYTES - len(header)])
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if received == 0:
            raise AttachmentError("Receipt file is empty")
        media_type, suffix = detect_attachment_type(bytes(header))
        filename = f"txn-{transaction_id}-{token}{suffix}"
        target = root / filename
        temporary.replace(target)
        return StoredAttachment(filename, target, media_type)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if owns_client:
            await http.aclose()


def remove_attachment(relative_path: str | None) -> None:
    """Best-effort caller-facing primitive for deleting a stored receipt."""
    if relative_path:
        resolve_attachment_path(relative_path).unlink(missing_ok=True)


async def attach_downloaded_attachment(
    session: AsyncSession,
    transaction_id: int,
    stored: StoredAttachment,
    *,
    caption: str | None = None,
    interaction_id: int | None = None,
    worker_token: str | None = None,
) -> AttachmentMutation:
    """Attach a downloaded file and create its audit action without committing.

    The transaction is deliberately reloaded after the network download.  A
    processing worker token, when supplied, fences the write to the interaction
    that owns it.  The returned old path must only be removed after the caller's
    commit succeeds; on any exception the caller should remove ``stored``.
    """
    if worker_token is not None:
        if interaction_id is None:
            raise AttachmentError("worker token requires an interaction")
        owner = await session.scalar(
            select(AuditInteraction.id).where(
                AuditInteraction.id == interaction_id,
                AuditInteraction.status == "processing",
                AuditInteraction.worker_token == worker_token,
            )
        )
        if owner is None:
            raise AttachmentError("interaction processing lease is no longer valid")

    # populate_existing makes a fresh database read even if the identity map
    # contains the row loaded before Telegram's file download completed.
    txn = await session.scalar(
        select(Transaction)
        .execution_options(populate_existing=True)
        .where(Transaction.id == transaction_id)
    )
    if txn is None:
        raise AttachmentError("transaction not found")
    old_path = txn.attachment_path
    before = {"attachment_path": old_path, "note": txn.note}
    txn.attachment_path = stored.relative_path
    if caption is not None:
        txn.note = caption
    await session.flush()
    after = {"attachment_path": txn.attachment_path, "note": txn.note}
    action = AuditAction(
        interaction_id=interaction_id,
        action_type="attach_transaction_file",
        target_type="transaction",
        target_id=transaction_id,
        arguments_json=json.dumps({"media_type": stored.media_type}, sort_keys=True),
        before_json=json.dumps(before, sort_keys=True),
        after_json=json.dumps(after, sort_keys=True),
        status="applied",
    )
    session.add(action)
    await session.flush()
    return AttachmentMutation(transaction_id, stored.relative_path, old_path, action.id)


def cleanup_replaced_attachment(relative_path: str | None) -> bool:
    """Delete an old file after commit, logging a recoverable warning on failure."""
    if not relative_path:
        return True
    try:
        remove_attachment(relative_path)
    except (AttachmentError, OSError) as exc:
        logger.warning(
            "Could not remove replaced transaction attachment %s: %s",
            relative_path,
            exc,
        )
        return False
    return True


async def record_attachment_cleanup_warning(
    session: AsyncSession, action_id: int, relative_path: str
) -> None:
    """Annotate an audit action when post-commit file cleanup needs attention."""
    action = await session.get(AuditAction, action_id)
    if action is not None:
        action.error_code = f"old_attachment_cleanup_failed:{relative_path}"
        await session.commit()
