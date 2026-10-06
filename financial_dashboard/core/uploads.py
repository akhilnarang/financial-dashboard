"""Upload-related helpers shared across web, api, and services layers."""

import re
from pathlib import Path

from fastapi import UploadFile

from financial_dashboard.exceptions import PayloadTooLargeException

STATEMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "statements"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
PDF_UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # Real statement and CAS PDFs are under 2 MB.


def safe_upload_filename(filename: str | None) -> str:
    """Strip any path components and restrict to a safe character set."""
    base = Path(filename or "statement.pdf").name or "statement.pdf"
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "statement.pdf"
    return cleaned[:120]


async def read_bounded_pdf(file: UploadFile) -> bytes:
    """Read an uploaded PDF. Reject it with a 413 when it exceeds 10 MB.

    The read stops one byte past the limit, so an oversized file never loads whole.
    """
    payload = await file.read(PDF_UPLOAD_MAX_BYTES + 1)
    if len(payload) > PDF_UPLOAD_MAX_BYTES:
        raise PayloadTooLargeException(detail="PDF exceeds 10 MB limit.")
    return payload
