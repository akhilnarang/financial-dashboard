"""Upload-related helpers shared across web, api, and services layers."""

import datetime
import re
import secrets
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


def save_statement_pdf(data: bytes, safe_name: str, prefix: str = "") -> Path:
    """Write a statement PDF to a new file in STATEMENTS_DIR.

    A random token keeps two saves in the same second apart. Exclusive
    create mode makes sure a save never overwrites an existing file. A
    failed write removes the partial file.

    Args:
        data: The PDF bytes.
        safe_name: A name from safe_upload_filename.
        prefix: Optional text to put before the name.

    Returns:
        The path of the new file.
    """
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    path = STATEMENTS_DIR / f"{ts}_{secrets.token_hex(4)}_{prefix}{safe_name}"
    try:
        with path.open("xb") as fh:
            fh.write(data)
    except FileExistsError:
        # The path belongs to another file. Do not remove it.
        raise
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return path


async def read_bounded_pdf(file: UploadFile) -> bytes:
    """Read an uploaded PDF. Reject it with a 413 when it exceeds 10 MB.

    The read stops one byte past the limit, so an oversized file never loads whole.
    """
    payload = await file.read(PDF_UPLOAD_MAX_BYTES + 1)
    if len(payload) > PDF_UPLOAD_MAX_BYTES:
        raise PayloadTooLargeException(detail="PDF exceeds 10 MB limit.")
    return payload
