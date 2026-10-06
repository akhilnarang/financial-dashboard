"""Upload-related helpers shared across web, api, and services layers."""

import datetime
import re
import secrets
from pathlib import Path

STATEMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "statements"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_upload_filename(filename: str | None) -> str:
    """Strip any path components and restrict to a safe character set."""
    base = Path(filename or "statement.pdf").name or "statement.pdf"
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "statement.pdf"
    return cleaned[:120]


def save_statement_pdf(data: bytes, safe_name: str, prefix: str = "") -> Path:
    """Write a statement PDF to a new file in STATEMENTS_DIR.

    A random token keeps two saves in the same second apart. Exclusive
    create mode makes sure a save never overwrites an existing file.

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
    with path.open("xb") as fh:
        fh.write(data)
    return path
