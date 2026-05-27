"""Upload-related helpers shared across web, api, and services layers."""

import re
from pathlib import Path

STATEMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "statements"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_upload_filename(filename: str | None) -> str:
    """Strip any path components and restrict to a safe character set."""
    base = Path(filename or "statement.pdf").name or "statement.pdf"
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "statement.pdf"
    return cleaned[:120]
