"""Shared web form helpers."""

from __future__ import annotations

import re
from pathlib import Path

STATEMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "statements"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_upload_filename(filename: str | None) -> str:
    """Strip any path components and restrict to a safe character set."""
    base = Path(filename or "statement.pdf").name or "statement.pdf"
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "statement.pdf"
    return cleaned[:120]


def _unlink_statement_file(path_str: str | None) -> None:
    """Delete a statement PDF, but only if it resolves inside STATEMENTS_DIR."""
    if not path_str:
        return
    try:
        target = Path(path_str).resolve()
        target.relative_to(STATEMENTS_DIR.resolve())
    except ValueError, OSError:
        return
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass
