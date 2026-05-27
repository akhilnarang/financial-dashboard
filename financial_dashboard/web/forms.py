"""Shared web form helpers."""

from __future__ import annotations

from pathlib import Path

from financial_dashboard.core.uploads import STATEMENTS_DIR


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
