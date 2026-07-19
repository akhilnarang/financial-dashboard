"""Shared helper for summarizing per-row import skips.

CC and bank statement imports tag each failed ``recon["missing"]`` entry in
place (``duplicate`` / ``import_failed`` / ``import_error``). This module
collapses those tags into the count + human-readable ``upload.error`` string
shown on the ``StatementUpload`` / ``BankStatementUpload`` row, so the CC and
bank pipelines (initial upload, polling, manual upload, retry, reprocess) all
report skipped rows identically.

Parse failures (entries whose amount/date can't be parsed) get an
``import_error`` tag for in-recon diagnostics but are deliberately excluded
from the summary count — matching the original bank-email behavior — because
they aren't actionable import failures (the row was already malformed before
the savepoint).
"""

from typing import NamedTuple


class ImportSkipSummary(NamedTuple):
    """Duplicate/error counts and the corresponding upload error message."""

    duplicate_count: int
    error_count: int
    error_message: str | None


def import_skip_summary(recon: dict) -> ImportSkipSummary:
    """Count duplicate / unexpected-error skips and build an ``error`` blurb.

    Args:
        recon: Reconciliation dict produced by ``reconcile_statement`` /
            ``reconcile_bank_statement`` and mutated in place by the import
            helpers.

    Returns:
        ``(duplicate_count, error_count, error_msg)`` where ``error_msg`` is
        ``None`` when nothing was skipped, otherwise a single-line summary
        like ``"Skipped 2 duplicate, 1 unexpected error row(s) during
        auto-import; see reconciliation details."``. Callers assign it
        directly to ``upload.error``.
    """
    duplicate_count = sum(1 for e in recon["missing"] if e.get("duplicate"))
    error_count = sum(1 for e in recon["missing"] if e.get("import_failed"))
    error_msg = None
    if duplicate_count or error_count:
        details = []
        if duplicate_count:
            details.append(f"{duplicate_count} duplicate")
        if error_count:
            details.append(f"{error_count} unexpected error")
        error_msg = (
            f"Skipped {', '.join(details)} row(s) during auto-import; "
            "see reconciliation details."
        )
    return ImportSkipSummary(duplicate_count, error_count, error_msg)
