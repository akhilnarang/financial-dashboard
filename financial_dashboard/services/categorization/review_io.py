"""CSV export/apply helpers for the human-in-the-loop category review workflow."""

from decimal import Decimal, InvalidOperation
from typing import NamedTuple, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.manual import assign_category_manual


class ReviewRow(TypedDict):
    """One row of the category-review CSV round-trip — every cell is a string.

    Identity columns (amount/date/direction) may be blank, which skips that
    check on apply. ``final_category`` is the only column the user edits.
    """

    id: str
    date: str
    direction: str
    amount: str
    currency: str
    channel: str
    counterparty: str
    raw_description: str
    suggested_category: str
    confidence: str
    suggested_method: str
    review_reason: str
    final_category: str


# CSV column order IS the TypedDict's field order — one source of truth, so the
# header and the row schema can't drift apart.
EXPORT_FIELDS: list[str] = list(ReviewRow.__annotations__)


class ApplyResult(NamedTuple):
    applied: int
    skipped: int
    invalid: list[str]


def build_export_row(txn: Transaction) -> ReviewRow:
    """Map a categorized Transaction to an export row dict.

    suggested_category and final_category are both pre-filled from txn.category
    so the user only needs to edit rows where the suggestion is wrong.
    """
    return {
        "id": str(txn.id),
        "date": str(txn.transaction_date or ""),
        "direction": txn.direction or "",
        "amount": str(txn.amount),
        "currency": txn.currency or "",
        "channel": txn.channel or "",
        "counterparty": txn.counterparty or "",
        "raw_description": txn.raw_description or "",
        "suggested_category": txn.category or "",
        "confidence": (
            str(txn.category_confidence) if txn.category_confidence is not None else ""
        ),
        "suggested_method": txn.category_method or "",
        "review_reason": txn.review_reason or "",
        "final_category": txn.category or "",
    }


def _identity_mismatch(txn: Transaction, row: ReviewRow) -> bool:
    """True if any non-blank CSV identity field disagrees with the DB row.

    Lenient: a blank CSV field skips its own check, so callers may omit any
    subset of amount/date/direction. Guards against the common footgun of
    exporting from a throwaway DB copy and applying to a different (prod) DB.
    """
    if csv_amount := row.get("amount", ""):
        try:
            if Decimal(str(csv_amount)) != txn.amount:
                return True
        except InvalidOperation:
            return True

    if (csv_date := row.get("date", "")) and csv_date != str(
        txn.transaction_date or ""
    ):
        return True

    if (csv_direction := row.get("direction", "")) and csv_direction != (
        txn.direction or ""
    ):
        return True

    return False


async def apply_reviewed_rows(
    session: AsyncSession, rows: list[ReviewRow]
) -> ApplyResult:
    """Write human-verified categories back to the DB.

    For each row: blank final_category → skip; otherwise verify identity
    fields (amount, date, direction) against the DB row, then call
    assign_category_manual which validates the slug, updates the txn, and
    commits.  Rows where verification fails or assign returns False are
    recorded in invalid.  A malformed id or unexpected error never aborts
    the entire batch.  The LLM / engine is never called.

    Verification is lenient: if a field is blank in the CSV that particular
    check is skipped, so callers may omit any subset of the identity fields.
    """
    applied = 0
    skipped = 0
    invalid: list[str] = []

    for row in rows:
        raw_id = row.get("id", "?")
        final_category = row.get("final_category", "")
        if not (final_category and final_category.strip()):
            skipped += 1
            continue

        # Parse id — a non-integer must not crash the whole batch.
        try:
            txn_id = int(raw_id)
        except ValueError, TypeError:
            invalid.append(f"{raw_id}:{final_category}")
            continue

        try:
            # Safety verification: load the txn and confirm identity fields
            # match the CSV before writing.  This guards against the common
            # pattern of exporting from a /tmp copy and applying to prod.
            if (txn := await session.get(Transaction, txn_id)) is None:
                invalid.append(f"{txn_id}:missing")
                continue

            if _identity_mismatch(txn, row):
                invalid.append(f"{txn_id}:mismatch")
                continue

            ok, _ = await assign_category_manual(session, txn_id, final_category)
            if ok:
                applied += 1
            else:
                invalid.append(f"{txn_id}:{final_category}")
        except Exception:
            invalid.append(f"{raw_id}:{final_category}")

    return ApplyResult(applied=applied, skipped=skipped, invalid=invalid)
