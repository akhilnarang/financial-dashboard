"""Apply human-verified categories from an edited export CSV back to the DB.

Reads the CSV produced by categorize_export.py, applies the final_category
column to each transaction row, and records it as a manual categorization.
The LLM / engine is never called — only human-edited final_category values
are written.

Usage:
    DB_URL=sqlite+aiosqlite:///path/to/db.db \\
        uv run python scripts/categorize_apply.py --in categorization_export.csv

    # Dry-run: parse, validate, and report without writing:
    uv run python scripts/categorize_apply.py --in categorization_export.csv --dry-run
"""

import argparse
import asyncio
import csv
import sys

from financial_dashboard.config import settings
from financial_dashboard.db import async_session, init_db
from financial_dashboard.services.categorization.merchant_rules import (
    load_merchant_rules,
)
from financial_dashboard.services.categorization.review_io import apply_reviewed_rows
from financial_dashboard.services.categorization.vocabulary import (
    canonicalize_slug,
    is_valid_slug,
)
from financial_dashboard.services.settings import load_all_settings


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply human-verified categories from an edited export CSV to the DB."
        )
    )
    parser.add_argument(
        "--in",
        dest="input",
        required=True,
        metavar="PATH",
        help="Path to the CSV produced by categorize_export.py (required)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing anything to the DB",
    )
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    required_headers = {"id", "final_category"}
    missing_headers = required_headers - set(reader.fieldnames or [])
    if missing_headers:
        print(
            f"error: CSV is missing required column(s): {', '.join(sorted(missing_headers))}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        would_apply = 0
        would_skip = 0
        invalid_ids: list[str] = []
        for row in rows:
            final_category = row.get("final_category", "")
            if not (final_category and final_category.strip()):
                would_skip += 1
            elif is_valid_slug(canonicalize_slug(final_category)):
                would_apply += 1
            else:
                invalid_ids.append(row.get("id", "?"))
        print(
            f"dry-run: {would_apply} would apply, {would_skip} would skip, "
            f"{len(invalid_ids)} invalid"
        )
        if invalid_ids:
            print(f"invalid ids: {', '.join(invalid_ids)}")
        return

    print(f"applying to {settings.db_url}")
    await init_db()
    await load_all_settings()
    await load_merchant_rules()

    async with async_session() as session:
        result = await apply_reviewed_rows(session, rows)

    print(f"applied : {result.applied}")
    print(f"skipped : {result.skipped}")
    if result.invalid:
        print(f"invalid ({len(result.invalid)}): {', '.join(result.invalid)}")


if __name__ == "__main__":
    asyncio.run(_main())
