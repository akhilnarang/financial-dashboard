"""Export all transactions with engine-suggested categories to a CSV for human review.

The session is never committed — DB is left untouched.  Edit the
final_category column in the output CSV then run categorize_apply.py to
persist the changes without calling the LLM.

Usage:
    DB_URL=sqlite+aiosqlite:///path/to/copy.db \\
        uv run python scripts/categorize_export.py [--limit N] [--llm] [--out PATH]
"""

import argparse
import asyncio
import collections
import csv
import logging
import sys

from sqlalchemy import select

from financial_dashboard.db import async_session, init_db
from financial_dashboard.db.models import Transaction
from financial_dashboard.services import settings as settings_mod
from financial_dashboard.services.categorization.engine import categorize_one
from financial_dashboard.services.categorization.merchant_rules import (
    load_merchant_rules,
)
from financial_dashboard.services.categorization.review_io import (
    EXPORT_FIELDS,
    build_export_row,
)
from financial_dashboard.services.settings import (
    get_active_llm_key,
    get_setting,
    load_all_settings,
)

logger = logging.getLogger(__name__)


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Export transactions with suggested categories to a CSV for review."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Export only the first N transactions (default: all)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also run the LLM for rows the rules don't match (requires a configured key)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["gemini", "openai"],
        help="LLM provider for this run. Default: the categorization.llm_provider setting.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="ID",
        help=(
            "Override the model id for the active provider this run "
            "(e.g. gemini-2.5-flash, gpt-4o-mini). Default: the provider's model setting."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Sleep this long between LLM calls to respect rate limits (default: 0)",
    )
    parser.add_argument(
        "--out",
        default="categorization_export.csv",
        metavar="PATH",
        help="CSV output path (default: categorization_export.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    await init_db()
    await load_all_settings()
    await load_merchant_rules()

    if args.provider:
        settings_mod._cache["categorization.llm_provider"] = args.provider

    provider = get_setting("categorization.llm_provider") or "gemini"

    if args.model:
        model_key = "openai.model" if provider == "openai" else "gemini.model"
        settings_mod._cache[model_key] = args.model

    if args.llm and not get_active_llm_key().strip():
        print(
            f"error: --llm was passed but no API key is configured for provider "
            f"'{provider}'.\nSet the key in .env or via the settings UI/DB.",
            file=sys.stderr,
        )
        sys.exit(1)

    mode = "rules + LLM" if args.llm else "rules-only"
    model_setting = "openai.model" if provider == "openai" else "gemini.model"
    model_note = (
        f" ({provider}: {get_setting(model_setting) or 'default'})" if args.llm else ""
    )
    print(f"mode: {mode}{model_note}", flush=True)

    rows: list[dict] = []

    # One session — never committed.  All ORM mutations (txn.category etc.) are
    # discarded when the context manager exits without a commit().
    async with async_session() as session:
        stmt = select(Transaction).order_by(Transaction.transaction_date)
        if args.limit is not None:
            stmt = stmt.limit(args.limit)
        txns = (await session.execute(stmt)).scalars().all()

        print(f"loaded {len(txns)} transactions — running engine…", flush=True)
        for i, txn in enumerate(txns):
            try:
                await categorize_one(session, txn, use_llm=args.llm)
            except Exception as exc:  # noqa: BLE001 — export must not abort mid-run
                error = f"{type(exc).__name__}: {exc}"
                txn.category = None
                txn.category_method = None
                txn.category_confidence = None
                txn.review_reason = error
            rows.append(build_export_row(txn))
            # Gentle pacing between LLM calls to respect API rate limits.
            if args.llm and args.delay and i < len(txns) - 1:
                await asyncio.sleep(args.delay)
        # session closes here with NO commit — DB is untouched.

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPORT_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    # Stdout summary
    cat_counts: collections.Counter = collections.Counter(
        r["suggested_category"] for r in rows
    )
    print(f"\nmode          : {mode}")
    print(f"rows written  : {len(rows)}")
    print(f"output        : {args.out}")
    print("\ncount by suggested_category:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>6}  {cat or '(blank)'}")


if __name__ == "__main__":
    asyncio.run(_main())
