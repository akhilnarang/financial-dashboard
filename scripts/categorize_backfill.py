"""Backfill transaction categorization.

Usage:
    uv run python scripts/categorize_backfill.py [--rules-only] [--batch-size N]
"""

import argparse
import asyncio
import logging

from financial_dashboard.db import init_db
from financial_dashboard.services.categorization.backfill import run_backfill
from financial_dashboard.services.categorization.merchant_rules import (
    load_merchant_rules,
)
from financial_dashboard.services.settings import load_all_settings


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    await init_db()
    await load_all_settings()
    await load_merchant_rules()
    await run_backfill(rules_only=args.rules_only, batch_size=args.batch_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
