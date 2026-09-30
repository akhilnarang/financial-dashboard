"""Categorize every uncategorized transaction now.

Runs the rule sweep until no untouched rows remain. The rule pass seeds the
few-shot examples. Then runs the LLM sweep until it stops making progress.

Usage:
    uv run python scripts/categorize_backfill.py [--rules-only] [--batch-size N]
"""

import argparse
import asyncio
from collections.abc import Awaitable, Callable

from financial_dashboard.db import init_db
from financial_dashboard.services.categorization.sweep import (
    run_llm_sweep,
    run_rule_sweep,
)
from financial_dashboard.services.settings import load_all_settings


async def _drain(sweep: Callable[..., Awaitable[int]], batch_limit: int) -> int:
    total = 0
    while n := await sweep(batch_limit=batch_limit):
        total += n
    return total


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rule sweep, then the LLM sweep, until both drain."
    )
    parser.add_argument("--rules-only", action="store_true", help="Skip the LLM sweep")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="LLM batch size (default 100)"
    )
    args = parser.parse_args()

    await init_db()
    await load_all_settings()

    # Each processed row leaves the untouched set, so this loop ends.
    rules = await _drain(run_rule_sweep, max(args.batch_size, 500))
    print(f"Rule pass processed {rules} rows.")
    if not args.rules_only:
        llm = await _drain(run_llm_sweep, args.batch_size)
        print(f"LLM sweep categorized {llm} rows.")


if __name__ == "__main__":
    asyncio.run(_main())
