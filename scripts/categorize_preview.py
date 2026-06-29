"""Dry-run preview of the transaction categorization engine.

Previews what the categorization engine would assign to each transaction and
writes a markdown report — WITHOUT committing anything to the DB.  The session
is discarded (never committed), so the source data is left untouched.  The only
persistent side-effect is schema migration / category seeding performed by
init_db() on the target DB copy (expected on a throwaway copy).

Usage:
    DB_URL=sqlite+aiosqlite:///path/to/copy.db \\
        uv run python scripts/categorize_preview.py [--limit N] [--llm] [--out PATH]
"""

import argparse
import asyncio
import collections
import logging
import sys
import time

from sqlalchemy import select

from financial_dashboard.db import async_session, init_db
from financial_dashboard.db.models import Transaction
from financial_dashboard.services.categorization.engine import categorize_one
from financial_dashboard.services.categorization.merchant_rules import (
    load_merchant_rules,
)
from financial_dashboard.services import settings as settings_mod
from financial_dashboard.services.settings import (
    get_active_llm_key,
    get_setting,
    load_all_settings,
)

logger = logging.getLogger(__name__)

_NARRATION_MAX = 80


def _escape_pipe(text: str | None) -> str:
    if not text:
        return ""
    return text.replace("|", r"\|").replace("\n", " ").replace("\r", "")


def _truncate(text: str | None, max_len: int) -> str:
    if not text:
        return ""
    cleaned = text.replace("\n", " ").replace("\r", "")
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "…"
    return cleaned


def _display_category(row: dict) -> str:
    method = row["category_method"]
    category = row["category"]
    review_reason = row["review_reason"]

    if row.get("error"):
        return f"⚠ error: {_escape_pipe(row['error'])[:60]}"
    if method == "pending_llm":
        return "— (no rule; needs LLM)"
    if category == "unknown":
        if review_reason:
            return f"unknown (review: {review_reason})"
        return "unknown"
    return category or ""


def _build_markdown(rows: list[dict], *, use_llm: bool, out_path: str) -> str:
    mode = "rules + LLM" if use_llm else "rules-only"
    total = len(rows)
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    lines: list[str] = []
    lines.append(f"# Categorization Preview — {mode}")
    lines.append("")
    lines.append(f"**Mode:** {mode}  |  **Transactions:** {total}  |  **Run at:** {ts}")
    lines.append("")

    # --- SUMMARY ---
    lines.append("## Summary")
    lines.append("")

    cat_counts: dict[str, int] = collections.Counter()
    method_counts: dict[str, int] = collections.Counter()
    for r in rows:
        cat_counts[_display_category(r)] += 1
        method_counts[r["category_method"] or "null"] += 1

    lines.append("### By detected category")
    lines.append("")
    lines.append("| detected_category | count |")
    lines.append("| --- | --- |")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {_escape_pipe(cat)} | {cnt} |")
    lines.append("")

    lines.append("### By method")
    lines.append("")
    lines.append("| method | count |")
    lines.append("| --- | --- |")
    for meth, cnt in sorted(method_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {meth} | {cnt} |")
    lines.append("")

    # --- DETAIL ---
    lines.append("## Detail")
    lines.append("")
    lines.append(
        "| # | date | direction | counterparty | amount | raw narration"
        " | detected_category | method | confidence |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(rows, 1):
        date = str(r["transaction_date"] or "")
        direction = r["direction"] or ""
        counterparty = _escape_pipe(r["counterparty"])
        amount = str(r["amount"])
        narration = _escape_pipe(_truncate(r["raw_description"], _NARRATION_MAX))
        detected = _escape_pipe(_display_category(r))
        method = r["category_method"] or "null"
        conf = r["category_confidence"]
        conf_str = f"{conf:.2f}" if conf is not None else ""
        lines.append(
            f"| {i} | {date} | {direction} | {counterparty} | {amount}"
            f" | {narration} | {detected} | {method} | {conf_str} |"
        )
    lines.append("")

    return "\n".join(lines)


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run preview: show what the categorization engine would assign."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Preview only the first N transactions (default: all)",
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
        help="Override the model id for the active provider this run "
        "(e.g. gemini-2.5-flash, gpt-4o-mini). Default: the provider's model setting.",
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
        default="categorization_preview.md",
        metavar="PATH",
        help="Markdown output path (default: categorization_preview.md)",
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

    results: list[dict] = []

    # One session — never committed.  All ORM mutations (txn.category etc.) are
    # discarded when the context manager exits without a commit().
    async with async_session() as session:
        stmt = select(Transaction).order_by(Transaction.transaction_date)
        if args.limit is not None:
            stmt = stmt.limit(args.limit)
        rows = (await session.execute(stmt)).scalars().all()

        print(f"loaded {len(rows)} transactions — running engine…", flush=True)
        for i, txn in enumerate(rows):
            error = None
            try:
                await categorize_one(session, txn, use_llm=args.llm)
            except Exception as exc:  # noqa: BLE001 — preview must not abort mid-run
                error = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "id": txn.id,
                    "transaction_date": txn.transaction_date,
                    "direction": txn.direction,
                    "counterparty": txn.counterparty,
                    "amount": txn.amount,
                    "currency": txn.currency,
                    "raw_description": txn.raw_description,
                    "category": txn.category,
                    "category_method": txn.category_method,
                    "category_confidence": txn.category_confidence,
                    "review_reason": txn.review_reason,
                    "error": error,
                }
            )
            # Gentle pacing between LLM calls to respect API rate limits.
            if args.llm and args.delay and i < len(rows) - 1:
                await asyncio.sleep(args.delay)
        # session closes here with NO commit — DB is untouched.

    md = _build_markdown(results, use_llm=args.llm, out_path=args.out)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)

    # Stdout summary
    top_cats = collections.Counter(_display_category(r) for r in results).most_common(5)
    print(f"\nmode          : {mode}")
    print(f"total rows    : {len(results)}")
    print(f"report written: {args.out}")
    print("\ntop categories:")
    for cat, cnt in top_cats:
        print(f"  {cnt:>6}  {cat}")


if __name__ == "__main__":
    asyncio.run(_main())
