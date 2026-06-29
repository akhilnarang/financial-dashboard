"""Deterministic CSV cleaner for transaction categorization exports.

Applies updated rules then the direction/polarity guardrail to an export CSV
without calling any LLM.

Processing order for each row:
  1. If the user edited it (final_category present AND != suggested_category):
     preserve the edit. A literal 'reimbursement' is silently mapped to
     'repayment' to align old hand edits with the renamed slug.
  2. Otherwise (untouched): run match_rules first. If a rule hits, use it.
     If no rule hits, apply resolve_direction(suggested or 'unknown', direction).

Merchant rules are loaded from the DB at startup when DB_URL is set.
If no DB is available, the script continues with empty merchant-rule cache
(rules still apply self_transfer detection via self-identifier tokens, and
any category from the suggested column is passed through the polarity guard).

Usage:
    uv run python scripts/categorize_postprocess.py --in PATH --out PATH
"""

import argparse
import asyncio
import collections
import csv
import logging

from financial_dashboard.services.categorization.polarity import resolve_direction
from financial_dashboard.services.categorization.review_io import EXPORT_FIELDS
from financial_dashboard.services.categorization.rules import (
    load_rule_config,
    match_rules,
)
from financial_dashboard.services.categorization.vocabulary import canonicalize_slug

logger = logging.getLogger(__name__)


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic CSV cleaner — applies rules + direction guardrail, no LLM."
        )
    )
    parser.add_argument(
        "--in",
        dest="input",
        required=True,
        metavar="PATH",
        help="Input CSV path (produced by categorize_export.py)",
    )
    parser.add_argument(
        "--out",
        dest="output",
        required=True,
        metavar="PATH",
        help="Output CSV path (input file is never modified)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # Load settings + merchant rules from the DB when available — READ-ONLY.
    # Deliberately NOT init_db(): this is a CSV→CSV tool and must never run
    # migrations/seeds against DB_URL. A missing DB / table is not fatal — the
    # script continues with empty caches (self_name_tokens=("self",), merchant_rules=()).
    try:
        from financial_dashboard.services.categorization.merchant_rules import (
            load_merchant_rules,
        )
        from financial_dashboard.services.settings import load_all_settings

        await load_all_settings()
        await load_merchant_rules()
    except Exception as exc:
        logger.debug("DB unavailable — running with empty caches: %s", exc)

    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    rule_config = load_rule_config()

    rules_changed = 0
    fallback_changed = 0
    manual_preserved = 0
    out_rows: list[dict] = []

    for row in rows:
        suggested = row.get("suggested_category", "") or ""
        direction = row.get("direction", "") or ""
        original_final = row.get("final_category", "") or ""

        out_row = dict(row)

        if original_final and original_final != suggested:
            # User typed something custom: preserve it, but map reimbursement→repayment.
            final = original_final
            if canonicalize_slug(final) == "reimbursement":
                final = "repayment"
            out_row["final_category"] = final
            manual_preserved += 1
        else:
            # Untouched: try rules first, then direction guardrail.
            rule_fields = {
                "counterparty": row.get("counterparty"),
                "raw_description": row.get("raw_description"),
                "channel": row.get("channel"),
                "direction": direction,
            }
            rule_hit = match_rules(rule_fields, rule_config)
            if rule_hit is not None:
                # Still pass through the polarity guard so a rule slug can't be
                # direction-impossible for this row.
                resolved, _ = resolve_direction(rule_hit.slug, direction)
                out_row["final_category"] = resolved
                rules_changed += 1
            else:
                resolved, _ = resolve_direction(suggested or "unknown", direction)
                out_row["final_category"] = resolved
                fallback_changed += 1

        out_rows.append(out_row)

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=EXPORT_FIELDS,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(out_rows)

    cat_counts: collections.Counter = collections.Counter(
        r["final_category"] for r in out_rows
    )
    print(f"rows processed      : {len(out_rows)}")
    print(f"changed by rule     : {rules_changed}")
    print(f"direction fallback  : {fallback_changed}")
    print(f"manual preserved    : {manual_preserved}")
    print("\ncount by final_category:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>6}  {cat or '(blank)'}")


if __name__ == "__main__":
    asyncio.run(_main())
