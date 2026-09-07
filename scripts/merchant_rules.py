"""Merchant-rule management CLI.

Manage the DB-backed merchant→category mapping table without touching code.

Usage:
    uv run python scripts/merchant_rules.py list
    uv run python scripts/merchant_rules.py add --pattern PATTERN --category SLUG [--priority N]
    uv run python scripts/merchant_rules.py import --csv PATH
    uv run python scripts/merchant_rules.py seed   # personal overrides (untracked)

The generic built-in defaults (merchant_defaults.py) are seeded automatically by
init_db. `seed` layers your personal/local overrides on top, from the local
gitignored merchant_seed_data.py.
"""

import argparse
import asyncio
import csv

from financial_dashboard.db import async_session, init_db
from financial_dashboard.services.categorization.merchant_rules import (
    add_merchant_rule,
    list_merchant_rules,
    load_merchant_rules,
)
from financial_dashboard.services.settings import load_all_settings


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage merchant categorization rules stored in the DB."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Print all merchant rules")

    add_p = sub.add_parser("add", help="Add or update a merchant rule")
    add_p.add_argument(
        "--pattern",
        required=True,
        help="Lowercased substring to match against narration",
    )
    add_p.add_argument(
        "--category", required=True, help="Category slug (e.g. bill_payment)"
    )
    add_p.add_argument(
        "--priority", type=int, default=100, help="Priority (default 100, lower=first)"
    )

    imp_p = sub.add_parser("import", help="Bulk-add rules from a CSV file")
    imp_p.add_argument(
        "--csv",
        required=True,
        dest="csv_path",
        metavar="PATH",
        help="CSV with columns: pattern, category[, priority]",
    )

    sub.add_parser(
        "seed", help="Bulk-add rules from the local (untracked) merchant_seed_data.py"
    )

    args = parser.parse_args()

    await init_db()
    await load_all_settings()

    if args.cmd == "list":
        async with async_session() as session:
            rules = await list_merchant_rules(session)
        if not rules:
            print("No merchant rules found.")
            return
        print(f"{'priority':<10} {'pattern':<32} {'category':<25} active")
        print("-" * 75)
        for r in rules:
            print(f"{r.priority:<10} {r.pattern:<32} {r.category:<25} {r.active}")

    elif args.cmd == "add":
        async with async_session() as session:
            await add_merchant_rule(
                session, args.pattern, args.category, priority=args.priority
            )
            await session.commit()
        await load_merchant_rules()
        print(
            f"Added: pattern={args.pattern.strip().lower()!r} "
            f"category={args.category!r} priority={args.priority}"
        )

    elif args.cmd == "import":
        with open(args.csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        added = 0
        errors: list[str] = []
        async with async_session() as session:
            for row in rows:
                pattern = row.get("pattern", "").strip()
                category = row.get("category", "").strip()
                raw_priority = row.get("priority", "100") or "100"
                try:
                    priority = int(raw_priority)
                except ValueError:
                    errors.append(f"{pattern!r}: bad priority value {raw_priority!r}")
                    continue
                if not pattern or not category:
                    errors.append(f"skipping blank row: {row}")
                    continue
                try:
                    await add_merchant_rule(
                        session, pattern, category, priority=priority
                    )
                    added += 1
                except ValueError as exc:
                    errors.append(f"{pattern!r}: {exc}")
            await session.commit()
        await load_merchant_rules()
        print(f"Imported {added} rules.")
        for err in errors:
            print(f"  warning: {err}")

    elif args.cmd == "seed":
        try:
            from financial_dashboard.services.categorization.merchant_seed_data import (
                SEED_MERCHANT_RULES,
            )
        except ModuleNotFoundError:
            print("No merchant_seed_data.py found (it's untracked) — nothing to seed.")
            return

        added = 0
        errors = []
        async with async_session() as session:
            for category, patterns in SEED_MERCHANT_RULES.items():
                for pattern in patterns:
                    try:
                        await add_merchant_rule(session, pattern, category)
                        added += 1
                    except ValueError as exc:
                        errors.append(f"{pattern!r}: {exc}")
            await session.commit()
        await load_merchant_rules()
        print(f"Seeded {added} rules from merchant_seed_data.py.")
        for err in errors:
            print(f"  warning: {err}")


if __name__ == "__main__":
    asyncio.run(_main())
