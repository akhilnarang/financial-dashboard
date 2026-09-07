"""Export audited assistant turns as JSONL for offline evaluation."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from financial_dashboard.db import async_session
from financial_dashboard.services.assistant.evals import iter_audit_eval_rows


async def export(path: Path | None) -> int:
    output = path.open("w", encoding="utf-8") if path is not None else sys.stdout
    count = 0
    try:
        async with async_session() as session:
            async for row in iter_audit_eval_rows(session):
                output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                count += 1
    finally:
        if path is not None:
            output.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="JSONL destination; stdout by default"
    )
    args = parser.parse_args()
    asyncio.run(export(args.output))


if __name__ == "__main__":
    main()
