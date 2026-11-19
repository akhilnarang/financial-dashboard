"""Give matched card rows the full statement narration as counterparty.

A card alert often cuts the merchant name short. The statement narration of
the same charge gives the full name. Each stored card-statement
reconciliation keeps the narration and the row id of each matched line. This
script applies the enrichment counterparty rule to those rows again.

By default the script only lists the changes and rolls them back. Use
--apply to write them.

Usage:
    uv run python scripts/backfill_statement_narration.py [--apply]
"""

import argparse
import asyncio
from typing import NamedTuple

from sqlalchemy import select

from financial_dashboard.db import async_session
from financial_dashboard.db.models import StatementUpload, Transaction
from financial_dashboard.services.categorization.engine import (
    requeue_after_enrichment,
)
from financial_dashboard.services.settings import load_all_settings
from financial_dashboard.services.statements.cc import (
    reconciliation_from_json,
    takes_narration,
)


class NarrationChange(NamedTuple):
    """One row whose counterparty takes a statement narration."""

    txn_id: int
    upload_id: int
    old: str | None
    new: str


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Give matched card rows the full statement narration."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write the changes (default: list only)"
    )
    args = parser.parse_args()

    await load_all_settings()

    changes: dict[int, NarrationChange] = {}
    async with async_session() as session:
        uploads = await session.scalars(
            select(StatementUpload)
            .where(StatementUpload.reconciliation_data.is_not(None))
            .order_by(StatementUpload.id)
        )
        for upload in uploads.all():
            recon = reconciliation_from_json(upload.reconciliation_data or "{}")
            for entry in recon.get("matched", []):
                if not (txn_id := entry.get("db_txn_id")):
                    continue
                txn = await session.get(Transaction, txn_id)
                if (
                    txn is None
                    or txn.account_id != upload.account_id
                    or str(txn.transaction_date) != entry.get("db_date")
                ):
                    continue
                narration = (entry.get("narration") or "").strip()
                if not takes_narration(txn, narration) or txn.counterparty == narration:
                    continue
                old = changes[txn_id].old if txn_id in changes else txn.counterparty
                changes[txn_id] = NarrationChange(txn_id, upload.id, old, narration)
                txn.counterparty = narration
                txn.counterparty_source = "bank"
                await requeue_after_enrichment(session, txn)

        for change in changes.values():
            print(
                f"txn {change.txn_id} (statement {change.upload_id}): "
                f"{change.old!r} -> {change.new!r}"
            )
        if args.apply:
            await session.commit()
            print(f"Updated {len(changes)} rows.")
        else:
            await session.rollback()
            print(f"Dry run: {len(changes)} rows would change. Use --apply to write.")


if __name__ == "__main__":
    asyncio.run(_main())
