#!/usr/bin/env python
"""Backfill balance snapshots from existing statement uploads."""

import asyncio

from sqlalchemy import select

from financial_dashboard.db import (
    BankStatementUpload,
    StatementUpload,
    async_session,
    init_db,
)
from financial_dashboard.services.snapshots import emit_bank_snapshot, emit_cc_snapshot


async def main() -> None:
    await init_db()
    bank_count = 0
    cc_count = 0
    async with async_session() as session:
        bank_uploads = (
            (await session.execute(select(BankStatementUpload))).scalars().all()
        )
        for upload in bank_uploads:
            if await emit_bank_snapshot(session, upload):
                bank_count += 1

        cc_uploads = (await session.execute(select(StatementUpload))).scalars().all()
        for upload in cc_uploads:
            if await emit_cc_snapshot(session, upload):
                cc_count += 1

        await session.commit()

    print(f"bank snapshots emitted: {bank_count}")
    print(f"cc snapshots emitted: {cc_count}")


if __name__ == "__main__":
    asyncio.run(main())
