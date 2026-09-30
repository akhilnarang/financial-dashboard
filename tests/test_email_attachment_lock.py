"""Email attachment locking protocol regressions."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Base, Transaction
from financial_dashboard.services.email_attachments import (
    TransactionSlotConflict,
    claim_transaction_source_slot,
)


@pytest.mark.anyio
async def test_destination_claim_rejects_stale_different_owner_and_allows_same_source(
    tmp_path,
):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'transaction-slot.sqlite'}"
    )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with maker() as stale_session:
            target = Transaction(
                bank="testbank",
                email_type="test_alert",
                direction="debit",
                amount=42,
                email_id=None,
            )
            stale_session.add(target)
            await stale_session.commit()
            target_id = target.id
            assert target.email_id is None

            async with maker() as winner_session:
                winner = await winner_session.get(Transaction, target.id)
                assert winner is not None
                winner.email_id = 101
                await winner_session.commit()

            with pytest.raises(TransactionSlotConflict):
                async with stale_session.begin():
                    await claim_transaction_source_slot(
                        stale_session, target, "email", 202
                    )

        async with maker() as check_session:
            stored = await check_session.get(Transaction, target_id)
            assert stored is not None
            assert stored.email_id == 101
            # Re-claiming the source already attached to the destination is
            # idempotent and leaves ownership unchanged.
            async with check_session.begin_nested():
                await claim_transaction_source_slot(check_session, stored, "email", 101)
            assert stored.email_id == 101
    finally:
        await engine.dispose()
