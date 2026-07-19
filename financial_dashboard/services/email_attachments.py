"""Concurrency helpers for source-to-transaction attachments."""

from typing import Literal, cast

from sqlalchemy import select, text, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import set_committed_value

from financial_dashboard.db import Email, Transaction

AttachmentChannel = Literal["sms", "email"]


class TransactionSlotConflict(Exception):
    """An apparently open destination slot was concurrently claimed."""

    def __init__(
        self,
        transaction_id: int,
        channel: AttachmentChannel,
        source_id: int,
    ) -> None:
        super().__init__(
            f"Transaction {transaction_id} {channel} slot could not be claimed "
            f"by source {source_id}"
        )
        self.transaction_id = transaction_id
        self.channel = channel
        self.source_id = source_id


async def lock_email_for_attachment(
    session: AsyncSession, email_id: int
) -> Email | None:
    """Lock an existing email before checking or filling its transaction slot.

    Call this as the first database operation in a fresh transaction. Databases
    with row-lock support serialize on ``SELECT ... FOR UPDATE``. SQLite ignores
    that clause, so it instead takes a reserved write lock before reading the
    email and its current attachments.
    """
    if session.get_bind().dialect.name == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))

    return (
        await session.scalars(
            select(Email).where(Email.id == email_id).with_for_update()
        )
    ).one_or_none()


async def claim_transaction_source_slot(
    session: AsyncSession,
    transaction: Transaction,
    channel: AttachmentChannel,
    source_id: int | None,
) -> None:
    """Atomically claim one destination channel slot without stealing it.

    The ORM object can be stale after matching. A conditional UPDATE makes the
    database arbitrate a race with another source targeting the same
    transaction. Re-attaching the source already stored in the slot is a no-op.
    A different owner already visible to the matcher is preserved for historical
    same-channel deduplication; if an apparently open slot loses the atomic
    claim, the helper raises before any enrichment is applied.
    """
    if source_id is None:
        return

    slot_name = "sms_message_id" if channel == "sms" else "email_id"
    current_source_id = getattr(transaction, slot_name)
    if current_source_id is not None:
        # Same-source replay is idempotent. Historically, a bank's duplicate
        # same-channel notification can also enrich/deduplicate against the
        # canonical source already stored here; preserve that behavior while
        # leaving the different owner's destination slot untouched.
        return

    slot_column = getattr(Transaction, slot_name)
    statement = (
        update(Transaction)
        .where(
            Transaction.id == transaction.id,
            (slot_column.is_(None)) | (slot_column == source_id),
        )
        .values({slot_name: source_id})
        .execution_options(synchronize_session=False)
    )
    # Do not let an unrelated pending ORM mutation flush before ownership is
    # established. The explicit resolver relies on this ordering for complete
    # rollback semantics when it loses the claim.
    with session.no_autoflush:
        result = cast(CursorResult, await session.execute(statement))
    if result.rowcount != 1:
        raise TransactionSlotConflict(transaction.id, channel, source_id)

    # Keep the identity-map object aligned with the successful Core UPDATE
    # without marking the attribute dirty and emitting an unconditional ORM
    # UPDATE at the next flush.
    set_committed_value(transaction, slot_name, source_id)
