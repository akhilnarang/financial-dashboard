"""Receivables ledger: money other people owe the user.

The ledger does not change cashflow figures. Paid and outstanding are
computed from the linked rows on each read.
"""

from decimal import Decimal
from typing import cast

from sqlalchemy import CursorResult, delete, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.enums import ReceivableRole, ReceivableStatus
from financial_dashboard.db.models import (
    Receivable,
    ReceivableLink,
    Transaction,
    utc_now,
)
from financial_dashboard.schemas.receivables import (
    ReceivableCreate,
    ReceivableRead,
    ReceivableUpdate,
)

ROLE_DIRECTION = {ReceivableRole.charge: "debit", ReceivableRole.payment: "credit"}


class LinkConflict(ValueError):
    """The link exists, or the payment row already pays another receivable."""


def receivable_read(receivable: Receivable) -> ReceivableRead:
    """Build the API view of a receivable.

    Args:
        receivable: A receivable with its links loaded.

    Returns:
        The receivable with its linked row ids, the sum of its payment
        rows, and the amount minus that sum.
    """
    paid = Decimal("0.00")
    ids = {role: [] for role in ReceivableRole}
    for link in sorted(receivable.links, key=lambda link: link.transaction_id):
        ids[ReceivableRole(link.role)].append(link.transaction_id)
        if link.role == ReceivableRole.payment:
            paid += link.transaction.amount
    return ReceivableRead(
        id=receivable.id,
        person=receivable.person,
        amount=receivable.amount,
        note=receivable.note,
        status=ReceivableStatus(receivable.status),
        created_at=receivable.created_at,
        closed_at=receivable.closed_at,
        paid=paid,
        outstanding=receivable.amount - paid,
        charge_transaction_ids=ids[ReceivableRole.charge],
        payment_transaction_ids=ids[ReceivableRole.payment],
    )


async def list_receivables(
    session: AsyncSession, status: ReceivableStatus | None
) -> list[Receivable]:
    """List receivables, oldest first.

    Args:
        session: The request session.
        status: Keep only this status, or ``None`` for all.

    Returns:
        The receivables with their links loaded.
    """
    query = select(Receivable).order_by(Receivable.created_at, Receivable.id)
    if status is not None:
        query = query.where(Receivable.status == status)
    return list(await session.scalars(query))


async def _linkable(
    session: AsyncSession, transaction_id: int, role: ReceivableRole
) -> Transaction:
    """Load the row a link names and check that it fits the role.

    Args:
        session: The request session.
        transaction_id: The row to link.
        role: The role the link takes.

    Returns:
        The transaction.

    Raises:
        ValueError: The row is unknown, the wrong direction, or not in INR.
    """
    txn = await session.get(Transaction, transaction_id)
    if txn is None:
        raise ValueError(f"Transaction {transaction_id} not found")
    if txn.direction != ROLE_DIRECTION[role]:
        raise ValueError(
            f"A {role} link needs a {ROLE_DIRECTION[role]} row; "
            f"transaction {transaction_id} is a {txn.direction}"
        )
    if (txn.currency or "INR") != "INR":
        raise ValueError(f"Transaction {transaction_id} is not in INR")
    return txn


async def _write_lock(session: AsyncSession) -> None:
    """Take the SQLite write lock. It holds until the commit or rollback.

    Args:
        session: A session that holds no write yet.

    Returns:
        None.
    """
    if session.get_bind().dialect.name == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))


async def _new_links(
    session: AsyncSession, wanted: list[tuple[int, ReceivableRole]]
) -> list[ReceivableLink]:
    """Build links under the write lock. Roll back and raise on a bad row.

    The SQLite write lock holds until the commit, so a merge or delete
    cannot remove a linked row first.

    Args:
        session: The request session. It must not hold a write yet.
        wanted: The row id and role of each link.

    Returns:
        The new, unsaved links.

    Raises:
        ValueError: A row does not fit its role.
    """
    await _write_lock(session)
    try:
        return [
            ReceivableLink(transaction=await _linkable(session, i, role), role=role)
            for i, role in wanted
        ]
    except ValueError:
        await session.rollback()
        raise


async def _commit_links(session: AsyncSession) -> None:
    """Commit, and turn a duplicate link into a conflict.

    Args:
        session: The session that holds the new links.

    Returns:
        None.

    Raises:
        LinkConflict: A unique index refused a link.
    """
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise LinkConflict(
            "A payment row pays one receivable, and a receivable links a row once"
        ) from exc


async def create_receivable(
    session: AsyncSession, payload: ReceivableCreate
) -> Receivable:
    """Create a receivable and its links, and commit.

    Args:
        session: The request session.
        payload: The validated request body.

    Returns:
        The new receivable.

    Raises:
        LinkConflict: A payment row already pays another receivable.
        ValueError: A transaction id is unknown, listed twice, or the wrong
            direction for its role.
    """
    wanted = [(i, ReceivableRole.charge) for i in payload.charge_transaction_ids]
    wanted += [(i, ReceivableRole.payment) for i in payload.payment_transaction_ids]
    ids = [transaction_id for transaction_id, _ in wanted]
    if len(ids) != len(set(ids)):
        raise ValueError("A transaction is listed twice")
    receivable = Receivable(
        person=payload.person,
        amount=payload.amount,
        note=payload.note,
        links=await _new_links(session, wanted),
    )
    session.add(receivable)
    await _commit_links(session)
    return receivable


async def update_receivable(
    session: AsyncSession, receivable: Receivable, payload: ReceivableUpdate
) -> Receivable:
    """Apply a partial edit, and commit.

    A move to ``settled`` or ``written_off`` stamps ``closed_at``. A move
    back to ``open`` clears it.

    Args:
        session: The request session.
        receivable: The receivable to change.
        payload: The fields to change.

    Returns:
        The changed receivable.
    """
    if payload.amount is not None:
        receivable.amount = payload.amount
    if "note" in payload.model_fields_set:
        receivable.note = payload.note
    if payload.status is not None and payload.status != receivable.status:
        receivable.status = payload.status
        is_open = payload.status == ReceivableStatus.open
        receivable.closed_at = None if is_open else utc_now()
    await session.commit()
    return receivable


async def add_link(
    session: AsyncSession,
    receivable: Receivable,
    transaction_id: int,
    role: ReceivableRole,
) -> Receivable:
    """Link a transaction to a receivable, and commit.

    Args:
        session: The request session.
        receivable: The receivable to link.
        transaction_id: The row to link.
        role: ``charge`` for a debit the money was for, ``payment`` for a
            credit that pays it back.

    Returns:
        The changed receivable.

    Raises:
        LinkConflict: The receivable already links the row, or the payment
            row already pays another receivable.
        ValueError: The row is unknown or the wrong direction for the role.
    """
    if any(link.transaction_id == transaction_id for link in receivable.links):
        raise LinkConflict(f"Transaction {transaction_id} is already linked")
    receivable.links.extend(await _new_links(session, [(transaction_id, role)]))
    await _commit_links(session)
    return receivable


async def remove_link(
    session: AsyncSession, receivable_id: int, transaction_id: int
) -> Receivable | None:
    """Unlink a transaction from a receivable, and commit.

    The delete names the receivable and the row, not the link id. SQLite can
    give a deleted link id to a new link of another receivable.

    Args:
        session: The request session. It must not hold a write yet.
        receivable_id: The receivable to change.
        transaction_id: The row to unlink.

    Returns:
        The changed receivable, or ``None`` when it does not link the row.
    """
    await _write_lock(session)
    result = await session.execute(
        delete(ReceivableLink).where(
            ReceivableLink.receivable_id == receivable_id,
            ReceivableLink.transaction_id == transaction_id,
        )
    )
    if not cast(CursorResult, result).rowcount:
        await session.rollback()
        return None
    await session.commit()
    return await session.scalar(
        select(Receivable)
        .where(Receivable.id == receivable_id)
        .execution_options(populate_existing=True)
    )


async def move_receivable_links(
    session: AsyncSession, old_id: int, new_id: int | None
) -> dict[str, int]:
    """Point the links of a deleted transaction at its keeper, or drop them.

    A link is dropped when its receivable already links the keeper, or when
    it is a payment link and the keeper already pays a receivable. Both rows
    show one event, so its money counts once.

    Args:
        session: The session that deletes ``old_id``.
        old_id: The transaction that goes away.
        new_id: The transaction that keeps the data, or ``None`` for a plain
            delete.

    Returns:
        The nonzero counts: ``receivable_links`` for links moved, or removed
        on a plain delete, and ``receivable_links.dropped`` for links dropped
        on a merge.
    """
    stale = ReceivableLink.transaction_id == old_id
    if new_id is None:
        result = await session.execute(delete(ReceivableLink).where(stale))
        removed = cast(CursorResult, result).rowcount
        return {"receivable_links": removed} if removed else {}
    keeper = ReceivableLink.transaction_id == new_id
    linked = select(ReceivableLink.receivable_id).where(keeper)
    pays = select(ReceivableLink.id).where(
        keeper, ReceivableLink.role == ReceivableRole.payment
    )
    dropped = await session.execute(
        delete(ReceivableLink).where(
            stale,
            or_(
                ReceivableLink.receivable_id.in_(linked),
                (ReceivableLink.role == ReceivableRole.payment) & pays.exists(),
            ),
        )
    )
    moved = await session.execute(
        update(ReceivableLink).where(stale).values(transaction_id=new_id)
    )
    counts = {
        "receivable_links": cast(CursorResult, moved).rowcount,
        "receivable_links.dropped": cast(CursorResult, dropped).rowcount,
    }
    return {key: n for key, n in counts.items() if n}
