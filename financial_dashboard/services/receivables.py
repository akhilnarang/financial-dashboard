"""Receivables ledger: money other people owe the user.

The ledger does not change cashflow figures. Paid and outstanding are
computed from the linked rows on each read.
"""

from decimal import Decimal
from typing import NamedTuple, cast

from sqlalchemy import CursorResult, delete, select, update
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
    """The receivable already links the transaction."""


class ReceivableTotals(NamedTuple):
    paid: Decimal
    outstanding: Decimal


def receivable_totals(receivable: Receivable) -> ReceivableTotals:
    """Compute what a person paid back and what they still owe.

    Args:
        receivable: A receivable with its links loaded.

    Returns:
        The sum of the linked payment rows, and the amount minus that sum.
    """
    paid = sum(
        (
            link.transaction.amount
            for link in receivable.links
            if link.role == ReceivableRole.payment
        ),
        Decimal("0.00"),
    )
    return ReceivableTotals(paid=paid, outstanding=receivable.amount - paid)


def receivable_read(receivable: Receivable) -> ReceivableRead:
    """Build the API view of a receivable.

    Args:
        receivable: A receivable with its links loaded.

    Returns:
        The receivable with its totals and linked row ids.
    """
    paid, outstanding = receivable_totals(receivable)
    ids = {role: [] for role in ReceivableRole}
    for link in sorted(receivable.links, key=lambda link: link.transaction_id):
        ids[ReceivableRole(link.role)].append(link.transaction_id)
    return ReceivableRead(
        id=receivable.id,
        person=receivable.person,
        amount=receivable.amount,
        note=receivable.note,
        status=ReceivableStatus(receivable.status),
        created_at=receivable.created_at,
        closed_at=receivable.closed_at,
        paid=paid,
        outstanding=outstanding,
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
    """Return the row a link can name, or raise ValueError with the reason."""
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
        ValueError: A transaction id is unknown, listed twice, or the wrong
            direction for its role.
    """
    wanted = [(i, ReceivableRole.charge) for i in payload.charge_transaction_ids]
    wanted += [(i, ReceivableRole.payment) for i in payload.payment_transaction_ids]
    seen = set()
    links = []
    for transaction_id, role in wanted:
        if transaction_id in seen:
            raise ValueError(f"Transaction {transaction_id} is listed twice")
        seen.add(transaction_id)
        txn = await _linkable(session, transaction_id, role)
        links.append(ReceivableLink(transaction=txn, role=role))
    receivable = Receivable(
        person=payload.person, amount=payload.amount, note=payload.note, links=links
    )
    session.add(receivable)
    await session.commit()
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
        LinkConflict: The receivable already links the row.
        ValueError: The row is unknown or the wrong direction for the role.
    """
    if any(link.transaction_id == transaction_id for link in receivable.links):
        raise LinkConflict(f"Transaction {transaction_id} is already linked")
    txn = await _linkable(session, transaction_id, role)
    receivable.links.append(ReceivableLink(transaction=txn, role=role))
    await session.commit()
    return receivable


async def remove_link(
    session: AsyncSession, receivable: Receivable, transaction_id: int
) -> bool:
    """Unlink a transaction from a receivable, and commit.

    Args:
        session: The request session.
        receivable: The receivable to change.
        transaction_id: The row to unlink.

    Returns:
        False when the receivable does not link the row.
    """
    link = next(
        (link for link in receivable.links if link.transaction_id == transaction_id),
        None,
    )
    if link is None:
        return False
    receivable.links.remove(link)
    await session.commit()
    return True


async def move_receivable_links(
    session: AsyncSession, old_id: int, new_id: int | None
) -> int:
    """Point the links of a deleted transaction at its keeper, or drop them.

    A link whose receivable already links the keeper is dropped, because a
    receivable links a row once.

    Args:
        session: The session that deletes ``old_id``.
        old_id: The transaction that goes away.
        new_id: The transaction that keeps the data, or ``None`` for a plain
            delete.

    Returns:
        The number of links moved or dropped.
    """
    stale = ReceivableLink.transaction_id == old_id
    if new_id is None:
        result = await session.execute(delete(ReceivableLink).where(stale))
        return cast(CursorResult, result).rowcount
    linked = select(ReceivableLink.receivable_id).where(
        ReceivableLink.transaction_id == new_id
    )
    dropped = await session.execute(
        delete(ReceivableLink).where(stale, ReceivableLink.receivable_id.in_(linked))
    )
    moved = await session.execute(
        update(ReceivableLink).where(stale).values(transaction_id=new_id)
    )
    return cast(CursorResult, dropped).rowcount + cast(CursorResult, moved).rowcount
