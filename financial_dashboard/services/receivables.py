"""Receivables ledger: money other people owe the user, and money the user owes.

The ledger does not change cashflow figures. Paid and outstanding are
computed from the linked rows on each read.
"""

from decimal import Decimal
from typing import cast

from sqlalchemy import CursorResult, delete, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.enums import (
    ReceivableKind,
    ReceivableRole,
    ReceivableStatus,
)
from financial_dashboard.db.models import (
    Receivable,
    ReceivableLink,
    ReceivableMatchPrompt,
    TelegramOutboundDelivery,
    Transaction,
    utc_now,
)
from financial_dashboard.exceptions import (
    ConflictException,
    UnprocessableEntityException,
)
from financial_dashboard.schemas.receivables import (
    ReceivableCreate,
    ReceivableRead,
    ReceivableUpdate,
)
from financial_dashboard.services.person_identifiers import (
    learn_identifiers,
    rename_identifiers,
)

ROLE_DIRECTION = {
    ReceivableKind.receivable: {
        ReceivableRole.charge: "debit",
        ReceivableRole.payment: "credit",
    },
    ReceivableKind.payable: {
        ReceivableRole.charge: "credit",
        ReceivableRole.payment: "debit",
    },
}


def receivable_read(receivable: Receivable) -> ReceivableRead:
    """Build the API view of a receivable.

    Args:
        receivable: A receivable with its links loaded.

    Returns:
        The receivable with its linked row ids, the sum of its payment
        rows, and the amount minus that sum.
    """
    paid = _paid(receivable)
    ids = {role: [] for role in ReceivableRole}
    links = [link for link in receivable.links if link.transaction is not None]
    for link in sorted(links, key=lambda link: link.transaction_id):
        ids[ReceivableRole(link.role)].append(link.transaction_id)
    return ReceivableRead(
        id=receivable.id,
        kind=ReceivableKind(receivable.kind),
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


def _paid(receivable: Receivable) -> Decimal:
    """Sum the payment rows of a receivable. A link whose row is gone adds nothing."""
    return sum(
        (
            link.transaction.amount
            for link in receivable.links
            if link.role == ReceivableRole.payment and link.transaction is not None
        ),
        Decimal("0.00"),
    )


def _settle_if_paid(receivable: Receivable) -> None:
    """Settle an open receivable that its payment links pay in full."""
    if (
        receivable.status == ReceivableStatus.open
        and _paid(receivable) >= receivable.amount
    ):
        receivable.status = ReceivableStatus.settled
        receivable.closed_at = utc_now()


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
    session: AsyncSession,
    transaction_id: int,
    kind: ReceivableKind,
    role: ReceivableRole,
) -> Transaction:
    """Load the row a link names and check that it fits the role.

    Args:
        session: The request session.
        transaction_id: The row to link.
        kind: The kind of the receivable the link belongs to.
        role: The role the link takes.

    Returns:
        The transaction.

    Raises:
        UnprocessableEntityException: The row is unknown, the wrong
            direction, or not in INR.
    """
    txn = await session.get(Transaction, transaction_id)
    if txn is None:
        raise UnprocessableEntityException(
            detail=f"Transaction {transaction_id} not found"
        )
    if txn.direction != (direction := ROLE_DIRECTION[kind][role]):
        raise UnprocessableEntityException(
            detail=f"A {kind} {role} link needs a {direction} row; "
            f"transaction {transaction_id} is a {txn.direction}"
        )
    if (txn.currency or "INR") != "INR":
        raise UnprocessableEntityException(
            detail=f"Transaction {transaction_id} is not in INR"
        )
    if txn.voided:
        raise UnprocessableEntityException(
            detail=f"Transaction {transaction_id} is voided"
        )
    return txn


async def write_lock(session: AsyncSession) -> None:
    """Take the SQLite write lock. It holds until the commit or rollback.

    Args:
        session: A session that holds no write yet.

    Returns:
        None.
    """
    if session.get_bind().dialect.name == "sqlite":
        await session.execute(text("BEGIN IMMEDIATE"))


async def commit_links(session: AsyncSession) -> None:
    """Commit, and turn a duplicate link into a conflict.

    Args:
        session: The session that holds the new links.

    Returns:
        None.

    Raises:
        ConflictException: A unique index refused a link.
    """
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise ConflictException(
            detail="A payment row pays one receivable, and a receivable links a row once"
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
        ConflictException: A payment row already pays another receivable.
        UnprocessableEntityException: A transaction id is unknown, listed
            twice, or the wrong direction for its role.
    """
    await write_lock(session)
    try:
        receivable = await build_receivable(session, payload)
    except UnprocessableEntityException:
        await session.rollback()
        raise
    await commit_links(session)
    return receivable


async def build_receivable(
    session: AsyncSession, payload: ReceivableCreate
) -> Receivable:
    """Add a receivable and its links to the session. Do not commit.

    The caller must hold the write lock. A payment link teaches the
    identifiers of its row.

    Args:
        session: A session that holds the write lock.
        payload: The validated request body.

    Returns:
        The new, unflushed receivable.

    Raises:
        UnprocessableEntityException: A transaction id is unknown, listed
            twice, or the wrong direction for its role.
    """
    wanted = [(i, ReceivableRole.charge) for i in payload.charge_transaction_ids]
    wanted += [(i, ReceivableRole.payment) for i in payload.payment_transaction_ids]
    ids = [transaction_id for transaction_id, _ in wanted]
    if len(ids) != len(set(ids)):
        raise UnprocessableEntityException(detail="A transaction is listed twice")
    links = [
        ReceivableLink(
            transaction=await _linkable(session, i, payload.kind, role), role=role
        )
        for i, role in wanted
    ]
    for link in links:
        if link.role == ReceivableRole.payment:
            await learn_identifiers(session, payload.person, link.transaction)
    receivable = Receivable(
        kind=payload.kind,
        person=payload.person,
        amount=payload.amount,
        note=payload.note,
        status=ReceivableStatus.open,
        links=links,
    )
    _settle_if_paid(receivable)
    session.add(receivable)
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
    await apply_update(session, receivable, payload)
    await session.commit()
    return receivable


async def apply_update(
    session: AsyncSession, receivable: Receivable, payload: ReceivableUpdate
) -> None:
    """Apply a partial edit to a receivable. Do not commit.

    A new person takes the identifiers of the old one.

    Args:
        session: The session to write in.
        receivable: The receivable to change.
        payload: The fields to change.

    Returns:
        None.
    """
    amount_changed = payload.amount not in (None, receivable.amount)
    if payload.amount is not None:
        receivable.amount = payload.amount
    if payload.person is not None:
        await rename_identifiers(session, receivable, payload.person)
        receivable.person = payload.person
    if "note" in payload.model_fields_set:
        receivable.note = payload.note
    if payload.status is not None and payload.status != receivable.status:
        receivable.status = payload.status
        is_open = payload.status == ReceivableStatus.open
        receivable.closed_at = None if is_open else utc_now()
    elif amount_changed and "status" not in payload.model_fields_set:
        _settle_if_paid(receivable)


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
        role: ``charge`` for the row the money was for, ``payment`` for a
            row that pays it back.

    Returns:
        The changed receivable.

    Raises:
        ConflictException: The receivable already links the row, or the
            payment row already pays another receivable.
        UnprocessableEntityException: The row is unknown or the wrong
            direction for the role.
    """
    await write_lock(session)
    try:
        await attach_link(session, receivable, transaction_id, role)
    except UnprocessableEntityException:
        await session.rollback()
        raise
    await commit_links(session)
    return receivable


async def attach_link(
    session: AsyncSession,
    receivable: Receivable,
    transaction_id: int,
    role: ReceivableRole,
) -> None:
    """Link a row to a receivable. Do not commit.

    The caller must hold the write lock, so a merge or delete cannot remove
    the row first. A payment link teaches the identifiers of its row.

    Args:
        session: A session that holds the write lock.
        receivable: The receivable to link.
        transaction_id: The row to link.
        role: The role the link takes.

    Returns:
        None.

    Raises:
        UnprocessableEntityException: The row is unknown or the wrong
            direction for the role.
    """
    txn = await _linkable(
        session, transaction_id, ReceivableKind(receivable.kind), role
    )
    if role == ReceivableRole.payment:
        await learn_identifiers(session, receivable.person, txn)
    receivable.links.append(ReceivableLink(transaction=txn, role=role))
    if role == ReceivableRole.payment:
        _settle_if_paid(receivable)


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
    await write_lock(session)
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
    show one event, so its money counts once. The match prompt of the old row
    goes away with its unsent message, so its buttons go stale.

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
    prompt = ReceivableMatchPrompt.transaction_id == old_id
    await session.execute(
        update(TelegramOutboundDelivery)
        .where(
            TelegramOutboundDelivery.id.in_(
                select(ReceivableMatchPrompt.delivery_id).where(prompt)
            ),
            TelegramOutboundDelivery.status.in_(
                ["pending", "delivering", "delivery_unknown"]
            ),
        )
        .values(status="cancelled", worker_token=None, delivery_lease_until=None)
    )
    await session.execute(delete(ReceivableMatchPrompt).where(prompt))
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
    for receivable in await session.scalars(
        select(Receivable)
        .join(ReceivableLink)
        .where(
            ReceivableLink.transaction_id == new_id,
            ReceivableLink.role == ReceivableRole.payment,
            Receivable.status == ReceivableStatus.open,
        )
        .execution_options(populate_existing=True)
    ):
        _settle_if_paid(receivable)
    counts = {
        "receivable_links": cast(CursorResult, moved).rowcount,
        "receivable_links.dropped": cast(CursorResult, dropped).rowcount,
    }
    return {key: n for key, n in counts.items() if n}
