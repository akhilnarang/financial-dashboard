"""JSON endpoints for the receivables ledger."""

from typing import Annotated

from fastapi import APIRouter, Path

from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.db.enums import ReceivableStatus
from financial_dashboard.db.models import Receivable
from financial_dashboard.exceptions import NotFoundException
from financial_dashboard.schemas.common import DatabaseId
from financial_dashboard.schemas.receivables import (
    PersonIdentifierCreate,
    PersonIdentifierListResponse,
    PersonIdentifierRead,
    ReceivableCreate,
    ReceivableLinkCreate,
    ReceivableListResponse,
    ReceivableRead,
    ReceivableUpdate,
)
from financial_dashboard.services.person_identifiers import (
    add_identifier,
    list_identifiers,
)
from financial_dashboard.services.receivables import (
    add_link,
    create_receivable,
    list_receivables,
    receivable_read,
    remove_link,
    update_receivable,
)

router = APIRouter()


async def _get(session: AsyncSessionDep, receivable_id: int) -> Receivable:
    """Load a receivable for a route.

    Args:
        session: The request session.
        receivable_id: The receivable to load.

    Returns:
        The receivable with its links.

    Raises:
        NotFoundException: No such receivable exists.
    """
    receivable = await session.get(Receivable, receivable_id)
    if receivable is None:
        raise NotFoundException(detail="Receivable not found")
    return receivable


@router.get("/receivables")
async def receivables_list(
    session: AsyncSessionDep, status: ReceivableStatus | None = None
) -> ReceivableListResponse:
    """List receivables with paid and outstanding sums.

    Args:
        session: The request session.
        status: Keep only this status, or all when absent.

    Returns:
        The matching receivables, oldest first.
    """
    rows = await list_receivables(session, status)
    return ReceivableListResponse(items=[receivable_read(row) for row in rows])


@router.post("/receivables", status_code=201)
async def receivables_create(
    payload: ReceivableCreate, session: AsyncSessionDep
) -> ReceivableRead:
    """Create a receivable, with optional charge and payment rows.

    Args:
        payload: The person, amount, note and linked row ids.
        session: The request session.

    Returns:
        The new receivable.
    """
    return receivable_read(await create_receivable(session, payload))


@router.patch("/receivables/{receivable_id}")
async def receivables_update(
    receivable_id: Annotated[DatabaseId, Path()],
    payload: ReceivableUpdate,
    session: AsyncSessionDep,
) -> ReceivableRead:
    """Change the amount, note, status or person of a receivable.

    Args:
        receivable_id: The receivable to change.
        payload: The fields to change.
        session: The request session.

    Returns:
        The changed receivable.
    """
    receivable = await _get(session, receivable_id)
    return receivable_read(await update_receivable(session, receivable, payload))


@router.post("/receivables/{receivable_id}/links", status_code=201)
async def receivables_link(
    receivable_id: Annotated[DatabaseId, Path()],
    payload: ReceivableLinkCreate,
    session: AsyncSessionDep,
) -> ReceivableRead:
    """Link a transaction to a receivable as a charge or a payment.

    Args:
        receivable_id: The receivable to link.
        payload: The row and its role.
        session: The request session.

    Returns:
        The changed receivable.
    """
    receivable = await _get(session, receivable_id)
    receivable = await add_link(
        session, receivable, payload.transaction_id, payload.role
    )
    return receivable_read(receivable)


@router.delete("/receivables/{receivable_id}/links/{transaction_id}")
async def receivables_unlink(
    receivable_id: Annotated[DatabaseId, Path()],
    transaction_id: Annotated[DatabaseId, Path()],
    session: AsyncSessionDep,
) -> ReceivableRead:
    """Unlink a transaction from a receivable.

    Args:
        receivable_id: The receivable to change.
        transaction_id: The row to unlink.
        session: The request session.

    Returns:
        The changed receivable.
    """
    receivable = await remove_link(session, receivable_id, transaction_id)
    if receivable is None:
        raise NotFoundException(detail="Link not found")
    return receivable_read(receivable)


@router.get("/receivables/identifiers")
async def identifiers_list(session: AsyncSessionDep) -> PersonIdentifierListResponse:
    """List the stored identifiers of every person.

    Args:
        session: The request session.

    Returns:
        The identifiers, by person.
    """
    rows = await list_identifiers(session)
    return PersonIdentifierListResponse(
        items=[
            PersonIdentifierRead.model_validate(row, from_attributes=True)
            for row in rows
        ]
    )


@router.post("/receivables/identifiers", status_code=201)
async def identifiers_create(
    payload: PersonIdentifierCreate, session: AsyncSessionDep
) -> PersonIdentifierRead:
    """Store a counterparty, VPA or phone that names a person.

    Args:
        payload: The person, kind and value.
        session: The request session.

    Returns:
        The stored identifier.
    """
    row = await add_identifier(session, payload)
    return PersonIdentifierRead.model_validate(row, from_attributes=True)
