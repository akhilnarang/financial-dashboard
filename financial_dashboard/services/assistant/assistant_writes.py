"""Receivable reads and confirmed receivable and category writes for the assistant.

A write runs only after the user taps Yes. The caller holds the SQLite write
lock and commits.
"""

import hashlib
from decimal import Decimal

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.templating import format_inr_exact
from financial_dashboard.db.enums import (
    PersonIdentifierKind,
    ReceivableRole,
)
from financial_dashboard.db.models import Receivable, Transaction
from financial_dashboard.exceptions import (
    ConflictException,
    UnprocessableEntityException,
)
from financial_dashboard.schemas.receivables import (
    PersonIdentifierCreate,
    ReceivableCreate,
    ReceivableRead,
    ReceivableUpdate,
)
from financial_dashboard.services.assistant.contracts import (
    AddPersonIdentifier,
    CreateCategory,
    CreateReceivable,
    LinkReceivable,
    AssistantWrite,
    UpdateReceivable,
)
from financial_dashboard.services.assistant.mutations import MutationRejected
from financial_dashboard.services.categorization.vocabulary import (
    canonicalize_slug,
    create_category,
)
from financial_dashboard.services.person_identifiers import (
    identifier_value,
    store_identifier,
)
from financial_dashboard.services.receivables import (
    apply_update,
    attach_link,
    build_receivable,
    receivable_read,
)


async def _receivable(session: AsyncSession, receivable_id: int) -> Receivable:
    """Load a receivable fresh from the database, or refuse the write."""
    if (
        receivable := await session.get(
            Receivable, receivable_id, populate_existing=True
        )
    ) is not None:
        return receivable
    raise MutationRejected(f"receivable #{receivable_id} not found")


def _update_payload(request: UpdateReceivable) -> ReceivableUpdate:
    """Turn the non-null fields of an update into a validated patch, or refuse it."""
    if fields := request.model_dump(
        exclude={"name", "receivable_id"}, exclude_none=True
    ):
        return ReceivableUpdate.model_validate(fields)
    raise MutationRejected("name a field to change")


async def apply_assistant_write(session: AsyncSession, request: AssistantWrite) -> str:
    """Apply one receivable or category write. Do not commit.

    Args:
        session: A session that holds the write lock.
        request: The confirmed write.

    Returns:
        The reply text.

    Raises:
        MutationRejected: The write is not valid.
    """
    try:
        match request:
            case CreateReceivable():
                created = await build_receivable(
                    session,
                    ReceivableCreate.model_validate(
                        request.model_dump(exclude={"name"}, exclude_none=True)
                    ),
                )
                await session.flush()
                return f"Created {created.person} #{created.id}."
            case AddPersonIdentifier():
                row = await store_identifier(
                    session,
                    PersonIdentifierCreate.model_validate(
                        request.model_dump(exclude={"name"})
                    ),
                )
                return f"Saved {row.kind} {row.value} for {row.person}."
            case CreateCategory():
                category = await create_category(session, request.slug, request.bucket)
                return f"Created {request.bucket} category {category.slug}."
            case LinkReceivable():
                receivable = await _receivable(session, request.receivable_id)
                await attach_link(
                    session,
                    receivable,
                    request.transaction_id,
                    ReceivableRole(request.role),
                )
            case UpdateReceivable():
                receivable = await _receivable(session, request.receivable_id)
                await apply_update(session, receivable, _update_payload(request))
        await session.flush()
    except ValidationError as exc:
        raise MutationRejected(exc.errors()[0]["msg"]) from exc
    except (UnprocessableEntityException, ConflictException) as exc:
        raise MutationRejected(str(exc.detail)) from exc
    except IntegrityError as exc:
        raise MutationRejected(
            "a payment row pays one receivable, and a receivable links a row once"
        ) from exc
    read = receivable_read(receivable)
    return (
        f"Saved {read.person} #{read.id}: {read.status}, "
        f"outstanding {format_inr_exact(read.outstanding)}."
    )


def _describe(read: ReceivableRead) -> str:
    """Name a receivable with its kind, status and outstanding for a question."""
    return (
        f"{read.person} #{read.id} ({read.kind}, {read.status}, "
        f"outstanding {format_inr_exact(read.outstanding)})"
    )


async def describe_assistant_write(
    session: AsyncSession, request: AssistantWrite
) -> str:
    """Render the confirmation question from stored data and the request.

    Args:
        session: The turn session.
        request: The proposed write. A dry run has validated it.

    Returns:
        The question.

    Raises:
        MutationRejected: A named receivable or row does not exist.
    """
    match request:
        case CreateReceivable():
            who = (
                f"{request.person} owes you"
                if request.kind == "receivable"
                else f"you owe {request.person}"
            )
            text = f"Record that {who} {format_inr_exact(Decimal(request.amount))}"
            if request.note:
                text += f" for '{request.note}'"
            if request.charge_transaction_ids:
                text += ", charges " + ", ".join(
                    f"#{i}" for i in request.charge_transaction_ids
                )
            return text + "?"
        case AddPersonIdentifier():
            value = identifier_value(PersonIdentifierKind(request.kind), request.value)
            return f"Save {request.kind} {value} for {request.person}?"
        case CreateCategory():
            return (
                f"Create the {request.bucket} category "
                f"{canonicalize_slug(request.slug)}?"
            )
        case LinkReceivable():
            read = receivable_read(await _receivable(session, request.receivable_id))
            if (txn := await session.get(Transaction, request.transaction_id)) is None:
                raise MutationRejected(
                    f"transaction #{request.transaction_id} not found"
                )
            currency = txn.currency or "INR"
            amount = (
                format_inr_exact(txn.amount)
                if currency == "INR"
                else f"{txn.amount} {currency}"
            )
            row = (
                f"#{txn.id} ({txn.direction} {amount}, "
                f"{(txn.counterparty or 'unknown')[:200]}, {txn.transaction_date})"
            )
            return f"Link {row} to {_describe(read)} as a {request.role}?"
        case UpdateReceivable():
            read = receivable_read(await _receivable(session, request.receivable_id))
            changes = [
                f"amount to {format_inr_exact(value)}"
                if field == "amount"
                else f"{field} to '{value}'"
                for field, value in _update_payload(request)
                .model_dump(exclude_unset=True)
                .items()
            ]
            return f"Change {_describe(read)}: set {', '.join(changes)}?"


async def assistant_write_state_hash(
    session: AsyncSession, request: AssistantWrite
) -> str | None:
    """Fingerprint the state that a confirmation of this write relies on.

    For a link or an update, the hash covers the amount, outstanding,
    status, person, note and links of the receivable. For a link, it also
    covers the amount, direction, currency, void state and counterparty of
    the row. A change in between voids the old button.

    Args:
        session: The turn session.
        request: The proposed write.

    Returns:
        The hash, or ``None`` when the receivable no longer exists.
    """
    state = [request.model_dump_json()]
    receivable_id = None
    row_fields: tuple[Decimal | str | bool | None, ...] = ()
    match request:
        case LinkReceivable():
            receivable_id = request.receivable_id
            txn = await session.get(
                Transaction, request.transaction_id, populate_existing=True
            )
            row_fields = (
                (txn.amount, txn.direction, txn.currency, txn.voided, txn.counterparty)
                if txn is not None
                else (None,)
            )
        case UpdateReceivable():
            receivable_id = request.receivable_id
        case CreateReceivable() | AddPersonIdentifier() | CreateCategory():
            pass
    if receivable_id is not None:
        receivable = await session.get(
            Receivable, receivable_id, populate_existing=True
        )
        if receivable is None:
            return None
        read = receivable_read(receivable)
        state += map(
            str,
            (
                read.outstanding,
                read.amount,
                read.status,
                read.person,
                read.note,
                read.charge_transaction_ids,
                read.payment_transaction_ids,
                *row_fields,
            ),
        )
    return hashlib.sha256("\x1f".join(state).encode()).hexdigest()
