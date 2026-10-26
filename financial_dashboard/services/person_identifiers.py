"""Person identifiers: the counterparty, VPA and phone that name a person.

Receivable matching reads them. A payment link and the user add them.
"""

import re
from typing import NamedTuple

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.enums import PersonIdentifierKind
from financial_dashboard.db.models import PersonIdentifier, Receivable, Transaction
from financial_dashboard.exceptions import UnprocessableEntityException
from financial_dashboard.schemas.receivables import PersonIdentifierCreate
from financial_dashboard.services.assistant.intent_policy import (
    is_generic_counterparty,
)
from financial_dashboard.services.categorization.normalize import (
    normalize_counterparty,
)

# A dot after the handle starts a domain, so an email address is not a VPA.
VPA = re.compile(r"[\w.-]+@[a-z]+(?!\w|\.[a-z])")
PHONE = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")


class RawIdentifiers(NamedTuple):
    vpas: set[str]
    phones: set[str]


def normalize_person(name: str) -> str:
    """Return the identifier key of a person: lowercase, single spaces.

    Args:
        name: The person as the receivable names them.

    Returns:
        The key.
    """
    return " ".join(name.lower().split())


def canonical_vpa(vpa: str) -> str:
    """Drop a narration prefix from a VPA.

    A dash-joined narration glues the name to the VPA, as in
    ``upi-name-handle@bank``. Only the part after the last dash is the VPA.

    Args:
        vpa: A lowercase VPA.

    Returns:
        The VPA without the prefix.
    """
    local, _, bank = vpa.partition("@")
    return f"{local.rstrip('-').rsplit('-', 1)[-1]}@{bank}"


def identifier_value(kind: PersonIdentifierKind, value: str) -> str | None:
    """Return the stored form of an identifier.

    Args:
        kind: The identifier kind.
        value: The value as the user or the bank wrote it.

    Returns:
        The alphanumerics of a counterparty, a lowercase VPA, or the 10
        digits of a phone. ``None`` when the value is too short or not valid.
    """
    match kind:
        case PersonIdentifierKind.counterparty:
            squashed = normalize_counterparty(value)
            return squashed if len(squashed) >= 4 else None
        case PersonIdentifierKind.vpa:
            value = value.strip().lower()
            return canonical_vpa(value) if VPA.fullmatch(value) else None
        case PersonIdentifierKind.phone:
            digits = "".join(ch for ch in value if ch.isdigit())
            # A typed number can carry the country code. Bank text is not cut.
            digits = digits.removeprefix("91") if len(digits) == 12 else digits
            return digits if PHONE.fullmatch(digits) else None


def raw_identifiers(raw: str | None) -> RawIdentifiers:
    """Find the VPAs and phones in the raw text of a row.

    Args:
        raw: The raw description of the row.

    Returns:
        The distinct VPAs and phones, in stored form.
    """
    raw = (raw or "").lower()
    vpas = {canonical_vpa(vpa) for vpa in VPA.findall(raw)}
    return RawIdentifiers(vpas, set(PHONE.findall(raw)))


async def _store_identifiers(
    session: AsyncSession,
    person: str,
    found: list[tuple[PersonIdentifierKind, str]],
) -> None:
    """Insert stored-form identifiers for a person, and skip the ones that exist.

    Args:
        session: The session to write in.
        person: The person, in any spelling. The key is its normalized form.
        found: The kind and stored-form value of each identifier.
    """
    rows = [
        {"person": normalize_person(person), "kind": kind, "value": value}
        for kind, value in dict.fromkeys(found)
    ]
    if rows:
        await session.execute(
            sqlite_insert(PersonIdentifier).values(rows).on_conflict_do_nothing()
        )


async def learn_identifiers(
    session: AsyncSession, person: str, txn: Transaction
) -> None:
    """Store the counterparty, VPA and phone of a payment row for a person.

    A VPA or a phone is stored only when the row has exactly one. A generic
    counterparty is not stored. Call it before a new link is pending. Its
    insert flushes the session.

    Args:
        session: The session that links the row.
        person: The person the row pays or is paid by.
        txn: The payment row.
    """
    raw = raw_identifiers(txn.raw_description)
    found = [(PersonIdentifierKind.vpa, vpa) for vpa in raw.vpas if len(raw.vpas) == 1]
    found += [
        (PersonIdentifierKind.phone, phone)
        for phone in raw.phones
        if len(raw.phones) == 1
    ]
    squashed = identifier_value(
        PersonIdentifierKind.counterparty, txn.counterparty or ""
    )
    if squashed is not None and not is_generic_counterparty(txn.counterparty):
        found.append((PersonIdentifierKind.counterparty, squashed))
    await _store_identifiers(session, person, found)


async def rename_identifiers(
    session: AsyncSession, receivable: Receivable, person: str
) -> None:
    """Give the identifiers of a receivable's old name to its new name.

    Move them when no other receivable uses the old name. Copy them otherwise.
    A new name that another receivable uses, or that has identifiers, is
    another person. It gets nothing.

    Args:
        session: The session that renames the receivable.
        receivable: The receivable, still with its old name.
        person: The new name.
    """
    old, new = normalize_person(receivable.person), normalize_person(person)
    others = set(
        map(
            normalize_person,
            await session.scalars(
                select(Receivable.person).where(Receivable.id != receivable.id)
            ),
        )
    )
    taken = await session.scalar(
        select(PersonIdentifier.id).where(PersonIdentifier.person == new)
    )
    if old == new or new in others or taken is not None:
        return
    rows = (
        await session.scalars(
            select(PersonIdentifier).where(PersonIdentifier.person == old)
        )
    ).all()
    await _store_identifiers(
        session, new, [(PersonIdentifierKind(row.kind), row.value) for row in rows]
    )
    if old not in others:
        await session.execute(
            delete(PersonIdentifier).where(PersonIdentifier.person == old)
        )


async def add_identifier(
    session: AsyncSession, payload: PersonIdentifierCreate
) -> PersonIdentifier:
    """Store an identifier of a person, and commit.

    Args:
        session: The request session.
        payload: The person, kind and value.

    Returns:
        The stored identifier. It can be one that already existed.
    """
    identifier = await store_identifier(session, payload)
    await session.commit()
    return identifier


async def store_identifier(
    session: AsyncSession, payload: PersonIdentifierCreate
) -> PersonIdentifier:
    """Store an identifier of a person. Do not commit.

    Args:
        session: The session to write in.
        payload: The person, kind and value.

    Returns:
        The stored identifier.

    Raises:
        UnprocessableEntityException: The value is not valid for its kind.
    """
    if (value := identifier_value(payload.kind, payload.value)) is None:
        raise UnprocessableEntityException(
            detail=f"Not a valid {payload.kind}: use 4+ letters for a counterparty, "
            "name@handle for a VPA, or a 10-digit mobile number"
        )
    await _store_identifiers(session, payload.person, [(payload.kind, value)])
    stored = await session.scalars(
        select(PersonIdentifier).where(
            PersonIdentifier.person == normalize_person(payload.person),
            PersonIdentifier.kind == payload.kind,
            PersonIdentifier.value == value,
        )
    )
    return stored.one()


async def list_identifiers(session: AsyncSession) -> list[PersonIdentifier]:
    """List every stored identifier, by person.

    Args:
        session: The request session.

    Returns:
        The identifiers.
    """
    query = select(PersonIdentifier).order_by(
        PersonIdentifier.person, PersonIdentifier.kind, PersonIdentifier.value
    )
    return list(await session.scalars(query))
