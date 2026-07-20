"""Deterministic id and reference helpers.

All stable ids are UUIDv5 over the :data:`~constants.SYNTH_NAMESPACE` so that
``(seed, profile, as_of)`` always produces byte-identical ids on every machine.
Money helpers build :class:`decimal.Decimal` from integer paisa so no float
ever touches a generated amount.
"""

import random
import uuid
from decimal import Decimal

from scripts.synth.constants import SYNTH_NAMESPACE, BULK_TXN_ID_BASE

#: Module-level binding so the hot generation loop (hundreds of thousands of
#: ids for a stress profile) avoids a per-call ``import`` + attribute lookup.
_uuid5 = uuid.uuid5


def stable_id(kind: str, *parts: str) -> str:
    """Return a deterministic 32-char hex id for ``kind`` keyed by ``parts``.

    ``kind`` is folded into the digest so an account id and a transaction id
    can never collide even if they share the same descriptive parts.
    """
    material = "|".join((kind, *(str(p) for p in parts)))
    return _uuid5(SYNTH_NAMESPACE, material).hex


def txn_reference(stable_txn_id: str) -> str:
    """The synthetic ``reference_number`` for a transaction.

    Prefixed and upper-cased so it is visually distinct from real bank refs
    and so the dashboard's unique index on
    ``(bank, reference_number, direction)`` provides natural-key idempotency
    for bulk-lane reruns.
    """
    return f"SYN-{stable_txn_id[:16].upper()}"


def bulk_txn_pk(index: int) -> int:
    """Primary key for the ``index``-th bulk-lane transaction.

    Offset by :data:`BULK_TXN_ID_BASE` so it stays clear of the autoincrement
    ids the fidelity lane assigns starting at 1.
    """
    return BULK_TXN_ID_BASE + index


def money(rng: random.Random, low_paise: int, high_paise: int) -> Decimal:
    """A deterministic money value in INR, built from paisa so it is exact."""
    paise = rng.randint(low_paise, high_paise)
    return (Decimal(paise) / Decimal(100)).quantize(Decimal("0.01"))


def quantize(value: Decimal) -> Decimal:
    """Two-decimal quantization matching the dashboard's ``Numeric(12,2)``."""
    return value.quantize(Decimal("0.01"))
