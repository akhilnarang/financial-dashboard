"""Typed values shared by investment transaction classification and consumers.

These DTOs contain only source-derived CAS facts.  Keeping them independent of
SQLAlchemy lets transaction classification stay pure
and reusable by ingestion, diagnostics, and projection code.
"""

import datetime
from decimal import Decimal
from typing import NamedTuple


class CompleteLot(NamedTuple):
    """Constructor-ready fields for a complete investment lot.

    Every field is an explicit, validated source fact.  Persistence turns this
    value into an ``InvestmentLot`` row; the type itself has no database
    dependency.
    """

    instrument_id: str
    instrument_name: str
    quantity: Decimal
    unit_cost: Decimal
    cost_basis: Decimal
    currency: str
    acquired_on: datetime.date
    source_ref: str
    transaction_type: str | None
    reference: str | None
    source_occurrence: int


class LotExclusion(NamedTuple):
    """Why a CAS transaction did not become a lot, with a stable reason."""

    reason: str
    detail: str
    instrument_id: str | None
    source_ref: str | None


class LotClassificationResult(NamedTuple):
    """A classified transaction: exactly one field is normally populated."""

    lot: CompleteLot | None
    exclusion: LotExclusion | None


class LotExtractionResult(NamedTuple):
    """Complete lots and exclusions extracted from one CAS payload."""

    lots: list[CompleteLot]
    exclusions: list[LotExclusion]


class CreateInvestmentLotsResult(NamedTuple):
    """Counts returned after persisting the complete facts from one upload."""

    created: int
    exclusions: list[LotExclusion]
