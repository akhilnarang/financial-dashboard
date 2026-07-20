"""Pure CAS investment transaction classification and disposal allocation.

This module deliberately knows nothing about SQLAlchemy.  It converts explicit
CAS transaction facts into complete acquisition lots or stable exclusions, and
conservatively nets disposals only when the source identifies an exact lot.
"""

import datetime
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from financial_dashboard.core.dates import parse_date
from financial_dashboard.services.investment_types import (
    CompleteLot,
    LotClassificationResult,
    LotConsumption,
    LotExclusion,
    LotExtractionResult,
    LotKey,
)

# CAS is an Indian depository statement; every amount it prints is INR.  This
# is a fact about the document, not a fabricated currency.
CAS_CURRENCY = "INR"

# CAS prints units/nav at limited precision, so the printed amount can differ
# from their full-precision product by sub-penny display noise.  The exclusive
# bound rejects a discrepancy of a full paisa or more.
_LOT_AGREEMENT_TOLERANCE = Decimal("0.01")

# The parser emits free-form transaction type strings, so these are matched as
# lower-cased substrings.  Unknown values remain ambiguous rather than guessed.
_ACQUISITION_TYPES = ("purchase", "switch_in", "switch-in", "buy", "allotment")
_DISPOSAL_TYPES = ("redemption", "switch_out", "switch-out", "sell", "sold")


class _AcquisitionFact(NamedTuple):
    key: LotKey
    quantity: Decimal


class _DisposalFact(NamedTuple):
    source_ref: str
    quantity: Decimal | None
    disposed_on: datetime.date | None
    reference: str | None
    order: int


def _to_decimal(value) -> Decimal | None:
    """Parse a CAS numeric field into a finite ``Decimal``, or ``None``.

    Strings, ints, floats and existing Decimals are accepted because the input
    is JSON-decoded.  Non-finite and unparseable values exclude their row
    instead of failing an entire ingest.
    """
    if value is None or value == "":
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation, ValueError, TypeError:
        return None
    if not amount.is_finite():
        return None
    return amount


def _classify_transaction(raw: dict) -> LotClassificationResult:
    """Classify one CAS transaction into a complete lot or an exclusion.

    Returns ``(lot, None)`` for a complete acquisition and
    ``(None, exclusion)`` otherwise.  ``LotClassificationResult`` preserves
    that positional interface while giving callers named fields.
    """
    scope = str(raw.get("scope") or "").lower()
    source_ref = raw.get("source_ref")
    source_ref_text = str(source_ref) if source_ref is not None else None
    isin = raw.get("isin")
    isin_text = (
        str(isin).strip().upper() if isinstance(isin, str) and isin.strip() else None
    )
    ttype_raw = raw.get("transaction_type")
    ttype = (
        str(ttype_raw).strip().lower()
        if isinstance(ttype_raw, str) and ttype_raw.strip()
        else ""
    )
    reference = raw.get("reference")
    reference_text = (
        str(reference).strip()
        if isinstance(reference, str) and reference.strip()
        else None
    )

    if scope != "mf":
        return LotClassificationResult(
            lot=None,
            exclusion=LotExclusion(
                reason="not_mutual_fund",
                detail=(
                    f"scope {scope!r}: only MF purchase transactions carry cost in CAS"
                ),
                instrument_id=isin_text,
                source_ref=source_ref_text,
            ),
        )

    if ttype and any(marker in ttype for marker in _DISPOSAL_TYPES):
        return LotClassificationResult(
            lot=None,
            exclusion=LotExclusion(
                reason="disposal_transaction",
                detail=f"transaction_type {ttype!r} is a disposal, not an acquisition",
                instrument_id=isin_text,
                source_ref=source_ref_text,
            ),
        )
    if not ttype or not any(marker in ttype for marker in _ACQUISITION_TYPES):
        return LotClassificationResult(
            lot=None,
            exclusion=LotExclusion(
                reason="ambiguous_transaction_type",
                detail=(
                    f"transaction_type {ttype!r} does not identify an acquisition; "
                    f"refusing to treat the date as an acquisition date"
                ),
                instrument_id=isin_text,
                source_ref=source_ref_text,
            ),
        )

    units = _to_decimal(raw.get("units"))
    nav = _to_decimal(raw.get("nav"))
    amount = _to_decimal(raw.get("amount"))
    date_raw = raw.get("date")
    acquired_on = parse_date(str(date_raw)) if date_raw is not None else None

    missing: list[str] = []
    if units is None or units <= 0:
        missing.append("units")
    if nav is None or nav <= 0:
        missing.append("nav")
    if amount is None or amount <= 0:
        missing.append("amount")
    if acquired_on is None:
        missing.append("date")
    if isin_text is None:
        missing.append("isin")
    if missing:
        return LotClassificationResult(
            lot=None,
            exclusion=LotExclusion(
                reason="missing_lot_facts",
                detail="missing/invalid: " + ", ".join(missing),
                instrument_id=isin_text,
                source_ref=source_ref_text,
            ),
        )

    assert units is not None and nav is not None and amount is not None
    product = units * nav
    discrepancy = abs(product - amount)
    if discrepancy >= _LOT_AGREEMENT_TOLERANCE:
        return LotClassificationResult(
            lot=None,
            exclusion=LotExclusion(
                reason="cost_basis_inconsistent",
                detail=(
                    f"amount {amount} differs from units*nav {product} by "
                    f"{discrepancy}; CAS values disagree beyond rounding"
                ),
                instrument_id=isin_text,
                source_ref=source_ref_text,
            ),
        )
    cost_basis = product.quantize(_LOT_AGREEMENT_TOLERANCE)

    description = raw.get("description")
    instrument_name = (
        str(description).strip()
        if isinstance(description, str) and description.strip()
        else isin_text
    )
    assert isin_text is not None
    assert acquired_on is not None
    assert instrument_name

    return LotClassificationResult(
        lot=CompleteLot(
            instrument_id=isin_text,
            instrument_name=instrument_name,
            quantity=units,
            unit_cost=nav,
            cost_basis=cost_basis,
            currency=CAS_CURRENCY,
            acquired_on=acquired_on,
            source_ref=str(source_ref),
            transaction_type=ttype or None,
            reference=reference_text,
            source_occurrence=0,
        ),
        exclusion=None,
    )


def extract_lots_from_payload(payload: dict) -> LotExtractionResult:
    """Split one CAS payload's transactions into lots and exclusions.

    Repeated complete rows retain source multiplicity and receive a zero-based
    occurrence within their source identity.  ``LotExtractionResult`` remains
    tuple-compatible for existing unpacking and index access.
    """
    lots: list[CompleteLot] = []
    exclusions: list[LotExclusion] = []
    occurrences: dict[tuple[str, str, datetime.date, str | None], int] = defaultdict(
        int
    )
    for raw in payload.get("transactions") or []:
        if not isinstance(raw, dict):
            continue
        lot, exclusion = _classify_transaction(raw)
        if lot is not None:
            key = (
                lot.source_ref,
                lot.instrument_id,
                lot.acquired_on,
                lot.reference,
            )
            occurrence = occurrences[key]
            occurrences[key] += 1
            lots.append(lot._replace(source_occurrence=occurrence))
        elif exclusion is not None:
            exclusions.append(exclusion)
    return LotExtractionResult(lots=lots, exclusions=exclusions)


def _txn_isin(raw: dict) -> str | None:
    isin = raw.get("isin")
    if isinstance(isin, str) and isin.strip():
        return isin.strip().upper()
    return None


def _txn_is_disposal(raw: dict) -> bool:
    """Whether a CAS transaction disposes units."""
    ttype_raw = raw.get("transaction_type")
    ttype = (
        str(ttype_raw).strip().lower()
        if isinstance(ttype_raw, str) and ttype_raw.strip()
        else ""
    )
    return bool(ttype) and any(marker in ttype for marker in _DISPOSAL_TYPES)


def _txn_is_acquisition(raw: dict) -> bool:
    """Whether a CAS transaction acquires units."""
    ttype_raw = raw.get("transaction_type")
    ttype = (
        str(ttype_raw).strip().lower()
        if isinstance(ttype_raw, str) and ttype_raw.strip()
        else ""
    )
    return bool(ttype) and any(marker in ttype for marker in _ACQUISITION_TYPES)


def _normalized_ref(value) -> str:
    return str(value).strip() if value is not None else ""


def _normalized_reference(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _allocate_exact(
    disposal: _DisposalFact,
    acquisitions: list[_AcquisitionFact],
    remaining: dict[LotKey, Decimal],
) -> bool:
    """Consume a disposal only when its source facts identify one lot."""
    active = [lot for lot in acquisitions if remaining[lot.key] > 0]
    if not active or disposal.quantity is None or disposal.quantity <= 0:
        return False

    if disposal.reference:
        exact = [lot for lot in active if lot.key.reference == disposal.reference]
        if len(exact) > 1:
            return False
        if len(exact) == 1:
            lot = exact[0]
            if (
                disposal.disposed_on is not None
                and lot.key.acquired_on > disposal.disposed_on
            ):
                return False
            if disposal.quantity > remaining[lot.key]:
                return False
            remaining[lot.key] -= disposal.quantity
            return True

    eligible = [
        lot
        for lot in active
        if disposal.disposed_on is None or lot.key.acquired_on <= disposal.disposed_on
    ]
    if len(eligible) != 1:
        return False
    lot = eligible[0]
    if disposal.quantity > remaining[lot.key]:
        return False
    remaining[lot.key] -= disposal.quantity
    return True


def resolve_lot_consumption(payloads) -> LotConsumption:
    """Conservatively net disposals from explicit CAS source facts.

    Allocation is scoped by instrument and source reference.  An exact
    transaction reference identifying one active acquisition wins; otherwise
    the bucket must contain exactly one active acquisition.  Missing,
    conflicting, future, or over-disposal facts mark the instrument unresolved
    instead of inferring FIFO or another allocation policy.
    """
    acquisitions: dict[str, dict[str, list[_AcquisitionFact]]] = defaultdict(
        lambda: defaultdict(list)
    )
    disposals: dict[str, list[_DisposalFact]] = defaultdict(list)
    ambiguous_acquisitions: dict[str, dict[str, set[str | None]]] = defaultdict(
        lambda: defaultdict(set)
    )
    occurrences: dict[tuple[str, str, datetime.date, str | None], int] = defaultdict(
        int
    )
    order = 0
    for payload in payloads or ():
        if not isinstance(payload, dict):
            continue
        for raw in payload.get("transactions") or []:
            if not isinstance(raw, dict):
                continue
            isin = _txn_isin(raw)
            if isin is None:
                continue
            order += 1
            if _txn_is_disposal(raw):
                units = _to_decimal(raw.get("units"))
                date_raw = raw.get("date")
                disposals[isin].append(
                    _DisposalFact(
                        source_ref=_normalized_ref(raw.get("source_ref")),
                        quantity=abs(units) if units is not None else None,
                        disposed_on=(
                            parse_date(str(date_raw)) if date_raw is not None else None
                        ),
                        reference=_normalized_reference(raw.get("reference")),
                        order=order,
                    )
                )
                continue
            if not _txn_is_acquisition(raw):
                continue
            lot, _exclusion = _classify_transaction(raw)
            if lot is None:
                ref = _normalized_ref(raw.get("source_ref"))
                if ref:
                    ambiguous_acquisitions[isin][ref].add(
                        _normalized_reference(raw.get("reference"))
                    )
                continue
            natural = (
                lot.instrument_id,
                _normalized_ref(lot.source_ref),
                lot.acquired_on,
                _normalized_reference(lot.reference),
            )
            occurrence = occurrences[natural]
            occurrences[natural] += 1
            key = LotKey(*natural, occurrence)
            acquisitions[isin][key.source_ref].append(
                _AcquisitionFact(key=key, quantity=lot.quantity)
            )

    unresolved: set[str] = set()
    original: dict[LotKey, Decimal] = {
        lot.key: lot.quantity
        for by_ref in acquisitions.values()
        for lots in by_ref.values()
        for lot in lots
    }
    resolved_remaining = dict(original)

    for isin, instrument_disposals in disposals.items():
        by_ref = acquisitions.get(isin, {})
        ordered = sorted(
            instrument_disposals,
            key=lambda disposal: (
                disposal.disposed_on is None,
                disposal.disposed_on or datetime.date.max,
                disposal.order,
            ),
        )
        for disposal in ordered:
            if not disposal.source_ref or disposal.quantity is None:
                unresolved.add(isin)
                break
            candidates = by_ref.get(disposal.source_ref, [])
            ambiguous_references = ambiguous_acquisitions.get(isin, {}).get(
                disposal.source_ref, set()
            )
            if ambiguous_references:
                exact_complete = [
                    lot
                    for lot in candidates
                    if resolved_remaining[lot.key] > 0
                    and disposal.reference
                    and lot.key.reference == disposal.reference
                ]
                if (
                    disposal.reference is None
                    or disposal.reference in ambiguous_references
                    or len(exact_complete) != 1
                ):
                    unresolved.add(isin)
                    break
            if not _allocate_exact(disposal, candidates, resolved_remaining):
                unresolved.add(isin)
                break

    remaining = {
        key: quantity
        for key, quantity in resolved_remaining.items()
        if key.instrument_id not in unresolved and quantity != original[key]
    }
    return LotConsumption(
        unresolved_instruments=unresolved,
        remaining=remaining,
    )


def unresolved_disposal_instruments(payloads) -> set[str]:
    """Return instrument ids with disposal history that needs guessing."""
    return resolve_lot_consumption(payloads).unresolved_instruments
