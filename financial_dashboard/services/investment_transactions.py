"""Pure CAS investment transaction classification.

This module deliberately knows nothing about SQLAlchemy.  It converts explicit
CAS transaction facts into complete acquisition lots or stable exclusions.
"""

import datetime
from collections import defaultdict
from decimal import Decimal, InvalidOperation

from financial_dashboard.core.dates import parse_date
from financial_dashboard.services.investment_types import (
    CompleteLot,
    LotClassificationResult,
    LotExclusion,
    LotExtractionResult,
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
