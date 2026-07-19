"""Investment service: source-faithful lot classification and read queries.

CAS payloads are modeled here **without fabrication.** A capital-gains lot is
only ever built from an explicit acquisition fact — an instrument id, a
quantity, a per-unit cost, a cost basis, a currency and an acquisition date
that are all present in the source and mutually consistent. Anything less is
reported with a stable reason so an operator can see what was and was not
projectable; it is never turned into a fake lot.

Source-data reality (``cas_parser`` schema):

* Demat holdings/securities carry ``isin``/``quantity``/``price``/``value`` but
  **no cost basis and no acquisition date** — CAS does not print them. These
  are value-only positions, kept as reconciliation/allocation data, never lots.
* Mutual-fund schemes carry an aggregate ``cost`` but **no acquisition date** —
  not a lot either.
* Mutual-fund *purchase* transactions carry ``units`` + ``nav`` + ``amount`` +
  ``date`` + ``isin`` — the one CAS fact set that fully determines a lot
  (quantity, unit cost, cost basis, currency, acquisition date). Those, and
  only those, become :class:`InvestmentLot` rows.

Acquisition dates and cost basis are never derived from a current market value.
"""

import datetime
import json
import logging
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import parse_date
from financial_dashboard.db.models import CasUpload, InvestmentLot

logger = logging.getLogger(__name__)

#: CAS is an Indian depository statement; every amount it prints is INR. That
#: is a fact about the document, not a fabricated currency.
CAS_CURRENCY = "INR"

#: Ingestion agreement tolerance between the reported ``amount`` and the
#: full-precision ``units * nav``. CAS prints units/nav at limited precision, so
#: the printed amount is a rounding of the true product and a sub-penny
#: disagreement is unavoidable display noise. A discrepancy of a full paisa or
#: more means the three printed numbers are mutually inconsistent beyond
#: rounding and the lot is excluded. The bound is **exclusive**: a difference of
#: exactly ``0.01`` is rejected (it cannot arise from a single 2dp rounding),
#: which closes the former 1-paisa grey zone that emitted lots whose stored
#: cost basis disagreed with their ``units * nav``.
_LOT_AGREEMENT_TOLERANCE = Decimal("0.01")

#: MF transaction_type substrings (lowercased) that denote an acquisition. The
#: CAS parser emits free-form type strings, so these are matched as substrings;
#: an unknown/blank type is treated as ambiguous and excluded conservatively
#: rather than guessed.
_ACQUISITION_TYPES = ("purchase", "switch_in", "switch-in", "buy", "allotment")

#: transaction_type substrings that denote a disposal — never a lot.
_DISPOSAL_TYPES = ("redemption", "switch_out", "switch-out", "sell", "sold")


class CompleteLot(NamedTuple):
    """Constructor-ready fields for a complete :class:`InvestmentLot`.

    Every field is an explicit, validated source fact. Returned by
    :func:`extract_lots_from_payload` and turned into ORM rows by
    :func:`create_investment_lots`.
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
    """Why a CAS transaction did not become a lot, with a stable ``reason``."""

    reason: str
    detail: str
    instrument_id: str | None
    source_ref: str | None


class CanonicalLotKey(NamedTuple):
    """Stable cross-upload identity for one acquisition occurrence.

    The source facts form a multiset: ``occurrence`` distinguishes genuinely
    repeated identical acquisitions, while the same occurrence repeated in an
    overlapping monthly CAS is represented only once.
    """

    portfolio_key: str
    source_ref: str
    instrument_id: str
    acquired_on: datetime.date
    reference: str | None
    quantity: Decimal
    unit_cost: Decimal
    cost_basis: Decimal
    currency: str
    transaction_type: str | None
    occurrence: int


class CasLotProvenance(NamedTuple):
    """One CAS upload that reported a canonical acquisition occurrence."""

    cas_upload_id: int
    statement_date: datetime.date
    depository_source: str


class CanonicalLot(NamedTuple):
    """One projection-ready acquisition occurrence with complete provenance.

    ``key`` carries only stable source/cost facts. ``canonical_*`` identifies
    the deterministic representative row (the earliest statement/upload/row),
    while ``provenance`` retains every overlapping upload that reported this
    occurrence. No current NAV/value is mixed into this acquisition-cost DTO.
    """

    key: CanonicalLotKey
    instrument_name: str
    canonical_lot_id: int
    canonical_cas_upload_id: int
    provenance: tuple[CasLotProvenance, ...]


class CurrentValuation(NamedTuple):
    """An explicit current holding valuation, separate from acquisition cost.

    Identity is ``(portfolio_key, scope, source_ref, instrument_id,
    occurrence)``. Keeping that identity prevents the same ISIN in two folios
    or demat accounts from overwriting another. Numeric fields remain ``None``
    when the source omitted them; value/price/quantity are never derived from
    each other.
    """

    portfolio_key: str
    cas_upload_id: int
    statement_date: datetime.date
    depository_source: str
    scope: str
    source_ref: str
    occurrence: int
    instrument_id: str
    instrument_name: str
    asset_class: str
    quantity: Decimal | None
    unit_price: Decimal | None
    value: Decimal | None
    currency: str


class LotBackfillResult(NamedTuple):
    """Sanitized counts from an investment-lot upgrade backfill."""

    uploads_scanned: int
    lots_created: int
    malformed_upload_ids: tuple[int, ...]


class _CanonicalCandidate(NamedTuple):
    lot: InvestmentLot
    upload: CasUpload


class _CanonicalBase(NamedTuple):
    portfolio_key: str
    source_ref: str
    instrument_id: str
    acquired_on: datetime.date
    reference: str | None
    quantity: Decimal
    unit_cost: Decimal
    cost_basis: Decimal
    currency: str
    transaction_type: str | None


class _PayloadSource(NamedTuple):
    upload: CasUpload
    payload: dict


class _RawTransactionCandidate(NamedTuple):
    upload: CasUpload
    raw: dict
    source_order: int


class LotKey(NamedTuple):
    """Stable source identity of one persisted acquisition lot.

    ``occurrence`` preserves multiplicity for the same source transaction
    identity. This compatibility key omits portfolio/cost facts; new projection
    code should use :class:`CanonicalLotKey`.
    """

    instrument_id: str
    source_ref: str
    acquired_on: datetime.date
    reference: str | None
    occurrence: int


class LotConsumption(NamedTuple):
    """Read-only remaining-lot model derived from preserved CAS facts.

    ``remaining`` contains only acquisition lots touched by deterministic
    disposal allocation, keyed by :class:`LotKey`. An absent key is untouched,
    zero means full consumption, and a positive value is the exact remaining
    quantity. The projection preserves the acquisition's explicit unit cost and
    computes its reduced reporting cost basis as ``remaining * unit_cost`` at
    money precision. Persisted :class:`InvestmentLot` rows are never changed.

    ``unresolved_instruments`` contains instruments for which at least one
    disposal cannot be allocated without guessing. Projection suppresses every
    lot for such an instrument, regardless of values in ``remaining``.
    """

    unresolved_instruments: set[str]
    remaining: dict[LotKey, Decimal]


class CanonicalLotConsumption(NamedTuple):
    """Disposal state keyed to :class:`CanonicalLotKey` for projection.

    ``unresolved`` is scoped by ``(portfolio_key, instrument_id)`` so one PAN's
    ambiguous redemption cannot suppress another portfolio's same ISIN.
    ``remaining`` has the same sparse semantics as :class:`LotConsumption`.
    """

    unresolved: set[tuple[str, str]]
    remaining: dict[CanonicalLotKey, Decimal]


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

    Strings, ints, floats and existing Decimals are all accepted (the payload
    arrives JSON-decoded); anything non-finite or unparseable is ``None`` so a
    malformed row is excluded rather than crashing the whole ingest.
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


def _classify_transaction(raw: dict) -> tuple[CompleteLot | None, LotExclusion | None]:
    """Classify one CAS transaction dict into a complete lot or an exclusion.

    Returns ``(lot, None)`` for a complete acquisition, ``(None, exclusion)``
    otherwise. Pure: no DB, no side effects, so the same function backs both
    ingest-time lot creation and query-time diagnostics.
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

    # MF transactions carry units/nav/amount; demat rows carry only quantity.
    if scope != "mf":
        return None, LotExclusion(
            reason="not_mutual_fund",
            detail=f"scope {scope!r}: only MF purchase transactions carry cost in CAS",
            instrument_id=isin_text,
            source_ref=source_ref_text,
        )

    if ttype and any(marker in ttype for marker in _DISPOSAL_TYPES):
        return None, LotExclusion(
            reason="disposal_transaction",
            detail=f"transaction_type {ttype!r} is a disposal, not an acquisition",
            instrument_id=isin_text,
            source_ref=source_ref_text,
        )
    if not ttype or not any(marker in ttype for marker in _ACQUISITION_TYPES):
        return None, LotExclusion(
            reason="ambiguous_transaction_type",
            detail=(
                f"transaction_type {ttype!r} does not identify an acquisition; "
                f"refusing to treat the date as an acquisition date"
            ),
            instrument_id=isin_text,
            source_ref=source_ref_text,
        )

    units = _to_decimal(raw.get("units"))
    nav = _to_decimal(raw.get("nav"))
    amount = _to_decimal(raw.get("amount"))
    date_raw = raw.get("date")
    acquired_on = parse_date(str(date_raw)) if date_raw is not None else None

    # Every lot field must be explicit and positive; a missing/zero/negative
    # value is reported, never filled in from another field or from value.
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
        return None, LotExclusion(
            reason="missing_lot_facts",
            detail="missing/invalid: " + ", ".join(missing),
            instrument_id=isin_text,
            source_ref=source_ref_text,
        )

    # Agreement + exact balance. CAS prints units/nav at limited precision, so
    # the printed ``amount`` is a rounding of the true ``units * nav``; a
    # sub-penny disagreement is unavoidable display noise, but a discrepancy of
    # a full paisa or more means the printed numbers are mutually inconsistent
    # beyond rounding and the lot is excluded with a stable reason. The bound is
    # exclusive: a difference of exactly one paisa is rejected — it cannot arise
    # from a single 2dp rounding — which removes the former 1-paisa grey zone.
    assert (
        units is not None and nav is not None and amount is not None
    )  # narrowed above
    product = units * nav  # full precision — what every backend recomputes
    discrepancy = abs(product - amount)
    if discrepancy >= _LOT_AGREEMENT_TOLERANCE:
        return None, LotExclusion(
            reason="cost_basis_inconsistent",
            detail=(
                f"amount {amount} differs from units*nav {product} by "
                f"{discrepancy}; CAS values disagree beyond rounding"
            ),
            instrument_id=isin_text,
            source_ref=source_ref_text,
        )
    # Store the quantized product — the parser's canonical lot value — NOT the
    # separately-printed amount. The renderer values the asset leg as
    # ``quantity * unit_cost`` and balance-checks at 2 dp, so a ``cost_basis``
    # equal to ``(quantity * unit_cost).quantize(0.01)`` makes the lot exactly
    # consistent (the renderer's guard passes with a zero diff, not up to a
    # paisa) and the emitted entry balances at the renderer's money precision
    # for every accepted lot. This is what closes the former 1-paisa grey zone.
    cost_basis = product.quantize(_LOT_AGREEMENT_TOLERANCE)

    description = raw.get("description")
    instrument_name = (
        str(description).strip()
        if isinstance(description, str) and description.strip()
        else isin_text
    )
    # The ``missing`` guard above guarantees these are non-None here; the asserts
    # narrow them for the type checker (matches projection.py's convention).
    assert isin_text is not None
    assert acquired_on is not None
    assert instrument_name  # isin_text is non-None here

    lot = CompleteLot(
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
    )
    return lot, None


def extract_lots_from_payload(
    payload: dict,
) -> tuple[list[CompleteLot], list[LotExclusion]]:
    """Split a CAS payload's transactions into complete lots and exclusions.

    Transactions are the only CAS section that can carry an acquisition date;
    holdings/folios (value-only) are not scanned for lots — they remain
    reconciliation data. Repeated complete rows are retained as source
    multiplicity and assigned a zero-based ``source_occurrence`` within their
    source transaction identity. Retry idempotency is enforced when persisting,
    not by collapsing source rows here.
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
    return lots, exclusions


# ---------------------------------------------------------------------------
# Disposal-history resolution (redemption safety)
# ---------------------------------------------------------------------------


def _txn_isin(raw: dict) -> str | None:
    isin = raw.get("isin")
    if isinstance(isin, str) and isin.strip():
        return isin.strip().upper()
    return None


def _txn_is_disposal(raw: dict) -> bool:
    """A CAS transaction that disposes units (redemption/switch_out/sell)."""
    ttype_raw = raw.get("transaction_type")
    ttype = (
        str(ttype_raw).strip().lower()
        if isinstance(ttype_raw, str) and ttype_raw.strip()
        else ""
    )
    return bool(ttype) and any(marker in ttype for marker in _DISPOSAL_TYPES)


def _txn_is_acquisition(raw: dict) -> bool:
    """A CAS transaction that acquires units (purchase/switch_in/buy)."""
    ttype_raw = raw.get("transaction_type")
    ttype = (
        str(ttype_raw).strip().lower()
        if isinstance(ttype_raw, str) and ttype_raw.strip()
        else ""
    )
    return bool(ttype) and any(marker in ttype for marker in _ACQUISITION_TYPES)


def unresolved_disposal_instruments(payloads) -> set[str]:
    """Instrument ids whose preserved CAS facts contain a disposal that cannot
    be truthfully allocated to specific acquisition lots.

    Thin wrapper over :func:`resolve_lot_consumption` returning just the
    unresolved set. Kept for backward compatibility with existing call sites
    and tests; new code should call :func:`resolve_lot_consumption` directly
    to also inspect the consumption map.
    """
    return resolve_lot_consumption(payloads).unresolved_instruments


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
    """Consume ``disposal`` only when its source facts identify one lot.

    Candidates are already scoped by exact instrument and ``source_ref``. An
    explicit transaction reference that matches exactly one active acquisition
    identifies that lot. Otherwise the bucket must contain exactly one active
    acquisition, which is the only deterministic identity left. Full and
    partial consumption of that one lot are both supported.

    Multiple possible acquisitions are unresolved. Acquisition date order is
    deliberately *not* an allocation fact: CAS does not say that a redemption
    consumed FIFO, so neither an exact aggregate quantity nor distinct dates
    authorize spreading a disposal across lots.
    """
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

    # A dated disposal cannot consume a future acquisition. A missing date does
    # not fabricate one; identity still comes solely from the explicit bucket.
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
    """Conservative lot consumption/netting from explicit CAS facts.

    Allocation is scoped first by ``(instrument_id, source_ref)``; a shared ref
    never crosses instruments. Within that explicit bucket, one transaction
    reference identifying one acquisition wins. Otherwise exactly one active
    acquisition is a deterministic identity. Multiple possible acquisitions
    are never allocated by inferred FIFO, acquisition date, or aggregate cost.
    A missing ref/units, an incomplete/conflicting acquisition in the same
    bucket, an unmatched disposal, future-lot consumption, or over-disposal
    marks the whole affected instrument unresolved and suppresses it
    conservatively.

    Full consumption records zero and partial consumption records the exact
    source quantity left; untouched lots are absent so projection continues to
    use their normalized persisted quantity. Projection retains the source
    acquisition date/unit cost and recomputes the proportional cost basis.
    Identical magnitudes are never used as a de-duplication key: separate source
    rows retain multiplicity; overlap canonicalization belongs to
    :func:`get_canonical_lot_consumption`, not this compatibility pure helper.

    Pure over its argument (an iterable of CAS payload dicts) so the same
    rule backs ingest-time diagnostics and projection-time consumption.
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
                    # A disposal sharing this bucket could have consumed the
                    # incomplete acquisition. Allocating it only among complete
                    # persisted lots would invent certainty/cost basis.
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
        # Date then source order makes repeated exact-lot consumption stable
        # while retaining multiplicity. This ordering never allocates one
        # disposal across acquisition lots.
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


async def create_investment_lots(
    session: AsyncSession, *, cas_upload_id: int, payload: dict
) -> tuple[int, list[LotExclusion]]:
    """Persist complete lots for one CAS upload; return ``(created, excluded)``.

    Idempotent within an upload: a retry compares every normalized source fact
    plus ``source_occurrence`` and inserts only absent rows. Re-ingestion is
    also handled by the caller (``ingest_cas_payload``) deleting the prior
    upload — and its lots — before creating a new one.
    """
    lots, exclusions = extract_lots_from_payload(payload)
    existing_rows = (
        (
            await session.execute(
                select(InvestmentLot).where(
                    InvestmentLot.cas_upload_id == cas_upload_id
                )
            )
        )
        .scalars()
        .all()
    )
    existing = {_persisted_lot_key(row) for row in existing_rows}
    created = 0
    for lot in lots:
        key = _complete_lot_key(lot)
        if key in existing:
            continue
        session.add(
            InvestmentLot(
                cas_upload_id=cas_upload_id,
                instrument_id=lot.instrument_id,
                instrument_name=lot.instrument_name,
                quantity=lot.quantity,
                unit_cost=lot.unit_cost,
                cost_basis=lot.cost_basis,
                currency=lot.currency,
                acquired_on=lot.acquired_on,
                source_ref=lot.source_ref,
                transaction_type=lot.transaction_type,
                reference=lot.reference,
                source_occurrence=lot.source_occurrence,
            )
        )
        existing.add(key)
        created += 1
    if created:
        await session.flush()
    return created, exclusions


def _complete_lot_key(lot: CompleteLot) -> tuple:
    return (
        lot.instrument_id,
        lot.instrument_name,
        lot.quantity,
        lot.unit_cost,
        lot.cost_basis,
        lot.currency,
        lot.acquired_on,
        lot.source_ref,
        lot.transaction_type,
        lot.reference,
        lot.source_occurrence,
    )


def _persisted_lot_key(lot: InvestmentLot) -> tuple:
    return (
        lot.instrument_id,
        lot.instrument_name,
        Decimal(lot.quantity),
        Decimal(lot.unit_cost),
        Decimal(lot.cost_basis),
        lot.currency,
        lot.acquired_on,
        lot.source_ref,
        lot.transaction_type,
        lot.reference,
        lot.source_occurrence,
    )


async def backfill_investment_lots(session: AsyncSession) -> LotBackfillResult:
    """Normalize complete lots from every preserved pre-upgrade CAS payload.

    The normal no-fabrication extractor/persister is used for every upload.
    Malformed JSON/top-level payloads are isolated and logged by upload id only;
    one bad historical row cannot prevent valid uploads from being backfilled.
    The persister is fact-and-occurrence idempotent, so existing lots and a
    direct rerun are never duplicated.
    """
    uploads = (
        (await session.execute(select(CasUpload).order_by(CasUpload.id)))
        .scalars()
        .all()
    )
    created = 0
    malformed: list[int] = []
    for upload in uploads:
        try:
            payload = json.loads(upload.raw_holdings_json)
            if not isinstance(payload, dict):
                raise TypeError("CAS payload must be a JSON object")
            upload_created, _ = await create_investment_lots(
                session,
                cas_upload_id=upload.id,
                payload=payload,
            )
        except json.JSONDecodeError, TypeError, ValueError, InvalidOperation:
            malformed.append(upload.id)
            logger.warning(
                "Skipping malformed CAS payload during investment-lot backfill "
                "(cas_upload_id=%s)",
                upload.id,
            )
            continue
        created += upload_created
    return LotBackfillResult(
        uploads_scanned=len(uploads),
        lots_created=created,
        malformed_upload_ids=tuple(malformed),
    )


# ---------------------------------------------------------------------------
# Read queries
# ---------------------------------------------------------------------------


class LotPosition(NamedTuple):
    """A net instrument position: current quantity/value plus known cost basis.

    ``quantity``/``unit_price``/``value`` come from the latest CAS holding that
    explicitly priced the instrument (a point-in-time allocation fact); the
    ``cost_*`` fields aggregate the complete lots projected for it, and are
    ``None``/zero when no complete lot exists (a value-only holding)."""

    instrument_id: str
    instrument_name: str
    asset_class: str
    quantity: Decimal | None
    unit_price: Decimal | None
    value: Decimal | None
    currency: str
    lot_quantity: Decimal
    lot_cost_basis: Decimal


def _canonical_base(lot: InvestmentLot, upload: CasUpload) -> _CanonicalBase:
    reference = (
        lot.reference.strip() if lot.reference and lot.reference.strip() else None
    )
    transaction_type = (
        lot.transaction_type.strip().lower()
        if lot.transaction_type and lot.transaction_type.strip()
        else None
    )
    return _CanonicalBase(
        upload.portfolio_key.strip().upper(),
        lot.source_ref.strip(),
        lot.instrument_id.strip().upper(),
        lot.acquired_on,
        reference,
        Decimal(lot.quantity),
        Decimal(lot.unit_cost),
        Decimal(lot.cost_basis),
        lot.currency.strip().upper(),
        transaction_type,
    )


async def get_canonical_lots(session: AsyncSession) -> list[CanonicalLot]:
    """Return the canonical acquisition multiset across overlapping CAS uploads.

    Grouping uses portfolio + stable source transaction and complete cost facts.
    Within each upload equal facts are an ordered multiset; cross-upload
    canonical multiplicity is the maximum occurrence count, not the sum. Thus a
    monthly overlap is removed while two genuine identical source occurrences
    remain two. The earliest ``(statement_date, upload_id, lot_id)`` candidate
    is the deterministic representative, and every contributing upload remains
    in ``CanonicalLot.provenance``.
    """
    rows = (
        await session.execute(
            select(InvestmentLot, CasUpload)
            .join(CasUpload, CasUpload.id == InvestmentLot.cas_upload_id)
            .order_by(
                CasUpload.statement_date,
                CasUpload.id,
                InvestmentLot.source_occurrence,
                InvestmentLot.id,
            )
        )
    ).all()
    grouped: dict[_CanonicalBase, dict[int, list[_CanonicalCandidate]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for lot, upload in rows:
        grouped[_canonical_base(lot, upload)][upload.id].append(
            _CanonicalCandidate(lot=lot, upload=upload)
        )

    canonical: list[CanonicalLot] = []
    for base, by_upload in grouped.items():
        ordered_by_upload = {
            upload_id: sorted(
                candidates,
                key=lambda candidate: (
                    candidate.lot.source_occurrence,
                    candidate.lot.id,
                ),
            )
            for upload_id, candidates in by_upload.items()
        }
        multiplicity = max(len(candidates) for candidates in ordered_by_upload.values())
        for occurrence in range(multiplicity):
            candidates = [
                upload_candidates[occurrence]
                for upload_candidates in ordered_by_upload.values()
                if len(upload_candidates) > occurrence
            ]
            representative = min(
                candidates,
                key=lambda candidate: (
                    candidate.upload.statement_date,
                    candidate.upload.id,
                    candidate.lot.id,
                ),
            )
            provenance = tuple(
                CasLotProvenance(
                    cas_upload_id=candidate.upload.id,
                    statement_date=candidate.upload.statement_date,
                    depository_source=candidate.upload.depository_source,
                )
                for candidate in sorted(
                    candidates,
                    key=lambda candidate: (
                        candidate.upload.statement_date,
                        candidate.upload.id,
                    ),
                )
            )
            canonical.append(
                CanonicalLot(
                    key=CanonicalLotKey(
                        portfolio_key=base.portfolio_key,
                        source_ref=base.source_ref,
                        instrument_id=base.instrument_id,
                        acquired_on=base.acquired_on,
                        reference=base.reference,
                        quantity=base.quantity,
                        unit_cost=base.unit_cost,
                        cost_basis=base.cost_basis,
                        currency=base.currency,
                        transaction_type=base.transaction_type,
                        occurrence=occurrence,
                    ),
                    instrument_name=representative.lot.instrument_name,
                    canonical_lot_id=representative.lot.id,
                    canonical_cas_upload_id=representative.upload.id,
                    provenance=provenance,
                )
            )
    canonical.sort(
        key=lambda lot: (
            lot.key.acquired_on,
            lot.key.portfolio_key,
            lot.key.instrument_id,
            lot.key.source_ref,
            lot.key.reference or "",
            lot.key.quantity,
            lot.key.unit_cost,
            lot.key.occurrence,
        )
    )
    return canonical


async def get_complete_lots(session: AsyncSession) -> list[InvestmentLot]:
    """Every raw persisted complete lot, including overlapping provenance.

    Projection code should use :func:`get_canonical_lots`; this accessor remains
    for audit/tests that need the normalized source rows themselves.
    """
    rows = (
        (
            await session.execute(
                select(InvestmentLot).order_by(
                    InvestmentLot.acquired_on, InvestmentLot.id
                )
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


async def get_incomplete_reasons(session: AsyncSession) -> list[LotExclusion]:
    """Re-derive the exclusion reasons from each upload's preserved raw payload.

    The persisted :class:`InvestmentLot` table stores only complete lots; the
    exclusions are recomputed from ``raw_holdings_json`` so the diagnostics
    always reflect the current classification logic without a separate store.
    """
    out: list[LotExclusion] = []
    uploads = (
        (await session.execute(select(CasUpload).order_by(CasUpload.id)))
        .scalars()
        .all()
    )
    for upload in uploads:
        try:
            payload = json.loads(upload.raw_holdings_json)
        except json.JSONDecodeError, TypeError:
            continue
        _, exclusions = extract_lots_from_payload(payload)
        out.extend(exclusions)
    return out


async def get_lot_consumption(session: AsyncSession) -> LotConsumption:
    """Lot consumption read from preserved CAS payloads.

    See :func:`resolve_lot_consumption` for the rule. Read-only: it only
    SELECTs ``CasUpload`` rows and never writes a core lot — the projection
    applies the returned consumption map without persisting it.
    """
    payloads: list[dict] = []
    uploads = (
        (await session.execute(select(CasUpload).order_by(CasUpload.id)))
        .scalars()
        .all()
    )
    for upload in uploads:
        try:
            payloads.append(json.loads(upload.raw_holdings_json))
        except json.JSONDecodeError, TypeError:
            continue
    return resolve_lot_consumption(payloads)


def _fact_token(value) -> str:
    if value is None or value == "":
        return ""
    decimal = _to_decimal(value)
    if decimal is not None:
        return str(decimal.normalize())
    return str(value).strip()


def _transaction_fingerprint(raw: dict) -> tuple[str, ...]:
    """Stable source facts used for overlap multiset canonicalization."""
    date_raw = raw.get("date")
    parsed_date = parse_date(str(date_raw)) if date_raw is not None else None
    return (
        str(raw.get("scope") or "").strip().lower(),
        _normalized_ref(raw.get("source_ref")),
        _txn_isin(raw) or "",
        parsed_date.isoformat() if parsed_date is not None else str(date_raw or ""),
        str(raw.get("transaction_type") or "").strip().lower(),
        _normalized_reference(raw.get("reference")) or "",
        _fact_token(raw.get("units")),
        _fact_token(raw.get("nav")),
        _fact_token(raw.get("amount")),
    )


def _canonical_raw_transactions(sources: list[_PayloadSource]) -> list[dict]:
    """Canonical transaction multiset for overlapping uploads of one portfolio."""
    grouped: dict[tuple[str, ...], dict[int, list[_RawTransactionCandidate]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for source in sources:
        for source_order, raw in enumerate(source.payload.get("transactions") or []):
            if not isinstance(raw, dict):
                continue
            grouped[_transaction_fingerprint(raw)][source.upload.id].append(
                _RawTransactionCandidate(
                    upload=source.upload,
                    raw=raw,
                    source_order=source_order,
                )
            )

    canonical: list[tuple[tuple[str, ...], int, _RawTransactionCandidate]] = []
    for fingerprint, by_upload in grouped.items():
        ordered_by_upload = {
            upload_id: sorted(
                candidates,
                key=lambda candidate: candidate.source_order,
            )
            for upload_id, candidates in by_upload.items()
        }
        multiplicity = max(len(candidates) for candidates in ordered_by_upload.values())
        for occurrence in range(multiplicity):
            candidates = [
                upload_candidates[occurrence]
                for upload_candidates in ordered_by_upload.values()
                if len(upload_candidates) > occurrence
            ]
            representative = min(
                candidates,
                key=lambda candidate: (
                    candidate.upload.statement_date,
                    candidate.upload.id,
                    candidate.source_order,
                ),
            )
            canonical.append((fingerprint, occurrence, representative))
    canonical.sort(key=lambda item: (item[0][3], item[0], item[1]))
    return [item[2].raw for item in canonical]


def _canonical_consumption_key_map(
    portfolio_key: str, transactions: list[dict]
) -> dict[LotKey, CanonicalLotKey]:
    old_occurrences: dict[tuple[str, str, datetime.date, str | None], int] = (
        defaultdict(int)
    )
    full_occurrences: dict[_CanonicalBase, int] = defaultdict(int)
    out: dict[LotKey, CanonicalLotKey] = {}
    for raw in transactions:
        lot, _ = _classify_transaction(raw)
        if lot is None:
            continue
        old_base = (
            lot.instrument_id,
            _normalized_ref(lot.source_ref),
            lot.acquired_on,
            _normalized_reference(lot.reference),
        )
        old_occurrence = old_occurrences[old_base]
        old_occurrences[old_base] += 1
        old_key = LotKey(*old_base, old_occurrence)
        full_base = _CanonicalBase(
            portfolio_key=portfolio_key.strip().upper(),
            source_ref=_normalized_ref(lot.source_ref),
            instrument_id=lot.instrument_id,
            acquired_on=lot.acquired_on,
            reference=_normalized_reference(lot.reference),
            quantity=lot.quantity,
            unit_cost=lot.unit_cost,
            cost_basis=lot.cost_basis,
            currency=lot.currency,
            transaction_type=lot.transaction_type,
        )
        occurrence = full_occurrences[full_base]
        full_occurrences[full_base] += 1
        out[old_key] = CanonicalLotKey(
            portfolio_key=full_base.portfolio_key,
            source_ref=full_base.source_ref,
            instrument_id=full_base.instrument_id,
            acquired_on=full_base.acquired_on,
            reference=full_base.reference,
            quantity=full_base.quantity,
            unit_cost=full_base.unit_cost,
            cost_basis=full_base.cost_basis,
            currency=full_base.currency,
            transaction_type=full_base.transaction_type,
            occurrence=occurrence,
        )
    return out


async def get_canonical_lot_consumption(
    session: AsyncSession,
) -> CanonicalLotConsumption:
    """Return source-faithful disposal state for canonical acquisition lots.

    Acquisition and disposal rows repeated by overlapping statements are first
    canonicalized as multisets per portfolio. The existing exact-only policy is
    then applied unchanged: no FIFO/average inference, and any ambiguous or
    unmatched disposal suppresses that portfolio/instrument. The result keys
    match :func:`get_canonical_lots` exactly.
    """
    uploads = (
        (await session.execute(select(CasUpload).order_by(CasUpload.id)))
        .scalars()
        .all()
    )
    by_portfolio: dict[str, list[_PayloadSource]] = defaultdict(list)
    for upload in uploads:
        try:
            payload = json.loads(upload.raw_holdings_json)
            if not isinstance(payload, dict):
                raise TypeError("CAS payload must be a JSON object")
        except json.JSONDecodeError, TypeError:
            logger.warning(
                "Skipping malformed CAS disposal history (cas_upload_id=%s)",
                upload.id,
            )
            continue
        by_portfolio[upload.portfolio_key.strip().upper()].append(
            _PayloadSource(upload=upload, payload=payload)
        )

    unresolved: set[tuple[str, str]] = set()
    remaining: dict[CanonicalLotKey, Decimal] = {}
    for portfolio_key, sources in by_portfolio.items():
        transactions = _canonical_raw_transactions(sources)
        consumption = resolve_lot_consumption([{"transactions": transactions}])
        normalized_portfolio = portfolio_key.strip().upper()
        unresolved.update(
            (normalized_portfolio, instrument_id)
            for instrument_id in consumption.unresolved_instruments
        )
        key_map = _canonical_consumption_key_map(portfolio_key, transactions)
        for old_key, quantity in consumption.remaining.items():
            canonical_key = key_map.get(old_key)
            if canonical_key is None:
                unresolved.add((normalized_portfolio, old_key.instrument_id))
                continue
            remaining[canonical_key] = quantity
    remaining = {
        key: quantity
        for key, quantity in remaining.items()
        if (key.portfolio_key, key.instrument_id) not in unresolved
    }
    return CanonicalLotConsumption(unresolved=unresolved, remaining=remaining)


async def get_unresolved_disposal_instruments(session: AsyncSession) -> set[str]:
    """Instrument ids with unresolvable disposal history, read from preserved
    CAS payloads. See :func:`unresolved_disposal_instruments` for the rule.

    Thin wrapper over :func:`get_lot_consumption` returning just the
    unresolved set. Kept for backward compatibility; new code should call
    :func:`get_lot_consumption` to also inspect the consumption map.
    """
    return (await get_lot_consumption(session)).unresolved_instruments


def _explicit_source_ref(raw: dict, *, scope: str, index: int) -> str:
    explicit = raw.get("source_ref")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    if scope == "folio":
        folio = raw.get("folio_number")
        if folio is not None and str(folio).strip():
            return str(folio).strip()
    else:
        parts = [
            str(raw.get(field)).strip()
            for field in ("depository", "dp_id", "client_id")
            if raw.get(field) is not None and str(raw.get(field)).strip()
        ]
        if parts:
            return ":".join(parts)
    # Preserve the source row instead of dropping/overwriting it. The index is
    # scoped to one immutable raw payload; no account identity is fabricated.
    return f"{scope}:{index}"


def _holding_valuations(upload: CasUpload, payload: dict) -> list[CurrentValuation]:
    """Identity-preserving explicit holding facts from one CAS upload."""
    valuations: list[CurrentValuation] = []
    occurrences: dict[tuple[str, str, str], int] = defaultdict(int)
    for account_index, account in enumerate(payload.get("accounts") or []):
        if not isinstance(account, dict):
            continue
        source_ref = _explicit_source_ref(account, scope="demat", index=account_index)
        for holding in account.get("holdings") or []:
            if not isinstance(holding, dict):
                continue
            isin = holding.get("isin")
            if not isinstance(isin, str) or not isin.strip():
                continue
            instrument_id = isin.strip().upper()
            identity = ("demat", source_ref, instrument_id)
            occurrence = occurrences[identity]
            occurrences[identity] += 1
            valuations.append(
                CurrentValuation(
                    portfolio_key=upload.portfolio_key.strip().upper(),
                    cas_upload_id=upload.id,
                    statement_date=upload.statement_date,
                    depository_source=upload.depository_source,
                    scope="demat",
                    source_ref=source_ref,
                    occurrence=occurrence,
                    instrument_id=instrument_id,
                    instrument_name=str(holding.get("name") or instrument_id),
                    asset_class=str(holding.get("asset_class") or "other"),
                    quantity=_to_decimal(holding.get("quantity")),
                    unit_price=_to_decimal(holding.get("price")),
                    value=_to_decimal(holding.get("value")),
                    currency=CAS_CURRENCY,
                )
            )
    for folio_index, folio in enumerate(payload.get("folios") or []):
        if not isinstance(folio, dict):
            continue
        source_ref = _explicit_source_ref(folio, scope="folio", index=folio_index)
        for scheme in folio.get("schemes") or []:
            if not isinstance(scheme, dict):
                continue
            isin = scheme.get("isin")
            if not isinstance(isin, str) or not isin.strip():
                continue
            instrument_id = isin.strip().upper()
            identity = ("folio", source_ref, instrument_id)
            occurrence = occurrences[identity]
            occurrences[identity] += 1
            valuations.append(
                CurrentValuation(
                    portfolio_key=upload.portfolio_key.strip().upper(),
                    cas_upload_id=upload.id,
                    statement_date=upload.statement_date,
                    depository_source=upload.depository_source,
                    scope="folio",
                    source_ref=source_ref,
                    occurrence=occurrence,
                    instrument_id=instrument_id,
                    instrument_name=str(scheme.get("scheme_name") or instrument_id),
                    asset_class="mutual_fund",
                    quantity=_to_decimal(scheme.get("units")),
                    unit_price=_to_decimal(scheme.get("nav")),
                    value=_to_decimal(scheme.get("value")),
                    currency=CAS_CURRENCY,
                )
            )
    return valuations


async def get_current_valuations(session: AsyncSession) -> list[CurrentValuation]:
    """Current explicit holding facts from the latest CAS per portfolio.

    This API is intentionally independent of :func:`get_canonical_lots`: it
    carries statement-date NAV/value/quantity facts, not acquisition cost. A
    malformed latest payload is logged and isolated to its portfolio; an older
    valuation is not silently relabelled as current.
    """
    uploads = (
        (
            await session.execute(
                select(CasUpload).order_by(
                    CasUpload.statement_date.desc(),
                    CasUpload.id.desc(),
                )
            )
        )
        .scalars()
        .all()
    )
    latest: dict[str, CasUpload] = {}
    for upload in uploads:
        latest.setdefault(upload.portfolio_key.strip().upper(), upload)

    valuations: list[CurrentValuation] = []
    for upload in latest.values():
        try:
            payload = json.loads(upload.raw_holdings_json)
            if not isinstance(payload, dict):
                raise TypeError("CAS payload must be a JSON object")
        except json.JSONDecodeError, TypeError:
            logger.warning(
                "Skipping malformed current CAS valuation (cas_upload_id=%s)",
                upload.id,
            )
            continue
        valuations.extend(_holding_valuations(upload, payload))
    valuations.sort(
        key=lambda valuation: (
            valuation.portfolio_key,
            valuation.scope,
            valuation.source_ref,
            valuation.instrument_id,
            valuation.occurrence,
        )
    )
    return valuations


async def get_latest_payload(session: AsyncSession) -> dict | None:
    """The most recent CAS upload's raw payload (by statement date), or None."""
    upload = (
        (
            await session.execute(
                select(CasUpload).order_by(
                    CasUpload.statement_date.desc(), CasUpload.id.desc()
                )
            )
        )
        .scalars()
        .first()
    )
    if upload is None:
        return None
    try:
        return json.loads(upload.raw_holdings_json)
    except json.JSONDecodeError, TypeError:
        return None


async def get_positions(session: AsyncSession) -> list[LotPosition]:
    """Compatibility aggregate by ISIN over the canonical split APIs.

    New projection code must consume :func:`get_current_valuations` and
    :func:`get_canonical_lots` separately so source identity and valuation-vs-
    cost semantics stay explicit. This legacy view sums current facts by ISIN
    without last-write-wins and joins canonical (not overlapping raw) costs.
    """
    valuations = await get_current_valuations(session)
    holdings: dict[str, list[CurrentValuation]] = defaultdict(list)
    for valuation in valuations:
        holdings[valuation.instrument_id].append(valuation)

    lots = await get_canonical_lots(session)
    lot_qty: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    lot_cost: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    lot_name: dict[str, str] = {}
    for lot in lots:
        lot_qty[lot.key.instrument_id] += lot.key.quantity
        lot_cost[lot.key.instrument_id] += lot.key.cost_basis
        lot_name.setdefault(lot.key.instrument_id, lot.instrument_name)

    instruments = sorted(set(holdings) | set(lot_qty))
    out: list[LotPosition] = []
    for isin in instruments:
        current = holdings.get(isin, [])
        quantities = [fact.quantity for fact in current if fact.quantity is not None]
        values = [fact.value for fact in current if fact.value is not None]
        prices = {fact.unit_price for fact in current if fact.unit_price is not None}
        asset_classes = {fact.asset_class for fact in current}
        out.append(
            LotPosition(
                instrument_id=isin,
                instrument_name=(
                    current[0].instrument_name
                    if current
                    else lot_name.get(isin) or isin
                ),
                asset_class=(
                    next(iter(asset_classes)) if len(asset_classes) == 1 else "other"
                ),
                quantity=(sum(quantities, Decimal("0")) if quantities else None),
                unit_price=(next(iter(prices)) if len(prices) == 1 else None),
                value=(sum(values, Decimal("0")) if values else None),
                currency=current[0].currency if current else CAS_CURRENCY,
                lot_quantity=lot_qty.get(isin, Decimal("0")),
                lot_cost_basis=lot_cost.get(isin, Decimal("0")),
            )
        )
    return out


async def get_latest_values(session: AsyncSession) -> dict[str, Decimal]:
    """Latest explicit market value per instrument (ISIN -> value) from CAS.

    Value-only holdings are included here — this is allocation/reconciliation
    data, not a lot. Instruments the CAS did not price are absent.
    """
    out: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for valuation in await get_current_valuations(session):
        if valuation.value is not None:
            out[valuation.instrument_id] += valuation.value
    return dict(out)
