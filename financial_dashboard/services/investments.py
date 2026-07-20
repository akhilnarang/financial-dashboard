"""Investment persistence/read service and compatibility facade.

Pure CAS transaction handling lives in :mod:`investment_transactions`; its
public values and functions are re-exported here so existing callers keep the
original service API. CAS payloads are modeled **without fabrication.** A lot is
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
from financial_dashboard.services.investment_transactions import (
    CAS_CURRENCY as CAS_CURRENCY,
    _classify_transaction,
    _normalized_ref,
    _normalized_reference,
    _to_decimal,
    _txn_isin,
    extract_lots_from_payload as extract_lots_from_payload,
    resolve_lot_consumption as resolve_lot_consumption,
    unresolved_disposal_instruments as unresolved_disposal_instruments,
)
from financial_dashboard.services.investment_types import (
    CompleteLot as CompleteLot,
    CreateInvestmentLotsResult as CreateInvestmentLotsResult,
    LotClassificationResult as LotClassificationResult,
    LotConsumption as LotConsumption,
    LotExclusion as LotExclusion,
    LotExtractionResult as LotExtractionResult,
    LotKey as LotKey,
)

logger = logging.getLogger(__name__)


def _decode_cas_payload(raw_payload: str | None) -> dict | None:
    """Decode one preserved CAS payload, accepting JSON objects only."""
    if raw_payload is None:
        return None
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError, TypeError:
        return None
    return payload if isinstance(payload, dict) else None


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


class CanonicalLotConsumption(NamedTuple):
    """Disposal state keyed to :class:`CanonicalLotKey` for projection.

    ``unresolved`` is scoped by ``(portfolio_key, instrument_id)`` so one PAN's
    ambiguous redemption cannot suppress another portfolio's same ISIN.
    ``remaining`` has the same sparse semantics as :class:`LotConsumption`.
    """

    unresolved: set[tuple[str, str]]
    remaining: dict[CanonicalLotKey, Decimal]


async def create_investment_lots(
    session: AsyncSession, *, cas_upload_id: int, payload: dict
) -> CreateInvestmentLotsResult:
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
    return CreateInvestmentLotsResult(created=created, exclusions=exclusions)


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
            payload = _decode_cas_payload(upload.raw_holdings_json)
            if payload is None:
                raise ValueError("CAS payload must be a JSON object")
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
        payload = _decode_cas_payload(upload.raw_holdings_json)
        if payload is None:
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
        payload = _decode_cas_payload(upload.raw_holdings_json)
        if payload is not None:
            payloads.append(payload)
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
        payload = _decode_cas_payload(upload.raw_holdings_json)
        if payload is None:
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
        payload = _decode_cas_payload(upload.raw_holdings_json)
        if payload is None:
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
    return _decode_cas_payload(upload.raw_holdings_json)


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
