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
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import parse_date
from financial_dashboard.db.models import CasUpload, InvestmentLot

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


class LotExclusion(NamedTuple):
    """Why a CAS transaction did not become a lot, with a stable ``reason``."""

    reason: str
    detail: str
    instrument_id: str | None
    source_ref: str | None


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
    )
    return lot, None


def extract_lots_from_payload(
    payload: dict,
) -> tuple[list[CompleteLot], list[LotExclusion]]:
    """Split a CAS payload's transactions into complete lots and exclusions.

    Transactions are the only CAS section that can carry an acquisition date;
    holdings/folios (value-only) are not scanned for lots — they remain
    reconciliation data. Deduplicated in memory by the natural key so a parser
    emitting the same movement twice cannot create duplicate lots (the DB
    UNIQUE constraint cannot dedupe NULL references on SQLite).
    """
    lots: list[CompleteLot] = []
    exclusions: list[LotExclusion] = []
    seen: set[tuple] = set()
    for raw in payload.get("transactions") or []:
        if not isinstance(raw, dict):
            continue
        lot, exclusion = _classify_transaction(raw)
        if lot is not None:
            key = (
                lot.source_ref,
                lot.instrument_id,
                lot.acquired_on.isoformat(),
                lot.reference or "",
            )
            if key in seen:
                continue
            seen.add(key)
            lots.append(lot)
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

    CAS prints mutual-fund redemptions/switch-outs only as aggregate units and
    amount against a scheme; it never identifies which acquisition lots a
    redemption settled. Projecting the gross acquisition lots would therefore
    overstate holdings (redeemed units would still appear held). To **never
    overstate**, every instrument with an *unresolvable* disposal is reported
    here so the projection can suppress all its lots conservatively.

    A disposal is resolvable ONLY when it is exactly tied to an acquisition by
    an explicit, shared ``source_ref`` whose units match in magnitude — a
    genuine linked switch leg. Aggregate/FIFO/average allocation is inference
    and is never performed here ("no inference"). Because CAS does not link a
    purchase to a later redemption, a free-standing redemption is always
    unresolvable in practice, so the default is conservative suppression by
    instrument.

    Pure over its argument (an iterable of CAS payload dicts) so the same rule
    backs ingest-time diagnostics and projection-time suppression.
    """
    # Per instrument: acquisitions keyed by source_ref → magnitude of units, and
    # the list of disposal (source_ref, magnitude) pairs to tie.
    acquisitions: dict[str, dict[str, Decimal]] = defaultdict(dict)
    disposals: dict[str, list[tuple[str, Decimal | None]]] = defaultdict(list)
    for payload in payloads or ():
        for raw in payload.get("transactions") or []:
            if not isinstance(raw, dict):
                continue
            isin = _txn_isin(raw)
            if isin is None:
                continue
            ref = str(raw.get("source_ref") or "").strip()
            units = _to_decimal(raw.get("units"))
            if _txn_is_disposal(raw):
                # A disposal with no usable ref/units can never be explicitly
                # tied to an acquisition; record it so it forces suppression.
                disposals[isin].append((ref, abs(units) if units is not None else None))
            elif _txn_is_acquisition(raw) and ref and units is not None:
                # Keep the first acquisition magnitude per ref; a duplicate ref
                # is a parser artefact, not a second tie-able acquisition.
                acquisitions[isin].setdefault(ref, abs(units))

    unresolved: set[str] = set()
    for isin, disp_list in disposals.items():
        acq_by_ref = acquisitions.get(isin, {})
        for ref, magnitude in disp_list:
            # Exactly tied only when the SAME source_ref carries both an
            # acquisition and a disposal of matching magnitude — an explicit
            # linked switch. A missing ref/units, or any mismatch, is
            # unresolvable without inference.
            if magnitude is None or not ref:
                unresolved.add(isin)
                break
            acq = acq_by_ref.get(ref)
            if acq is None or acq != magnitude:
                unresolved.add(isin)
                break
    return unresolved


async def create_investment_lots(
    session: AsyncSession, *, cas_upload_id: int, payload: dict
) -> tuple[int, list[LotExclusion]]:
    """Persist complete lots for one CAS upload; return ``(created, excluded)``.

    Idempotent within an upload: in-memory dedup keys on the natural identity.
    Re-ingestion is handled by the caller (``ingest_cas_payload``) deleting the
    prior upload — and its lots — before creating a new one.
    """
    lots, exclusions = extract_lots_from_payload(payload)
    for lot in lots:
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
            )
        )
    if lots:
        await session.flush()
    return len(lots), exclusions


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


async def get_complete_lots(session: AsyncSession) -> list[InvestmentLot]:
    """Every complete, projection-eligible lot across all uploads."""
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
    uploads = (await session.execute(select(CasUpload))).scalars().all()
    for upload in uploads:
        try:
            payload = json.loads(upload.raw_holdings_json)
        except json.JSONDecodeError, TypeError:
            continue
        _, exclusions = extract_lots_from_payload(payload)
        out.extend(exclusions)
    return out


async def get_unresolved_disposal_instruments(session: AsyncSession) -> set[str]:
    """Instrument ids with unresolvable disposal history, read from preserved
    CAS payloads. See :func:`unresolved_disposal_instruments` for the rule.

    The projection uses this to suppress gross acquisition lots for any
    instrument whose redemptions cannot be truthfully allocated, so the default
    lot projection never overstates holdings. Read-only: it only SELECTs
    ``CasUpload`` rows.
    """
    payloads: list[dict] = []
    uploads = (await session.execute(select(CasUpload))).scalars().all()
    for upload in uploads:
        try:
            payloads.append(json.loads(upload.raw_holdings_json))
        except json.JSONDecodeError, TypeError:
            continue
    return unresolved_disposal_instruments(payloads)


def _holding_positions(payload: dict) -> dict[str, dict]:
    """Latest explicit per-instrument holding facts from a CAS payload.

    Demat holdings (``accounts[].holdings[]``) and MF schemes
    (``folios[].schemes[]``) are flattened by ISIN; only instruments the CAS
    explicitly priced (``quantity``/``price`` or ``units``/``nav`` present) are
    returned, so a value-only total never becomes a fabricated position.
    """
    positions: dict[str, dict] = {}
    for account in payload.get("accounts") or []:
        for holding in account.get("holdings") or []:
            isin = holding.get("isin")
            if not isinstance(isin, str) or not isin.strip():
                continue
            key = isin.strip().upper()
            positions[key] = {
                "instrument_id": key,
                "instrument_name": holding.get("name") or key,
                "asset_class": str(holding.get("asset_class") or "other"),
                "quantity": _to_decimal(holding.get("quantity")),
                "unit_price": _to_decimal(holding.get("price")),
                "value": _to_decimal(holding.get("value")),
                "currency": CAS_CURRENCY,
            }
    for folio in payload.get("folios") or []:
        for scheme in folio.get("schemes") or []:
            isin = scheme.get("isin")
            if not isinstance(isin, str) or not isin.strip():
                continue
            key = isin.strip().upper()
            positions[key] = {
                "instrument_id": key,
                "instrument_name": scheme.get("scheme_name") or key,
                "asset_class": "mutual_fund",
                "quantity": _to_decimal(scheme.get("units")),
                "unit_price": _to_decimal(scheme.get("nav")),
                "value": _to_decimal(scheme.get("value")),
                "currency": CAS_CURRENCY,
            }
    return positions


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
    """Net instrument positions from the latest CAS, joined with known lots.

    Holdings supply the current allocation (quantity/price/value the CAS
    explicitly states); complete lots supply the cost basis for capital-gains.
    An instrument with a holding but no complete lot still appears, with zero
    lot fields — it is a value-only position, not a fabricated lot.
    """
    payload = await get_latest_payload(session)
    holdings = _holding_positions(payload) if payload else {}

    lots = await get_complete_lots(session)
    lot_qty: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    lot_cost: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    lot_name: dict[str, str] = {}
    for lot in lots:
        lot_qty[lot.instrument_id] += lot.quantity
        lot_cost[lot.instrument_id] += lot.cost_basis
        lot_name.setdefault(lot.instrument_id, lot.instrument_name)

    instruments = sorted(set(holdings) | set(lot_qty))
    out: list[LotPosition] = []
    for isin in instruments:
        holding = holdings.get(isin, {})
        out.append(
            LotPosition(
                instrument_id=isin,
                instrument_name=(
                    holding.get("instrument_name") or lot_name.get(isin) or isin
                ),
                asset_class=holding.get("asset_class", "other"),
                quantity=holding.get("quantity"),
                unit_price=holding.get("unit_price"),
                value=holding.get("value"),
                currency=holding.get("currency", CAS_CURRENCY),
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
    payload = await get_latest_payload(session)
    if not payload:
        return {}
    out: dict[str, Decimal] = {}
    for isin, holding in _holding_positions(payload).items():
        value = holding.get("value")
        if value is not None:
            out[isin] = value
    return out
