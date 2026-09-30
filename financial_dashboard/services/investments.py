"""Investment persistence and current-valuation service.

Pure CAS transaction handling lives in :mod:`investment_transactions`. CAS
payloads are modeled **without fabrication.** A lot is
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

from financial_dashboard.db.models import CasUpload, InvestmentLot
from financial_dashboard.services.investment_transactions import (
    CAS_CURRENCY,
    _to_decimal,
    extract_lots_from_payload,
)
from financial_dashboard.services.investment_types import (
    CompleteLot,
    CreateInvestmentLotsResult,
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

    It carries statement-date NAV/value/quantity facts, not acquisition cost. A
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
