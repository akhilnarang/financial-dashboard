"""CAS import service."""

from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import parse_date
from financial_dashboard.db.enums import SnapshotCategory, SnapshotKind, SnapshotSource
from financial_dashboard.db.models import BalanceSnapshot, CasUpload, SnapshotHolding


class CasIngestError(ValueError):
    pass


def _decimal(value) -> Decimal:
    return Decimal(str(value))


async def ingest_cas_payload(
    session: AsyncSession, payload: dict, *, force_replace: bool = False
) -> CasUpload:
    meta = payload.get("meta") or {}
    summary = payload.get("summary") or {}
    recon = payload.get("reconciliation") or {}

    if summary.get("grand_total") is None:
        raise CasIngestError("CAS has no portfolio grand_total; refusing to import.")
    if not meta.get("statement_period_end"):
        raise CasIngestError("CAS has no statement period end; refusing to import.")

    statement_date = parse_date(meta["statement_period_end"], dayfirst=True)
    if statement_date is None:
        raise CasIngestError("CAS statement period end could not be parsed.")

    grand_total = _decimal(summary["grand_total"])
    portfolio_key = (meta.get("pan") or "primary").upper()
    incoming_source = str(meta.get("source") or "unknown").lower()

    prior_uploads = (
        (
            await session.execute(
                select(CasUpload).where(
                    CasUpload.portfolio_key == portfolio_key,
                    CasUpload.statement_date == statement_date,
                )
            )
        )
        .scalars()
        .all()
    )
    # NSDL embeds CDSL — refuse to silently overwrite an NSDL upload with a CDSL one.
    if not force_replace:
        for prior in prior_uploads:
            if prior.depository_source == "nsdl" and incoming_source == "cdsl":
                raise CasIngestError(
                    f"An NSDL CAS already exists for this PAN on "
                    f"{statement_date.isoformat()}. NSDL embeds CDSL, so the "
                    f"incoming CDSL statement would lose data. Re-upload with "
                    f"the override option if you really want to replace it."
                )
    for upload in prior_uploads:
        snapshot_ids = (
            (
                await session.execute(
                    select(BalanceSnapshot.id).where(
                        BalanceSnapshot.cas_upload_id == upload.id
                    )
                )
            )
            .scalars()
            .all()
        )
        if snapshot_ids:
            await session.execute(
                delete(SnapshotHolding).where(
                    SnapshotHolding.snapshot_id.in_(snapshot_ids)
                )
            )
        await session.execute(
            delete(BalanceSnapshot).where(BalanceSnapshot.cas_upload_id == upload.id)
        )
        await session.execute(delete(CasUpload).where(CasUpload.id == upload.id))

    upload = CasUpload(
        portfolio_key=portfolio_key,
        depository_source=str(meta.get("source") or "unknown").lower(),
        investor_name=meta.get("investor_name"),
        statement_date=statement_date,
        grand_total=grand_total,
        portfolio_ok=bool(recon.get("portfolio_ok", True)),
        portfolio_delta=(
            _decimal(recon["portfolio_delta"])
            if recon.get("portfolio_delta") is not None
            else None
        ),
        raw_holdings_json=json.dumps(payload, default=str),
    )
    session.add(upload)
    await session.flush()

    snapshot = BalanceSnapshot(
        cas_upload_id=upload.id,
        portfolio_key=portfolio_key,
        kind=SnapshotKind.asset.value,
        category=SnapshotCategory.investment.value,
        as_of_date=statement_date,
        value=grand_total,
        source=SnapshotSource.cas.value,
    )
    session.add(snapshot)
    await session.flush()

    by_class: dict[str, Decimal] = defaultdict(lambda: Decimal("0.00"))
    for account in payload.get("accounts") or []:
        for holding in account.get("holdings") or []:
            if holding.get("value") is not None:
                by_class[str(holding.get("asset_class") or "other")] += _decimal(
                    holding["value"]
                )
    for folio in payload.get("folios") or []:
        if folio.get("total_value") is not None:
            by_class["mutual_fund"] += _decimal(folio["total_value"])

    classified = sum(by_class.values(), Decimal("0.00"))
    remainder = grand_total - classified
    if remainder > Decimal("0.00"):
        by_class["other"] += remainder

    for asset_class, value in by_class.items():
        session.add(
            SnapshotHolding(
                snapshot_id=snapshot.id,
                asset_class=asset_class,
                label=asset_class.replace("_", " ").title(),
                value=value,
            )
        )
    return upload


async def ingest_cas_pdf(
    session: AsyncSession,
    pdf_path: str | Path,
    password: str | None = None,
    *,
    force_replace: bool = False,
) -> CasUpload:
    from financial_dashboard.integrations.parsers import parse_cas_pdf

    parsed = parse_cas_pdf(Path(pdf_path), password=password)
    return await ingest_cas_payload(
        session, parsed.model_dump(mode="json"), force_replace=force_replace
    )
