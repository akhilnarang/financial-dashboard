"""CAS email auto-fetch: rule management + per-email handler."""

import datetime
import logging
from pathlib import Path
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.enums import EmailKind
from financial_dashboard.db.models import CasUpload, EmailSource, FetchRule, utc_now
from financial_dashboard.services.cas_ingestion import CasIngestError, ingest_cas_pdf
from financial_dashboard.services.settings import get_setting, get_setting_bool
from financial_dashboard.core.uploads import STATEMENTS_DIR, safe_upload_filename

logger = logging.getLogger(__name__)


class CasSender(NamedTuple):
    address: str
    bank: str


class CasEmailProcessResult(NamedTuple):
    upload_result: dict | None
    error: str | None


CAS_SENDERS: tuple[CasSender, ...] = (
    CasSender("NSDL-CAS@nsdl.co.in", "cas_nsdl"),
    CasSender("eCAS@cdslstatement.com", "cas_cdsl"),
)
CAS_COOLDOWN = datetime.timedelta(hours=24)


def _cas_pan() -> str:
    return (get_setting("cas_pan") or "").strip()


def _cas_enabled() -> bool:
    return get_setting_bool("cas_auto_fetch_enabled")


async def ensure_cas_fetch_rules(session: AsyncSession) -> None:
    """Reconcile auto-managed CAS FetchRules with the settings + cooldown.

    Called inside poll_all() before the enabled-rules query. Never deletes
    rules — only toggles `enabled` — so audit history is preserved.
    """
    auto_rules = (
        (
            await session.execute(
                select(FetchRule).where(FetchRule.auto_managed.is_(True))
            )
        )
        .scalars()
        .all()
    )

    if not _cas_enabled() or not _cas_pan():
        if not _cas_enabled():
            reason = "toggle off"
        else:
            reason = "no CAS PAN configured"
            logger.warning(
                "CAS auto-fetch is enabled but cas_pan is empty; "
                "no CAS emails will be processed"
            )
        for rule in auto_rules:
            if rule.enabled:
                rule.enabled = False
                logger.info("Disabled auto-managed CAS rule %s: %s", rule.id, reason)
        await session.flush()
        return

    sources = (
        (await session.execute(select(EmailSource).where(EmailSource.active.is_(True))))
        .scalars()
        .all()
    )

    now = utc_now()
    active_source_ids = {source.id for source in sources}
    existing_by_key = {(rule.source_id, rule.sender): rule for rule in auto_rules}

    # Disable auto-managed rules whose source went inactive/away.
    for rule in auto_rules:
        if rule.source_id not in active_source_ids and rule.enabled:
            rule.enabled = False
            logger.info(
                "Disabled auto-managed CAS rule %s: source %s inactive",
                rule.id,
                rule.source_id,
            )

    for source in sources:
        on_cooldown = (
            source.cas_last_polled_at is not None
            and (now - source.cas_last_polled_at) < CAS_COOLDOWN
        )
        for sender in CAS_SENDERS:
            rule = existing_by_key.get((source.id, sender.address))
            if rule is None:
                # initial_backfill_done_at left None so the provider runs its
                # standard 3-month backscan on first poll — important for CAS
                # since source.last_synced_at is already recent from existing
                # transactional polls, so without backfill we'd miss the most
                # recent monthly CAS statement that arrived before the toggle
                # was turned on. After a successful first fetch, the orchestrator
                # stamps initial_backfill_done_at and future polls fall back to
                # incremental source.last_synced_at filtering.
                rule = FetchRule(
                    provider=source.provider,
                    source_id=source.id,
                    sender=sender.address,
                    bank=sender.bank,
                    email_kind=EmailKind.CAS_STATEMENT.value,
                    enabled=not on_cooldown,
                    auto_managed=True,
                )
                session.add(rule)
                logger.info(
                    "Created auto-managed CAS rule for source=%s sender=%s",
                    source.id,
                    sender.address,
                )
            else:
                rule.enabled = not on_cooldown
    await session.flush()


async def process_cas_email(
    session: AsyncSession,
    raw_bytes: bytes,
    *,
    source_id: int | None,
    log_ref: str,
) -> CasEmailProcessResult:
    """Parse + ingest a CAS email.

    Returns a ``CasEmailProcessResult`` NamedTuple of (upload_result,
    error_message); positional unpacking is still supported.
    """
    from financial_dashboard.services.statements.cc import extract_pdf_from_email

    pan = _cas_pan()
    if not pan:
        logger.warning("CAS email %s arrived but cas_pan is not set", log_ref)
        return CasEmailProcessResult(None, "CAS PAN not configured")

    attachments = extract_pdf_from_email(raw_bytes) or []
    pdfs = [(name, data) for name, data in attachments if data]
    if not pdfs:
        logger.warning("CAS email %s has no PDF attachment", log_ref)
        return CasEmailProcessResult(None, "no PDF attachment")
    if len(pdfs) > 1:
        logger.info(
            "CAS email %s has %d PDFs; ingesting first only (%s)",
            log_ref,
            len(pdfs),
            pdfs[0][0],
        )

    filename, pdf_bytes = pdfs[0]
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    safe_name = safe_upload_filename(filename or "cas.pdf")
    file_path: Path = STATEMENTS_DIR / f"{ts}_src{source_id}_{safe_name}"
    file_path.write_bytes(pdf_bytes)

    try:
        upload = await ingest_cas_pdf(session, file_path, password=pan)
    except (CasIngestError, ValueError) as exc:
        logger.warning("CAS ingest failed for %s: %s", log_ref, exc)
        file_path.unlink(missing_ok=True)
        return CasEmailProcessResult(None, str(exc))
    except Exception as exc:
        logger.exception("CAS ingest crashed for %s", log_ref)
        file_path.unlink(missing_ok=True)
        return CasEmailProcessResult(None, f"unexpected {type(exc).__name__}: {exc}")

    return CasEmailProcessResult({"cas_upload_id": upload.id}, None)


async def link_cas_upload_email(
    session: AsyncSession, cas_upload_id: int, email_id: int
) -> None:
    upload = await session.get(CasUpload, cas_upload_id)
    if upload is not None:
        upload.email_id = email_id
