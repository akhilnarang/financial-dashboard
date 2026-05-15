"""Rule HTML routes."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import (
    EmailSource,
    FetchRule,
)
from bank_email_parser import SUPPORTED_BANKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()


@router.get("/rules", response_class=HTMLResponse)
async def rule_list(
    request: FastAPIRequest,
    bank: Annotated[str | None, Query(description="Filter by bank")] = None,
    source_id: Annotated[
        str | None, Query(description="Filter by email source ID")
    ] = None,
    enabled: Annotated[
        str | None, Query(description="Filter by enabled status: true or false")
    ] = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(FetchRule)
    if bank:
        stmt = stmt.where(FetchRule.bank == bank)
    if source_id:
        try:
            stmt = stmt.where(FetchRule.source_id == int(source_id))
        except ValueError:
            pass
    if enabled == "true":
        stmt = stmt.where(FetchRule.enabled.is_(True))
    elif enabled == "false":
        stmt = stmt.where(FetchRule.enabled.is_(False))
    stmt = stmt.order_by(FetchRule.id)
    rules = (await session.execute(stmt)).scalars().all()

    bank_result = await session.execute(select(FetchRule.bank).distinct())
    banks = sorted([row[0] for row in bank_result.all() if row[0]])

    source_result = await session.execute(
        select(EmailSource).where(EmailSource.active.is_(True)).order_by(EmailSource.id)
    )
    sources = source_result.scalars().all()

    filters = {"bank": bank, "source_id": source_id, "enabled": enabled}

    return templates.TemplateResponse(
        request,
        "rules.html",
        {
            "active_page": "rules",
            "rules": rules,
            "sources": sources,
            "banks": banks,
            "supported_banks": SUPPORTED_BANKS,
            "filters": filters,
        },
    )


@router.post("/rules")
async def rule_create(
    request: FastAPIRequest,
    bank: str = Form(...),
    sender: str = Form(""),
    subject: str = Form(""),
    folder: str = Form(""),
    source_id: int = Form(None),
    provider: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    resolved_provider = provider.strip()
    if source_id:
        source = await session.get(EmailSource, source_id)
        if source:
            resolved_provider = source.provider

    rule = FetchRule(
        provider=resolved_provider or "gmail",
        source_id=source_id if source_id else None,
        bank=bank.strip(),
        sender=sender.strip() or None,
        subject=subject.strip() or None,
        folder=folder.strip() or None,
    )
    session.add(rule)
    await session.commit()

    return RedirectResponse(url="/rules", status_code=303)


@router.get("/rules/{rule_id}/edit", response_class=HTMLResponse)
async def rule_edit_form(
    request: FastAPIRequest,
    rule_id: int,
    session: AsyncSession = Depends(get_session),
):
    rule = await session.get(FetchRule, rule_id)
    if not rule:
        return RedirectResponse(url="/rules", status_code=303)

    source_result = await session.execute(
        select(EmailSource).where(EmailSource.active.is_(True)).order_by(EmailSource.id)
    )
    sources = source_result.scalars().all()

    return templates.TemplateResponse(
        request,
        "rule_edit.html",
        {
            "active_page": "rules",
            "rule": rule,
            "sources": sources,
            "supported_banks": SUPPORTED_BANKS,
        },
    )


@router.post("/rules/{rule_id}/edit")
async def rule_update(
    request: FastAPIRequest,
    rule_id: int,
    bank: str = Form(...),
    sender: str = Form(""),
    subject: str = Form(""),
    folder: str = Form(""),
    enabled: bool = Form(False),
    source_id: int = Form(None),
    provider: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    rule = await session.get(FetchRule, rule_id)
    if not rule:
        return RedirectResponse(url="/rules", status_code=303)

    resolved_provider = provider.strip()
    if source_id:
        source = await session.get(EmailSource, source_id)
        if source:
            resolved_provider = source.provider

    rule.provider = resolved_provider or rule.provider
    rule.source_id = source_id if source_id else None
    rule.bank = bank.strip()
    rule.sender = sender.strip() or None
    rule.subject = subject.strip() or None
    rule.folder = folder.strip() or None
    rule.enabled = enabled
    await session.commit()

    return RedirectResponse(url="/rules", status_code=303)


@router.post("/rules/{rule_id}/delete")
async def rule_delete(
    rule_id: int,
    session: AsyncSession = Depends(get_session),
):
    rule = await session.get(FetchRule, rule_id)
    if rule:
        await session.delete(rule)
        await session.commit()

    return RedirectResponse(url="/rules", status_code=303)


@router.post("/rules/{rule_id}/toggle")
async def rule_toggle(
    rule_id: int,
    session: AsyncSession = Depends(get_session),
):
    rule = await session.get(FetchRule, rule_id)
    if rule:
        rule.enabled = not rule.enabled
        await session.commit()

    return RedirectResponse(url="/rules", status_code=303)
