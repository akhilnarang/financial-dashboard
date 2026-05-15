"""Source HTML routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.crypto import encrypt_credentials
from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import (
    EmailSource,
    FetchRule,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()


@router.get("/sources", response_class=HTMLResponse)
async def source_list(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(EmailSource).order_by(EmailSource.id))
    sources = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "sources.html",
        {
            "active_page": "sources",
            "sources": sources,
        },
    )


@router.post("/sources")
async def create_source(
    request: FastAPIRequest,
    provider: str = Form(...),
    label: str = Form(...),
    account_identifier: str = Form(""),
    credential_value: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    # Build creds dict based on provider
    if provider == "gmail":
        creds = {
            "user": account_identifier.strip(),
            "app_password": credential_value.strip(),
        }
    elif provider == "fastmail":
        creds = {"token": credential_value.strip()}
    else:
        return RedirectResponse(url="/sources", status_code=303)

    encrypted = encrypt_credentials(creds)

    source = EmailSource(
        provider=provider.strip(),
        label=label.strip(),
        account_identifier=account_identifier.strip() or None,
        credentials=encrypted,
    )
    session.add(source)
    await session.commit()

    return RedirectResponse(url="/sources", status_code=303)


@router.get("/sources/{source_id}/edit", response_class=HTMLResponse)
async def edit_source_form(
    source_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    source = await session.get(EmailSource, source_id)
    if not source:
        return RedirectResponse(url="/sources", status_code=303)

    return templates.TemplateResponse(
        request,
        "source_edit.html",
        {
            "active_page": "sources",
            "source": source,
        },
    )


@router.post("/sources/{source_id}/edit")
async def update_source(
    source_id: int,
    request: FastAPIRequest,
    provider: str = Form(...),
    label: str = Form(...),
    account_identifier: str = Form(""),
    credential_value: str = Form(""),
    active: bool = Form(False),
    session: AsyncSession = Depends(get_session),
):
    source = await session.get(EmailSource, source_id)
    if not source:
        return RedirectResponse(url="/sources", status_code=303)

    source.provider = provider.strip()
    source.label = label.strip()
    source.account_identifier = account_identifier.strip() or None
    source.active = active

    if credential_value.strip():
        if provider == "gmail":
            creds = {
                "user": account_identifier.strip(),
                "app_password": credential_value.strip(),
            }
        elif provider == "fastmail":
            creds = {"token": credential_value.strip()}
        else:
            creds = {}
        if creds:
            source.credentials = encrypt_credentials(creds)

    await session.commit()

    return RedirectResponse(url="/sources", status_code=303)


@router.post("/sources/{source_id}/delete")
async def delete_source(
    source_id: int,
    session: AsyncSession = Depends(get_session),
):
    source = await session.get(EmailSource, source_id)
    if source:
        await session.execute(
            update(FetchRule)
            .where(FetchRule.source_id == source_id)
            .values(source_id=None)
        )
        await session.delete(source)
        await session.commit()

    return RedirectResponse(url="/sources", status_code=303)
