"""Settings HTML routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.services.settings import (
    get_grouped_settings,
    parse_form_updates,
    restart_services,
    save_settings,
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


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "active_page": "settings",
            "grouped_settings": get_grouped_settings(),
        },
    )


@router.post("/settings")
async def save_settings_route(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    form = await request.form()
    updates, errors = parse_form_updates({str(k): v for k, v in form.items()})

    if errors:
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "active_page": "settings",
                "grouped_settings": get_grouped_settings(),
                "errors": errors,
            },
            status_code=422,
        )

    try:
        changed_keys = await save_settings(updates)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "active_page": "settings",
                "grouped_settings": get_grouped_settings(),
                "errors": [str(exc)],
            },
            status_code=422,
        )

    telegram_restart_keys = {
        "telegram.bot_token",
        "telegram.chat_id",
        "telegram.enabled",
    }
    if changed_keys & telegram_restart_keys:
        await restart_services()

    return RedirectResponse(url="/settings?saved", status_code=303)
