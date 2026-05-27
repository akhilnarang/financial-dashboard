"""Polling HTML routes."""

import logging

from fastapi import APIRouter, Depends, Request as FastAPIRequest
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()


@router.post("/poll")
async def trigger_poll(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    logger.info("Manual poll triggered")
    fetch_service = getattr(request.app.state, "fetch_service", None)
    if fetch_service is None:
        return RedirectResponse(url="/", status_code=303)
    if not await fetch_service.trigger_poll():
        logger.info("Manual poll ignored because a background poll is already active")
        return RedirectResponse(url="/", status_code=303)
    return RedirectResponse(url="/", status_code=303)
