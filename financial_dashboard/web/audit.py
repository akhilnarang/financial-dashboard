"""Authenticated HTML views for assistant audit records."""

import datetime
import json
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.services.audit_reads import (
    AUDIT_PAGE_SIZE,
    get_audit_detail,
    list_terminal_proactive_deliveries,
    list_audit_interactions,
)

router = APIRouter()
templates = get_templates()


@router.get("/audit", response_class=HTMLResponse)
async def audit_list(
    request: Request,
    page: int = Query(default=1, ge=1),
    status: str | None = None,
    trigger: str | None = None,
    transaction_id: int | None = Query(default=None, ge=1),
    action_type: str | None = None,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    session: AsyncSession = Depends(get_session),
) -> Response:
    filters = {
        "status": status,
        "trigger": trigger,
        "transaction_id": transaction_id,
        "action_type": action_type,
        "date_from": date_from,
        "date_to": date_to,
    }
    pagination_query = urlencode(
        {
            key: value.isoformat() if isinstance(value, datetime.date) else value
            for key, value in filters.items()
            if value is not None and value != ""
        }
    )
    rows = await list_audit_interactions(
        session, offset=(page - 1) * AUDIT_PAGE_SIZE, **filters
    )
    terminal_proactive = (
        await list_terminal_proactive_deliveries(session) if page == 1 else []
    )
    return templates.TemplateResponse(
        request,
        "audit.html",
        {
            "active_page": "audit",
            "interactions": rows,
            "page": page,
            "has_next": len(rows) == AUDIT_PAGE_SIZE,
            "filters": filters,
            "pagination_query": pagination_query,
            "terminal_proactive": terminal_proactive,
        },
    )


@router.get("/audit/{interaction_id}", response_class=HTMLResponse)
async def audit_detail(
    interaction_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Response:
    detail = await get_audit_detail(session, interaction_id)
    if detail is None:
        return HTMLResponse("<p>Audit interaction not found.</p>", status_code=404)
    return templates.TemplateResponse(
        request,
        "partials/audit_detail.html",
        {"detail": detail, "pretty_json": _pretty_json},
    )


def _pretty_json(value: str | None) -> str:
    if not value:
        return "{}"
    try:
        return json.dumps(
            json.loads(value), ensure_ascii=False, indent=2, sort_keys=True
        )
    except TypeError, ValueError:
        return value
