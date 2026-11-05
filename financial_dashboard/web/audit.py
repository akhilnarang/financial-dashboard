"""Authenticated HTML views for assistant audit records."""

import datetime
import json
from typing import NamedTuple
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import JsonValue
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


class AuditJsonBlock(NamedTuple):
    """Holds the stored JSON as text and, when it parses, its compact and tree forms."""

    text: str
    compact: str | None
    tree: str | None


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
    try:
        output = json.loads(detail.interaction.model_output_json or "{}")
    except ValueError:
        output = {}
    search = output.get("merchant_search") if isinstance(output, dict) else None
    sources = search.get("sources") if isinstance(search, dict) else None
    merchant_sources = (
        [
            source
            for source in sources[:3]
            if isinstance(source, dict)
            and isinstance(source.get("url"), str)
            and source["url"].startswith(("https://", "http://"))
        ]
        if isinstance(sources, list)
        else []
    )
    return templates.TemplateResponse(
        request,
        "partials/audit_detail.html",
        {
            "detail": detail,
            "json_block": _json_block,
            "merchant_sources": merchant_sources,
        },
    )


def _json_block(value: str | None) -> AuditJsonBlock:
    """Format a JSON string for display and decode embedded JSON for the tree.

    Args:
        value: The raw JSON text. The text can be empty or None.

    Returns:
        The stored JSON, pretty-printed and compact, and the tree with
        embedded JSON decoded. Compact and tree are None when the text is
        not JSON.
    """
    if not value:
        return AuditJsonBlock("{}", "{}", "{}")
    try:
        loaded = json.loads(value)
        tree = json.dumps(_decode_embedded(loaded), ensure_ascii=False)
    except TypeError, ValueError, RecursionError:
        return AuditJsonBlock(value, None, None)
    return AuditJsonBlock(
        json.dumps(loaded, ensure_ascii=False, indent=2),
        json.dumps(loaded, ensure_ascii=False, separators=(",", ":")),
        tree,
    )


def _decode_embedded(value: JsonValue) -> JsonValue:
    """Decode JSON text that is nested inside string values.

    Args:
        value: The JSON value to expand.

    Returns:
        The value with every decodable nested JSON string expanded.
    """
    match value:
        case str():
            if (decoded := _load_container(value)) is not None:
                return _decode_embedded(decoded)
            return _text_segments(value) if "\n" in value else value
        case list():
            return [_decode_embedded(item) for item in value]
        case dict():
            return {key: _decode_embedded(item) for key, item in value.items()}
        case _:
            return value


def _load_container(text: str) -> list[JsonValue] | dict[str, JsonValue] | None:
    """Parse text as a JSON array or object.

    Args:
        text: The text to parse.

    Returns:
        The parsed array or object. None when the text is not one.
    """
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list | dict) else None


def _text_segments(text: str) -> str | list[JsonValue]:
    """Split multiline text into plain text parts and JSON parts.

    Args:
        text: The multiline string value to split.

    Returns:
        The list of parts, or the original text when no line is JSON.
    """
    buffer: list[str] = []
    segments: list[JsonValue] = []
    for line in text.splitlines():
        if (decoded := _load_container(line)) is None:
            buffer.append(line)
            continue
        if buffer:
            segments.append("\n".join(buffer))
            buffer.clear()
        segments.append(_decode_embedded(decoded))
    if not segments:
        return text
    if buffer:
        segments.append("\n".join(buffer))
    return segments
