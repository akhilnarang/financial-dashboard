"""HTML routes for the extension framework and the Paisa extension surface.

Pages live under ``/extensions`` (the list) and ``/extensions/paisa`` (the
Paisa configuration page). All configuration mutations are POST-Redirect-GET
so a refresh never resubmits. Status is fetched client-side (a network probe
has no business blocking a page render); preview is rendered server-side
because it is a pure local read.

Every DB-touching handler takes ``session: AsyncSession = Depends(get_session)``.
"""

import json
import logging
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.extensions import BUILTIN_EXTENSIONS
from financial_dashboard.schemas.extensions import PaisaConfigInput
from financial_dashboard.services.paisa import surface
from financial_dashboard.services.paisa.renderers.beancount import quote_string

logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()

_SETUP_PATH_PLACEHOLDER = "/absolute/path/to/financial-dashboard.journal"


def _manifests(request: FastAPIRequest):
    manager = getattr(request.app.state, "extension_manager", None)
    if manager is not None:
        return manager.all()
    return BUILTIN_EXTENSIONS


@router.get("/extensions", response_class=HTMLResponse)
async def extensions_index(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    manifests = _manifests(request)
    extensions = [surface.extension_info(m) for m in manifests]
    return templates.TemplateResponse(
        request,
        "extensions/index.html",
        {"active_page": "extensions", "extensions": extensions},
    )


# ---------------------------------------------------------------------------
# Paisa configuration page + actions
# ---------------------------------------------------------------------------


def _flash(response: RedirectResponse, key: str, value: str) -> RedirectResponse:
    """Append a one-shot ``?key=value`` flash to a PRG redirect location.

    Both ``key`` and ``value`` are passed through ``urlencode`` so spaces,
    ``&``, ``#``, and Unicode cannot split into extra params or a fragment and
    corrupt the Location header. ``request.query_params`` URL-decodes on read,
    so the original value round-trips on the next GET.
    """
    sep = "&" if "?" in response.headers["location"] else "?"
    response.headers["location"] = (
        f"{response.headers['location']}{sep}{urlencode({key: value})}"
    )
    return response


async def _paisa_context(session: AsyncSession, request: FastAPIRequest) -> dict:
    """Build the render context for the Paisa page.

    Config and account choices are cheap, local reads. Preview is also local
    (no network) and only runs when the mode permits it; its failure is shaped
    into the same ``{ok, reason}`` envelope so the template renders a notice
    instead of erroring. Status is left to the client-side fetch.
    """
    cfg = surface.config_view()
    choices = await surface.account_choices(session)
    try:
        prev = await surface.preview_projection(session)
        preview_payload = prev.model_dump()
    except Exception as exc:  # optional-extension isolation
        logger.warning("Paisa preview on page render failed: %s", exc, exc_info=True)
        preview_payload = {"ok": False, "reason": str(exc)}

    setup_path = cfg.generated_path or _SETUP_PATH_PLACEHOLDER
    setup_include_line = f"include {setup_path}"
    if cfg.ledger_cli == "beancount":
        setup_include_line = f"include {quote_string(setup_path)}"

    return {
        "active_page": "extensions",
        "config": cfg.model_dump(),
        "accounts": [a.model_dump() for a in choices.accounts],
        "preview": preview_payload,
        "safe_link": surface.safe_link(),
        "setup_include_line": setup_include_line,
        "flash": {
            "saved": request.query_params.get("saved"),
            "generated": request.query_params.get("generated"),
            "synced": request.query_params.get("synced"),
            "outcome": request.query_params.get("outcome"),
            "error": request.query_params.get("error"),
        },
        "errors": [],
    }


@router.get("/extensions/paisa", response_class=HTMLResponse)
async def paisa_page(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    ctx = await _paisa_context(session, request)
    return templates.TemplateResponse(request, "extensions/paisa.html", ctx)


def _parse_mappings(keys: list[str], values: list[str]) -> dict[str, str]:
    """Zip parallel key/value field lists into a {str: str} mapping.

    Empty rows (both sides blank) are dropped; a row with only one side filled
    is dropped too so a stray blank box cannot inject an empty account name.
    """
    out: dict[str, str] = {}
    for key, value in zip(keys, values, strict=False):
        k = (key or "").strip()
        v = (value or "").strip()
        if k and v:
            out[k] = v
    return out


def _form_to_input(form) -> PaisaConfigInput:
    """Translate the submitted HTML form into the shared config input model."""
    raw_account_ids = form.getlist("selected_account_ids")
    selected_ids: list[int] = []
    for raw in raw_account_ids:
        text = (raw or "").strip()
        if not text:
            continue
        try:
            selected_ids.append(int(text))
        except ValueError:
            continue

    account_mappings = _parse_mappings(
        form.getlist("account_mapping_key"),
        form.getlist("account_mapping_value"),
    )
    category_mappings = _parse_mappings(
        form.getlist("category_mapping_key"),
        form.getlist("category_mapping_value"),
    )

    timeout_raw = (form.get("request_timeout_seconds") or "").strip()
    try:
        timeout = int(timeout_raw) if timeout_raw else 15
    except ValueError:
        timeout = 15

    # FX rate rows: three parallel field lists (currency/date/rate). Rows with
    # both currency and date and rate are kept; empty rows are dropped. The
    # shared validator in surface.save_config enforces positivity + date shape.
    fx_rates = []
    fx_ccys = form.getlist("fx_currency")
    fx_dates = form.getlist("fx_date")
    fx_rates_raw = form.getlist("fx_rate")
    for ccy, date, rate in zip(fx_ccys, fx_dates, fx_rates_raw, strict=False):
        if (ccy or "").strip() or (date or "").strip() or (rate or "").strip():
            fx_rates.append(
                {
                    "currency": (ccy or "").strip().upper(),
                    "date": (date or "").strip(),
                    "rate": (rate or "").strip(),
                }
            )

    min_interval_raw = (form.get("auto_sync_min_interval_minutes") or "").strip()
    try:
        min_interval = int(min_interval_raw) if min_interval_raw else 30
    except ValueError:
        min_interval = 30
    ttl_raw = (form.get("report_cache_ttl_seconds") or "").strip()
    try:
        ttl = int(ttl_raw) if ttl_raw else 60
    except ValueError:
        ttl = 60

    return PaisaConfigInput(
        mode=(form.get("mode") or "").strip(),
        base_url=(form.get("base_url") or "").strip(),
        external_url=(form.get("external_url") or "").strip(),
        allow_remote=form.get("allow_remote") == "true",
        auth_username=(form.get("auth_username") or "").strip(),
        auth_password=(form.get("auth_password") or ""),
        generated_path=(form.get("generated_path") or "").strip(),
        selected_account_ids=selected_ids,
        project_since=(form.get("project_since") or "").strip(),
        account_mappings=account_mappings,
        category_mappings=category_mappings,
        non_inr_policy=(form.get("non_inr_policy") or "skip").strip(),
        request_timeout_seconds=timeout,
        ledger_cli=(form.get("ledger_cli") or "ledger").strip().lower(),
        fx_rates=fx_rates,
        report_cache_ttl_seconds=ttl,
        auto_sync_enabled=form.get("auto_sync_enabled") == "true",
        auto_sync_min_interval_minutes=min_interval,
        notify_sync_failures=form.get("notify_sync_failures") == "true",
        project_investments=form.get("project_investments") == "true",
    )


@router.post("/extensions/paisa")
async def paisa_save(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    form = await request.form()
    data = _form_to_input(form)
    result = await surface.save_config(session, data)
    if not result.ok:
        # Re-render the page with validation errors so the operator's input is
        # not lost to a bare redirect.
        ctx = await _paisa_context(session, request)
        ctx["errors"] = result.errors
        return templates.TemplateResponse(
            request, "extensions/paisa.html", ctx, status_code=422
        )
    return _flash(
        RedirectResponse(url="/extensions/paisa", status_code=303), "saved", "1"
    )


@router.post("/extensions/paisa/generate")
async def paisa_generate_action(
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await surface.generate_now_audited(session, trigger="web")
    except Exception as exc:  # optional-extension isolation
        logger.warning("Paisa generate action failed: %s", exc, exc_info=True)
        return _flash(
            RedirectResponse(url="/extensions/paisa", status_code=303),
            "error",
            _short_error(exc),
        )
    if result.ok:
        return _flash(
            RedirectResponse(url="/extensions/paisa", status_code=303),
            "generated",
            "1",
        )
    return _flash(
        RedirectResponse(url="/extensions/paisa", status_code=303),
        "error",
        result.reason or "generate_failed",
    )


@router.post("/extensions/paisa/sync")
async def paisa_sync_action(
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await surface.sync_now_audited(session, trigger="web")
    except Exception as exc:  # optional-extension isolation
        logger.warning("Paisa sync action failed: %s", exc, exc_info=True)
        return _flash(
            RedirectResponse(url="/extensions/paisa", status_code=303),
            "error",
            _short_error(exc),
        )
    if result.ok:
        return _flash(
            RedirectResponse(url="/extensions/paisa", status_code=303),
            "synced",
            "1",
        )
    return _flash(
        RedirectResponse(url="/extensions/paisa", status_code=303),
        "outcome",
        result.outcome,
    )


# ---------------------------------------------------------------------------
# Additive read-only pages: audit, reports, reconciliation
# ---------------------------------------------------------------------------


@router.get("/extensions/paisa/audit", response_class=HTMLResponse)
async def paisa_audit_page(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    audit = await surface.audit_view(session)
    return templates.TemplateResponse(
        request,
        "extensions/paisa_audit.html",
        {
            "active_page": "extensions",
            "audit": audit.model_dump(),
            "flash": {"error": request.query_params.get("error")},
        },
    )


@router.get("/extensions/paisa/reports/{report}", response_class=HTMLResponse)
async def paisa_report_page(
    request: FastAPIRequest,
    report: str,
):
    # Reports are fetched client-side from the JSON API so a slow/down Paisa
    # never blocks the page render; this route only shells the template.
    return templates.TemplateResponse(
        request,
        "extensions/paisa_report.html",
        {
            "active_page": "extensions",
            "report": report,
            "safe_link": surface.safe_link(),
        },
    )


@router.get("/extensions/paisa/reconciliation", response_class=HTMLResponse)
async def paisa_reconciliation_page(
    request: FastAPIRequest,
):
    # The reconciliation table is fetched client-side (it may hit the upstream);
    # the page render is local and never blocks on the network.
    cfg = surface.config_view()
    return templates.TemplateResponse(
        request,
        "extensions/paisa_reconciliation.html",
        {
            "active_page": "extensions",
            "config": cfg.model_dump(),
            "safe_link": surface.safe_link(),
        },
    )


def _short_error(exc: Exception) -> str:
    """First-line message for an ``?error=`` flash param.

    ``_flash`` URL-encodes the value, so this only needs to collapse a multi-line
    exception to its first non-empty line (``json.dumps`` also neutralises any
    embedded quotes/control characters) — never a security boundary.

    ``ensure_ascii=False`` preserves Unicode (e.g. ``ñ``, ``₹``) as real
    characters rather than ``\\uXXXX`` escapes, so the round-tripped flash
    reads naturally; the URL-encoding in :func:`_flash` remains the safety
    boundary that keeps those characters from splitting the Location header.
    """
    text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return json.dumps(text, ensure_ascii=False)[1:-1]
