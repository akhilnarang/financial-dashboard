"""JSON routes for the extension framework and the Paisa extension surface.

All routes live under ``/api/extensions``. The Paisa-specific routes live one
level deeper under ``/api/extensions/paisa``. Every DB-touching handler takes
``session: AsyncSession = Depends(get_session)``.

Failure isolation: probe/preview/generate/sync dispatch through the
``services.paisa.surface`` adapter, which catches the core Paisa/Projection
errors into typed ``ok=False`` responses. Any *unexpected* error is caught at
the route boundary and returned as a typed :class:`ExtensionErrorResponse` so
an optional-extension failure can never 500 or affect a core route.
"""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.extensions import BUILTIN_EXTENSIONS
from financial_dashboard.schemas.extensions import (
    ExtensionAuditResponse,
    ExtensionErrorResponse,
    ExtensionListResponse,
    PaisaAccountChoicesResponse,
    PaisaConfigInput,
    PaisaConfigSaveResponse,
    PaisaGenerateResponse,
    PaisaPreviewResponse,
    PaisaReconcileResponse,
    PaisaReportSummary,
    PaisaStatusResponse,
    PaisaSyncResponse,
)
from financial_dashboard.services.paisa import surface

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extensions")


def _manifests(request: Request):
    """The registered extension manifests.

    Reads from ``app.state.extension_manager`` when the lifespan has run, and
    falls back to ``BUILTIN_EXTENSIONS`` otherwise (e.g. in route-level tests
    that build the app without the lifespan). This is the documented
    legitimate form of ``getattr(request.app.state, ...)``.
    """
    manager = getattr(request.app.state, "extension_manager", None)
    if manager is not None:
        return manager.all()
    return BUILTIN_EXTENSIONS


@router.get("", response_model=ExtensionListResponse)
async def list_extensions(request: Request) -> ExtensionListResponse:
    return ExtensionListResponse(
        extensions=[surface.extension_info(m) for m in _manifests(request)]
    )


# ---------------------------------------------------------------------------
# Paisa
# ---------------------------------------------------------------------------


paisa_router = APIRouter(prefix="/paisa")


@paisa_router.get("/config")
async def get_paisa_config() -> JSONResponse:
    try:
        cfg = surface.config_view()
    except Exception as exc:  # optional-extension isolation
        logger.warning("Paisa config read failed: %s", exc, exc_info=True)
        return _typed_error("paisa_config_unavailable", str(exc))
    return JSONResponse(cfg.model_dump())


@paisa_router.get("/accounts")
async def get_paisa_accounts(
    session: AsyncSession = Depends(get_session),
) -> PaisaAccountChoicesResponse:
    return await surface.account_choices(session)


@paisa_router.post("/config", response_model=PaisaConfigSaveResponse)
async def save_paisa_config(
    payload: PaisaConfigInput = Body(...),
    session: AsyncSession = Depends(get_session),
) -> PaisaConfigSaveResponse:
    return await surface.save_config(session, payload)


@paisa_router.get("/status")
async def paisa_status() -> JSONResponse:
    """Probe the Paisa instance (connect/project modes only) and return status.

    Read-only: ping + config + diagnosis. Never writes. A failure is returned
    as a typed ``ok=False`` body, never a 500, so the dashboard's status widget
    degrades gracefully.
    """
    try:
        status: PaisaStatusResponse = await surface.probe_status()
    except Exception as exc:  # optional-extension isolation
        logger.warning("Paisa status probe failed: %s", exc, exc_info=True)
        return _typed_error("paisa_status_failed", str(exc))
    return JSONResponse(status.model_dump())


@paisa_router.post("/probe", response_model=PaisaStatusResponse)
async def paisa_probe(
    session: AsyncSession = Depends(get_session),
) -> PaisaStatusResponse:
    """Explicit probe action (same semantics as GET /status), audited."""
    try:
        return await surface.probe_status_audited(session, trigger="api")
    except Exception as exc:  # noqa: BLE001 — optional-extension isolation
        logger.warning("Paisa probe failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=ExtensionErrorResponse(
                error="paisa_probe_failed", detail=str(exc)
            ).model_dump(),
        ) from exc


@paisa_router.post("/preview", response_model=PaisaPreviewResponse)
async def paisa_preview(
    session: AsyncSession = Depends(get_session),
) -> PaisaPreviewResponse:
    try:
        return await surface.preview_projection(session)
    except Exception as exc:  # noqa: BLE001 — optional-extension isolation
        logger.warning("Paisa preview failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=ExtensionErrorResponse(
                error="paisa_preview_failed", detail=str(exc)
            ).model_dump(),
        ) from exc


@paisa_router.post("/generate", response_model=PaisaGenerateResponse)
async def paisa_generate(
    session: AsyncSession = Depends(get_session),
) -> PaisaGenerateResponse:
    try:
        return await surface.generate_now_audited(session, trigger="api")
    except Exception as exc:  # noqa: BLE001 — optional-extension isolation
        logger.warning("Paisa generate failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=ExtensionErrorResponse(
                error="paisa_generate_failed", detail=str(exc)
            ).model_dump(),
        ) from exc


@paisa_router.post("/sync", response_model=PaisaSyncResponse)
async def paisa_sync(
    session: AsyncSession = Depends(get_session),
) -> PaisaSyncResponse:
    try:
        return await surface.sync_now_audited(session, trigger="api")
    except Exception as exc:  # noqa: BLE001 — optional-extension isolation
        logger.warning("Paisa sync failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=ExtensionErrorResponse(
                error="paisa_sync_failed", detail=str(exc)
            ).model_dump(),
        ) from exc


# ---------------------------------------------------------------------------
# Audit + curated reports + reconciliation (additive, read-only)
# ---------------------------------------------------------------------------


@paisa_router.get("/audit", response_model=ExtensionAuditResponse)
async def paisa_audit(
    session: AsyncSession = Depends(get_session),
) -> ExtensionAuditResponse:
    """Recent ExtensionRun rows + last success/error for the Paisa extension.

    Read-only; never surfaces credentials or raw journal text.
    """
    return await surface.audit_view(session)


@paisa_router.get("/reports/{report}")
async def paisa_report(request: Request, report: str) -> JSONResponse:
    """One curated Paisa report through the per-app TTL cache.

    Disabled mode → no upstream calls. A failure is a typed ``ok=False`` body,
    never a 500. The five report pages read this; reconciliation reads its own
    assets/liabilities balances through the same cache.
    """
    try:
        summary: PaisaReportSummary = await surface.report_summary(
            request.app.state, report
        )
    except Exception as exc:  # noqa: BLE001 — optional-extension isolation
        logger.warning("Paisa report %s failed: %s", report, exc, exc_info=True)
        body = ExtensionErrorResponse(error="paisa_report_failed", detail=str(exc))
        return JSONResponse(body.model_dump(), status_code=200)
    return JSONResponse(summary.model_dump())


@paisa_router.get("/reconciliation", response_model=PaisaReconcileResponse)
async def paisa_reconciliation(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> PaisaReconcileResponse:
    """Read-only reconciliation view. Never writes a core row."""
    return await surface.reconciliation_view(session, request.app.state)


def _typed_error(code: str, detail: str) -> JSONResponse:
    body = ExtensionErrorResponse(error=code, detail=detail)
    return JSONResponse(body.model_dump(), status_code=200)


router.include_router(paisa_router)
