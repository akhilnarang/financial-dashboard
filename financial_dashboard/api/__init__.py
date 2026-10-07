"""API router aggregation."""

from fastapi import APIRouter

from .accounts import router as accounts_router
from .cas import router as cas_router
from .cashflow import router as cashflow_router
from .emails import router as emails_router
from .extensions import get_router as get_extensions_router
from .networth import router as networth_router
from .polling import router as polling_router
from .reconcile import router as reconcile_router
from .sms import router as sms_router
from .sources import router as sources_router
from .statements import router as statements_router
from .system import router as system_router
from .transactions import router as transactions_router


def get_router(*, paisa_enabled: bool) -> APIRouter:
    router = APIRouter(prefix="/api")
    router.include_router(accounts_router)
    router.include_router(cas_router)
    router.include_router(cashflow_router)
    router.include_router(emails_router)
    router.include_router(get_extensions_router(paisa_enabled=paisa_enabled))
    router.include_router(networth_router)
    router.include_router(polling_router)
    router.include_router(reconcile_router)
    router.include_router(transactions_router)
    router.include_router(sources_router)
    router.include_router(sms_router)
    router.include_router(statements_router)
    router.include_router(system_router)
    return router


__all__ = ["get_router"]
