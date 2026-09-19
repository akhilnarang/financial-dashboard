"""Web router aggregation."""

from fastapi import APIRouter

from financial_dashboard.web import (
    accounts,
    audit,
    bank_statements,
    cas,
    cashflow,
    dashboard,
    emails,
    extensions,
    polling,
    rules,
    settings,
    sms,
    sources,
    statements,
    transactions,
    networth,
)


def get_router(*, paisa_enabled: bool = True) -> APIRouter:
    router = APIRouter()
    router.include_router(dashboard.router)
    router.include_router(audit.router)
    router.include_router(transactions.router)
    router.include_router(cashflow.router)
    router.include_router(emails.router)
    router.include_router(accounts.router)
    router.include_router(networth.router)
    router.include_router(cas.router)
    router.include_router(sources.router)
    router.include_router(rules.router)
    router.include_router(settings.router)
    router.include_router(sms.router)
    router.include_router(extensions.get_router(paisa_enabled=paisa_enabled))
    # Preserve route ordering: bank statement detail paths must precede the
    # generic statement detail route.
    router.include_router(bank_statements.router)
    router.include_router(statements.router)
    router.include_router(polling.router)
    return router


# Backwards-compatible full router for direct imports.
router = get_router(paisa_enabled=True)


__all__ = ["get_router", "router"]
