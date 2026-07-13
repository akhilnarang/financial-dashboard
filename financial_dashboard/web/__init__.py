"""Web router aggregation."""

from fastapi import APIRouter

from financial_dashboard.web import (
    accounts,
    bank_statements,
    cas,
    cashflow,
    dashboard,
    emails,
    polling,
    rules,
    settings,
    sms,
    sources,
    statements,
    transactions,
    networth,
)

router = APIRouter()
router.include_router(dashboard.router)
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
router.include_router(bank_statements.router)
router.include_router(statements.router)
router.include_router(polling.router)


def get_router() -> APIRouter:
    return router


__all__ = ["get_router", "router"]
