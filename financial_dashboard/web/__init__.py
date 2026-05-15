"""Web router aggregation."""

from fastapi import APIRouter

from financial_dashboard.web import (
    accounts,
    bank_statements,
    dashboard,
    emails,
    polling,
    rules,
    settings,
    sources,
    statements,
    transactions,
)

router = APIRouter()
router.include_router(dashboard.router)
router.include_router(transactions.router)
router.include_router(emails.router)
router.include_router(accounts.router)
router.include_router(sources.router)
router.include_router(rules.router)
router.include_router(settings.router)
router.include_router(bank_statements.router)
router.include_router(statements.router)
router.include_router(polling.router)


def get_router() -> APIRouter:
    return router


__all__ = ["get_router", "router"]
