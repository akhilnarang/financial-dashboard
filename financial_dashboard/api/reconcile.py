"""Read-only bank reconciliation endpoint."""

import datetime

from fastapi import APIRouter

from financial_dashboard.api.query import validate_date_range
from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.schemas.reconcile import ReconcileReport
from financial_dashboard.services.reconcile import reconcile

router = APIRouter()


@router.get("/reconcile")
async def get_reconcile(
    session: AsyncSessionDep,
    date_from: datetime.date,
    date_to: datetime.date,
) -> ReconcileReport:
    """Reconcile the bank accounts for one inclusive date range.

    Return the statement gaps, the months without a statement, the unpaired
    self-transfers and the duplicate candidates. Read only.
    """
    validate_date_range(date_from, date_to)
    return await reconcile(session, date_from, date_to)
