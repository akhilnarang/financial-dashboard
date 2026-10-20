"""Read-only HTML view of open receivables."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db.enums import ReceivableStatus
from financial_dashboard.services.receivables import list_receivables, receivable_read

router = APIRouter()
templates = get_templates()


@router.get("/receivables", response_class=HTMLResponse)
async def receivables_page(
    request: Request, session: AsyncSession = Depends(get_session)
) -> Response:
    """Show who still owes money, with paid and outstanding sums.

    Args:
        request: The page request.
        session: The request session.

    Returns:
        The rendered page.
    """
    rows = await list_receivables(session, ReceivableStatus.open)
    return templates.TemplateResponse(
        request,
        "receivables.html",
        {
            "active_page": "receivables",
            "receivables": [receivable_read(row) for row in rows],
        },
    )
