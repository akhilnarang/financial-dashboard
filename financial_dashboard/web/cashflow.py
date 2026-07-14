"""Cashflow HTML route.

Server-renders every figure and every drill-through link from the same service
the JSON API calls, so the page is complete and correct with JavaScript off; the
two SVG charts are progressive enhancement over that.

The breakdown chart draws the range the page already shows, so it is handed the
summary the page was rendered from, serialized into the document. Fetching it
back would re-run the same eight aggregate queries for numbers already in the
response. The trend chart *does* fetch, because it is a trailing-12-month series
that ignores the selected range and so is not on the page at all.

The range is normalized by the service's ``resolve_range``, which is what keeps
the page, the charts and the ``/transactions`` links it emits all describing the
one range: whatever the query string said, the *resolved* bounds are the ones
echoed into the form, the payload and every link.
"""

import datetime

from fastapi import APIRouter, Depends, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.dates import DEFAULT_TREND_MONTHS, month_start
from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.services.cashflow.report import (
    cashflow_summary,
    resolve_range,
    trend_ranges,
)

templates = get_templates()
router = APIRouter()


def _presets(today: datetime.date) -> list[dict[str, str]]:
    """The one-click ranges, each pre-resolved to concrete ISO bounds.

    Built here rather than in the template so the page never has to do date
    arithmetic, and so a preset link is the same kind of URL as a hand-typed
    range: two explicit bounds the route re-resolves like any other.
    """
    this_month = month_start(today)
    last_month = month_start(today, 1)
    return [
        {
            "label": "This month",
            "date_from": this_month.isoformat(),
            "date_to": today.isoformat(),
        },
        {
            "label": "Last month",
            "date_from": last_month.isoformat(),
            # The day before this month's first is the last day of last month,
            # with no month-length special cases.
            "date_to": (this_month - datetime.timedelta(days=1)).isoformat(),
        },
        {
            "label": "Last 3 months",
            "date_from": month_start(today, 2).isoformat(),
            "date_to": today.isoformat(),
        },
        {
            "label": "This year",
            "date_from": today.replace(month=1, day=1).isoformat(),
            "date_to": today.isoformat(),
        },
    ]


@router.get("/cashflow", response_class=HTMLResponse)
async def cashflow_index(
    request: FastAPIRequest,
    date_from: str | None = None,
    date_to: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """Render the cashflow report for one range: tiles, charts, tables, footnotes.

    Both bounds are optional ISO ``YYYY-MM-DD`` strings; anything missing or
    unparseable is defaulted by ``resolve_range`` rather than rejected, and the
    *resolved* bounds are what the form, the charts and every drill-through link
    are built from.

    The summary is aggregated once, here, and serialized into the page for the
    breakdown chart to hydrate from. The chart must not re-fetch it: it would be
    the identical range, i.e. the same eight aggregate queries run a second time to
    produce numbers already on the page. Only the trend is fetched, because its
    trailing-window figures are genuinely not on the page.
    """
    start, end = resolve_range(date_from, date_to)
    summary = await cashflow_summary(session, start, end)
    return templates.TemplateResponse(
        request,
        "cashflow/index.html",
        {
            "active_page": "cashflow",
            "summary": summary,
            # The resolved bounds, not the raw query params: an unparseable
            # ?date_from= must not leak back into the form or the drill links.
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
            "presets": _presets(datetime.date.today()),
            "trend_months": DEFAULT_TREND_MONTHS,
            # Clicking a month on the trend chart sets the range to that month, so
            # the chart needs each month's bounds. They are computed here, from the
            # same window the trend endpoint uses, rather than derived in
            # JavaScript: a month's days are a server fact, and a clicked month must
            # select exactly the days whose bars were drawn.
            "trend_ranges": trend_ranges(DEFAULT_TREND_MONTHS),
        },
    )
