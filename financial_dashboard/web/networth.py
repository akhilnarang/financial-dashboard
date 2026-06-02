"""Net-worth HTML routes."""

import datetime
from collections.abc import Awaitable, Callable
from decimal import Decimal
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db.models import BalanceSnapshot, ManualItem
from financial_dashboard.services.manual_items import (
    create_item,
    deactivate,
    delete_snapshot,
    edit_snapshot,
    update_value,
)
from financial_dashboard.services.networth import current_networth

templates = get_templates()
router = APIRouter()


@router.get("/networth", response_class=HTMLResponse)
async def networth_index(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    summary = await current_networth(session)
    return templates.TemplateResponse(
        request,
        "networth/index.html",
        {"active_page": "networth", "summary": summary},
    )


@router.get("/networth/manual", response_class=HTMLResponse)
async def manual_items_index(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    items = (
        (
            await session.execute(
                select(ManualItem).order_by(ManualItem.active.desc(), ManualItem.name)
            )
        )
        .scalars()
        .all()
    )

    # Latest snapshot per manual_item (current value + as-of date for the table).
    latest_subq = (
        select(
            BalanceSnapshot.manual_item_id,
            func.max(BalanceSnapshot.as_of_date).label("max_date"),
        )
        .where(BalanceSnapshot.manual_item_id.is_not(None))
        .group_by(BalanceSnapshot.manual_item_id)
        .subquery()
    )
    latest_rows = (
        (
            await session.execute(
                select(BalanceSnapshot)
                .join(
                    latest_subq,
                    (BalanceSnapshot.manual_item_id == latest_subq.c.manual_item_id)
                    & (BalanceSnapshot.as_of_date == latest_subq.c.max_date),
                )
                # Deterministic tiebreak when an item has >1 snapshot on the
                # same as_of_date: ascending id means the highest id (most
                # recently inserted) wins the dict-comprehension last-write.
                .order_by(BalanceSnapshot.id)
            )
        )
        .scalars()
        .all()
    )
    latest_by_item: dict[int, BalanceSnapshot] = {
        s.manual_item_id: s for s in latest_rows if s.manual_item_id is not None
    }

    all_rows = (
        (
            await session.execute(
                select(BalanceSnapshot)
                .where(BalanceSnapshot.manual_item_id.is_not(None))
                .order_by(BalanceSnapshot.as_of_date.desc(), BalanceSnapshot.id.desc())
            )
        )
        .scalars()
        .all()
    )
    history_by_item: dict[int, list[BalanceSnapshot]] = {}
    for s in all_rows:
        if s.manual_item_id is None:
            continue
        history_by_item.setdefault(s.manual_item_id, []).append(s)

    return templates.TemplateResponse(
        request,
        "networth/manual.html",
        {
            "active_page": "networth",
            "items": items,
            "latest_by_item": latest_by_item,
            "history_by_item": history_by_item,
            "today_iso": datetime.date.today().isoformat(),
            "error": request.query_params.get("error"),
        },
    )


@router.post("/networth/manual")
async def manual_item_create(
    name: str = Form(...),
    kind: str = Form(...),
    category: str = Form("other"),
    value: Decimal = Form(...),
    as_of_date: datetime.date = Form(...),
    notes: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    await create_item(
        session,
        name=name,
        kind=kind,
        category=category,
        value=value,
        as_of_date=as_of_date,
        notes=notes,
    )
    await session.commit()
    return RedirectResponse(url="/networth/manual", status_code=303)


@router.post("/networth/manual/{item_id}/value")
async def manual_item_update_value(
    item_id: int,
    value: Decimal = Form(...),
    as_of_date: datetime.date = Form(...),
    session: AsyncSession = Depends(get_session),
):
    await update_value(session, item_id=item_id, value=value, as_of_date=as_of_date)
    await session.commit()
    return RedirectResponse(url="/networth/manual", status_code=303)


@router.post("/networth/manual/{item_id}/deactivate")
async def manual_item_deactivate(
    item_id: int,
    session: AsyncSession = Depends(get_session),
):
    await deactivate(session, item_id=item_id)
    await session.commit()
    return RedirectResponse(url="/networth/manual", status_code=303)


async def _snapshot_action(
    session: AsyncSession, action: Callable[[], Awaitable[object]]
) -> RedirectResponse:
    """Run a snapshot mutation and redirect back to the manual page.

    Commits and redirects clean on success; on a ValueError (e.g. a date
    collision) redirects back with the message in an ``?error=`` query param
    for the page to render, without committing.
    """
    try:
        await action()
    except ValueError as exc:
        return RedirectResponse(
            url=f"/networth/manual?error={quote(str(exc))}", status_code=303
        )
    await session.commit()
    return RedirectResponse(url="/networth/manual", status_code=303)


@router.post("/networth/manual/snapshot/{snapshot_id}/edit")
async def manual_snapshot_edit(
    snapshot_id: int,
    value: Decimal = Form(...),
    as_of_date: datetime.date = Form(...),
    session: AsyncSession = Depends(get_session),
):
    return await _snapshot_action(
        session,
        lambda: edit_snapshot(
            session, snapshot_id=snapshot_id, value=value, as_of_date=as_of_date
        ),
    )


@router.post("/networth/manual/snapshot/{snapshot_id}/delete")
async def manual_snapshot_delete(
    snapshot_id: int,
    session: AsyncSession = Depends(get_session),
):
    return await _snapshot_action(
        session, lambda: delete_snapshot(session, snapshot_id=snapshot_id)
    )
