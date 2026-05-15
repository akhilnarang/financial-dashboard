# ty: ignore
"""Transaction HTML routes."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.core.templating import get_templates
from financial_dashboard.db import (
    Account,
    Card,
    Email,
    Transaction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()

PAGE_SIZE = 50
SORT_COLUMNS = {
    "amount": Transaction.amount,
    "bank": Transaction.bank,
    "counterparty": Transaction.counterparty,
    "date": Transaction.transaction_date,
}


@router.get("/transactions", response_class=HTMLResponse)
async def transaction_list(
    request: FastAPIRequest,
    bank: Annotated[str | None, Query(description="Filter by bank name")] = None,
    account_id: Annotated[str | None, Query(description="Filter by account ID")] = None,
    card_id: Annotated[str | None, Query(description="Filter by card ID")] = None,
    direction: Annotated[
        str | None, Query(description="Filter by direction: debit or credit")
    ] = None,
    date_from: Annotated[
        str | None, Query(description="Transaction date on/after (YYYY-MM-DD)")
    ] = None,
    date_to: Annotated[
        str | None, Query(description="Transaction date on/before (YYYY-MM-DD)")
    ] = None,
    sort: Annotated[str, Query(description="Sort column")] = "date",
    order: Annotated[str, Query(description="Sort order: asc or desc")] = "desc",
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Transaction)

    if bank:
        stmt = stmt.where(Transaction.bank == bank)
    if account_id:
        try:
            stmt = stmt.where(Transaction.account_id == int(account_id))
        except ValueError:
            pass
    if card_id:
        try:
            stmt = stmt.where(Transaction.card_id == int(card_id))
        except ValueError:
            pass
    if direction:
        stmt = stmt.where(Transaction.direction == direction)
    if date_from:
        try:
            stmt = stmt.where(
                Transaction.transaction_date >= date.fromisoformat(date_from)
            )
        except ValueError:
            pass
    if date_to:
        try:
            stmt = stmt.where(
                Transaction.transaction_date <= date.fromisoformat(date_to)
            )
        except ValueError:
            pass

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_count = (await session.execute(count_stmt)).scalar()

    sort_col = SORT_COLUMNS.get(sort, Transaction.transaction_date)
    if order not in ("asc", "desc"):
        order = "desc"
    if order == "asc":
        stmt = stmt.order_by(sort_col.asc().nullslast(), Transaction.id.asc())
    else:
        stmt = stmt.order_by(sort_col.desc().nullslast(), Transaction.id.desc())

    total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
    if page < 1:
        page = 1
    if page > total_pages and total_pages > 0:
        page = total_pages
    stmt = stmt.limit(PAGE_SIZE).offset((page - 1) * PAGE_SIZE)

    result = await session.execute(stmt)
    transactions = result.scalars().all()

    bank_result = await session.execute(select(Transaction.bank).distinct())
    banks = sorted([row[0] for row in bank_result.all()])

    accounts = (
        (
            await session.execute(
                select(Account)
                .where(Account.active.is_(True))
                .order_by(Account.bank, Account.label)
            )
        )
        .scalars()
        .all()
    )

    cards = (
        (
            await session.execute(
                select(Card).where(Card.active.is_(True)).order_by(Card.card_mask)
            )
        )
        .scalars()
        .all()
    )

    # Build JSON for dependent dropdowns
    cards_by_account: dict[int, list] = {}
    for c in cards:
        cards_by_account.setdefault(c.account_id, []).append(
            {
                "id": c.id,
                "mask": c.card_mask,
                "label": c.label or c.card_mask,
            }
        )
    accounts_by_bank: dict[str, list] = {}
    for a in accounts:
        accounts_by_bank.setdefault(a.bank, []).append(
            {
                "id": a.id,
                "label": a.label,
                "type": a.type,
            }
        )

    # Build base query string for pagination/sort links
    filters = {
        "bank": bank,
        "account_id": account_id,
        "card_id": card_id,
        "direction": direction,
        "date_from": date_from,
        "date_to": date_to,
    }
    base_qs = urlencode({k: v for k, v in filters.items() if v})

    # Page window: show pages around current
    def page_window():
        pages = set()
        pages.add(1)
        pages.add(total_pages)
        for p in range(max(1, page - 2), min(total_pages, page + 2) + 1):
            pages.add(p)
        return sorted(pages)

    return templates.TemplateResponse(
        request,
        "transactions.html",
        {
            "active_page": "transactions",
            "transactions": transactions,
            "banks": banks,
            "accounts": accounts,
            "accounts_json": json.dumps(accounts_by_bank),
            "cards_json": json.dumps(cards_by_account),
            "filters": filters,
            "sort": sort,
            "order": order,
            "page": page,
            "total_count": total_count,
            "total_pages": total_pages,
            "page_size": PAGE_SIZE,
            "page_window": page_window(),
            "base_qs": base_qs,
        },
    )


async def _load_transaction(
    session: AsyncSession, txn_id: int
) -> tuple[Transaction, Email | None, Account | None] | None:
    """Load a Transaction with its linked Email and Account, or None."""
    result = await session.execute(
        select(Transaction, Email, Account)
        .outerjoin(Email, Transaction.email_id == Email.id)
        .outerjoin(Account, Transaction.account_id == Account.id)
        .where(Transaction.id == txn_id)
    )
    row = result.first()
    if not row:
        return None
    return tuple(row)  # type: ignore[return-value]


@router.get("/transactions/{txn_id}/detail", response_class=HTMLResponse)
async def transaction_detail(
    txn_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    loaded = await _load_transaction(session, txn_id)
    if loaded is None:
        return HTMLResponse("<p>Transaction not found.</p>", 404)
    txn, email, account = loaded
    return templates.TemplateResponse(
        request,
        "partials/transaction_detail.html",
        {"txn": txn, "email": email, "account": account},
    )


@router.get("/transactions/{txn_id}", response_class=HTMLResponse)
async def transaction_page(
    txn_id: int,
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    loaded = await _load_transaction(session, txn_id)
    if loaded is None:
        return HTMLResponse("<p>Transaction not found.</p>", 404)
    txn, email, account = loaded
    return templates.TemplateResponse(
        request,
        "transaction_page.html",
        {
            "active_page": "transactions",
            "txn": txn,
            "email": email,
            "account": account,
        },
    )
