"""Dashboard HTML routes."""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from bank_email_fetcher.core.deps import get_session
from bank_email_fetcher.core.templating import get_templates
from bank_email_fetcher.db import (
    Account,
    Email,
    FetchRule,
    PaymentStatus,
    StatementUpload,
    Transaction,
)
from bank_email_fetcher.services.statements.cc import parse_cc_amount, parse_cc_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_session),
):
    fetch_service = getattr(request.app.state, "fetch_service", None)
    today = date.today()

    total_emails = (await session.execute(select(func.count(Email.id)))).scalar() or 0
    active_rules = (
        await session.execute(
            select(func.count(FetchRule.id)).where(FetchRule.enabled.is_(True))
        )
    ).scalar() or 0

    result = await session.execute(
        select(Transaction)
        .order_by(
            Transaction.transaction_date.desc().nullslast(), Transaction.id.desc()
        )
        .limit(20)
    )
    transactions = result.scalars().all()

    credit_card_accounts = (
        (
            await session.execute(
                select(Account)
                .where(
                    Account.type == "credit_card",
                    Account.active.is_not(False),
                )
                .order_by(Account.label)
            )
        )
        .scalars()
        .all()
    )

    account_ids = [account.id for account in credit_card_accounts]
    latest_upload_by_account: dict[int, StatementUpload] = {}
    if account_ids:
        uploads = (
            (
                await session.execute(
                    select(StatementUpload)
                    .where(StatementUpload.account_id.in_(account_ids))
                    .order_by(
                        StatementUpload.account_id,
                        StatementUpload.created_at.desc(),
                        StatementUpload.id.desc(),
                    )
                )
            )
            .scalars()
            .all()
        )

        # Why: backfilled older statements can have a newer created_at than the true-latest
        # cycle, so primary sort is parsed due_date. When due_date is unparseable we fall
        # back to created_at on that side only — otherwise a newer upload with a malformed
        # date would lose to an older parseable one. All keys are normalized to a date via
        # created_at.date() fallback so comparisons don't mix date and datetime.
        def upload_sort_key(upload: StatementUpload) -> tuple[date, datetime, int]:
            created = upload.created_at or datetime.min
            primary_date = created.date()
            if upload.due_date:
                try:
                    primary_date = parse_cc_date(upload.due_date)
                except Exception:
                    pass
            return (primary_date, created, upload.id)

        for upload in uploads:
            current = latest_upload_by_account.get(upload.account_id)
            if current is None or upload_sort_key(upload) > upload_sort_key(current):
                latest_upload_by_account[upload.account_id] = upload

    outstanding_rows: list[dict] = []
    paid_rows: list[dict] = []
    grand_total = Decimal(0)
    cards_with_outstanding = 0
    cards_paid = 0

    for account in credit_card_accounts:
        if (upload := latest_upload_by_account.get(account.id)) is None:
            continue

        payment_status = upload.payment_status
        paid_amount = upload.payment_paid_amount or Decimal(0)

        try:
            amount_due = (
                parse_cc_amount(total_amount_due)
                if (total_amount_due := upload.total_amount_due)
                else None
            )
        except Exception:
            amount_due = None

        if amount_due is None:
            continue

        row: dict = {
            "account": account,
            "amount_due": amount_due,
            "paid_amount": paid_amount if paid_amount else None,
            "outstanding": None,
            "due_date_display": None,
            "overdue": False,
            "status": "unpaid",
            "status_label": "unpaid",
        }

        if due_date := upload.due_date:
            try:
                parsed_due = parse_cc_date(due_date)
                row["due_date_display"] = parsed_due.strftime("%d %b %Y")
                if parsed_due < today and payment_status != PaymentStatus.PAID:
                    row["overdue"] = True
            except Exception:
                # Why: due_date comes from raw PDF parsing and can be malformed — show the raw string rather than hiding it.
                row["due_date_display"] = due_date

        if payment_status == PaymentStatus.PAID or amount_due <= 0:
            cards_paid += 1
            row["status"] = "paid"
            row["status_label"] = "paid"
            row["paid_amount"] = paid_amount
            row["outstanding"] = Decimal(0)
            row["overdue"] = False
            paid_rows.append(row)
            continue

        # Why: clamp negative outstanding (overpayment credit balance) to 0 so it doesn't reduce the grand total.
        outstanding = max(amount_due - paid_amount, Decimal(0))
        # Why: column is stored as String, so loaded rows give back a str, not a PaymentStatus instance.
        status_value = (
            str(payment_status) if payment_status else PaymentStatus.UNPAID.value
        )
        row["status"] = status_value
        row["status_label"] = status_value.replace("_", " ")
        row["outstanding"] = outstanding
        outstanding_rows.append(row)
        if outstanding > 0:
            cards_with_outstanding += 1
            grand_total += outstanding

    cc_outstanding = {
        "outstanding_rows": outstanding_rows,
        "paid_rows": paid_rows,
        "summary": {
            "grand_total": grand_total,
            "cards_with_outstanding": cards_with_outstanding,
            "cards_paid": cards_paid,
            "has_any": bool(outstanding_rows or paid_rows),
        },
    }

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "active_page": "dashboard",
            "poll_status": (
                fetch_service.get_poll_status()
                if fetch_service
                else {
                    "state": "idle",
                    "started_at": None,
                    "finished_at": None,
                    "last_stats": None,
                    "last_error": None,
                    "progress": None,
                }
            ),
            "ops_stats": {
                "total_emails": total_emails,
                "active_rules": active_rules,
            },
            "transactions": transactions,
            "cc_outstanding": cc_outstanding,
        },
    )
