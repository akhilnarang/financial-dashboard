"""Dashboard HTML routes."""

from __future__ import annotations

import logging
from datetime import date
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
from bank_email_fetcher.integrations.parsers import get_supported_banks
from bank_email_fetcher.services.statements.cc import parse_cc_amount, parse_cc_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = get_templates()
SUPPORTED_BANKS = get_supported_banks()
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
        for upload in uploads:
            if upload.account_id not in latest_upload_by_account:
                latest_upload_by_account[upload.account_id] = upload

    rows = []
    grand_total = Decimal(0)
    cards_with_outstanding = 0
    cards_paid = 0
    cards_missing = 0
    cards_error = 0

    for account in credit_card_accounts:
        row: dict = {
            "account": account,
            "status": "missing",
            "status_label": "missing",
            "amount_due": None,
            "paid_amount": None,
            "outstanding": None,
            "due_date_display": None,
            "overdue": False,
        }
        rows.append(row)

        upload = latest_upload_by_account.get(account.id)
        if upload is None:
            cards_missing += 1
            continue

        payment_status = upload.payment_status
        paid_amount = upload.payment_paid_amount or Decimal(0)
        row["paid_amount"] = paid_amount if paid_amount else None

        if upload.due_date:
            try:
                parsed_due = parse_cc_date(upload.due_date)
                row["due_date_display"] = parsed_due.strftime("%d %b %Y")
                if parsed_due < today and payment_status != PaymentStatus.PAID:
                    row["overdue"] = True
            except Exception:
                # Why: due_date comes from raw PDF parsing and can be malformed — show the raw string rather than hiding it.
                row["due_date_display"] = upload.due_date

        try:
            amount_due = (
                parse_cc_amount(upload.total_amount_due)
                if upload.total_amount_due
                else None
            )
        except Exception:
            amount_due = None

        if amount_due is None:
            cards_error += 1
            row["status"] = "error"
            row["status_label"] = "parse error"
            continue

        row["amount_due"] = amount_due

        if payment_status == PaymentStatus.PAID:
            cards_paid += 1
            row["status"] = "paid"
            row["status_label"] = "paid"
            row["paid_amount"] = paid_amount
            row["outstanding"] = Decimal(0)
            row["overdue"] = False
            continue

        # Why: clamp negative outstanding (overpayment credit balance) to 0 so it doesn't reduce the grand total.
        outstanding = max(amount_due - paid_amount, Decimal(0))
        # Why: column is stored as String, so loaded rows give back a str, not a PaymentStatus instance.
        status_value = str(payment_status) if payment_status else PaymentStatus.UNPAID.value
        row["status"] = status_value
        row["status_label"] = status_value.replace("_", " ")
        row["paid_amount"] = paid_amount if paid_amount else None
        row["outstanding"] = outstanding
        if outstanding > 0:
            cards_with_outstanding += 1
            grand_total += outstanding

    cc_outstanding = {
        "rows": rows,
        "summary": {
            "grand_total": grand_total,
            "total_cards": len(credit_card_accounts),
            "cards_with_outstanding": cards_with_outstanding,
            "cards_paid": cards_paid,
            "cards_missing": cards_missing,
            "cards_error": cards_error,
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
