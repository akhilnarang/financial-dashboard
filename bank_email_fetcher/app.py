"""FastAPI web service for bank-email-fetcher.

Provides:
- Web UI routes (Jinja2 templates) for the dashboard, transactions, emails,
  accounts, cards, email sources, fetch rules, and CC statement reconciliation.
- REST API endpoints under /api/ for poll status, transaction notes, and
  source connectivity tests.
- Background asyncio poll loop that calls fetcher.poll_all() on a fixed interval.
- Lifespan handler that initialises the database and starts/stops the poll loop.

All HTML routes follow the POST-Redirect-GET pattern. API routes return JSON.

Start with:
    uv run fastapi dev

Note: APScheduler is intentionally NOT used. The plain asyncio loop is
sufficient for single-user, single-process deployment. Revisit if misfire
handling, job persistence, or multiple scheduled jobs are ever needed.
"""

import asyncio
import imaplib
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from decimal import InvalidOperation
from pathlib import Path
from urllib.request import Request, urlopen

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Query,
    Request as FastAPIRequest,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, func, or_, select, update

from bank_email_fetcher.config import settings
from bank_email_fetcher.deps import verify_credentials
from bank_email_fetcher.crypto import encrypt_credentials, decrypt_credentials
from urllib.parse import urlencode

from bank_email_fetcher.db import (
    Account,
    BankStatementUpload,
    Card,
    Email,
    EmailSource,
    FetchRule,
    PaymentStatus,
    StatementUpload,
    Transaction,
    async_session,
    init_db,
)
from bank_email_fetcher.fetcher import (
    get_poll_status,
    poll_all,
    _extract_html_body,
    _extract_text_body,
    _fetch_gmail_single_sync,
    _fetch_fastmail_single_sync,
    _process_email,
)
from bank_email_fetcher.linker import build_link_context, link_transaction
from bank_email_parser import SUPPORTED_BANKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def format_inr_compact(value) -> str:
    amount = value or 0
    abs_amount = abs(float(amount))
    if abs_amount >= 1_00_00_000:  # 1 crore
        scaled, suffix = float(amount) / 1_00_00_000, "Cr"
    elif abs_amount >= 1_00_000:  # 1 lakh
        scaled, suffix = float(amount) / 1_00_000, "L"
    elif abs_amount >= 1_000:
        scaled, suffix = float(amount) / 1_000, "K"
    else:
        return f"₹{float(amount):,.2f}"
    decimals = 1 if abs(scaled) >= 10 else 2
    formatted = f"{scaled:.{decimals}f}".rstrip("0").rstrip(".")
    return f"₹{formatted}{suffix}"


templates.env.filters["inr_compact"] = format_inr_compact

poll_task = None


async def _poll_loop() -> None:
    """Background loop that polls email sources on a fixed interval.

    TODO: revisit APScheduler if we need catch-up semantics, job persistence,
    or multiple scheduled jobs. Plain asyncio loop is sufficient for single-user,
    single-process deployment.
    """
    while True:
        try:
            await poll_all()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background poll failed")

        try:
            from bank_email_fetcher.reminders import check_and_send_reminders

            if sent := await check_and_send_reminders():
                logger.info("Sent %d payment reminder(s)", sent)
        except Exception:
            logger.exception("Reminder check failed")

        from bank_email_fetcher.settings_service import get_setting_int

        interval = max(1, get_setting_int("poll_interval_minutes", 15)) * 60
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from bank_email_fetcher.settings_service import start_services, stop_services

    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready")

    await start_services()

    if not settings.auth_enabled:
        logger.warning(
            "No AUTH_USERNAME/AUTH_PASSWORD set — running without authentication. "
            "Only run on a trusted network or behind a reverse proxy with auth."
        )

    loop_task = asyncio.create_task(_poll_loop())

    yield

    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass

    await stop_services()


app = FastAPI(
    title="Email Fetcher",
    lifespan=lifespan,
    dependencies=[Depends(verify_credentials)],
)
app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)

api_router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: FastAPIRequest):
    today = date.today()
    month_start = today.replace(day=1)

    async with async_session() as session:
        # Month-to-date financial stats
        row = (
            await session.execute(
                select(
                    func.count(Transaction.id),
                    func.coalesce(
                        func.sum(
                            case(
                                (Transaction.direction == "debit", Transaction.amount),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (Transaction.direction == "credit", Transaction.amount),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                ).where(Transaction.transaction_date >= month_start)
            )
        ).one()
        month_txns, month_debit, month_credit = row
        net_flow = month_credit - month_debit

        # Operational stats
        total_emails = (
            await session.execute(select(func.count(Email.id)))
        ).scalar() or 0
        active_rules = (
            await session.execute(
                select(func.count(FetchRule.id)).where(FetchRule.enabled.is_(True))
            )
        ).scalar() or 0

        # Recent transactions
        result = await session.execute(
            select(Transaction)
            .order_by(
                Transaction.transaction_date.desc().nullslast(), Transaction.id.desc()
            )
            .limit(20)
        )
        transactions = result.scalars().all()

    period_label = month_start.strftime("%B %Y")

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "active_page": "dashboard",
            "poll_status": get_poll_status(),
            "stats": {
                "month_debit": month_debit,
                "month_credit": month_credit,
                "month_transactions": month_txns,
                "net_flow": net_flow,
                "period_label": period_label,
            },
            "ops_stats": {
                "total_emails": total_emails,
                "active_rules": active_rules,
            },
            "transactions": transactions,
        },
    )


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

PAGE_SIZE = 50

SORT_COLUMNS = {
    "date": Transaction.transaction_date,
    "amount": Transaction.amount,
    "bank": Transaction.bank,
    "counterparty": Transaction.counterparty,
}


@app.get("/transactions", response_class=HTMLResponse)
async def transaction_list(
    request: FastAPIRequest,
    bank: Annotated[str | None, Query(description="Filter by bank name")] = None,
    account_id: Annotated[
        str | None, Query(description="Filter by account ID")
    ] = None,
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
):
    async with async_session() as session:
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

        # Count total matching rows
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_count = (await session.execute(count_stmt)).scalar()

        # Sorting
        sort_col = SORT_COLUMNS.get(sort, Transaction.transaction_date)
        if order not in ("asc", "desc"):
            order = "desc"
        if order == "asc":
            stmt = stmt.order_by(sort_col.asc().nullslast(), Transaction.id.asc())
        else:
            stmt = stmt.order_by(sort_col.desc().nullslast(), Transaction.id.desc())

        # Pagination
        total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
        if page < 1:
            page = 1
        if page > total_pages and total_pages > 0:
            page = total_pages
        stmt = stmt.limit(PAGE_SIZE).offset((page - 1) * PAGE_SIZE)

        result = await session.execute(stmt)
        transactions = result.scalars().all()

        # Filter dropdown data
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


# ---------------------------------------------------------------------------
# Transaction Notes
# ---------------------------------------------------------------------------


@api_router.post("/transactions/{txn_id}/note")
async def update_note(txn_id: int, request: FastAPIRequest):
    body = await request.json()
    note = body.get("note", "").strip()
    async with async_session() as session:
        txn = await session.get(Transaction, txn_id)
        if not txn:
            return JSONResponse({"error": "Not found"}, 404)
        txn.note = note or None
        await session.commit()
    return JSONResponse({"ok": True, "note": note})


# ---------------------------------------------------------------------------
# Transaction Detail
# ---------------------------------------------------------------------------


@app.get("/transactions/{txn_id}/detail", response_class=HTMLResponse)
async def transaction_detail(txn_id: int, request: FastAPIRequest):
    async with async_session() as session:
        result = await session.execute(
            select(Transaction, Email, Account)
            .outerjoin(Email, Transaction.email_id == Email.id)
            .outerjoin(Account, Transaction.account_id == Account.id)
            .where(Transaction.id == txn_id)
        )
        row = result.first()
        if not row:
            return HTMLResponse("<p>Transaction not found.</p>", 404)
        txn, email, account = row
    return templates.TemplateResponse(
        request,
        "partials/transaction_detail.html",
        {
            "txn": txn,
            "email": email,
            "account": account,
        },
    )


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------

@app.get("/emails", response_class=HTMLResponse)
async def email_list(
    request: FastAPIRequest,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    bank: Annotated[str | None, Query(description="Filter by bank (via rule)")] = None,
    provider: Annotated[
        str | None, Query(description="Filter by email provider")
    ] = None,
    status: Annotated[
        str | None, Query(description="Filter by processing status")
    ] = None,
    date_from: Annotated[
        str | None, Query(description="Received on/after (YYYY-MM-DD)")
    ] = None,
    date_to: Annotated[
        str | None, Query(description="Received on/before (YYYY-MM-DD)")
    ] = None,
    q: Annotated[
        str | None, Query(description="Case-insensitive search over sender and subject")
    ] = None,
):
    async with async_session() as session:
        stmt = select(Email)
        needs_rule_join = bool(bank)
        if needs_rule_join:
            stmt = stmt.join(FetchRule, Email.rule_id == FetchRule.id).where(
                FetchRule.bank == bank
            )
        if provider:
            stmt = stmt.where(Email.provider == provider)
        if status:
            stmt = stmt.where(Email.status == status)
        if date_from:
            try:
                stmt = stmt.where(Email.received_at >= date.fromisoformat(date_from))
            except ValueError:
                pass
        if date_to:
            try:
                end_of_day = datetime.combine(
                    date.fromisoformat(date_to), datetime.max.time()
                )
                stmt = stmt.where(Email.received_at <= end_of_day)
            except ValueError:
                pass
        if q:
            like = f"%{q}%"
            stmt = stmt.where(or_(Email.sender.ilike(like), Email.subject.ilike(like)))

        # Pagination
        total_count = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar() or 0
        failed_count = (await session.execute(select(func.count(Email.id)).where(Email.status == "failed"))).scalar() or 0
        total_pages = max(1, (total_count + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        offset = (page - 1) * page_size
        page_window = sorted(set([1] + list(range(max(1, page - 2), min(total_pages, page + 2) + 1)) + [total_pages]))

        stmt = stmt.order_by(Email.id.desc()).offset(offset).limit(page_size)
        emails = (await session.execute(stmt)).scalars().all()

        # Distinct values for filter dropdowns
        bank_result = await session.execute(select(FetchRule.bank).distinct())
        banks = sorted([row[0] for row in bank_result.all() if row[0]])
        provider_result = await session.execute(select(Email.provider).distinct())
        providers = sorted([row[0] for row in provider_result.all() if row[0]])

    filters = {
        "bank": bank,
        "provider": provider,
        "status": status,
        "date_from": date_from,
        "date_to": date_to,
        "q": q,
    }
    qs_items = {k: v for k, v in filters.items() if v}
    if page_size != 50:
        qs_items["page_size"] = page_size
    base_qs = urlencode(qs_items)

    return templates.TemplateResponse(
        request,
        "emails.html",
        {
            "active_page": "emails",
            "emails": emails,
            "banks": banks,
            "providers": providers,
            "filters": filters,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "page_window": page_window,
            "failed_count": failed_count,
            "base_qs": base_qs,
        },
    )


# ---------------------------------------------------------------------------
# Email Detail
# ---------------------------------------------------------------------------


@app.get("/emails/{email_id}/detail", response_class=HTMLResponse)
async def email_detail(email_id: int, request: FastAPIRequest):
    async with async_session() as session:
        result = await session.execute(
            select(Email, Transaction)
            .outerjoin(Transaction, Transaction.email_id == Email.id)
            .where(Email.id == email_id)
        )
        row = result.first()
        if not row:
            return HTMLResponse("<p>Email not found.</p>", 404)
        email_row, txn = row
    return templates.TemplateResponse(
        request,
        "partials/email_detail.html",
        {
            "email": email_row,
            "txn": txn,
        },
    )


# ---------------------------------------------------------------------------
# View Original Email
# ---------------------------------------------------------------------------


@app.get("/emails/{email_id}/original", response_class=HTMLResponse)
async def view_original_email(email_id: int):
    async with async_session() as session:
        email_row = await session.get(Email, email_id)
        if not email_row or not email_row.source_id or not email_row.remote_id:
            return HTMLResponse("<p>Original email not available.</p>", 404)
        source = await session.get(EmailSource, email_row.source_id)
        if not source:
            return HTMLResponse("<p>Email source not found.</p>", 404)

    creds = decrypt_credentials(source.credentials)

    if source.provider == "gmail":
        raw = await asyncio.to_thread(
            _fetch_gmail_single_sync,
            creds["user"],
            creds["app_password"],
            email_row.remote_id,
        )
    elif source.provider == "fastmail":
        raw = await asyncio.to_thread(
            _fetch_fastmail_single_sync, creds["token"], email_row.remote_id
        )
    else:
        return HTMLResponse("<p>Unknown provider.</p>", 400)

    if not raw:
        return HTMLResponse("<p>Could not fetch original email from provider.</p>", 502)

    html_body = _extract_html_body(raw)
    if not html_body:
        # Fallback to plain text
        text_body = _extract_text_body(raw)
        import html

        html_body = (
            f"<pre>{html.escape(text_body)}</pre>"
            if text_body
            else "<p>No content.</p>"
        )

    # Return with restrictive headers
    return HTMLResponse(
        html_body,
        headers={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data: https:;",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


async def auto_link_transactions(session, account):
    """Link orphan transactions to a newly created/updated account.

    Builds the full link context (so addon-card -> parent-account chains
    are respected) then runs link_transaction over every unlinked
    transaction for the same bank.
    """
    bank_lower = account.bank.strip().lower()
    ctx = await build_link_context(session)

    stmt = (
        select(Transaction)
        .where(func.lower(Transaction.bank) == bank_lower)
        .where(Transaction.account_id.is_(None))
    )
    result = await session.execute(stmt)
    for txn in result.scalars().all():
        link_transaction(ctx, txn)
    await session.commit()


@app.get("/accounts", response_class=HTMLResponse)
async def account_list(
    request: FastAPIRequest,
    bank: Annotated[str | None, Query(description="Filter by bank name")] = None,
    type: Annotated[
        str | None,
        Query(description="Filter by account type: bank_account, credit_card, debit_card"),
    ] = None,
    active: Annotated[
        str | None, Query(description="Filter by active status: true or false")
    ] = None,
):
    async with async_session() as session:
        stmt = select(Account)
        if bank:
            stmt = stmt.where(Account.bank == bank)
        if type:
            stmt = stmt.where(Account.type == type)
        if active == "true":
            stmt = stmt.where(Account.active.is_(True))
        elif active == "false":
            stmt = stmt.where(Account.active.is_(False))
        stmt = stmt.order_by(Account.id)
        accounts = (await session.execute(stmt)).scalars().all()

        # Distinct banks across accounts + transactions (union) for filter + add form.
        # Fall back to SUPPORTED_BANKS so the Add Account form is usable on a fresh
        # install where no accounts/transactions exist yet.
        acct_banks = await session.execute(select(Account.bank).distinct())
        txn_banks = await session.execute(select(Transaction.bank).distinct())
        all_banks = {row[0] for row in acct_banks.all() if row[0]} | {
            row[0] for row in txn_banks.all() if row[0]
        }
        if not all_banks:
            all_banks = {b.upper() for b in SUPPORTED_BANKS}
        banks = sorted(all_banks)

    filters = {"bank": bank, "type": type, "active": active}

    return templates.TemplateResponse(
        request,
        "accounts.html",
        {
            "active_page": "accounts",
            "accounts": accounts,
            "banks": banks,
            "filters": filters,
        },
    )


@app.post("/accounts")
async def account_create(
    request: FastAPIRequest,
    bank: str = Form(...),
    label: str = Form(...),
    type: str = Form(...),
    account_number: str = Form(""),
):
    async with async_session() as session:
        account = Account(
            bank=bank.strip(),
            label=label.strip(),
            type=type,
            account_number=account_number.strip() or None,
        )
        session.add(account)
        await session.commit()
        await session.refresh(account)

    async with async_session() as session:
        await auto_link_transactions(session, account)

    return RedirectResponse(url="/accounts", status_code=303)


@app.get("/accounts/{account_id}/edit", response_class=HTMLResponse)
async def account_edit_form(request: FastAPIRequest, account_id: int):
    async with async_session() as session:
        account = await session.get(Account, account_id)
        if not account:
            return RedirectResponse(url="/accounts", status_code=303)

        acct_bank_result = await session.execute(select(Account.bank).distinct())
        txn_bank_result = await session.execute(select(Transaction.bank).distinct())
        bank_set = {row[0] for row in acct_bank_result.all() if row[0]} | {
            row[0] for row in txn_bank_result.all() if row[0]
        }
        # Always include the current account's bank so the <select> has a match,
        # even if an older row used a casing/spelling not in the distinct set.
        if account.bank:
            bank_set.add(account.bank)
        banks = sorted(bank_set)

    # Decrypt statement password for display
    statement_password_plain = ""
    if account.statement_password:
        try:
            from bank_email_fetcher.config import get_fernet

            statement_password_plain = (
                get_fernet().decrypt(account.statement_password.encode()).decode()
            )
        except Exception:
            pass

    return templates.TemplateResponse(
        request,
        "account_edit.html",
        {
            "active_page": "accounts",
            "account": account,
            "banks": banks,
            "statement_password_plain": statement_password_plain,
        },
    )


async def retry_password_required_statements(account_id: int, password: str) -> dict:
    """Retry all password_required statements for the given account.

    Returns a dict with counts of successfully retried statements:
    {
        "cc_retried": int,
        "bank_retried": int,
        "cc_failed": int,
        "bank_failed": int,
    }
    """
    from bank_email_fetcher.statements import (
        parse_statement,
        reconcile_statement,
        reconciliation_to_json,
        enrich_matched_transactions,
    )
    from bank_email_fetcher.bank_statements import (
        parse_bank_statement,
        reconcile_bank_statement,
    )
    from bank_email_fetcher.linker import build_link_context, link_transaction

    result = {"cc_retried": 0, "bank_retried": 0, "cc_failed": 0, "bank_failed": 0}

    # Retry CC statements
    async with async_session() as session:
        cc_uploads = (
            await session.execute(
                select(StatementUpload)
                .where(
                    StatementUpload.account_id == account_id,
                    StatementUpload.status == "password_required",
                )
            )
        ).scalars().all()

    for upload in cc_uploads:
        pdf_path = Path(upload.file_path) if upload.file_path else None
        if not pdf_path or not pdf_path.exists():
            result["cc_failed"] += 1
            continue

        try:
            parsed = await asyncio.to_thread(parse_statement, pdf_path, password)

            # Get transactions for reconciliation
            async with async_session() as session:
                db_txns = (
                    await session.execute(
                        select(Transaction).where(Transaction.account_id == account_id)
                    )
                ).scalars().all()

            recon = reconcile_statement(parsed, db_txns, account_id)
            await enrich_matched_transactions(recon)

            # Update the upload with parsed data
            async with async_session() as session:
                upload_row = await session.get(StatementUpload, upload.id)
                upload_row.status = "parsed"
                upload_row.card_number = parsed.card_number
                upload_row.statement_name = parsed.name
                upload_row.due_date = parsed.due_date
                upload_row.total_amount_due = parsed.statement_total_amount_due
                upload_row.parsed_txn_count = len(recon["matched"]) + len(recon["missing"])
                upload_row.matched_count = len(recon["matched"])
                upload_row.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
                upload_row.reconciliation_data = reconciliation_to_json(recon)
                upload_row.error = None
                await session.commit()

            result["cc_retried"] += 1
            logger.info(
                "Successfully retried CC statement %d for account %d",
                upload.id,
                account_id,
            )
        except Exception as e:
            result["cc_failed"] += 1
            logger.warning(
                "Failed to retry CC statement %d for account %d: %s",
                upload.id,
                account_id,
                e,
            )

    # Retry bank statements
    async with async_session() as session:
        bank_uploads = (
            await session.execute(
                select(BankStatementUpload)
                .where(
                    BankStatementUpload.account_id == account_id,
                    BankStatementUpload.status == "password_required",
                )
            )
        ).scalars().all()

    for upload in bank_uploads:
        pdf_path = Path(upload.file_path) if upload.file_path else None
        if not pdf_path or not pdf_path.exists():
            result["bank_failed"] += 1
            continue

        try:
            parsed = await asyncio.to_thread(parse_bank_statement, pdf_path, upload.bank, password)

            # Get transactions for reconciliation
            async with async_session() as session:
                db_txns = (
                    await session.execute(
                        select(Transaction).where(Transaction.account_id == account_id)
                    )
                ).scalars().all()

            recon = reconcile_bank_statement(parsed, db_txns, account_id)
            await enrich_matched_transactions(recon)

            # Update the upload with parsed data
            async with async_session() as session:
                upload_row = await session.get(BankStatementUpload, upload.id)
                upload_row.status = "parsed"
                upload_row.account_number = parsed.account_number
                upload_row.account_holder_name = parsed.account_holder_name
                upload_row.opening_balance = parsed.opening_balance
                upload_row.closing_balance = parsed.closing_balance
                upload_row.statement_period_start = parsed.statement_period_start
                upload_row.statement_period_end = parsed.statement_period_end
                upload_row.parsed_txn_count = len(recon["matched"]) + len(recon["missing"])
                upload_row.matched_count = len(recon["matched"])
                upload_row.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
                upload_row.reconciliation_data = reconciliation_to_json(recon)
                upload_row.error = None
                await session.commit()

            result["bank_retried"] += 1
            logger.info(
                "Successfully retried bank statement %d for account %d",
                upload.id,
                account_id,
            )
        except Exception as e:
            result["bank_failed"] += 1
            logger.warning(
                "Failed to retry bank statement %d for account %d: %s",
                upload.id,
                account_id,
                e,
            )

    return result


@app.post("/accounts/{account_id}/edit")
async def account_update(
    request: FastAPIRequest,
    account_id: int,
    bank: str = Form(...),
    label: str = Form(...),
    type: str = Form(...),
    account_number: str = Form(""),
    statement_password: str = Form(""),
    statement_password_hint: str = Form(""),
):
    password_changed = False
    new_password_plain = None

    async with async_session() as session:
        account = await session.get(Account, account_id)
        if not account:
            return RedirectResponse(url="/accounts", status_code=303)

        account.bank = bank.strip()
        account.label = label.strip()
        account.type = type
        account.account_number = account_number.strip() or None
        account.statement_password_hint = statement_password_hint.strip() or None

        # Encrypt and store statement password
        if statement_password.strip():
            from bank_email_fetcher.config import get_fernet

            account.statement_password = (
                get_fernet().encrypt(statement_password.strip().encode()).decode()
            )
            password_changed = True
            new_password_plain = statement_password.strip()
        elif not statement_password:
            account.statement_password = None

        await session.commit()
        await session.refresh(account)

    async with async_session() as session:
        await auto_link_transactions(session, account)

    # Automatically retry password-required statements when password is set/changed
    if password_changed and new_password_plain:
        try:
            retry_result = await retry_password_required_statements(account_id, new_password_plain)
            total_retried = retry_result["cc_retried"] + retry_result["bank_retried"]
            if total_retried > 0:
                logger.info(
                    "Automatically retried %d password-required statements for account %d",
                    total_retried,
                    account_id,
                )
        except Exception as e:
            logger.warning(
                "Failed to automatically retry password-required statements for account %d: %s",
                account_id,
                e,
            )

    return RedirectResponse(url="/accounts", status_code=303)


@app.post("/accounts/{account_id}/delete")
async def account_delete(account_id: int):
    async with async_session() as session:
        account = await session.get(Account, account_id)
        if account:
            # Unlink transactions first
            await session.execute(
                update(Transaction)
                .where(Transaction.account_id == account_id)
                .values(account_id=None)
            )
            await session.delete(account)
            await session.commit()

    return RedirectResponse(url="/accounts", status_code=303)


@app.post("/accounts/{account_id}/cards")
async def card_add(
    account_id: int,
    card_mask: str = Form(...),
    label: str = Form(""),
    is_primary: str = Form(""),
):
    async with async_session() as session:
        account = await session.get(Account, account_id)
        if not account:
            return RedirectResponse(url="/accounts", status_code=303)
        card = Card(
            account_id=account_id,
            card_mask=card_mask.strip(),
            label=label.strip() or "self",
            is_primary=is_primary == "1",
            active=True,
        )
        session.add(card)
        await session.commit()
    return RedirectResponse(url=f"/accounts/{account_id}/edit", status_code=303)


@app.post("/accounts/{account_id}/cards/{card_id}/edit")
async def card_edit(
    account_id: int, card_id: int, label: str = Form(...), is_primary: str = Form("")
):
    async with async_session() as session:
        card = await session.get(Card, card_id)
        if card and card.account_id == account_id:
            card.label = label.strip() or card.label
            card.is_primary = is_primary == "1"
            await session.commit()
    return RedirectResponse(url=f"/accounts/{account_id}/edit", status_code=303)


@app.post("/accounts/{account_id}/cards/{card_id}/delete")
async def card_delete(account_id: int, card_id: int):
    async with async_session() as session:
        card = await session.get(Card, card_id)
        if card and card.account_id == account_id:
            # Unlink transactions from this card
            await session.execute(
                update(Transaction)
                .where(Transaction.card_id == card_id)
                .values(card_id=None)
            )
            await session.delete(card)
            await session.commit()
    return RedirectResponse(url=f"/accounts/{account_id}/edit", status_code=303)


# ---------------------------------------------------------------------------
# Email Sources
# ---------------------------------------------------------------------------


@app.get("/sources", response_class=HTMLResponse)
async def source_list(request: FastAPIRequest):
    async with async_session() as session:
        result = await session.execute(select(EmailSource).order_by(EmailSource.id))
        sources = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "sources.html",
        {
            "active_page": "sources",
            "sources": sources,
        },
    )


@app.post("/sources")
async def create_source(
    request: FastAPIRequest,
    provider: str = Form(...),
    label: str = Form(...),
    account_identifier: str = Form(""),
    credential_value: str = Form(...),
):
    # Build creds dict based on provider
    if provider == "gmail":
        creds = {
            "user": account_identifier.strip(),
            "app_password": credential_value.strip(),
        }
    elif provider == "fastmail":
        creds = {"token": credential_value.strip()}
    else:
        return RedirectResponse(url="/sources", status_code=303)

    encrypted = encrypt_credentials(creds)

    async with async_session() as session:
        source = EmailSource(
            provider=provider.strip(),
            label=label.strip(),
            account_identifier=account_identifier.strip() or None,
            credentials=encrypted,
        )
        session.add(source)
        await session.commit()

    return RedirectResponse(url="/sources", status_code=303)


@app.get("/sources/{source_id}/edit", response_class=HTMLResponse)
async def edit_source_form(source_id: int, request: FastAPIRequest):
    async with async_session() as session:
        source = await session.get(EmailSource, source_id)
        if not source:
            return RedirectResponse(url="/sources", status_code=303)

    return templates.TemplateResponse(
        request,
        "source_edit.html",
        {
            "active_page": "sources",
            "source": source,
        },
    )


@app.post("/sources/{source_id}/edit")
async def update_source(
    source_id: int,
    request: FastAPIRequest,
    provider: str = Form(...),
    label: str = Form(...),
    account_identifier: str = Form(""),
    credential_value: str = Form(""),
    active: bool = Form(False),
):
    async with async_session() as session:
        source = await session.get(EmailSource, source_id)
        if not source:
            return RedirectResponse(url="/sources", status_code=303)

        source.provider = provider.strip()
        source.label = label.strip()
        source.account_identifier = account_identifier.strip() or None
        source.active = active

        # Only re-encrypt credentials if a new value was provided
        if credential_value.strip():
            if provider == "gmail":
                creds = {
                    "user": account_identifier.strip(),
                    "app_password": credential_value.strip(),
                }
            elif provider == "fastmail":
                creds = {"token": credential_value.strip()}
            else:
                creds = {}
            if creds:
                source.credentials = encrypt_credentials(creds)

        await session.commit()

    return RedirectResponse(url="/sources", status_code=303)


@app.post("/sources/{source_id}/delete")
async def delete_source(source_id: int):
    async with async_session() as session:
        source = await session.get(EmailSource, source_id)
        if source:
            # Unlink rules referencing this source
            await session.execute(
                update(FetchRule)
                .where(FetchRule.source_id == source_id)
                .values(source_id=None)
            )
            await session.delete(source)
            await session.commit()

    return RedirectResponse(url="/sources", status_code=303)


@api_router.post("/sources/{source_id}/test")
async def test_source(source_id: int):
    """Test connectivity for an email source. Returns JSON with success/failure."""
    async with async_session() as session:
        source = await session.get(EmailSource, source_id)
        if not source:
            return JSONResponse(
                {"ok": False, "error": "Source not found"}, status_code=404
            )

    try:
        creds = decrypt_credentials(source.credentials)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Decryption failed: {e}"})

    def _test_gmail(user, password):
        conn = imaplib.IMAP4_SSL("imap.gmail.com")
        conn.login(user, password)
        conn.logout()
        return f"Gmail IMAP login successful for {user}"

    def _test_fastmail(token):
        jmap_session_url = "https://api.fastmail.com/jmap/session"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        req = Request(jmap_session_url, headers=headers)
        with urlopen(req) as resp:
            data = json.loads(resp.read())
        acct = data.get("primaryAccounts", {}).get("urn:ietf:params:jmap:mail")
        if not acct:
            raise ValueError("JMAP session returned but no mail account found")
        return f"Fastmail JMAP session OK (account: {acct})"

    if source.provider == "gmail":
        user = creds.get("user", "")
        password = creds.get("app_password", "")
        if not user or not password:
            return JSONResponse(
                {"ok": False, "error": "Missing user or app_password in credentials"}
            )
        try:
            msg = await asyncio.to_thread(_test_gmail, user, password)
            return JSONResponse({"ok": True, "message": msg})
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Gmail test failed: {e}"})

    elif source.provider == "fastmail":
        token = creds.get("token", "")
        if not token:
            return JSONResponse({"ok": False, "error": "Missing token in credentials"})
        try:
            msg = await asyncio.to_thread(_test_fastmail, token)
            return JSONResponse({"ok": True, "message": msg})
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Fastmail test failed: {e}"})

    else:
        return JSONResponse(
            {"ok": False, "error": f"Unknown provider: {source.provider}"}
        )


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@app.get("/rules", response_class=HTMLResponse)
async def rule_list(
    request: FastAPIRequest,
    bank: Annotated[str | None, Query(description="Filter by bank")] = None,
    source_id: Annotated[
        str | None, Query(description="Filter by email source ID")
    ] = None,
    enabled: Annotated[
        str | None, Query(description="Filter by enabled status: true or false")
    ] = None,
):
    async with async_session() as session:
        stmt = select(FetchRule)
        if bank:
            stmt = stmt.where(FetchRule.bank == bank)
        if source_id:
            try:
                stmt = stmt.where(FetchRule.source_id == int(source_id))
            except ValueError:
                pass
        if enabled == "true":
            stmt = stmt.where(FetchRule.enabled.is_(True))
        elif enabled == "false":
            stmt = stmt.where(FetchRule.enabled.is_(False))
        stmt = stmt.order_by(FetchRule.id)
        rules = (await session.execute(stmt)).scalars().all()

        # Distinct banks present in rules (for filter dropdown)
        bank_result = await session.execute(select(FetchRule.bank).distinct())
        banks = sorted([row[0] for row in bank_result.all() if row[0]])

        # Load sources for the add-rule form dropdown AND the filter dropdown
        source_result = await session.execute(
            select(EmailSource)
            .where(EmailSource.active.is_(True))
            .order_by(EmailSource.id)
        )
        sources = source_result.scalars().all()

    filters = {"bank": bank, "source_id": source_id, "enabled": enabled}

    return templates.TemplateResponse(
        request,
        "rules.html",
        {
            "active_page": "rules",
            "rules": rules,
            "sources": sources,
            "banks": banks,
            "supported_banks": SUPPORTED_BANKS,
            "filters": filters,
        },
    )


@app.post("/rules")
async def rule_create(
    request: FastAPIRequest,
    bank: str = Form(...),
    sender: str = Form(""),
    subject: str = Form(""),
    folder: str = Form(""),
    source_id: int = Form(None),
    provider: str = Form(""),
):
    async with async_session() as session:
        # Resolve provider from source if source_id is provided
        resolved_provider = provider.strip()
        if source_id:
            source = await session.get(EmailSource, source_id)
            if source:
                resolved_provider = source.provider

        rule = FetchRule(
            provider=resolved_provider or "gmail",
            source_id=source_id if source_id else None,
            bank=bank.strip(),
            sender=sender.strip() or None,
            subject=subject.strip() or None,
            folder=folder.strip() or None,
        )
        session.add(rule)
        await session.commit()

    return RedirectResponse(url="/rules", status_code=303)


@app.get("/rules/{rule_id}/edit", response_class=HTMLResponse)
async def rule_edit_form(request: FastAPIRequest, rule_id: int):
    async with async_session() as session:
        rule = await session.get(FetchRule, rule_id)
        if not rule:
            return RedirectResponse(url="/rules", status_code=303)

        # Load sources for the dropdown
        source_result = await session.execute(
            select(EmailSource)
            .where(EmailSource.active.is_(True))
            .order_by(EmailSource.id)
        )
        sources = source_result.scalars().all()

    return templates.TemplateResponse(
        request,
        "rule_edit.html",
        {
            "active_page": "rules",
            "rule": rule,
            "sources": sources,
            "supported_banks": SUPPORTED_BANKS,
        },
    )


@app.post("/rules/{rule_id}/edit")
async def rule_update(
    request: FastAPIRequest,
    rule_id: int,
    bank: str = Form(...),
    sender: str = Form(""),
    subject: str = Form(""),
    folder: str = Form(""),
    enabled: bool = Form(False),
    source_id: int = Form(None),
    provider: str = Form(""),
):
    async with async_session() as session:
        rule = await session.get(FetchRule, rule_id)
        if not rule:
            return RedirectResponse(url="/rules", status_code=303)

        # Resolve provider from source if source_id is provided
        resolved_provider = provider.strip()
        if source_id:
            source = await session.get(EmailSource, source_id)
            if source:
                resolved_provider = source.provider

        rule.provider = resolved_provider or rule.provider
        rule.source_id = source_id if source_id else None
        rule.bank = bank.strip()
        rule.sender = sender.strip() or None
        rule.subject = subject.strip() or None
        rule.folder = folder.strip() or None
        rule.enabled = enabled
        await session.commit()

    return RedirectResponse(url="/rules", status_code=303)


@app.post("/rules/{rule_id}/delete")
async def rule_delete(rule_id: int):
    async with async_session() as session:
        rule = await session.get(FetchRule, rule_id)
        if rule:
            await session.delete(rule)
            await session.commit()

    return RedirectResponse(url="/rules", status_code=303)


@app.post("/rules/{rule_id}/toggle")
async def rule_toggle(rule_id: int):
    async with async_session() as session:
        rule = await session.get(FetchRule, rule_id)
        if rule:
            rule.enabled = not rule.enabled
            await session.commit()

    return RedirectResponse(url="/rules", status_code=303)


# ---------------------------------------------------------------------------
# Reparse
# ---------------------------------------------------------------------------


@app.post("/emails/{email_id}/reparse")
async def reparse_email(email_id: int):
    """Re-parse a failed email from the failed spool (.eml file).

    Returns JSON so the caller can update the UI without a full-page redirect.
    """
    import re as _re

    SPOOL_DIR = Path(__file__).parent / "data" / "failed"

    async with async_session() as session:
        email_row = await session.get(Email, email_id)
        if not email_row:
            return JSONResponse(
                {"ok": False, "error": "Email not found"}, status_code=404
            )

        rule = (
            await session.get(FetchRule, email_row.rule_id)
            if email_row.rule_id
            else None
        )

    if not rule:
        return JSONResponse(
            {"ok": False, "error": "No fetch rule associated with this email"},
            status_code=400,
        )

    # Locate the .eml file using the same sanitisation as _save_failed_email
    safe_id = _re.sub(r"[^\w\-.]", "_", email_row.message_id)
    spool_path = SPOOL_DIR / f"{email_row.provider}_{safe_id}.eml"
    if not spool_path.exists():
        return JSONResponse(
            {
                "ok": False,
                "error": f"Spool file not found ({spool_path.name}). It may have been cleaned up or never saved.",
            },
            status_code=404,
        )

    raw_bytes = spool_path.read_bytes()

    # Try standard transaction parse first
    error, txn_data, password_hint = _process_email(rule.bank, raw_bytes)

    # If that failed (or parsed as statement with no transaction), try statement path
    stmt_result = None
    if not txn_data:
        try:
            from bank_email_fetcher.statements import process_statement_email

            stmt_result = await process_statement_email(
                rule.bank,
                raw_bytes,
                email_row.subject or "",
                source_id=email_row.source_id,
            )
        except Exception as stmt_err:
            logger.warning(
                "CC statement processing error during reparse of email %d: %s",
                email_id,
                stmt_err,
            )

        if stmt_result is None:
            try:
                from bank_email_fetcher.bank_statements import (
                    process_bank_statement_email,
                )

                stmt_result = await process_bank_statement_email(
                    rule.bank,
                    raw_bytes,
                    email_row.subject or "",
                    source_id=email_row.source_id,
                    password_hint=password_hint,
                )
            except Exception as stmt_err:
                logger.warning(
                    "Bank statement processing error during reparse of email %d: %s",
                    email_id,
                    stmt_err,
                )

    if not txn_data and not stmt_result:
        # Parsing still fails — update error message so it's fresh, but keep status=failed
        async with async_session() as session:
            em = await session.get(Email, email_id)
            if em:
                em.error = error
                await session.commit()
        return JSONResponse(
            {
                "ok": False,
                "error": error or "Parsing failed (no transaction or statement found)",
            }
        )

    # Success — update the email row and create transaction if needed
    from sqlalchemy.exc import IntegrityError as _IntegrityError

    async with async_session() as session:
        async with session.begin():
            em = await session.get(Email, email_id)
            if not em:
                return JSONResponse(
                    {"ok": False, "error": "Email disappeared"}, status_code=500
                )

            em.status = "parsed"
            em.error = None

            if stmt_result and stmt_result.get("statement_upload_id"):
                from bank_email_fetcher.db import StatementUpload

                su = await session.get(
                    StatementUpload, stmt_result["statement_upload_id"]
                )
                if su:
                    su.email_id = em.id
            elif stmt_result and stmt_result.get("bank_statement_upload_id"):
                from bank_email_fetcher.db import BankStatementUpload

                su = await session.get(
                    BankStatementUpload, stmt_result["bank_statement_upload_id"]
                )
                if su:
                    su.email_id = em.id

            txn_id = None
            if txn_data:
                try:
                    async with session.begin_nested():
                        from bank_email_fetcher.linker import (
                            build_link_context,
                            link_transaction as _link_txn,
                        )

                        txn_row = Transaction(email_id=em.id, **txn_data)
                        session.add(txn_row)
                        await session.flush()
                        _link_ctx = await build_link_context(session)
                        _link_txn(_link_ctx, txn_row)
                        await session.flush()
                        txn_id = txn_row.id
                        # Enrich txn_data with pre-rendered label for notification
                        from bank_email_fetcher.telegram_bot import (
                            build_account_label,
                        )
                        txn_data["account_label"] = build_account_label(
                            txn_row.account, txn_row.card
                        )
                        txn_data["channel"] = txn_row.channel
                except _IntegrityError:
                    em.status = "skipped"
                    em.error = "Duplicate transaction skipped because an identical transaction row already exists"
                    return JSONResponse({"ok": False, "error": em.error})

    # Send Telegram notification for the new transaction
    from bank_email_fetcher.settings_service import (
        should_notify_transactions,
        get_telegram_chat_id,
    )

    if txn_id and txn_data and should_notify_transactions():
        try:
            from bank_email_fetcher.telegram_bot import send_transaction_notification

            await send_transaction_notification(
                txn_id, txn_data, get_telegram_chat_id()
            )
        except Exception as tg_err:
            logger.warning(
                "Telegram notification failed for reparsed txn #%s: %s", txn_id, tg_err
            )

    msg = "Email re-parsed successfully"
    if stmt_result:
        stmt_kind = "Bank" if stmt_result.get("bank_statement_upload_id") else "CC"
        msg = f"{stmt_kind} statement re-processed (matched={stmt_result.get('matched', 0)}, imported={stmt_result.get('imported', 0)})"
    logger.info("Reparse of email %d succeeded: %s", email_id, msg)
    return JSONResponse(
        {"ok": True, "message": msg, "new_status": "parsed", "txn_id": txn_id}
    )


@app.post("/emails/reparse-all-failed")
async def reparse_all_failed():
    """Re-parse all emails with status='failed' from the failed spool."""
    import re as _re
    from sqlalchemy.exc import IntegrityError as _IntegrityError
    from bank_email_fetcher.statements import process_statement_email
    from bank_email_fetcher.db import StatementUpload
    from bank_email_fetcher.linker import (
        build_link_context,
        link_transaction as _link_txn,
    )

    SPOOL_DIR = Path(__file__).parent / "data" / "failed"
    succeeded = 0
    skipped = 0
    still_failed = 0

    async with async_session() as session:
        rows = (
            await session.execute(
                select(Email, FetchRule)
                .outerjoin(FetchRule, Email.rule_id == FetchRule.id)
                .where(Email.status == "failed")
            )
        ).all()

    for email_row, rule in rows:
        if not rule:
            still_failed += 1
            continue

        safe_id = _re.sub(r"[^\w\-.]", "_", email_row.message_id)
        spool_path = SPOOL_DIR / f"{email_row.provider}_{safe_id}.eml"
        if not spool_path.exists():
            still_failed += 1
            continue

        raw_bytes = spool_path.read_bytes()
        error, txn_data, _ = _process_email(rule.bank, raw_bytes)

        stmt_result = None
        if error and not txn_data:
            try:
                stmt_result = await process_statement_email(
                    rule.bank,
                    raw_bytes,
                    email_row.subject or "",
                    source_id=email_row.source_id,
                )
            except Exception:
                pass

        if not txn_data and not stmt_result:
            still_failed += 1
            continue

        was_skipped = False
        async with async_session() as session:
            async with session.begin():
                em = await session.get(Email, email_row.id)
                if not em:
                    continue
                em.status = "parsed"
                em.error = None

                if stmt_result and stmt_result.get("statement_upload_id"):
                    su = await session.get(
                        StatementUpload, stmt_result["statement_upload_id"]
                    )
                    if su:
                        su.email_id = em.id

                if txn_data:
                    try:
                        async with session.begin_nested():
                            txn_row = Transaction(email_id=em.id, **txn_data)
                            session.add(txn_row)
                            await session.flush()
                            _link_ctx = await build_link_context(session)
                            _link_txn(_link_ctx, txn_row)
                            await session.flush()
                    except _IntegrityError:
                        em.status = "skipped"
                        em.error = "Duplicate transaction skipped"
                        was_skipped = True

        if was_skipped:
            skipped += 1
        else:
            succeeded += 1

    return JSONResponse(
        {
            "ok": True,
            "succeeded": succeeded,
            "skipped": skipped,
            "failed": still_failed,
        }
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: FastAPIRequest):
    from bank_email_fetcher.settings_service import get_grouped_settings

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "active_page": "settings",
            "grouped_settings": get_grouped_settings(),
        },
    )


@app.post("/settings")
async def save_settings_route(request: FastAPIRequest):
    from bank_email_fetcher.settings_service import (
        parse_form_updates,
        save_settings,
        get_grouped_settings,
        restart_services,
    )

    form = await request.form()
    updates, errors = parse_form_updates(form)

    if errors:
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "active_page": "settings",
                "grouped_settings": get_grouped_settings(),
                "errors": errors,
            },
            status_code=422,
        )

    changed_keys = await save_settings(updates)

    telegram_restart_keys = {
        "telegram.bot_token",
        "telegram.chat_id",
        "telegram.enabled",
    }
    if changed_keys & telegram_restart_keys:
        await restart_services()

    return RedirectResponse(url="/settings?saved", status_code=303)


# ---------------------------------------------------------------------------
# Poll trigger
# ---------------------------------------------------------------------------


def _track_poll_task(task) -> None:
    global poll_task
    try:
        stats = task.result()
        logger.info("Background manual poll complete: %s", stats)
    except Exception:
        logger.exception("Background manual poll failed")
    finally:
        poll_task = None


# ---------------------------------------------------------------------------
# Statements (CC statement reconciliation)
# ---------------------------------------------------------------------------

STATEMENTS_DIR = Path(__file__).parent / "data" / "statements"


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_upload_filename(filename: str | None) -> str:
    """Strip any path components and restrict to a safe character set."""
    base = Path(filename or "statement.pdf").name or "statement.pdf"
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "statement.pdf"
    return cleaned[:120]


def _unlink_statement_file(path_str: str | None) -> None:
    """Delete a statement PDF, but only if it resolves inside STATEMENTS_DIR."""
    if not path_str:
        return
    try:
        target = Path(path_str).resolve()
        target.relative_to(STATEMENTS_DIR.resolve())
    except (ValueError, OSError):
        return
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass


@app.get("/statements", response_class=HTMLResponse)
async def statements_list(
    request: FastAPIRequest,
    type: Annotated[
        str | None, Query(description="Filter by statement type: cc or bank")
    ] = None,
    bank: Annotated[str | None, Query(description="Filter by bank name")] = None,
    account_id: Annotated[
        str | None, Query(description="Filter by account ID")
    ] = None,
    status: Annotated[
        str | None, Query(description="Filter by upload status")
    ] = None,
    date_from: Annotated[
        str | None, Query(description="Uploaded on/after (YYYY-MM-DD)")
    ] = None,
    date_to: Annotated[
        str | None, Query(description="Uploaded on/before (YYYY-MM-DD)")
    ] = None,
):
    def _apply_common_filters(stmt, model):
        if bank:
            stmt = stmt.where(model.bank == bank)
        if account_id:
            try:
                stmt = stmt.where(model.account_id == int(account_id))
            except ValueError:
                pass
        if status:
            stmt = stmt.where(model.status == status)
        if date_from:
            try:
                stmt = stmt.where(model.created_at >= date.fromisoformat(date_from))
            except ValueError:
                pass
        if date_to:
            try:
                end_of_day = datetime.combine(
                    date.fromisoformat(date_to), datetime.max.time()
                )
                stmt = stmt.where(model.created_at <= end_of_day)
            except ValueError:
                pass
        return stmt

    async with async_session() as session:
        if type == "bank":
            cc_uploads = []
        else:
            cc_stmt = _apply_common_filters(select(StatementUpload), StatementUpload)
            cc_stmt = cc_stmt.order_by(StatementUpload.created_at.desc())
            cc_uploads = (await session.execute(cc_stmt)).scalars().all()

        if type == "cc":
            bank_uploads = []
        else:
            bank_stmt = _apply_common_filters(
                select(BankStatementUpload), BankStatementUpload
            )
            bank_stmt = bank_stmt.order_by(BankStatementUpload.created_at.desc())
            bank_uploads = (await session.execute(bank_stmt)).scalars().all()

        cc_accounts = (
            (
                await session.execute(
                    select(Account)
                    .where(Account.type == "credit_card", Account.active.is_(True))
                    .order_by(Account.bank, Account.label)
                )
            )
            .scalars()
            .all()
        )
        bank_accounts = (
            (
                await session.execute(
                    select(Account)
                    .where(Account.type == "bank_account", Account.active.is_(True))
                    .order_by(Account.bank, Account.label)
                )
            )
            .scalars()
            .all()
        )

        # Distinct banks across both upload tables for filter dropdown
        cc_banks = (
            await session.execute(select(StatementUpload.bank).distinct())
        ).all()
        bank_banks = (
            await session.execute(select(BankStatementUpload.bank).distinct())
        ).all()
        banks = sorted({row[0] for row in cc_banks + bank_banks if row[0]})

    # Tag each upload with its type so templates can distinguish them
    for u in cc_uploads:
        u._statement_type = "cc"
    for u in bank_uploads:
        u._statement_type = "bank"
    uploads = sorted(
        [*cc_uploads, *bank_uploads],
        key=lambda u: u.created_at or datetime.min,
        reverse=True,
    )

    # Build JSON for cascading account dropdown (bank -> accounts)
    accounts_by_bank: dict[str, list] = {}
    for a in [*cc_accounts, *bank_accounts]:
        accounts_by_bank.setdefault(a.bank, []).append(
            {
                "id": a.id,
                "label": a.label,
                "type": a.type,
            }
        )

    # Map account id -> password hint, for inline display in the upload form
    account_hints = {
        a.id: a.statement_password_hint or ""
        for a in [*cc_accounts, *bank_accounts]
    }

    filters = {
        "type": type,
        "bank": bank,
        "account_id": account_id,
        "status": status,
        "date_from": date_from,
        "date_to": date_to,
    }

    return templates.TemplateResponse(
        request,
        "statements.html",
        {
            "active_page": "statements",
            "uploads": uploads,
            "cc_accounts": cc_accounts,
            "bank_accounts": bank_accounts,
            "banks": banks,
            "accounts_json": json.dumps(accounts_by_bank),
            "account_hints_json": json.dumps(account_hints),
            "filters": filters,
        },
    )


@app.post("/statements/upload")
async def statement_upload(
    request: FastAPIRequest,
    account_id: int = Form(...),
    password: str = Form(""),
    file: UploadFile = File(...),
):
    from bank_email_fetcher.statements import (
        parse_statement,
        reconcile_statement,
        reconciliation_to_json,
        enrich_matched_transactions,
        parse_cc_amount,
        parse_cc_date,
        last4_from_card,
        _extract_digits,
    )
    from bank_email_fetcher.linker import build_link_context, link_transaction

    async with async_session() as session:
        account = await session.get(Account, account_id)
        if not account or account.type != "credit_card":
            return RedirectResponse(url="/statements", status_code=303)

    # Save PDF to disk
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_upload_filename(file.filename)
    file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
    content = await file.read()
    file_path.write_bytes(content)

    # Parse the PDF
    try:
        parsed = await asyncio.to_thread(parse_statement, file_path, password or None)
    except Exception as e:
        error_msg = str(e)
        is_encrypted = "encrypt" in error_msg.lower() or "password" in error_msg.lower()
        async with async_session() as session:
            upload = StatementUpload(
                account_id=account_id,
                bank=account.bank,
                filename=safe_name,
                file_path=str(file_path),
                status="password_required" if is_encrypted else "parse_error",
                error=error_msg,
            )
            session.add(upload)
            await session.commit()
            upload_id = upload.id
        return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)

    # Reconcile against DB transactions for this account
    async with async_session() as session:
        db_txns = (
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id == account_id)
                )
            )
            .scalars()
            .all()
        )

    recon = reconcile_statement(parsed, db_txns, account_id)

    # Enrich matched DB transactions with statement narrations
    await enrich_matched_transactions(recon)

    # Create upload and auto-import missing transactions
    async with async_session() as session:
        upload = StatementUpload(
            account_id=account_id,
            bank=parsed.bank,
            filename=safe_name,
            file_path=str(file_path),
            status="parsed",
            card_number=parsed.card_number,
            statement_name=parsed.name,
            due_date=parsed.due_date,
            total_amount_due=parsed.statement_total_amount_due,
            parsed_txn_count=len(recon["matched"]) + len(recon["missing"]),
            matched_count=len(recon["matched"]),
            missing_count=len(recon["missing"]),
            reconciliation_data=reconciliation_to_json(recon),
        )
        session.add(upload)
        await session.flush()

        # Auto-import all missing transactions
        link_ctx = await build_link_context(session)
        acct_cards = (
            (await session.execute(select(Card).where(Card.account_id == account_id)))
            .scalars()
            .all()
        )
        _card_l4s = [v for v in (last4_from_card(c.card_mask) for c in acct_cards) if v]

        def _resolve_card_mask(raw: str | None) -> str | None:
            l4 = last4_from_card(raw)
            if l4:
                return l4
            partial = _extract_digits(raw)
            if partial:
                for cl4 in _card_l4s:
                    if cl4.endswith(partial):
                        return cl4
            return last4_from_card(account.account_number)

        imported = 0
        for entry in recon["missing"]:
            try:
                amount = parse_cc_amount(entry["amount"])
                txn_date = parse_cc_date(entry["date"])
            except ValueError, KeyError:
                continue
            txn = Transaction(
                statement_upload_id=upload.id,
                account_id=account_id,
                bank=parsed.bank,
                email_type="cc_statement",
                direction=entry["direction"],
                amount=amount,
                currency="INR",
                transaction_date=txn_date,
                counterparty=entry.get("narration"),
                card_mask=_resolve_card_mask(entry.get("card_number")),
                channel="cc_statement",
                raw_description=entry.get("narration"),
            )
            session.add(txn)
            await session.flush()
            link_transaction(link_ctx, txn)
            await session.flush()
            entry["imported"] = True
            entry["imported_txn_id"] = txn.id
            imported += 1

        upload.imported_count = imported
        upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
        upload.reconciliation_data = reconciliation_to_json(recon)
        if upload.missing_count == 0:
            upload.status = "imported"
        await session.commit()
        upload_id = upload.id

    from bank_email_fetcher.reminders import init_payment_tracking

    await init_payment_tracking(upload_id)

    return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)


@app.post("/statements/upload-bank")
async def bank_statement_upload(
    request: FastAPIRequest,
    account_id: int = Form(...),
    password: str = Form(""),
    file: UploadFile = File(...),
):
    from bank_email_fetcher.bank_statements import (
        parse_bank_statement,
        reconcile_bank_statement,
        reconciliation_to_json,
        enrich_matched_transactions,
        _parse_amount,
        _parse_date,
        _last4,
    )
    from bank_email_fetcher.linker import build_link_context, link_transaction

    async with async_session() as session:
        account = await session.get(Account, account_id)
        if not account or account.type != "bank_account":
            return RedirectResponse(url="/statements", status_code=303)

    # Save PDF to disk
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_upload_filename(file.filename)
    file_path = STATEMENTS_DIR / f"{ts}_{safe_name}"
    content = await file.read()
    file_path.write_bytes(content)

    # Parse the PDF
    try:
        parsed = await asyncio.to_thread(
            parse_bank_statement, file_path, account.bank, password or None
        )
    except Exception as e:
        error_msg = str(e)
        is_encrypted = "encrypt" in error_msg.lower() or "password" in error_msg.lower()
        async with async_session() as session:
            upload = BankStatementUpload(
                account_id=account_id,
                bank=account.bank,
                filename=safe_name,
                file_path=str(file_path),
                status="password_required" if is_encrypted else "parse_error",
                error=error_msg,
            )
            session.add(upload)
            await session.commit()
            upload_id = upload.id
        return RedirectResponse(url=f"/statements/bank/{upload_id}", status_code=303)

    # Reconcile against DB transactions for this account
    async with async_session() as session:
        db_txns = (
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id == account_id)
                )
            )
            .scalars()
            .all()
        )

    recon = reconcile_bank_statement(parsed, db_txns, account_id)
    await enrich_matched_transactions(recon)

    # Create upload and auto-import missing transactions
    async with async_session() as session:
        upload = BankStatementUpload(
            account_id=account_id,
            bank=parsed.bank or account.bank,
            filename=safe_name,
            file_path=str(file_path),
            status="parsed",
            account_number=parsed.account_number,
            account_holder_name=parsed.account_holder_name,
            opening_balance=parsed.opening_balance,
            closing_balance=parsed.closing_balance,
            statement_period_start=parsed.statement_period_start,
            statement_period_end=parsed.statement_period_end,
            parsed_txn_count=len(recon["matched"]) + len(recon["missing"]),
            matched_count=len(recon["matched"]),
            missing_count=len(recon["missing"]),
            reconciliation_data=reconciliation_to_json(recon),
        )
        session.add(upload)
        await session.flush()

        link_ctx = await build_link_context(session)

        imported = 0
        for entry in recon["missing"]:
            try:
                amount = _parse_amount(entry["amount"])
                txn_date = _parse_date(entry["date"])
            except ValueError, KeyError:
                continue
            txn = Transaction(
                bank_statement_upload_id=upload.id,
                account_id=account_id,
                bank=parsed.bank or account.bank,
                email_type="bank_statement",
                direction=entry["direction"],
                amount=amount,
                currency="INR",
                transaction_date=txn_date,
                counterparty=entry.get("narration"),
                account_mask=_last4(parsed.account_number),
                reference_number=entry.get("reference_number"),
                channel=entry.get("channel") or "bank_statement",
                raw_description=entry.get("narration"),
            )
            session.add(txn)
            await session.flush()
            link_transaction(link_ctx, txn)
            await session.flush()
            entry["imported"] = True
            entry["imported_txn_id"] = txn.id
            imported += 1

        upload.imported_count = imported
        upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
        upload.reconciliation_data = reconciliation_to_json(recon)
        if upload.missing_count == 0:
            upload.status = "imported"
        elif imported > 0:
            upload.status = "partial_import"
        await session.commit()
        upload_id = upload.id

    return RedirectResponse(url=f"/statements/bank/{upload_id}", status_code=303)


# ---------------------------------------------------------------------------
# Bank statement routes (separate model: BankStatementUpload)
# Must be registered BEFORE /statements/{upload_id} wildcard routes to avoid
# FastAPI matching "bank" as an upload_id parameter.
# ---------------------------------------------------------------------------


@app.get("/statements/bank/{upload_id}", response_class=HTMLResponse)
async def bank_statement_detail(upload_id: int, request: FastAPIRequest):
    from bank_email_fetcher.bank_statements import reconciliation_from_json

    async with async_session() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        if not upload:
            return HTMLResponse("<p>Bank statement not found.</p>", 404)

    recon = None
    if upload.reconciliation_data:
        recon = reconciliation_from_json(upload.reconciliation_data)

    return templates.TemplateResponse(
        request,
        "bank_statement_reconcile.html",
        {
            "active_page": "statements",
            "upload": upload,
            "recon": recon,
            "error": request.query_params.get("error"),
        },
    )


@app.post("/statements/bank/{upload_id}/retry")
async def bank_statement_retry(
    upload_id: int,
    password: str = Form(...),
    save_password: str = Form(""),
):
    from bank_email_fetcher.bank_statements import (
        parse_bank_statement,
        reconcile_bank_statement,
        reconciliation_to_json,
        enrich_matched_transactions,
        _parse_amount,
        _parse_date,
        _last4,
    )
    from bank_email_fetcher.linker import build_link_context, link_transaction

    async with async_session() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        if not upload:
            return RedirectResponse(url="/statements", status_code=303)
        account_id = upload.account_id
        file_path = upload.file_path

    try:
        parsed = await asyncio.to_thread(
            parse_bank_statement, Path(file_path), upload.bank, password
        )
    except Exception as e:
        msg = str(e)
        is_password_error = "encrypt" in msg.lower() or "password" in msg.lower()
        async with async_session() as session:
            upload = await session.get(BankStatementUpload, upload_id)
            upload.error = msg
            if not is_password_error:
                upload.status = "parse_error"
            await session.commit()
        return RedirectResponse(url=f"/statements/bank/{upload_id}", status_code=303)

    if save_password == "1":
        from bank_email_fetcher.config import get_fernet

        encrypted = get_fernet().encrypt(password.encode()).decode()
        async with async_session() as session:
            account = await session.get(Account, account_id)
            if account:
                account.statement_password = encrypted
                await session.commit()

    async with async_session() as session:
        db_txns = (
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id == account_id)
                )
            )
            .scalars()
            .all()
        )

    recon = reconcile_bank_statement(parsed, db_txns, account_id)
    await enrich_matched_transactions(recon)

    async with async_session() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        upload.status = "parsed"
        upload.account_number = parsed.account_number
        upload.account_holder_name = parsed.account_holder_name
        upload.opening_balance = parsed.opening_balance
        upload.closing_balance = parsed.closing_balance
        upload.statement_period_start = parsed.statement_period_start
        upload.statement_period_end = parsed.statement_period_end
        upload.parsed_txn_count = len(recon["matched"]) + len(recon["missing"])
        upload.matched_count = len(recon["matched"])
        upload.missing_count = len(recon["missing"])
        upload.reconciliation_data = reconciliation_to_json(recon)
        upload.error = None

        link_ctx = await build_link_context(session)
        imported = 0
        for entry in recon["missing"]:
            if entry.get("imported"):
                continue
            try:
                amount = _parse_amount(entry["amount"])
                txn_date = _parse_date(entry["date"])
            except ValueError, KeyError:
                continue
            txn = Transaction(
                bank_statement_upload_id=upload_id,
                account_id=account_id,
                bank=parsed.bank,
                email_type="bank_statement",
                direction=entry["direction"],
                amount=amount,
                currency="INR",
                transaction_date=txn_date,
                counterparty=entry.get("narration"),
                account_mask=_last4(parsed.account_number),
                reference_number=entry.get("reference_number"),
                channel=entry.get("channel") or "bank_statement",
                raw_description=entry.get("narration"),
            )
            session.add(txn)
            await session.flush()
            link_transaction(link_ctx, txn)
            await session.flush()
            entry["imported"] = True
            entry["imported_txn_id"] = txn.id
            imported += 1

        upload.imported_count = imported
        upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
        upload.reconciliation_data = reconciliation_to_json(recon)
        if upload.missing_count == 0:
            upload.status = "imported"
        elif imported > 0:
            upload.status = "partial_import"
        await session.commit()

    return RedirectResponse(url=f"/statements/bank/{upload_id}", status_code=303)


@app.post("/statements/bank/{upload_id}/delete")
async def bank_statement_delete(upload_id: int):
    async with async_session() as session:
        upload = await session.get(BankStatementUpload, upload_id)
        if not upload:
            return RedirectResponse(url="/statements", status_code=303)
        await session.execute(
            update(Transaction)
            .where(Transaction.bank_statement_upload_id == upload_id)
            .values(bank_statement_upload_id=None)
        )
        _unlink_statement_file(upload.file_path)
        await session.delete(upload)
        await session.commit()

    return RedirectResponse(url="/statements", status_code=303)


# ---------------------------------------------------------------------------
# CC statement detail/retry/delete/reprocess routes (wildcard {upload_id})
# ---------------------------------------------------------------------------


@app.get("/statements/{upload_id}", response_class=HTMLResponse)
async def statement_detail(upload_id: int, request: FastAPIRequest):
    from bank_email_fetcher.statements import (
        reconciliation_from_json,
        group_recon_by_person,
    )

    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        if not upload:
            return HTMLResponse("<p>Statement not found.</p>", 404)

    recon = None
    person_groups = []
    card_summaries = []
    if upload.reconciliation_data:
        recon = reconciliation_from_json(upload.reconciliation_data)
        person_groups = group_recon_by_person(recon)
        card_summaries = recon.get("card_summaries", [])

    return templates.TemplateResponse(
        request,
        "statement_reconcile.html",
        {
            "active_page": "statements",
            "upload": upload,
            "recon": recon,
            "person_groups": person_groups,
            "card_summaries": card_summaries,
            "error": request.query_params.get("error"),
        },
    )


@app.post("/statements/{upload_id}/retry")
async def statement_retry(
    upload_id: int,
    password: str = Form(...),
    save_password: str = Form(""),
):
    from bank_email_fetcher.statements import (
        parse_statement,
        reconcile_statement,
        reconciliation_to_json,
        enrich_matched_transactions,
        parse_cc_amount,
        parse_cc_date,
        last4_from_card,
        _extract_digits,
    )
    from bank_email_fetcher.linker import build_link_context, link_transaction

    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        if not upload:
            return RedirectResponse(url="/statements", status_code=303)
        account_id = upload.account_id
        file_path = upload.file_path

    try:
        parsed = await asyncio.to_thread(parse_statement, Path(file_path), password)
    except Exception as e:
        async with async_session() as session:
            upload = await session.get(StatementUpload, upload_id)
            upload.error = str(e)
            await session.commit()
        return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)

    # Save password to account if requested
    if save_password == "1":
        from bank_email_fetcher.config import get_fernet

        encrypted = get_fernet().encrypt(password.encode()).decode()
        async with async_session() as session:
            account = await session.get(Account, account_id)
            if account:
                account.statement_password = encrypted
                await session.commit()
                logger.info("Saved statement password for account %s", account.label)

    async with async_session() as session:
        db_txns = (
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id == account_id)
                )
            )
            .scalars()
            .all()
        )

    recon = reconcile_statement(parsed, db_txns, account_id)
    await enrich_matched_transactions(recon)

    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        account = await session.get(Account, account_id)
        upload.status = "parsed"
        upload.bank = parsed.bank
        upload.card_number = parsed.card_number
        upload.statement_name = parsed.name
        upload.due_date = parsed.due_date
        upload.total_amount_due = parsed.statement_total_amount_due
        upload.parsed_txn_count = len(recon["matched"]) + len(recon["missing"])
        upload.matched_count = len(recon["matched"])
        upload.missing_count = len(recon["missing"])
        upload.reconciliation_data = reconciliation_to_json(recon)
        upload.error = None

        # Auto-import missing transactions
        link_ctx = await build_link_context(session)
        acct_cards = (
            (await session.execute(select(Card).where(Card.account_id == account_id)))
            .scalars()
            .all()
        )
        _card_l4s = [v for v in (last4_from_card(c.card_mask) for c in acct_cards) if v]

        def _resolve_card_mask(raw: str | None) -> str | None:
            l4 = last4_from_card(raw)
            if l4:
                return l4
            partial = _extract_digits(raw)
            if partial:
                for cl4 in _card_l4s:
                    if cl4.endswith(partial):
                        return cl4
            return last4_from_card(account.account_number) if account else None

        imported = 0
        for entry in recon["missing"]:
            if entry.get("imported"):
                continue
            try:
                amount = parse_cc_amount(entry["amount"])
                txn_date = parse_cc_date(entry["date"])
            except ValueError, KeyError:
                continue
            txn = Transaction(
                statement_upload_id=upload_id,
                account_id=account_id,
                bank=parsed.bank,
                email_type="cc_statement",
                direction=entry["direction"],
                amount=amount,
                currency="INR",
                transaction_date=txn_date,
                counterparty=entry.get("narration"),
                card_mask=_resolve_card_mask(entry.get("card_number")),
                channel="cc_statement",
                raw_description=entry.get("narration"),
            )
            session.add(txn)
            await session.flush()
            link_transaction(link_ctx, txn)
            await session.flush()
            entry["imported"] = True
            entry["imported_txn_id"] = txn.id
            imported += 1

        upload.imported_count = imported
        upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
        upload.reconciliation_data = reconciliation_to_json(recon)
        if upload.missing_count == 0:
            upload.status = "imported"
        await session.commit()

    from bank_email_fetcher.reminders import init_payment_tracking

    await init_payment_tracking(upload_id)

    return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)


@app.post("/statements/{upload_id}/delete")
async def statement_delete(upload_id: int):
    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        if not upload:
            return RedirectResponse(url="/statements", status_code=303)
        # Unlink imported transactions (keep them, just clear the FK)
        await session.execute(
            update(Transaction)
            .where(Transaction.statement_upload_id == upload_id)
            .values(statement_upload_id=None)
        )
        _unlink_statement_file(upload.file_path)
        await session.delete(upload)
        await session.commit()

    return RedirectResponse(url="/statements", status_code=303)


@app.post("/statements/{upload_id}/payment")
async def statement_payment(upload_id: int, action: str = Form(...)):
    """Manually toggle a CC statement's payment status (mirrors the Telegram button).

    action=mark_paid   -> set PAID, stamp payment_paid_at, fill payment_paid_amount
                         from total_amount_due if parseable.
    action=mark_unpaid -> revert to UNPAID and replay reminders. Preserves any
                         real partial payment amount (from bank auto-detection)
                         so history isn't lost; only clears the manual full-pay
                         marker. No-op if there is no due date tracked.
    """
    from bank_email_fetcher.statements import parse_cc_amount

    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        if not upload:
            return RedirectResponse(url="/statements", status_code=303)

        if action == "mark_paid":
            if upload.payment_status != PaymentStatus.PAID:
                upload.payment_status = PaymentStatus.PAID
                upload.payment_paid_at = datetime.now(timezone.utc)
                if upload.total_amount_due:
                    try:
                        upload.payment_paid_amount = parse_cc_amount(
                            upload.total_amount_due
                        )
                    except (ValueError, InvalidOperation):
                        pass
                await session.commit()
        elif action == "mark_unpaid":
            if upload.payment_status is not None:
                was_partial = upload.payment_status == PaymentStatus.PARTIALLY_PAID
                upload.payment_status = (
                    PaymentStatus.PARTIALLY_PAID if was_partial else PaymentStatus.UNPAID
                )
                upload.payment_paid_at = None
                if not was_partial:
                    upload.payment_paid_amount = 0
                upload.payment_sent_offsets = "[]"
                upload.payment_last_reminded_at = None
                await session.commit()

    return RedirectResponse(url="/statements", status_code=303)


@app.post("/statements/{upload_id}/reprocess")
async def statement_reprocess(upload_id: int):
    """Re-parse the saved CC statement PDF and rebuild reconciliation data."""
    from bank_email_fetcher.statements import (
        parse_statement,
        reconcile_statement,
        reconciliation_to_json,
        enrich_matched_transactions,
        extract_pdf_from_email,
        _parse_pdf_bytes_sync,
    )

    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        if not upload:
            return RedirectResponse(url="/statements", status_code=303)
        account_id = upload.account_id
        file_path = upload.file_path
        email_id = upload.email_id

    # Try to get password if needed
    password = None
    async with async_session() as session:
        account = await session.get(Account, account_id)
        if account and account.statement_password:
            from bank_email_fetcher.config import get_fernet

            try:
                password = (
                    get_fernet().decrypt(account.statement_password.encode()).decode()
                )
            except Exception:
                pass

    def _reprocess_fail(msg: str):
        logger.warning("Reprocess failed for statement %d: %s", upload_id, msg)
        return RedirectResponse(
            url=f"/statements/{upload_id}?{urlencode({'error': msg})}",
            status_code=303,
        )

    pdf_path = Path(file_path) if file_path else None
    if pdf_path and pdf_path.exists():
        try:
            parsed = await asyncio.to_thread(parse_statement, pdf_path, password)
        except Exception as e:
            return _reprocess_fail(f"Parse error: {e}")
    elif email_id:
        async with async_session() as session:
            email_row = await session.get(Email, email_id)
            if not email_row or not email_row.source_id or not email_row.remote_id:
                return _reprocess_fail("Original email not available for re-fetch")
            source = await session.get(EmailSource, email_row.source_id)
            if not source:
                return _reprocess_fail("Email source not found")
            remote_id = email_row.remote_id
            provider = source.provider
            encrypted_creds = source.credentials

        try:
            creds = decrypt_credentials(encrypted_creds)
        except Exception:
            return _reprocess_fail("Email source credential decryption failed")

        if provider == "gmail":
            raw = await asyncio.to_thread(
                _fetch_gmail_single_sync,
                creds["user"],
                creds["app_password"],
                remote_id,
            )
        elif provider == "fastmail":
            raw = await asyncio.to_thread(
                _fetch_fastmail_single_sync, creds["token"], remote_id
            )
        else:
            raw = None

        if not raw:
            return _reprocess_fail("Could not fetch email from provider")

        pdfs = extract_pdf_from_email(raw)
        if not pdfs:
            return _reprocess_fail("No PDF attachment found in re-fetched email")

        try:
            parsed = await asyncio.to_thread(
                _parse_pdf_bytes_sync, pdfs[0][1], password
            )
        except Exception as e:
            return _reprocess_fail(f"Parse error: {e}")
    else:
        return _reprocess_fail("PDF file missing and no linked email to re-fetch from")

    async with async_session() as session:
        db_txns = (
            (
                await session.execute(
                    select(Transaction).where(Transaction.account_id == account_id)
                )
            )
            .scalars()
            .all()
        )

    recon = reconcile_statement(parsed, db_txns, account_id)
    await enrich_matched_transactions(recon)

    async with async_session() as session:
        upload = await session.get(StatementUpload, upload_id)
        upload.card_number = parsed.card_number
        upload.statement_name = parsed.name
        if (
            upload.due_date != parsed.due_date
            or upload.total_amount_due != parsed.statement_total_amount_due
        ):
            upload.payment_status = None
            upload.payment_paid_amount = 0
            upload.payment_paid_at = None
            upload.payment_sent_offsets = "[]"
            upload.payment_last_reminded_at = None

        upload.due_date = parsed.due_date
        upload.total_amount_due = parsed.statement_total_amount_due
        upload.parsed_txn_count = len(recon["matched"]) + len(recon["missing"])
        upload.matched_count = len(recon["matched"])
        upload.missing_count = sum(1 for e in recon["missing"] if not e.get("imported"))
        upload.reconciliation_data = reconciliation_to_json(recon)
        await session.commit()

    from bank_email_fetcher.reminders import init_payment_tracking

    await init_payment_tracking(upload_id)

    return RedirectResponse(url=f"/statements/{upload_id}", status_code=303)


@app.post("/statements/reprocess-failed")
async def statements_reprocess_failed():
    """Reprocess failed emails that have PDF attachments as statements (CC or bank)."""
    from bank_email_fetcher.statements import (
        process_statement_email,
        extract_pdf_from_email,
    )
    from bank_email_fetcher.bank_statements import process_bank_statement_email

    SPOOL_DIR = Path(__file__).parent / "data" / "failed"
    if not SPOOL_DIR.exists():
        return RedirectResponse(url="/statements", status_code=303)

    async with async_session() as session:
        failed_emails = (
            await session.execute(
                select(Email, FetchRule)
                .join(FetchRule, Email.rule_id == FetchRule.id)
                .where(Email.status == "failed")
            )
        ).all()

    processed = 0
    for email_row, rule in failed_emails:
        import re as _re

        safe_id = _re.sub(r"[^\w\-.]", "_", email_row.message_id)
        spool_name = f"{email_row.provider}_{safe_id}.eml"
        matches = [SPOOL_DIR / spool_name] if (SPOOL_DIR / spool_name).exists() else []
        if not matches:
            continue

        raw_bytes = matches[0].read_bytes()
        pdfs = extract_pdf_from_email(raw_bytes)
        if not pdfs:
            continue

        # Try CC statement first, then bank account statement
        result = None
        try:
            result = await process_statement_email(
                rule.bank,
                raw_bytes,
                email_row.subject or "",
                source_id=email_row.source_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to reprocess email %s as CC statement: %s", email_row.id, e
            )

        if result is None:
            try:
                result = await process_bank_statement_email(
                    rule.bank,
                    raw_bytes,
                    email_row.subject or "",
                    source_id=email_row.source_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to reprocess email %s as bank statement: %s",
                    email_row.id,
                    e,
                )

        if result:
            async with async_session() as session:
                em = await session.get(Email, email_row.id)
                if em:
                    em.status = "parsed"
                    em.error = None
                if result.get("statement_upload_id"):
                    su = await session.get(
                        StatementUpload, result["statement_upload_id"]
                    )
                    if su:
                        su.email_id = email_row.id
                elif result.get("bank_statement_upload_id"):
                    su = await session.get(
                        BankStatementUpload, result["bank_statement_upload_id"]
                    )
                    if su:
                        su.email_id = email_row.id
                await session.commit()
            processed += 1

    logger.info("Reprocessed %d failed emails as statements", processed)
    return RedirectResponse(url="/statements", status_code=303)


@api_router.get("/poll/status")
async def poll_status():
    return JSONResponse(get_poll_status())


@app.post("/poll")
async def trigger_poll():
    global poll_task
    logger.info("Manual poll triggered")
    status = get_poll_status()
    if status["state"] == "polling" or (poll_task and not poll_task.done()):
        logger.info("Manual poll ignored because a background poll is already active")
        return RedirectResponse(url="/", status_code=303)

    poll_task = asyncio.create_task(poll_all())
    poll_task.add_done_callback(_track_poll_task)
    return RedirectResponse(url="/", status_code=303)


app.include_router(api_router)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
