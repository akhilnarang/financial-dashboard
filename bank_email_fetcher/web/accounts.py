"""Account HTML routes."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from bank_email_fetcher.config import get_fernet
from bank_email_fetcher.core.deps import get_session
from bank_email_fetcher.core.templating import get_templates
from bank_email_fetcher.db import (
    Account,
    Card,
    Transaction,
)
from bank_email_parser import SUPPORTED_BANKS
from bank_email_fetcher.services.accounts import (
    auto_link_account,
    retry_password_required_statements as accounts_retry_password_required_statements,
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


@router.get("/accounts", response_class=HTMLResponse)
async def account_list(
    request: FastAPIRequest,
    bank: Annotated[str | None, Query(description="Filter by bank name")] = None,
    type: Annotated[
        str | None,
        Query(
            description="Filter by account type: bank_account, credit_card, debit_card"
        ),
    ] = None,
    active: Annotated[
        str | None, Query(description="Filter by active status: true or false")
    ] = None,
    session: AsyncSession = Depends(get_session),
):
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


@router.post("/accounts")
async def account_create(
    request: FastAPIRequest,
    bank: str = Form(...),
    label: str = Form(...),
    type: str = Form(...),
    account_number: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    account = Account(
        bank=bank.strip(),
        label=label.strip(),
        type=type,
        account_number=account_number.strip() or None,
    )
    session.add(account)
    await session.commit()
    await session.refresh(account)
    await auto_link_account(session, account)

    return RedirectResponse(url="/accounts", status_code=303)


@router.get("/accounts/{account_id}/edit", response_class=HTMLResponse)
async def account_edit_form(
    request: FastAPIRequest,
    account_id: int,
    session: AsyncSession = Depends(get_session),
):
    account = await session.get(Account, account_id)
    if not account:
        return RedirectResponse(url="/accounts", status_code=303)

    acct_bank_result = await session.execute(select(Account.bank).distinct())
    txn_bank_result = await session.execute(select(Transaction.bank).distinct())
    bank_set = {row[0] for row in acct_bank_result.all() if row[0]} | {
        row[0] for row in txn_bank_result.all() if row[0]
    }
    if account.bank:
        bank_set.add(account.bank)
    banks = sorted(bank_set)

    # Decrypt statement password for display
    statement_password_plain = ""
    if account.statement_password:
        try:
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


@router.post("/accounts/{account_id}/edit")
async def account_update(
    request: FastAPIRequest,
    account_id: int,
    bank: str = Form(...),
    label: str = Form(...),
    type: str = Form(...),
    account_number: str = Form(""),
    statement_password: str = Form(""),
    statement_password_hint: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    account = await session.get(Account, account_id)
    if not account:
        return RedirectResponse(url="/accounts", status_code=303)

    account.bank = bank.strip()
    account.label = label.strip()
    account.type = type
    account.account_number = account_number.strip() or None
    account.statement_password_hint = statement_password_hint.strip() or None

    if statement_password.strip():
        account.statement_password = (
            get_fernet().encrypt(statement_password.strip().encode()).decode()
        )
    elif not statement_password:
        account.statement_password = None

    await session.commit()
    await session.refresh(account)
    await auto_link_account(session, account)

    # Automatically retry password-required statements when password is provided
    if statement_password.strip():
        try:
            retry_result = await accounts_retry_password_required_statements(
                session, account_id, statement_password.strip()
            )
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


@router.post("/accounts/{account_id}/delete")
async def account_delete(
    account_id: int,
    session: AsyncSession = Depends(get_session),
):
    account = await session.get(Account, account_id)
    if account:
        await session.execute(
            update(Transaction)
            .where(Transaction.account_id == account_id)
            .values(account_id=None)
        )
        await session.delete(account)
        await session.commit()

    return RedirectResponse(url="/accounts", status_code=303)


@router.post("/accounts/{account_id}/cards")
async def card_add(
    account_id: int,
    card_mask: str = Form(...),
    label: str = Form(""),
    is_primary: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
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


@router.post("/accounts/{account_id}/cards/{card_id}/edit")
async def card_edit(
    account_id: int,
    card_id: int,
    label: str = Form(...),
    is_primary: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    card = await session.get(Card, card_id)
    if card and card.account_id == account_id:
        card.label = label.strip() or card.label
        card.is_primary = is_primary == "1"
        await session.commit()
    return RedirectResponse(url=f"/accounts/{account_id}/edit", status_code=303)


@router.post("/accounts/{account_id}/cards/{card_id}/delete")
async def card_delete(
    account_id: int,
    card_id: int,
    session: AsyncSession = Depends(get_session),
):
    card = await session.get(Card, card_id)
    if card and card.account_id == account_id:
        await session.execute(
            update(Transaction)
            .where(Transaction.card_id == card_id)
            .values(card_id=None)
        )
        await session.delete(card)
        await session.commit()
    return RedirectResponse(url=f"/accounts/{account_id}/edit", status_code=303)
