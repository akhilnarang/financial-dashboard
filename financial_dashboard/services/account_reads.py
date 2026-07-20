"""Bounded, redacted account queries for the JSON API."""

from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from financial_dashboard.core.masks import display_mask
from financial_dashboard.db import (
    Account,
    BalanceSnapshot,
    BankStatementUpload,
    Card,
    StatementUpload,
    Transaction,
)
from financial_dashboard.schemas import accounts as account_schemas

_CARD_LIMIT_PER_ACCOUNT = 50
_BALANCE_CATEGORY_LIMIT = 50


def _card_read(row) -> account_schemas.AccountCardRead:
    """Map one projected card row to its redacted API schema."""
    return account_schemas.AccountCardRead(
        id=row.id,
        label=row.label,
        is_primary=bool(row.is_primary),
        active=bool(row.active),
        card_mask=display_mask(row.card_mask),
    )


async def _cards_by_account(
    session: AsyncSession,
    account_ids: list[int],
) -> dict[int, tuple[list[account_schemas.AccountCardRead], bool]]:
    """Load capped card summaries for several accounts in one query."""
    if not account_ids:
        return {}

    ranked = (
        select(
            Card.id.label("id"),
            Card.account_id.label("account_id"),
            Card.card_mask.label("card_mask"),
            Card.label.label("label"),
            Card.is_primary.label("is_primary"),
            Card.active.label("active"),
            func.row_number()
            .over(
                partition_by=Card.account_id,
                order_by=(Card.is_primary.desc(), Card.id.asc()),
            )
            .label("position"),
        )
        .where(Card.account_id.in_(account_ids))
        .subquery()
    )
    rows = (
        await session.execute(
            select(ranked)
            .where(ranked.c.position <= _CARD_LIMIT_PER_ACCOUNT + 1)
            .order_by(ranked.c.account_id, ranked.c.position)
            .execution_options(autoflush=False)
        )
    ).all()

    grouped: defaultdict[int, list] = defaultdict(list)
    for row in rows:
        grouped[row.account_id].append(row)

    result: dict[int, tuple[list[account_schemas.AccountCardRead], bool]] = {}
    for account_id in account_ids:
        account_rows = grouped[account_id]
        result[account_id] = (
            [_card_read(row) for row in account_rows[:_CARD_LIMIT_PER_ACCOUNT]],
            len(account_rows) > _CARD_LIMIT_PER_ACCOUNT,
        )
    return result


def _account_read(
    account: Account,
    cards: tuple[list[account_schemas.AccountCardRead], bool],
) -> account_schemas.AccountRead:
    """Map one account and its bounded cards to the summary schema."""
    card_items, cards_truncated = cards
    return account_schemas.AccountRead(
        id=account.id,
        bank=account.bank,
        label=account.label,
        type=account.type,
        active=bool(account.active),
        account_mask=display_mask(account.account_number),
        cards=card_items,
        cards_truncated=cards_truncated,
    )


async def list_accounts(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
    bank: str | None,
    account_type: str | None,
    active: bool | None,
) -> account_schemas.AccountListResponse:
    """Return one stable, filtered account page with bounded cards."""
    filters = []
    if bank is not None:
        filters.append(func.lower(Account.bank) == bank.strip().lower())
    if account_type is not None:
        filters.append(Account.type == account_type.strip())
    if active is not None:
        filters.append(Account.active.is_(active))

    with session.no_autoflush:
        total_count = await session.scalar(
            select(func.count(Account.id))
            .where(*filters)
            .execution_options(autoflush=False)
        )
        accounts = (
            (
                await session.execute(
                    select(Account)
                    .options(noload(Account.cards))
                    .where(*filters)
                    .order_by(Account.id)
                    .offset(offset)
                    .limit(limit)
                    .execution_options(autoflush=False)
                )
            )
            .scalars()
            .all()
        )
        cards_by_account = await _cards_by_account(
            session, [account.id for account in accounts]
        )

    items = [
        _account_read(account, cards_by_account.get(account.id, ([], False)))
        for account in accounts
    ]
    return account_schemas.AccountListResponse(
        items=items,
        returned_count=len(items),
        total_count=total_count or 0,
        limit=limit,
        offset=offset,
    )


async def _latest_balance_snapshots(
    session: AsyncSession,
    account_id: int,
) -> list[account_schemas.AccountBalanceSnapshotRead]:
    """Load the newest snapshot in each bounded balance category."""
    ranked = (
        select(
            BalanceSnapshot.id.label("snapshot_id"),
            BalanceSnapshot.category.label("category"),
            BalanceSnapshot.as_of_date.label("as_of_date"),
            BalanceSnapshot.value.label("value"),
            BalanceSnapshot.currency.label("currency"),
            func.row_number()
            .over(
                partition_by=BalanceSnapshot.category,
                order_by=(
                    BalanceSnapshot.as_of_date.desc(),
                    BalanceSnapshot.id.desc(),
                ),
            )
            .label("position"),
        )
        .where(BalanceSnapshot.account_id == account_id)
        .subquery()
    )
    rows = (
        await session.execute(
            select(ranked)
            .where(ranked.c.position == 1)
            .order_by(ranked.c.category)
            .limit(_BALANCE_CATEGORY_LIMIT)
            .execution_options(autoflush=False)
        )
    ).all()
    return [
        account_schemas.AccountBalanceSnapshotRead(
            snapshot_id=row.snapshot_id,
            category=row.category,
            as_of_date=row.as_of_date,
            value=row.value,
            currency=row.currency,
        )
        for row in rows
    ]


async def get_account_detail(
    session: AsyncSession,
    account_id: int,
) -> account_schemas.AccountDetailResponse | None:
    """Return one account's bounded cards, counts, and latest balances."""
    with session.no_autoflush:
        account = await session.scalar(
            select(Account)
            .options(noload(Account.cards))
            .where(Account.id == account_id)
            .execution_options(autoflush=False)
        )
        if account is None:
            return None

        cards = await _cards_by_account(session, [account_id])
        counts = (
            await session.execute(
                select(
                    select(func.count(Transaction.id))
                    .where(Transaction.account_id == account_id)
                    .scalar_subquery()
                    .label("transaction_count"),
                    select(func.count(StatementUpload.id))
                    .where(StatementUpload.account_id == account_id)
                    .scalar_subquery()
                    .label("cc_statement_count"),
                    select(func.count(BankStatementUpload.id))
                    .where(BankStatementUpload.account_id == account_id)
                    .scalar_subquery()
                    .label("bank_statement_count"),
                ).execution_options(autoflush=False)
            )
        ).one()
        snapshots = await _latest_balance_snapshots(session, account_id)

    summary = _account_read(account, cards.get(account_id, ([], False)))
    return account_schemas.AccountDetailResponse(
        **summary.model_dump(),
        transaction_count=counts.transaction_count,
        cc_statement_count=counts.cc_statement_count,
        bank_statement_count=counts.bank_statement_count,
        latest_balance_snapshots=snapshots,
    )
