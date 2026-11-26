"""The filters, totals and order of the transactions page.

The web page, its links and the assistant's transaction report share these, so
a report and the page that its link opens list the same rows.
"""

import datetime
from decimal import Decimal
from typing import Literal, NamedTuple
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict
from sqlalchemy import ColumnElement, Select, and_, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Card, Transaction
from financial_dashboard.services.cashflow.buckets import internal_slugs_for_scope
from financial_dashboard.services.cashflow.report import (
    BLANK_CATEGORY,
    BLANK_COUNTERPARTY,
    INR_OR_NULL,
    NON_INR,
    NOT_VOIDED,
    UNCATEGORIZED,
    is_blank_counterparty,
)
from financial_dashboard.services.cashflow.scope import Scope, scope_predicate
from financial_dashboard.services.transaction_reads import (
    build_transaction_filter_clauses,
)

SortKey = Literal["amount", "bank", "category", "counterparty", "date"]
SortOrder = Literal["asc", "desc"]

SORT_COLUMNS = {
    "amount": Transaction.amount,
    "bank": Transaction.bank,
    "category": Transaction.category,
    "counterparty": Transaction.counterparty,
    "date": Transaction.transaction_date,
}


class TransactionTotals(NamedTuple):
    """The count and the INR amounts of the filtered rows.

    A foreign-currency row is a card alert that no statement converted to
    rupees yet, so the sums leave it out and count it. A voided row moved no
    money.

    Attributes:
        count: The number of filtered rows.
        credits: The sum of the INR credits.
        debits: The sum of the INR debits.
        net: Credits minus debits.
        foreign_count: The live rows not in rupees, which the sums leave out.
    """

    count: int
    credits: Decimal
    debits: Decimal
    net: Decimal
    foreign_count: int


class TransactionFilters(BaseModel):
    """The filters of the transactions page, as its URL query names them.

    The page, its links and the assistant's transaction report use this one
    model, so a report and the page that its link opens list the same rows.
    Every field is optional, and an empty model selects every row.

    Attributes:
        bank: The bank, matched without case.
        account_ids: Rows on any of these accounts.
        card_ids: Rows with any of these cards. A listed card narrows its own
            listed account to that card.
        direction: ``debit`` or ``credit``.
        date_from: The first date.
        date_to: The last date.
        category: The category slug.
        counterparty: The exact counterparty. A blank text selects the rows
            with no counterparty.
        card_holder: The cardholder that the card statement names.
        uncategorized: Only rows outside the known category buckets.
        category_null: Only rows with no category.
        internal: Only internal-movement rows for the scope.
        non_inr: True for foreign rows, False for rupee rows, None for both.
        undated: Only live rows with no date.
        excluded: True for rows out of the cashflow, False for rows in it,
            None for both.
        provisional: Only screenshot rows that no bank source confirms yet.
        scope: The account perimeter of a cashflow figure.
        q: Text to find in the row's text fields.
        amount: The exact amount.
        source: The exact row source.
        review_status: The category review status.
        reference: A payment reference.
    """

    model_config = ConfigDict(frozen=True)

    bank: str | None = None
    account_ids: tuple[int, ...] = ()
    card_ids: tuple[int, ...] = ()
    direction: str | None = None
    date_from: datetime.date | None = None
    date_to: datetime.date | None = None
    category: str | None = None
    counterparty: str | None = None
    card_holder: str | None = None
    uncategorized: bool = False
    category_null: bool = False
    internal: bool = False
    non_inr: bool | None = None
    undated: bool = False
    excluded: bool | None = None
    provisional: bool = False
    scope: Scope | None = None
    q: str | None = None
    amount: Decimal | None = None
    source: str | None = None
    review_status: str | None = None
    reference: str | None = None

    def clauses(self) -> list[ColumnElement[bool]]:
        """Return the SQL clauses that select the filtered rows.

        Returns:
            Clauses to AND together on ``Transaction``.
        """
        clauses = build_transaction_filter_clauses(
            bank=self.bank,
            direction=self.direction,
            date_from=self.date_from,
            date_to=self.date_to,
            category=self.category,
            amount=self.amount,
            source=self.source,
            review_status=self.review_status,
            reference=self.reference,
            search=self.q,
            excluded=self.excluded,
        )
        if self.account_ids or self.card_ids:
            clauses.append(self._account_or_card())
        if self.counterparty is not None:
            clauses.append(
                BLANK_COUNTERPARTY
                if is_blank_counterparty(self.counterparty)
                else Transaction.counterparty == self.counterparty
            )
        if self.card_holder:
            clauses.append(Transaction.card_holder == self.card_holder)
        if self.uncategorized:
            clauses.append(UNCATEGORIZED)
        if self.category_null:
            # Narrower than uncategorized: only the rows with no category, so a
            # report line that counts them links to exactly those rows.
            clauses.append(BLANK_CATEGORY)
        if self.internal:
            # The account scope decides which slugs are internal: over the bank a
            # card bill is money that leaves it.
            clauses.append(
                Transaction.category.in_(tuple(internal_slugs_for_scope(self.scope)))
            )
        if self.non_inr is True:
            clauses.append(NON_INR)
        elif self.non_inr is False:
            clauses.append(INR_OR_NULL)
        if self.undated:
            clauses.extend((Transaction.transaction_date.is_(None), NOT_VOIDED))
        if self.provisional:
            # A bank match appends its source, so a bare "screenshot" is unconfirmed.
            clauses.extend((Transaction.source == "screenshot", NOT_VOIDED))
        if (account_scope := scope_predicate(self.scope)) is not None:
            clauses.append(account_scope)
        return clauses

    def _account_or_card(self) -> ColumnElement[bool]:
        """Select the rows on a listed account or with a listed card.

        A listed card narrows its own account to that card, so choosing an
        account and then one of its cards lists that card's rows.

        Returns:
            One clause for the listed accounts and cards.
        """
        on_card = Transaction.card_id.in_(self.card_ids)
        if not self.account_ids:
            return on_card
        on_account = Transaction.account_id.in_(self.account_ids)
        if not self.card_ids:
            return on_account
        narrowed = select(Card.account_id).where(Card.id.in_(self.card_ids))
        return or_(
            on_card,
            and_(on_account, Transaction.account_id.not_in(narrowed)),
        )

    def query(self) -> list[tuple[str, str]]:
        """Return the URL query pairs of the transactions page for these filters.

        Returns:
            The pairs in the order the page shows them. Each listed id is its
            own pair. A blank counterparty stays, because it narrows the rows.
        """
        pairs: list[tuple[str, str | None]] = [
            ("bank", self.bank),
            *(("account_id", str(value)) for value in self.account_ids),
            *(("card_id", str(value)) for value in self.card_ids),
            ("direction", self.direction),
            ("date_from", self.date_from.isoformat() if self.date_from else None),
            ("date_to", self.date_to.isoformat() if self.date_to else None),
            ("category", self.category),
            ("counterparty", self.counterparty),
            ("card_holder", self.card_holder),
            ("uncategorized", _flag(self.uncategorized)),
            ("category_null", _flag(self.category_null)),
            ("internal", _flag(self.internal)),
            ("non_inr", _three_way(self.non_inr)),
            ("undated", _flag(self.undated)),
            ("excluded", _three_way(self.excluded)),
            ("provisional", _flag(self.provisional)),
            ("scope", self.scope),
            ("q", self.q),
            ("amount", None if self.amount is None else str(self.amount)),
            ("source", self.source),
            ("review_status", self.review_status),
            ("reference", self.reference),
        ]
        return [
            (key, value)
            for key, value in pairs
            if value is not None and (value != "" or key == "counterparty")
        ]


def _flag(value: bool) -> str | None:
    """Return the "1" that an on/off filter takes in the URL.

    Args:
        value: The filter.

    Returns:
        "1" when the filter is on, or None for no parameter.
    """
    return "1" if value else None


def _three_way(value: bool | None) -> str | None:
    """Return the "1" or "0" that a three-way filter takes in the URL.

    Args:
        value: True, False, or None for no filter.

    Returns:
        "1", "0", or None for no parameter.
    """
    return None if value is None else _flag(value) or "0"


def transactions_path(
    filters: TransactionFilters, sort: SortKey = "date", order: SortOrder = "desc"
) -> str:
    """Return the path of the transactions page that lists the filtered rows.

    Args:
        filters: The filters.
        sort: The sort column.
        order: The sort order.

    Returns:
        The path and query, such as ``/transactions?account_id=1&sort=date``.
    """
    return f"/transactions?{urlencode([*filters.query(), ('sort', sort), ('order', order)])}"


def ordered(stmt: Select, sort: SortKey, order: SortOrder) -> Select:
    """Order a transaction query as the transactions page does.

    Args:
        stmt: The query.
        sort: The sort column.
        order: The sort order. The id breaks ties, so pages do not overlap.

    Returns:
        The ordered query.
    """
    column = SORT_COLUMNS[sort]
    if order == "asc":
        return stmt.order_by(column.asc().nullslast(), Transaction.id.asc())
    return stmt.order_by(column.desc().nullslast(), Transaction.id.desc())


async def transaction_totals(session: AsyncSession, stmt: Select) -> TransactionTotals:
    """Count the filtered rows and sum their INR credits and debits.

    Args:
        session: The open database session.
        stmt: The filtered query, before order and page.

    Returns:
        The row count and the totals.
    """
    filtered = stmt.subquery()
    live = filtered.c.voided.is_not(True)
    is_inr = (
        func.coalesce(
            func.nullif(func.upper(func.trim(filtered.c.currency)), ""), "INR"
        )
        == "INR"
    ) & live
    row = (
        await session.execute(
            select(
                func.count(),
                func.sum(
                    case(
                        (is_inr & (filtered.c.direction == "credit"), filtered.c.amount)
                    )
                ),
                func.sum(
                    case(
                        (is_inr & (filtered.c.direction == "debit"), filtered.c.amount)
                    )
                ),
                func.count().filter(
                    live & ~is_inr & filtered.c.direction.in_(("credit", "debit"))
                ),
            )
        )
    ).one()
    count, credits, debits, foreign = row
    credits = Decimal(str(credits or 0))
    debits = Decimal(str(debits or 0))
    return TransactionTotals(
        count=count,
        credits=credits,
        debits=debits,
        net=credits - debits,
        foreign_count=foreign,
    )
