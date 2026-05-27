"""Balance snapshot helpers.

The `on_conflict_do_update` calls below use the SQLite dialect's INSERT ...
ON CONFLICT syntax targeting partial unique indexes via `index_where`. This
is SQLite-specific (the dashboard is SQLite-only via aiosqlite); porting to
another backend would require a dialect-aware upsert.
"""

import datetime
import json
from decimal import Decimal
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from financial_dashboard.db.models import BalanceSnapshot
from financial_dashboard.db.enums import SnapshotCategory, SnapshotKind, SnapshotSource


def _enum_value(value: str | StrEnum) -> str:
    return value.value if isinstance(value, StrEnum) else str(value)


async def upsert_account_snapshot(
    session: AsyncSession,
    *,
    account_id: int,
    kind: str | StrEnum,
    category: str | StrEnum,
    as_of_date: datetime.date,
    value: Decimal,
    source: str | StrEnum,
    currency: str = "INR",
) -> None:
    category_value = _enum_value(category)
    stmt = sqlite_insert(BalanceSnapshot).values(
        account_id=account_id,
        kind=_enum_value(kind),
        category=category_value,
        as_of_date=as_of_date,
        value=value,
        source=_enum_value(source),
        currency=currency,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["account_id", "category", "as_of_date"],
        index_where=BalanceSnapshot.account_id.isnot(None),
        set_={
            "kind": _enum_value(kind),
            "value": value,
            "source": _enum_value(source),
            "currency": currency,
        },
    )
    await session.execute(stmt)


async def upsert_manual_snapshot(
    session: AsyncSession,
    *,
    manual_item_id: int,
    kind: str | StrEnum,
    category: str | StrEnum,
    as_of_date: datetime.date,
    value: Decimal,
    source: str | StrEnum,
    currency: str = "INR",
) -> None:
    stmt = sqlite_insert(BalanceSnapshot).values(
        manual_item_id=manual_item_id,
        kind=_enum_value(kind),
        category=_enum_value(category),
        as_of_date=as_of_date,
        value=value,
        source=_enum_value(source),
        currency=currency,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["manual_item_id", "as_of_date"],
        index_where=BalanceSnapshot.manual_item_id.isnot(None),
        set_={
            "kind": _enum_value(kind),
            "category": _enum_value(category),
            "value": value,
            "source": _enum_value(source),
            "currency": currency,
        },
    )
    await session.execute(stmt)


async def emit_bank_snapshot(session: AsyncSession, upload) -> bool:
    from financial_dashboard.services.statements.bank import _parse_amount, _parse_date

    if not upload.closing_balance or not upload.statement_period_end:
        return False
    try:
        value = _parse_amount(upload.closing_balance)
        as_of_date = _parse_date(upload.statement_period_end)
    except Exception:
        return False

    await upsert_account_snapshot(
        session,
        account_id=upload.account_id,
        kind=SnapshotKind.asset,
        category=SnapshotCategory.bank_balance,
        as_of_date=as_of_date,
        value=value,
        source=SnapshotSource.bank_statement,
    )
    return True


def _cc_statement_date(upload) -> datetime.date:
    from financial_dashboard.services.statements.cc import parse_cc_date

    dates: list[datetime.date] = []
    if upload.reconciliation_data:
        try:
            recon = json.loads(upload.reconciliation_data)
        except json.JSONDecodeError:
            recon = {}
        for key in ("matched", "missing"):
            for entry in recon.get(key) or []:
                if date_text := entry.get("date"):
                    try:
                        dates.append(parse_cc_date(date_text))
                    except Exception:
                        pass
        for pair in recon.get("adjustment_pairs") or []:
            for key in ("debit_date", "credit_date"):
                if date_text := pair.get(key):
                    try:
                        dates.append(parse_cc_date(date_text))
                    except Exception:
                        pass
    if dates:
        return max(dates)
    created_at = upload.created_at or datetime.datetime.now(datetime.UTC)
    return created_at.date()


async def emit_cc_snapshot(session: AsyncSession, upload) -> bool:
    from financial_dashboard.db.enums import PaymentStatus
    from financial_dashboard.services.statements.cc import parse_cc_amount

    if not upload.total_amount_due:
        return False
    try:
        amount_due = parse_cc_amount(upload.total_amount_due)
    except Exception:
        return False

    paid_amount = upload.payment_paid_amount or Decimal("0.00")
    status_value = str(upload.payment_status) if upload.payment_status else None
    if status_value == PaymentStatus.PAID.value or amount_due <= 0:
        value = Decimal("0.00")
    else:
        value = max(amount_due - paid_amount, Decimal("0.00"))

    await upsert_account_snapshot(
        session,
        account_id=upload.account_id,
        kind=SnapshotKind.liability,
        category=SnapshotCategory.cc_outstanding,
        as_of_date=_cc_statement_date(upload),
        value=value,
        source=SnapshotSource.cc_statement,
    )
    return True
