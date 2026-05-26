from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from sqlalchemy import and_, or_, select
from sqlalchemy import func as sa_func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db import Email, PaisaExport, SmsMessage, async_session
from financial_dashboard.services.parser_quirks import AMBIGUOUS_12H_TIME_EMAIL_TYPES
from financial_dashboard.services.settings import PaisaConfig, get_paisa_config
from financial_dashboard.services.txn_merge import (
    _FUZZY_MATCH_WINDOW_MINUTES,
    _counterparty_match,
    _normalize_counterparty,
)

PaisaSource = Literal["sms", "email"]
PaisaMatchKind = Literal["standard", "am_pm_alias"]
_MASK_FIELDS = ("card_mask", "account_mask")
_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
_DEFAULT_TZ_TIME_DATE = dt.date(2000, 1, 1)
# Process-local lock only; cross-process safety depends on atomic os.replace().
_REWRITE_LOCK = asyncio.Lock()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaisaAccounts:
    source_account: str
    counterparty_account: str
    missing_account_mapping: bool
    comment: str | None = None


@dataclass(frozen=True)
class PaisaExportOutcome:
    status: Literal["parsed", "skipped", "error"]
    export_id: int | None
    needs_journal_rewrite: bool = False
    error: str | None = None


def _digits_only(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value if ch.isdigit())


def _normalized_bank(value: object) -> str:
    return str(value or "").strip().lower()


def _normalized_direction(value: object) -> str:
    return str(value or "").strip().lower()


def _normalized_reference(value: object) -> str:
    return str(value or "").strip()


def _normalized_currency(value: object) -> str:
    cur = str(value or "").strip().upper()
    return cur or "INR"


def _quantized_amount(value: object) -> Decimal:
    dec = Decimal(str(value or "0"))
    return dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _as_date(value: object) -> dt.date | None:
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            value = value.astimezone(_IST)
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


def _as_time(value: object) -> dt.time | None:
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            value = value.astimezone(_IST)
        value = value.time()
    if isinstance(value, dt.time):
        if value.tzinfo is not None and value.utcoffset() is not None:
            aware = dt.datetime.combine(_DEFAULT_TZ_TIME_DATE, value)
            value = aware.astimezone(_IST).time().replace(tzinfo=None)
        return value.replace(microsecond=0)
    return None


def _normalized_wall_date_time(
    date_value: object, time_value: object
) -> tuple[dt.date | None, dt.time | None]:
    date_part = _as_date(date_value)
    time_part = _as_time(time_value)
    if (
        isinstance(time_value, dt.time)
        and time_value.tzinfo is not None
        and time_value.utcoffset() is not None
        and date_part is not None
    ):
        aware = dt.datetime.combine(date_part, time_value)
        wall = aware.astimezone(_IST)
        return wall.date(), wall.time().replace(microsecond=0, tzinfo=None)
    return date_part, time_part


def _looks_like_parser_shape_collision(txn_data: dict) -> bool:
    if _digits_only(txn_data.get("card_mask")):
        return False
    if _digits_only(txn_data.get("account_mask")):
        return False
    if _normalize_counterparty(txn_data.get("counterparty")):
        return False
    return True


def build_paisa_idempotency_key(txn_data: dict) -> str:
    bank = _normalized_bank(txn_data.get("bank"))
    direction = _normalized_direction(txn_data.get("direction"))

    reference = _normalized_reference(txn_data.get("reference_number"))
    if reference:
        return f"ref|{bank}|{reference}|{direction}"

    amount = _quantized_amount(txn_data.get("amount"))
    currency = _normalized_currency(txn_data.get("currency"))
    txn_date, txn_time = _normalized_wall_date_time(
        txn_data.get("transaction_date"),
        txn_data.get("transaction_time"),
    )
    counterparty = _normalize_counterparty(txn_data.get("counterparty"))
    card_mask = _digits_only(txn_data.get("card_mask"))
    account_mask = _digits_only(txn_data.get("account_mask"))

    pieces = [
        "fp",
        bank,
        direction,
        f"{amount:.2f}",
        currency,
        txn_date.isoformat() if txn_date else "",
        txn_time.isoformat() if txn_time else "",
        counterparty,
        card_mask,
        account_mask,
    ]
    if _looks_like_parser_shape_collision(txn_data):
        pieces.append(str(txn_data.get("email_type") or "").strip().lower())
    return "|".join(pieces)


def resolve_paisa_accounts(txn_data: dict, config: PaisaConfig) -> PaisaAccounts:
    bank = _normalized_bank(txn_data.get("bank"))
    direction = _normalized_direction(txn_data.get("direction"))
    card_mask_digits = _digits_only(txn_data.get("card_mask"))
    account_mask_digits = _digits_only(txn_data.get("account_mask"))

    source_key = ""
    source_account = ""
    missing_map = False

    if card_mask_digits:
        source_key = f"{bank}:card:{card_mask_digits}"
        source_account = config.account_map.get(source_key, "")
        if not source_account:
            source_account = config.fallback_liability_account
            missing_map = True
    elif account_mask_digits:
        source_key = f"{bank}:account:{account_mask_digits}"
        source_account = config.account_map.get(source_key, "")
        if not source_account:
            source_account = config.fallback_asset_account
            missing_map = True
    else:
        source_account = config.fallback_asset_account
        missing_map = True

    counterparty_account = (
        config.default_income_account
        if direction == "credit"
        else config.default_expense_account
    )

    comment: str | None = None
    if missing_map:
        if card_mask_digits and source_key:
            comment = f"missing-map card-mask={card_mask_digits} source={source_key}"
        elif account_mask_digits and source_key:
            comment = (
                f"missing-map account-mask={account_mask_digits} source={source_key}"
            )
        else:
            comment = f"missing-map {bank}:account-mask="

    return PaisaAccounts(
        source_account=source_account,
        counterparty_account=counterparty_account,
        missing_account_mapping=missing_map,
        comment=comment,
    )


def _sanitize_ledger_text(value: object, *, fallback: str = "") -> str:
    if value is None:
        return fallback
    normalized = []
    for ch in str(value):
        if ch in ("\r", "\n") or ord(ch) < 32 or ord(ch) == 127:
            normalized.append(" ")
        else:
            normalized.append(ch)
    compact = " ".join("".join(normalized).split())
    return compact or fallback


def _ledger_posting(account: str, amount: Decimal) -> str:
    account = _sanitize_ledger_text(account, fallback="Unknown")
    signed = f"{amount:.2f}"
    account_pad = max(0, 34 - len(account))
    amount_pad = max(0, 8 - len(signed))
    gap = max(2, account_pad + amount_pad)
    return f"    {account}{' ' * gap}{signed} INR\n"


def _ledger_meta(value: object) -> str:
    return _sanitize_ledger_text(value)


def _missing_map_comment(export: PaisaExport) -> str:
    bank = _normalized_bank(export.bank)
    card_mask_digits = _digits_only(export.card_mask)
    account_mask_digits = _digits_only(export.account_mask)
    if card_mask_digits:
        return f"missing-map card-mask={card_mask_digits} source={bank}:card:{card_mask_digits}"
    if account_mask_digits:
        return f"missing-map account-mask={account_mask_digits} source={bank}:account:{account_mask_digits}"
    return f"missing-map {bank}:account-mask="


def render_paisa_journal_entry(export: PaisaExport) -> str:
    if export.transaction_date is None:
        logger.warning(
            "Skipping journal render for paisa_export=%s without date", export.id
        )
        return ""
    txn_date = export.transaction_date
    payee = _sanitize_ledger_text(export.counterparty, fallback="Unknown")
    amount = _quantized_amount(export.amount)

    if _normalized_direction(export.direction) == "credit":
        first_account = export.source_account or "Assets:Unknown"
        second_account = export.counterparty_account or "Income:Uncategorized"
        first_amount = amount
        second_amount = -amount
    else:
        first_account = export.counterparty_account or "Expenses:Uncategorized"
        second_account = export.source_account or "Liabilities:Unknown"
        first_amount = amount
        second_amount = -amount

    lines = [
        f"{txn_date:%Y/%m/%d} {payee}\n",
        _ledger_posting(first_account, first_amount),
        _ledger_posting(second_account, second_amount),
    ]
    if export.missing_account_mapping:
        lines.append(f"    ; {_ledger_meta(_missing_map_comment(export))}\n")
    lines.append(
        "    ; financial-dashboard:id="
        f"{_ledger_meta(export.id)} "
        f"source={_ledger_meta(export.source)} "
        f"sms_id={_ledger_meta(export.sms_message_id)} "
        f"email_id={_ledger_meta(export.email_id)} "
        f"ref={_ledger_meta(export.reference_number)}\n"
    )
    return "".join(lines)


def _incoming_time_is_earlier(
    *,
    existing_date: dt.date | None,
    existing_time: dt.time | None,
    incoming_date: dt.date | None,
    incoming_time: dt.time,
) -> bool:
    if existing_time is None:
        return True
    incoming_date = incoming_date or existing_date
    if existing_date is not None and incoming_date is not None:
        return dt.datetime.combine(incoming_date, incoming_time) < dt.datetime.combine(
            existing_date, existing_time
        )
    return incoming_time < existing_time


async def _find_match_with_kind(
    session: AsyncSession,
    txn_data: dict,
    *,
    source: PaisaSource,
    email_id: int | None,
    sms_message_id: int | None,
) -> tuple[PaisaExport, PaisaMatchKind] | None:
    del source  # source is intentionally ignored for matching.

    if sms_message_id is not None:
        sms_result = await session.execute(
            select(PaisaExport).where(PaisaExport.sms_message_id == sms_message_id)
        )
        sms_rows = sms_result.scalars().all()
        if len(sms_rows) == 1:
            return sms_rows[0], "standard"
        if len(sms_rows) > 1:
            return None

    if email_id is not None:
        email_result = await session.execute(
            select(PaisaExport).where(PaisaExport.email_id == email_id)
        )
        email_rows = email_result.scalars().all()
        if len(email_rows) == 1:
            return email_rows[0], "standard"
        if len(email_rows) > 1:
            return None

    bank = _normalized_bank(txn_data.get("bank"))
    direction = _normalized_direction(txn_data.get("direction"))
    reference = (txn_data.get("reference_number") or "").strip()
    if reference:
        result = await session.execute(
            select(PaisaExport).where(
                PaisaExport.bank == bank,
                PaisaExport.direction == direction,
                PaisaExport.reference_number == reference,
            )
        )
        rows = result.scalars().all()
        if len(rows) == 1:
            return rows[0], "standard"
        if len(rows) > 1:
            return None

    txn_date = _as_date(txn_data.get("transaction_date"))
    if txn_date is None:
        return None

    incoming_time = _as_time(txn_data.get("transaction_time"))
    incoming_currency = _normalized_currency(txn_data.get("currency"))

    if incoming_time is not None:
        anchor = dt.datetime.combine(txn_date, incoming_time)
        lower = anchor - dt.timedelta(minutes=_FUZZY_MATCH_WINDOW_MINUTES)
        upper = anchor + dt.timedelta(minutes=_FUZZY_MATCH_WINDOW_MINUTES)
        date_lower = lower.date()
        date_upper = upper.date()
    else:
        lower = upper = None
        date_lower = txn_date
        date_upper = txn_date

    result = await session.execute(
        select(PaisaExport).where(
            PaisaExport.bank == bank,
            PaisaExport.direction == direction,
            PaisaExport.amount == _quantized_amount(txn_data.get("amount")),
            sa_func.coalesce(PaisaExport.currency, "INR") == incoming_currency,
            PaisaExport.status != "error",
            PaisaExport.transaction_date.is_not(None),
            PaisaExport.transaction_date >= date_lower,
            PaisaExport.transaction_date <= date_upper,
        )
    )
    candidates = list(result.scalars().all())

    if incoming_time is not None and lower is not None and upper is not None:

        def _in_window(row: PaisaExport) -> bool:
            if row.transaction_time is None or row.transaction_date is None:
                return True
            row_dt = dt.datetime.combine(row.transaction_date, row.transaction_time)
            return lower <= row_dt <= upper

        candidates = [row for row in candidates if _in_window(row)]

    incoming_counterparty = txn_data.get("counterparty")
    if candidates:
        if incoming_time is None or any(
            row.transaction_time is None for row in candidates
        ):
            filtered = [
                row
                for row in candidates
                if _counterparty_match(row.counterparty, incoming_counterparty)
            ]
            if len(filtered) == 1:
                return filtered[0], "standard"
            return None

        if len(candidates) == 1:
            return candidates[0], "standard"

        filtered = [
            row
            for row in candidates
            if _counterparty_match(row.counterparty, incoming_counterparty)
        ]
        if len(filtered) == 1:
            return filtered[0], "standard"
        return None

    if incoming_time is None:
        return None

    aliased = await _find_am_pm_alias_match(
        session,
        txn_data,
        txn_date=txn_date,
        incoming_time=incoming_time,
        incoming_currency=incoming_currency,
        incoming_cp=incoming_counterparty,
    )
    if aliased is not None:
        return aliased, "am_pm_alias"
    return None


async def _find_am_pm_alias_match(
    session: AsyncSession,
    txn_data: dict,
    *,
    txn_date: dt.date,
    incoming_time: dt.time,
    incoming_currency: str,
    incoming_cp: str | None,
) -> PaisaExport | None:
    if not incoming_cp:
        return None

    search_minus = incoming_time.hour >= 12
    search_plus = incoming_time.hour < 12
    if not search_minus and not search_plus:
        return None

    anchor = dt.datetime.combine(txn_date, incoming_time)

    def _window(offset_hours: int) -> tuple[dt.datetime, dt.datetime]:
        center = anchor + dt.timedelta(hours=offset_hours)
        lo = center - dt.timedelta(minutes=_FUZZY_MATCH_WINDOW_MINUTES)
        hi = center + dt.timedelta(minutes=_FUZZY_MATCH_WINDOW_MINUTES)
        return lo, hi

    date_clauses = []
    minus_lo = minus_hi = plus_lo = plus_hi = None
    if search_minus:
        minus_lo, minus_hi = _window(-12)
        date_clauses.append(
            and_(
                PaisaExport.transaction_date >= minus_lo.date(),
                PaisaExport.transaction_date <= minus_hi.date(),
            )
        )
    if search_plus:
        plus_lo, plus_hi = _window(+12)
        date_clauses.append(
            and_(
                PaisaExport.transaction_date >= plus_lo.date(),
                PaisaExport.transaction_date <= plus_hi.date(),
            )
        )

    result = await session.execute(
        select(PaisaExport).where(
            PaisaExport.bank == _normalized_bank(txn_data.get("bank")),
            PaisaExport.direction == _normalized_direction(txn_data.get("direction")),
            PaisaExport.amount == _quantized_amount(txn_data.get("amount")),
            sa_func.coalesce(PaisaExport.currency, "INR") == incoming_currency,
            PaisaExport.status != "error",
            PaisaExport.email_type.in_(AMBIGUOUS_12H_TIME_EMAIL_TYPES),
            PaisaExport.transaction_time.is_not(None),
            PaisaExport.transaction_date.is_not(None),
            or_(*date_clauses),
        )
    )
    rows = list(result.scalars().all())

    def _matches_minus(row: PaisaExport) -> bool:
        if not search_minus or minus_lo is None or minus_hi is None:
            return False
        if row.transaction_date is None or row.transaction_time is None:
            return False
        if not (1 <= row.transaction_time.hour < 12):
            return False
        row_dt = dt.datetime.combine(row.transaction_date, row.transaction_time)
        return minus_lo <= row_dt <= minus_hi

    def _matches_plus(row: PaisaExport) -> bool:
        if not search_plus or plus_lo is None or plus_hi is None:
            return False
        if row.transaction_date is None or row.transaction_time is None:
            return False
        if row.transaction_time.hour != 12:
            return False
        row_dt = dt.datetime.combine(row.transaction_date, row.transaction_time)
        return plus_lo <= row_dt <= plus_hi

    candidates = [row for row in rows if _matches_minus(row) or _matches_plus(row)]
    candidates = [
        row for row in candidates if _counterparty_match(row.counterparty, incoming_cp)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


async def find_matching_paisa_export(
    session: AsyncSession,
    txn_data: dict,
    *,
    source: PaisaSource,
    email_id: int | None = None,
    sms_message_id: int | None = None,
) -> PaisaExport | None:
    result = await _find_match_with_kind(
        session,
        txn_data,
        source=source,
        email_id=email_id,
        sms_message_id=sms_message_id,
    )
    return result[0] if result is not None else None


def _is_duplicate_paisa_export_error(exc: IntegrityError) -> bool:
    message = str(exc.orig)
    return "uq_paisa_exports_idempotency_key" in message or (
        "UNIQUE constraint failed:" in message
        and "paisa_exports.idempotency_key" in message
    )


def _merge_source(existing_source: str | None, incoming_source: PaisaSource) -> str:
    if not existing_source:
        return incoming_source
    if existing_source == incoming_source:
        return existing_source
    if existing_source == "sms+email":
        return existing_source
    return "sms+email"


def _is_same_source_reparse(
    row: PaisaExport,
    *,
    source: PaisaSource,
    email_id: int | None,
    sms_message_id: int | None,
) -> bool:
    if source == "email" and email_id is not None:
        return row.email_id == email_id
    if source == "sms" and sms_message_id is not None:
        return row.sms_message_id == sms_message_id
    return False


def _resolve_and_apply_accounts(row: PaisaExport, config: PaisaConfig) -> None:
    account_view = {
        "bank": row.bank,
        "direction": row.direction,
        "card_mask": row.card_mask,
        "account_mask": row.account_mask,
    }
    resolved = resolve_paisa_accounts(account_view, config)
    row.source_account = resolved.source_account
    row.counterparty_account = resolved.counterparty_account
    row.missing_account_mapping = resolved.missing_account_mapping


def _apply_same_source_reparse_refresh(
    row: PaisaExport,
    txn_data: dict,
    *,
    source: PaisaSource,
    email_id: int | None,
    sms_message_id: int | None,
    config: PaisaConfig,
) -> None:
    row.bank = _normalized_bank(txn_data.get("bank"))
    row.direction = _normalized_direction(txn_data.get("direction"))
    row.amount = _quantized_amount(txn_data.get("amount"))
    row.currency = _normalized_currency(txn_data.get("currency"))
    row.transaction_date = _as_date(txn_data.get("transaction_date"))
    row.transaction_time = _as_time(txn_data.get("transaction_time"))
    row.counterparty = (
        str(txn_data.get("counterparty")) if txn_data.get("counterparty") else None
    )
    row.reference_number = (txn_data.get("reference_number") or "").strip() or None
    row.card_mask = (
        str(txn_data.get("card_mask")) if txn_data.get("card_mask") else None
    )
    row.account_mask = (
        str(txn_data.get("account_mask")) if txn_data.get("account_mask") else None
    )
    row.email_type = (
        str(txn_data.get("email_type")) if txn_data.get("email_type") else None
    )
    row.source = _merge_source(row.source, source)
    if sms_message_id is not None:
        row.sms_message_id = sms_message_id
    if email_id is not None:
        row.email_id = email_id
    _resolve_and_apply_accounts(row, config)
    row.status = "exported"
    row.error = None
    row.exported_at = dt.datetime.now(dt.UTC)


def _apply_enrichment(
    row: PaisaExport,
    txn_data: dict,
    *,
    source: PaisaSource,
    email_id: int | None,
    sms_message_id: int | None,
    match_kind: PaisaMatchKind,
    config: PaisaConfig,
) -> None:
    def _set_field(name: str, incoming_value: object) -> None:
        if incoming_value is None:
            return
        current = getattr(row, name, None)
        if current is None:
            setattr(row, name, incoming_value)
            return
        if current == incoming_value:
            return
        if name in _MASK_FIELDS and _digits_only(str(current)) == _digits_only(
            str(incoming_value)
        ):
            return
        if source == "email":
            setattr(row, name, incoming_value)

    incoming_reference = (txn_data.get("reference_number") or "").strip() or None
    incoming_counterparty = (
        str(txn_data.get("counterparty")) if txn_data.get("counterparty") else None
    )
    incoming_card_mask = (
        str(txn_data.get("card_mask")) if txn_data.get("card_mask") else None
    )
    incoming_account_mask = (
        str(txn_data.get("account_mask")) if txn_data.get("account_mask") else None
    )
    incoming_transaction_date = _as_date(txn_data.get("transaction_date"))
    incoming_transaction_time = _as_time(txn_data.get("transaction_time"))

    _set_field("reference_number", incoming_reference)
    _set_field("counterparty", incoming_counterparty)
    _set_field("card_mask", incoming_card_mask)
    _set_field("account_mask", incoming_account_mask)

    if hasattr(row, "raw_description"):
        incoming_raw_description = txn_data.get("raw_description")
        _set_field(
            "raw_description",
            str(incoming_raw_description) if incoming_raw_description else None,
        )

    if incoming_transaction_time is not None:
        existing_transaction_date = row.transaction_date
        existing_transaction_time = row.transaction_time
        if row.transaction_time is None:
            row.transaction_time = incoming_transaction_time
            if row.transaction_date is None and incoming_transaction_date is not None:
                row.transaction_date = incoming_transaction_date
        elif source == "email" and _incoming_time_is_earlier(
            existing_date=existing_transaction_date,
            existing_time=existing_transaction_time,
            incoming_date=incoming_transaction_date,
            incoming_time=incoming_transaction_time,
        ):
            row.transaction_time = incoming_transaction_time
            if incoming_transaction_date is not None:
                row.transaction_date = incoming_transaction_date

    if match_kind == "am_pm_alias":
        alias_time = _as_time(txn_data.get("transaction_time"))
        if alias_time is not None:
            row.transaction_time = alias_time

    row.source = _merge_source(row.source, source)
    if sms_message_id is not None and row.sms_message_id is None:
        row.sms_message_id = sms_message_id
    if email_id is not None and row.email_id is None:
        row.email_id = email_id

    _resolve_and_apply_accounts(row, config)
    row.status = "exported"
    row.error = None
    row.exported_at = dt.datetime.now(dt.UTC)


def _new_export_from_txn_data(
    txn_data: dict,
    *,
    source: PaisaSource,
    email_id: int | None,
    sms_message_id: int | None,
    config: PaisaConfig,
) -> PaisaExport:
    accounts = resolve_paisa_accounts(txn_data, config)
    return PaisaExport(
        source=source,
        email_id=email_id,
        sms_message_id=sms_message_id,
        idempotency_key=build_paisa_idempotency_key(txn_data),
        bank=_normalized_bank(txn_data.get("bank")),
        email_type=(
            str(txn_data.get("email_type")) if txn_data.get("email_type") else None
        ),
        direction=_normalized_direction(txn_data.get("direction")),
        amount=_quantized_amount(txn_data.get("amount")),
        currency=_normalized_currency(txn_data.get("currency")),
        transaction_date=_as_date(txn_data.get("transaction_date")),
        transaction_time=_as_time(txn_data.get("transaction_time")),
        counterparty=(
            str(txn_data.get("counterparty")) if txn_data.get("counterparty") else None
        ),
        reference_number=((txn_data.get("reference_number") or "").strip() or None),
        card_mask=(
            str(txn_data.get("card_mask")) if txn_data.get("card_mask") else None
        ),
        account_mask=(
            str(txn_data.get("account_mask")) if txn_data.get("account_mask") else None
        ),
        source_account=accounts.source_account,
        counterparty_account=accounts.counterparty_account,
        missing_account_mapping=accounts.missing_account_mapping,
        status="exported",
        error=None,
        exported_at=dt.datetime.now(dt.UTC),
    )


def _build_error_export_row(
    txn_data: dict,
    *,
    source: PaisaSource,
    email_id: int | None,
    sms_message_id: int | None,
) -> PaisaExport:
    return PaisaExport(
        source=source,
        email_id=email_id,
        sms_message_id=sms_message_id,
        idempotency_key=build_paisa_idempotency_key(txn_data),
        bank=_normalized_bank(txn_data.get("bank")),
        email_type=(
            str(txn_data.get("email_type")) if txn_data.get("email_type") else None
        ),
        direction=_normalized_direction(txn_data.get("direction")),
        amount=_quantized_amount(txn_data.get("amount")),
        currency=_normalized_currency(txn_data.get("currency")),
        transaction_date=_as_date(txn_data.get("transaction_date")),
        transaction_time=_as_time(txn_data.get("transaction_time")),
        counterparty=(
            str(txn_data.get("counterparty")) if txn_data.get("counterparty") else None
        ),
        reference_number=((txn_data.get("reference_number") or "").strip() or None),
        card_mask=(
            str(txn_data.get("card_mask")) if txn_data.get("card_mask") else None
        ),
        account_mask=(
            str(txn_data.get("account_mask")) if txn_data.get("account_mask") else None
        ),
        status="error",
        error=None,
        exported_at=None,
    )


async def _mark_export_error(
    session: AsyncSession,
    *,
    source: PaisaSource,
    txn_data: dict,
    email_id: int | None,
    sms_message_id: int | None,
    error: str,
) -> int | None:
    key = build_paisa_idempotency_key(txn_data)
    row = await session.scalar(
        select(PaisaExport).where(PaisaExport.idempotency_key == key).limit(1)
    )
    if row is None:
        row = _build_error_export_row(
            txn_data,
            source=source,
            email_id=email_id,
            sms_message_id=sms_message_id,
        )
        session.add(row)
        try:
            await session.flush()
        except IntegrityError as exc:
            if not _is_duplicate_paisa_export_error(exc):
                raise
            row = await session.scalar(
                select(PaisaExport).where(PaisaExport.idempotency_key == key).limit(1)
            )
            if row is None:
                raise

    row.status = "error"
    row.error = error
    row.exported_at = None
    if row.email_id is None and email_id is not None:
        row.email_id = email_id
    if row.sms_message_id is None and sms_message_id is not None:
        row.sms_message_id = sms_message_id
    await session.flush()
    return row.id


async def process_paisa_transaction(
    session: AsyncSession,
    *,
    source: PaisaSource,
    txn_data: dict,
    email_row: Email | None = None,
    sms_row: SmsMessage | None = None,
) -> PaisaExportOutcome:
    email_id = email_row.id if email_row is not None else None
    sms_message_id = sms_row.id if sms_row is not None else None

    try:
        config = get_paisa_config()
        match_result = await _find_match_with_kind(
            session,
            txn_data,
            source=source,
            email_id=email_id,
            sms_message_id=sms_message_id,
        )
        if match_result is not None:
            match, match_kind = match_result
            async with session.begin_nested():
                if _is_same_source_reparse(
                    match,
                    source=source,
                    email_id=email_id,
                    sms_message_id=sms_message_id,
                ):
                    _apply_same_source_reparse_refresh(
                        match,
                        txn_data,
                        source=source,
                        email_id=email_id,
                        sms_message_id=sms_message_id,
                        config=config,
                    )
                else:
                    _apply_enrichment(
                        match,
                        txn_data,
                        source=source,
                        email_id=email_id,
                        sms_message_id=sms_message_id,
                        match_kind=match_kind,
                        config=config,
                    )
                await session.flush()
            return PaisaExportOutcome(
                status="parsed",
                export_id=match.id,
                needs_journal_rewrite=True,
                error=None,
            )

        try:
            async with session.begin_nested():
                created = _new_export_from_txn_data(
                    txn_data,
                    source=source,
                    email_id=email_id,
                    sms_message_id=sms_message_id,
                    config=config,
                )
                session.add(created)
                await session.flush()
        except IntegrityError as exc:
            if not _is_duplicate_paisa_export_error(exc):
                raise
            race_match = await _find_match_with_kind(
                session,
                txn_data,
                source=source,
                email_id=email_id,
                sms_message_id=sms_message_id,
            )
            if race_match is None:
                raise
            match, match_kind = race_match
            if _is_same_source_reparse(
                match,
                source=source,
                email_id=email_id,
                sms_message_id=sms_message_id,
            ):
                async with session.begin_nested():
                    _apply_same_source_reparse_refresh(
                        match,
                        txn_data,
                        source=source,
                        email_id=email_id,
                        sms_message_id=sms_message_id,
                        config=config,
                    )
                    await session.flush()
            else:
                async with session.begin_nested():
                    _apply_enrichment(
                        match,
                        txn_data,
                        source=source,
                        email_id=email_id,
                        sms_message_id=sms_message_id,
                        match_kind=match_kind,
                        config=config,
                    )
                    await session.flush()
            export_id = match.id
        else:
            export_id = created.id

        return PaisaExportOutcome(
            status="parsed",
            export_id=export_id,
            needs_journal_rewrite=True,
            error=None,
        )
    except Exception as exc:
        error_text = str(exc)
        logger.exception(
            "Paisa export failed (source=%s email_id=%s sms_id=%s): %s",
            source,
            email_id,
            sms_message_id,
            error_text,
        )
        try:
            export_id = await _mark_export_error(
                session,
                source=source,
                txn_data=txn_data,
                email_id=email_id,
                sms_message_id=sms_message_id,
                error=error_text,
            )
        except Exception:
            logger.exception(
                "Failed to persist paisa export error row (source=%s email_id=%s sms_id=%s)",
                source,
                email_id,
                sms_message_id,
            )
            export_id = None
        return PaisaExportOutcome(
            status="error",
            export_id=export_id,
            needs_journal_rewrite=False,
            error=error_text,
        )


async def rewrite_paisa_journal(config: PaisaConfig) -> None:
    async with _REWRITE_LOCK:
        async with async_session() as session:
            result = await session.execute(
                select(PaisaExport)
                .where(PaisaExport.status == "exported")
                .order_by(PaisaExport.id)
            )
            rows = list(result.scalars().all())

        ordered = sorted(
            rows,
            key=lambda row: (
                row.transaction_date is None,
                row.transaction_date or dt.date.max,
                row.transaction_time or dt.time.max,
                row.id,
            ),
        )
        dated_rows = [row for row in ordered if row.transaction_date is not None]
        skipped_undated = len(ordered) - len(dated_rows)
        if skipped_undated:
            logger.warning(
                "Skipping %d exported paisa rows without transaction_date during rewrite",
                skipped_undated,
            )

        entries = [render_paisa_journal_entry(row) for row in dated_rows]
        body = "\n".join(entry.rstrip("\n") for entry in entries if entry)
        if body:
            body = f"{body}\n"

        target = Path(config.generated_journal_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp_name: str | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                delete=False,
            ) as tmp_file:
                tmp_name = tmp_file.name
                tmp_file.write(body)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())

            os.replace(tmp_name, target)
        except Exception:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
            raise
