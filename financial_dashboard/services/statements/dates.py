"""Statement date helpers."""

from datetime import date

from financial_dashboard.core.dates import parse_date


def cc_stmt_date_range(parsed) -> tuple[date, date] | None:
    dates: list[date] = []
    for txn in (parsed.transactions or []) + (parsed.payments_refunds or []):
        try:
            parsed_value = parse_date(txn.date, dayfirst=True)
        except TypeError, ValueError:
            parsed_value = None
        if parsed_value is not None:
            dates.append(parsed_value)
    return (min(dates), max(dates)) if dates else None


def bank_stmt_date_range(parsed) -> tuple[date, date] | None:
    if parsed.statement_period_start and parsed.statement_period_end:
        try:
            start = parse_date(parsed.statement_period_start, dayfirst=True)
            end = parse_date(parsed.statement_period_end, dayfirst=True)
        except TypeError, ValueError:
            start = end = None
        if start and end:
            return start, end

    dates: list[date] = []
    for txn in parsed.transactions or []:
        try:
            parsed_value = parse_date(txn.date, dayfirst=True)
        except TypeError, ValueError:
            parsed_value = None
        if parsed_value is not None:
            dates.append(parsed_value)
    return (min(dates), max(dates)) if dates else None
