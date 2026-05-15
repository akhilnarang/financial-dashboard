"""Shared date parsing helpers backed by python-dateutil."""

from __future__ import annotations

from datetime import date, datetime

from dateutil import parser


def parse_datetime(
    value: str | datetime | None, *, dayfirst: bool = False
) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    return parser.parse(text, dayfirst=dayfirst)


def parse_date(
    value: str | date | datetime | None, *, dayfirst: bool = False
) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = parse_datetime(value, dayfirst=dayfirst)
    return parsed.date() if parsed else None


def format_ddmmyyyy(value: date | datetime | None) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%d/%m/%Y")
