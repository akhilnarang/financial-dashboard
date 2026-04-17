"""Thin adapters around sibling parser packages."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from bank_email_parser import SUPPORTED_BANKS as SUPPORTED_BANKS
from bank_email_parser.api import parse_email as _parse_email
from bank_email_parser.exceptions import (
    ParseError as ParseError,
    UnsupportedEmailTypeError as UnsupportedEmailTypeError,
)
from bank_statement_parser.extractor import extract_raw_pdf as _extract_bank_pdf
from bank_statement_parser.parsers.factory import (
    get_parser as _get_bank_statement_parser,
)
from cc_parser.extractor import extract_raw_pdf as _extract_cc_pdf
from cc_parser.parsers.factory import get_parser as _get_cc_parser

from bank_email_fetcher.core.dates import parse_date


def get_supported_banks() -> tuple[str, ...]:
    return tuple(SUPPORTED_BANKS)


def parse_transaction_email(bank: str, html: str):
    return _parse_email(bank, html)


def parse_cc_statement_pdf(
    pdf_path: Path, password: str | None = None, bank: str = "auto"
):
    raw_data = _extract_cc_pdf(pdf_path, include_blocks=True, password=password or None)
    parser = _get_cc_parser(bank, raw_data)
    return parser.parse(raw_data)


def parse_bank_statement_pdf(pdf_path: Path, bank: str, password: str | None = None):
    raw_data = _extract_bank_pdf(
        pdf_path, include_blocks=False, password=password or None
    )
    parser = _get_bank_statement_parser(bank)
    return parser.parse(raw_data)


def parse_cc_amount(amount_str: str) -> Decimal:
    return Decimal(amount_str.replace(",", ""))


def parse_cc_token_amount(amount_str: str) -> Decimal:
    return Decimal(amount_str.replace(",", ""))


def format_cc_amount(amount: Decimal) -> str:
    return f"{amount:,.2f}"


def parse_statement_date(date_str: str):
    parsed = parse_date(date_str, dayfirst=True)
    if parsed is None:
        raise ValueError(f"Could not parse statement date: {date_str!r}")
    return parsed
