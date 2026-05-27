"""PDF adapters over sibling parser packages.

One call in, ``ParsedStatement`` out. Used by the CC and bank statement
pipelines. Everything else (amount/date format helpers, parser types,
exceptions) lives in the sibling packages or in the statement services.
"""

from pathlib import Path
from importlib import import_module

from bank_statement_parser.extractor import extract_raw_pdf as _extract_bank_pdf
from bank_statement_parser.parsers.factory import (
    get_parser as _get_bank_statement_parser,
)
from cc_parser.extractor import extract_raw_pdf as _extract_cc_pdf
from cc_parser.parsers.factory import get_parser as _get_cc_parser


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


def parse_cas_pdf(pdf_path: Path, password: str | None = None):
    extractor = import_module("cas_parser.extractor")
    factory = import_module("cas_parser.parsers.factory")

    candidates = None
    if password:
        candidates = list(dict.fromkeys([password, password.upper()]))
    raw_data = extractor.extract_raw_pdf(pdf_path, passwords=candidates)
    return factory.get_parser(factory.detect_source(raw_data)).parse(raw_data)
