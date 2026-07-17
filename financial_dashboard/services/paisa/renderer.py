"""Compatibility facade over the multi-backend renderer registry.

Historically this module held the only renderer (Ledger-CLI). The renderer is
now a strategy selected by the ``paisa.ledger_cli`` backend id (ledger,
hledger, beancount); the strategies live in :mod:`financial_dashboard.services
.paisa.renderers`. This module remains as a stable import surface so every
existing caller — the projection, the surface adapter, the public package
``__init__``, and the v1 tests — keeps importing the same names and gets the
ledger strategy as the default.

To render for a specific backend, call :func:`render_document_for_backend` (or
``renderers.render_document(doc, backend)``); to validate/normalize an account
name for a backend, use :func:`validate_account_name` / :func:
`normalize_default_account` with the ``backend`` argument.
"""

from decimal import Decimal

from financial_dashboard.services.paisa.renderers import (
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    get_renderer,
    render_document as _render_for_backend,
    validate_account_name as _validate_for_backend,
    normalize_default_account as _normalize_for_backend,
    validate_backend,
)
from financial_dashboard.services.paisa.renderers import base
from financial_dashboard.services.paisa.renderers.base import (
    AMOUNT_COLUMN,
    CARD_PAYMENT_CLEARING,
    EQUITY_OPENING,
    INR,
    INVESTMENT_ASSET_ROOT,
    INVESTMENT_EQUITY_OPENING,
    InvalidAccountName,
    InvestmentLotEntry,
    LedgerAccount,
    LedgerDocument,
    LedgerPosting,
    OpeningBalance,
    PriceDirective,
    ProjectedEntry,
    UnbalancedEntry,
    check_balanced,
    check_lot_consistent,
    fmt_amount,
    format_posting_line,
    investment_asset_account,
    sanitize_commodity,
    sanitize_text,
)
from financial_dashboard.services.paisa.renderers.base import (
    validate as _ledger_validate,
)

#: The backend the v1 functions default to. Pinned to ``ledger`` so a caller
#: that ignores backends entirely renders exactly the v1 output.
DEFAULT_LEDGER_CLI = DEFAULT_BACKEND

__all__ = [
    "AMOUNT_COLUMN",
    "CARD_PAYMENT_CLEARING",
    "DEFAULT_BACKEND",
    "DEFAULT_LEDGER_CLI",
    "EQUITY_OPENING",
    "INR",
    "INVESTMENT_ASSET_ROOT",
    "INVESTMENT_EQUITY_OPENING",
    "InvalidAccountName",
    "InvestmentLotEntry",
    "LedgerAccount",
    "LedgerDocument",
    "LedgerPosting",
    "OpeningBalance",
    "PriceDirective",
    "ProjectedEntry",
    "SUPPORTED_BACKENDS",
    "UnbalancedEntry",
    "check_balanced",
    "check_lot_consistent",
    "fmt_amount",
    "format_posting_line",
    "investment_asset_account",
    "normalize_default_account",
    "render_document",
    "render_document_for_backend",
    "sanitize_commodity",
    "sanitize_text",
    "validate_account_name",
    "validate_backend",
]


def render_document(doc: LedgerDocument) -> str:
    """Render ``doc`` with the default (ledger) strategy — the v1 entry point.

    Backed by :func:`render_document_for_backend` with ``backend="ledger"``;
    the output is byte-identical to the pre-multi-backend renderer for any
    document that carries only INR postings and no price directives.
    """
    return _render_for_backend(doc, DEFAULT_BACKEND)


def render_document_for_backend(doc: LedgerDocument, backend: str | None) -> str:
    """Render ``doc`` with the named backend strategy."""
    return _render_for_backend(doc, backend)


def validate_account_name(name: str, backend: str | None = DEFAULT_BACKEND) -> str:
    """Validate ``name``; defaults to the ledger grammar (v1 behavior)."""
    return _validate_for_backend(name, backend)


def normalize_default_account(name: str, backend: str | None = DEFAULT_BACKEND) -> str:
    """Deterministically transform a default account name for ``backend``."""
    return _normalize_for_backend(name, backend)


def _format_posting_line(
    account: str, amount: Decimal, *, indent: bool, commodity: str = INR
) -> str:
    """v1 posting-line formatter: ledger grammar, INR by default.

    Kept as a named import target for tests that pin the ledger amount/account
    separation contract. Delegates to :func:`format_posting_line` with the
    ledger validator so the output matches the strategy exactly.
    """
    return format_posting_line(
        account, amount, indent=indent, commodity=commodity, validate=_ledger_validate
    )


# Re-export the registry getter for callers that want the strategy object.
def get_strategy(backend: str | None = DEFAULT_BACKEND):
    return get_renderer(backend)


# Silence unused-import lints for names re-exported as the public surface.
_ = (
    base,
    LedgerAccount,
    sanitize_text,
    check_balanced,
    fmt_amount,
    format_posting_line,
    check_lot_consistent,
)
