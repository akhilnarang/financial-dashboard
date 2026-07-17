"""Shared renderer for the ledger-CLI syntax family: ``ledger`` and ``hledger``.

The two CLIs accept overlapping — not identical — syntax. This module emits the
common subset both parse: ``account``/``commodity`` declarations, dated
transactions, ``;`` comments, ``P`` price directives, and explicit commodities
on amounts. hledger is *not* a strict superset of ledger (it rejects a bare
alphanumeric commodity in a ``commodity`` declaration or ``P`` directive that
ledger tolerates), so the commodity token is quoted consistently in every
position via :func:`base.commodity_token`. The per-backend modules supply only
the validator (identical for the two — spaces are legal in account names under
both).

Syntax conventions
------------------
* Assets are debit-normal (positive balance = increase); liabilities, income
  and equity are credit-normal. Signed amounts are emitted directly so every
  transaction balances to zero per commodity without relying on auto-balance.
* Account names use the ``Top:Sub:Leaf`` hierarchy Paisa expects.
* One ``; txn:<id>`` comment per entry carries the dashboard transaction id.
* A priced foreign entry posts both legs in the foreign commodity (balanced in
  that currency) and a ``P <date> <CCY> <rate> INR`` directive lets the backend
  value it in INR — no implicit conversion, no fabricated INR leg.
"""

from decimal import Decimal

from financial_dashboard.services.paisa.renderers import base
from financial_dashboard.services.paisa.renderers.base import (
    AMOUNT_COLUMN,
    EQUITY_OPENING,
    InvestmentLotEntry,
    LedgerDocument,
    ProjectedEntry,
    check_balanced,
    check_lot_consistent,
    commodity_token,
    fmt_price_rate,
    format_posting_line,
    investment_asset_account,
    normalize_investment_equity,
    sanitize_text,
    validate as _ledger_validate,
)

_POSTING_INDENT = "    "


def validate(name: str) -> str:
    """Ledger/hledger account-name validation: a ``:``-joined hierarchy with no
    empty segments and no character the parser treats specially (newline, tab,
    ``;``, ``{``, ``}`). Spaces are legal in both backends, so a long default
    name like ``Assets:Bank:HDFC:Salary Plus`` survives verbatim."""
    return _ledger_validate(name)


def normalize_default_account(name: str) -> str:
    """Defaults need no transform for the ledger family — spaces are legal."""
    return name


def _format_posting(
    account: str,
    amount: Decimal,
    commodity: str,
    *,
    amount_formatter=base.fmt_amount,
) -> str:
    # The amount commodity is the ledger-family token: sanitized, then quoted
    # when not pure ASCII letters (an alphanumeric ISIN). Ledger 3.3.2 rejects a
    # bare digit-bearing symbol in a posting amount; pure-letter currency codes
    # (INR, USD) stay bare. The same token is used in declarations and P
    # directives (hledger rejects a bare alphanumeric there too).
    return format_posting_line(
        account,
        amount,
        indent=True,
        commodity=commodity_token(commodity),
        validate=validate,
        amount_formatter=amount_formatter,
    )


def render_document(doc: LedgerDocument) -> str:
    """Render a :class:`LedgerDocument` to ledger/hledger text.

    Output ordering is deterministic: ``account`` declarations (sorted), then
    the opening-balances entry (dated at the cutover), then entries in the order
    given (the projection sorts them by date/id), then investment-lot openings
    (dated at each lot's acquisition date), then deduplicated ``P`` price
    directives. The result has no trailing blank line beyond a single newline
    so byte-equality is stable.
    """
    lines: list[str] = []

    # Instrument commodities a lot introduces are declared so hledger/ledger
    # both recognize them; bank-account declarations come from the projection.
    lot_commodities = sorted(
        {base.sanitize_commodity(lot.instrument) for lot in doc.lot_postings}
    )
    for name in doc.accounts_declared:
        lines.append(f"account {validate(name)}")
    if doc.accounts_declared:
        lines.append("")
    if lot_commodities:
        # Commodity declarations (``commodity <token>``), distinct from account
        # declarations — an instrument is not an account. The token is quoted
        # when not pure ASCII letters so hledger parses it too.
        for sym in lot_commodities:
            lines.append(f"commodity {commodity_token(sym)}")
        lines.append("")

    lines.extend(_render_openings(doc))

    for entry in doc.entries:
        lines.extend(_render_entry(entry))

    for lot in doc.lot_postings:
        lines.extend(_render_lot(lot))

    if doc.price_directives:
        lines.extend(_render_prices(doc))

    text = "\n".join(lines)
    return text + "\n" if text else ""


def _meta_tag_lines(meta, indent_level: int = 1) -> list[str]:
    """Render ``((key, value), ...)`` metadata as one ``; key: value`` tag per
    line. ``indent_level`` 1 = entry-level (4 spaces), 2 = posting-level (8).

    Ledger/hledger tag syntax: one tag per line, a space after the colon, and
    no comma separator — each line is its own ``; key: value`` so a tag query
    (``ledger --tag key``) finds it.
    """
    indent = _POSTING_INDENT * indent_level
    out: list[str] = []
    for key, value in meta:
        out.append(f"{indent}; {key}: {value}")
    return out


def _render_openings(doc: LedgerDocument) -> list[str]:
    """Render the single Equity:Opening Balances entry (always INR).

    Only nonzero openings are emitted — a zero line is noise and would churn
    the diff. The equity balancing line is always the final posting and carries
    whatever signed sum zeroes the entry. Each opening posting carries its
    account-id/source/as-of metadata as posting-level tags.
    """
    nonzero = [ob for ob in doc.openings if ob.amount]
    if not nonzero:
        return []
    if doc.cutover_date is None:
        # Defensive: projection guarantees a cutover when openings exist.
        return []
    lines: list[str] = [f"{doc.cutover_date.isoformat()} * Opening Balances"]
    # Entry-level kind tag so the opening is identifiable as dashboard data.
    lines.append(f"{_POSTING_INDENT}; dashboard_kind: opening")
    equity_total = Decimal("0.00")
    for ob in nonzero:
        equity_total += ob.amount
        lines.append(_format_posting(ob.account_name, ob.amount, base.INR))
        # Posting-level metadata: account id, source, as-of date.
        if ob.meta:
            lines.extend(_meta_tag_lines(ob.meta, indent_level=2))
    # Equity takes the opposite sign so the entry balances to zero.
    lines.append(_format_posting(EQUITY_OPENING, -equity_total, base.INR))
    lines.append("")
    return lines


def _render_entry(entry: ProjectedEntry) -> list[str]:
    check_balanced(entry)
    date_line = entry.date.isoformat()
    payee = sanitize_text(entry.payee)
    header = f"{date_line} * {payee}".rstrip()
    txn_comment = ", ".join(str(t) for t in entry.txn_ids)
    if txn_comment:
        header = f"{header}    ; txn:{txn_comment}"
    lines: list[str] = [header]
    # Entry-level canonical metadata: one ``; key: value`` tag per line.
    if entry.meta:
        lines.extend(_meta_tag_lines(entry.meta, indent_level=1))
    for posting in entry.postings:
        lines.append(
            _format_posting(posting.account, posting.amount, posting.commodity)
        )
        if posting.meta:
            lines.extend(_meta_tag_lines(posting.meta, indent_level=2))
    if entry.note:
        # Note as a ledger comment on its own indented line, sanitized.
        cleaned = sanitize_text(entry.note)[:160]
        if cleaned:
            lines.append(f"{_POSTING_INDENT}; note: {cleaned}")
    lines.append("")
    return lines


def _render_prices(doc: LedgerDocument) -> list[str]:
    """Deduplicated ``P <date> <token> <rate> INR`` directives.

    The projection already deduplicates the directives it hands the document;
    sorted here by (date, currency) for a stable byte output. A price line is
    only present when an explicit configured rate backed a priced foreign entry.
    The currency is the ledger-family token (quoted when not pure ASCII letters)
    so hledger parses the directive as well as ledger.
    """
    lines: list[str] = []
    for price in sorted(doc.price_directives, key=lambda p: (p.date, p.currency)):
        rate_text = fmt_price_rate(price.rate, price.unit)
        lines.append(
            f"P {price.date.isoformat()} {commodity_token(price.currency)} {rate_text}"
        )
    return lines


def _render_lot(lot: InvestmentLotEntry) -> list[str]:
    """One conservative investment-lot opening, dated at the acquisition date.

    Posts ``quantity`` of the instrument commodity into the dedicated
    ``Assets:Investments:<instrument>`` account with a per-unit cost annotation
    ``{unit_cost INR}`` and the acquisition-date lot note ``[acquired_on]``,
    balanced against :data:`base.INVESTMENT_EQUITY_OPENING` for the exact
    full-precision negative product ``-(quantity * unit_cost)``. No bank/cash
    leg is inferred. ledger (and hledger) value the asset leg's INR cost as
    ``quantity * unit_cost`` at full precision, so the equity leg emits that
    same product (via :func:`base.fmt_lot_money`) and the entry balances to the
    penny *and* to full precision for every lot — including one whose product
    is not 2-dp exact. The stored 2-dp ``cost_basis`` is reporting /
    source-agreement metadata only (still validated by check_lot_consistent).
    The acquisition date also appears as the transaction date so the lot is
    legible even under a backend that ignores the ``[date]`` notation.
    """
    check_lot_consistent(lot)
    commodity = base.sanitize_commodity(lot.instrument)
    asset_account = validate(investment_asset_account(lot.instrument))
    equity_account = validate(normalize_investment_equity())
    payee = (
        sanitize_text(f"Investment Lot - {lot.instrument_name}")[:160]
        or "Investment Lot"
    )
    qty_text = base.fmt_lot_decimal(lot.quantity)
    cost_text = base.fmt_lot_decimal(lot.unit_cost)
    header = f"{lot.acquired_on.isoformat()} * {payee}"
    # The instrument commodity in the *amount* is the ledger-family token (an
    # alphanumeric ISIN is not bare-safe for Ledger 3.3.2, and hledger rejects a
    # bare one in every position); the cost-annotation currency
    # (``{cost INR}``) and the lot-date bracket stay bare, as does the separate
    # ``commodity``/``P`` directive which renders this symbol via the token.
    amount_commodity = commodity_token(commodity)
    # The equity leg is the exact full-precision negative product so it cancels
    # the asset leg's cost (``quantity × unit_cost``) to the penny AND to full
    # precision in every backend. The stored 2-dp ``cost_basis`` is reporting /
    # source-agreement metadata only (still validated by check_lot_consistent);
    # emitting it here would leave a sub-cent imbalance for a non-2dp lot.
    equity_amount = -(lot.quantity * lot.unit_cost)
    lines: list[str] = [header]
    # Lot metadata as entry-level tags (non-sensitive CAS provenance).
    if lot.meta:
        lines.extend(_meta_tag_lines(lot.meta, indent_level=1))
    lines.extend(
        [
            (
                f"{_POSTING_INDENT}{asset_account}"
                f"  {qty_text} {amount_commodity}"
                f" {{{cost_text} {lot.currency}}}"
                f" [{lot.acquired_on.isoformat()}]"
            ),
            _format_posting(
                equity_account,
                equity_amount,
                lot.currency,
                amount_formatter=base.fmt_lot_money,
            ),
            "",
        ]
    )
    return lines


# Pinned for tests that import the alignment constant from a backend module.
__all__ = ["AMOUNT_COLUMN", "normalize_default_account", "render_document", "validate"]
