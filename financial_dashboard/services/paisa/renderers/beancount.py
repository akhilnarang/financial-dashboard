"""Beancount renderer strategy.

Beancount (https://github.com/beancount/beancount) is stricter than the ledger
family in three ways this module honors:

* **Account components** must each match ``[A-Z][A-Za-z0-9]*`` and the root
  must be one of ``Assets``/``Liabilities``/``Equity``/``Income``/``Expenses``.
  Spaces are illegal, so a projection *default* name like
  ``Equity:Opening Balances`` is deterministically PascalCased to
  ``Equity:OpeningBalances``; an operator *override* with a space is rejected
  rather than silently rewritten.
* Every account that receives a posting must be opened with a dated ``open``
  directive listing the commodities it may hold; a commodity must be declared
  with a dated ``commodity`` directive. The renderer derives both from the
  document's postings/openings so the file is self-contained and valid.
* String values are double-quoted, with backslashes escaped before quotes so
  Beancount cannot consume a source backslash as an escape sequence.

A priced foreign entry posts both legs in the foreign commodity (balanced in
that currency) and a ``<date> price <CCY> <rate> INR`` directive lets beancount
value it in INR — no implicit conversion, no fabricated INR leg.
"""

import datetime
import re
from decimal import Decimal

from financial_dashboard.services.paisa.renderers import base
from financial_dashboard.services.paisa.renderers.base import (
    EQUITY_OPENING,
    InvestmentLotEntry,
    LedgerDocument,
    ProjectedEntry,
    check_balanced,
    check_lot_consistent,
    fmt_price_rate,
    sanitize_text,
)

_POSTING_INDENT = "    "

#: The only account roots beancount accepts (beancount.core.account.ROOT_RE).
_ROOTS = ("Assets", "Liabilities", "Equity", "Income", "Expenses")

#: Each non-root component must start with an uppercase letter and contain only
#: ASCII letters and digits. Beancount also tolerates a few separator chars in
#: some configurations, but letters+digits is the universally portable subset.
_COMPONENT_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")


def validate(name: str) -> str:
    """Validate a beancount account name and return it unchanged.

    Rejects spaces, empty segments, a missing/unknown root, or any component
    that does not match ``[A-Z][A-Za-z0-9]*``. A name that passed the
    ledger-family validator but is illegal under beancount (e.g.
    ``Assets:Bank:HDFC:Savings Account``) raises here, surfacing an
    incompatible override instead of silently rewriting it.
    """
    if not isinstance(name, str) or not name.strip():
        raise base.InvalidAccountName("account name is empty")
    for ch in base._UNSAFE_TEXT_CHARS:
        if ch in name:
            raise base.InvalidAccountName(f"account name contains {ch!r}: {name!r}")
    if " " in name:
        raise base.InvalidAccountName(
            f"beancount account name must not contain spaces: {name!r}"
        )
    segments = name.split(":")
    if len(segments) < 2:
        raise base.InvalidAccountName(
            f"account name must be a ':', separated hierarchy: {name!r}"
        )
    if any(seg == "" for seg in segments):
        raise base.InvalidAccountName(
            f"account name has an empty hierarchy segment: {name!r}"
        )
    root, rest = segments[0], segments[1:]
    if root not in _ROOTS:
        raise base.InvalidAccountName(
            f"beancount root must be one of {list(_ROOTS)}: got {root!r} in {name!r}"
        )
    for seg in rest:
        if not _COMPONENT_RE.match(seg):
            raise base.InvalidAccountName(
                f"beancount component {seg!r} must match [A-Z][A-Za-z0-9]* in {name!r}"
            )
    return name


def normalize_default_account(name: str) -> str:
    """Deterministically transform a projection default into a beancount-legal
    name. Each component is reduced to its alphanumeric words and PascalCased
    (``Opening Balances`` → ``OpeningBalances``, ``HDFC`` → ``HDFC``). The root
    is preserved. Used for defaults only; overrides are validated, not mutated.
    """
    segments = name.split(":")
    normalized: list[str] = []
    for i, seg in enumerate(segments):
        if i == 0 and seg in _ROOTS:
            normalized.append(seg)
            continue
        words = re.findall(r"[A-Za-z0-9]+", seg)
        pascal = "".join(w[:1].upper() + w[1:] for w in words)
        if not pascal:
            raise base.InvalidAccountName(
                f"beancount segment has no legal characters: {seg!r} in {name!r}"
            )
        # Guarantee an uppercase leading char even for all-lowerword defaults.
        if not pascal[0].isupper():
            pascal = pascal[0].upper() + pascal[1:]
        normalized.append(pascal)
    out = ":".join(normalized)
    # Validate the transformed name so a pathological default surfaces loudly
    # rather than emitting something beancount rejects downstream.
    return validate(out)


def quote_string(text: str) -> str:
    """Render ``text`` as a Beancount string literal.

    Backslashes must be escaped *before* quotes. Otherwise a source value such
    as ``C:\\new\\file`` is parsed with ``\\n``/``\\f`` escapes and no longer
    has the same value, while an unknown escape silently loses its backslash.
    This helper is used for every data-derived Beancount string position.
    """
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _account_currencies(doc: LedgerDocument) -> dict[str, set[str]]:
    """Map every posted account to the set of commodities it holds.

    beancount's ``open`` directive may list the allowed currencies; deriving
    the exact set per account keeps ``bean-check`` from flagging a posting in a
    currency the account was not opened with. Investment-lot asset accounts
    hold the instrument commodity; their equity contra holds the cost currency.
    """
    currencies: dict[str, set[str]] = {}
    for ob in doc.openings:
        if ob.amount:
            currencies.setdefault(ob.account_name, set()).add(base.INR)
    for entry in doc.entries:
        for posting in entry.postings:
            currencies.setdefault(posting.account, set()).add(posting.commodity)
    for lot in doc.lot_postings:
        commodity = base.sanitize_commodity(lot.instrument)
        asset_account = validate(f"{base.INVESTMENT_ASSET_ROOT}:{commodity}")
        currencies.setdefault(asset_account, set()).add(commodity)
        equity_account = normalize_default_account(base.INVESTMENT_EQUITY_OPENING)
        currencies.setdefault(equity_account, set()).add(lot.currency)
    return currencies


def _commodities_used(
    doc: LedgerDocument, account_ccy: dict[str, set[str]]
) -> list[str]:
    """Every commodity the document touches (INR always present when there is
    any posting)."""
    used: set[str] = {base.INR}
    for ccys in account_ccy.values():
        used |= ccys
    return sorted(used)


def _lot_open_dates(
    doc: LedgerDocument, anchor: datetime.date | None
) -> dict[str, datetime.date]:
    """Earliest ``open`` date each lot asset/equity account needs.

    Beancount requires an account to be opened on or before its first posting.
    A lot acquired *before* the cutover would post to its asset/equity account
    earlier than a cutover-dated ``open``, which bean-check rejects as an
    "inactive account". Each such account is therefore opened at
    ``min(cutover, earliest relevant lot.acquired_on)``; accounts whose lots are
    all on/after the cutover stay cutover-dated (``min`` collapses to cutover).
    Normal (non-lot) accounts are unaffected — they only ever post on/after the
    cutover.
    """
    out: dict[str, datetime.date] = {}
    for lot in doc.lot_postings:
        commodity = base.sanitize_commodity(lot.instrument)
        accounts = (
            validate(f"{base.INVESTMENT_ASSET_ROOT}:{commodity}"),
            normalize_default_account(base.INVESTMENT_EQUITY_OPENING),
        )
        for acct in accounts:
            cur = out.get(acct)
            if cur is None or lot.acquired_on < cur:
                out[acct] = lot.acquired_on
    if anchor is None:
        return out
    return {acct: min(d, anchor) for acct, d in out.items()}


def _format_posting(
    account: str,
    amount: Decimal,
    commodity: str,
    *,
    amount_formatter=base.fmt_amount,
) -> str:
    line = base.format_posting_line(
        account,
        amount,
        indent=True,
        commodity=commodity,
        validate=validate,
        amount_formatter=amount_formatter,
    )
    return line


def render_document(doc: LedgerDocument) -> str:
    """Render a :class:`LedgerDocument` to beancount text.

    Ordering: ``commodity`` declarations, then ``open`` directives (each listing
    the account's currencies), then dated transactions, then deduplicated
    ``price`` directives. Every dated directive needs a date; when no cutover is
    present the declarations are skipped (the projection always supplies a
    cutover when the document is non-empty).
    """
    lines: list[str] = []
    anchor = doc.cutover_date

    account_ccy = _account_currencies(doc)
    commodities = _commodities_used(doc, account_ccy)
    # Lot asset/equity accounts may need an earlier open when a lot was acquired
    # before the cutover; every other account opens at the cutover (anchor).
    lot_dates = _lot_open_dates(doc, anchor)

    # Declarations are only meaningful when something is actually posted — a
    # bare ``commodity``/``open`` block with no transactions is noise and would
    # make an empty document render non-empty (breaking byte-parity with the
    # ledger family's empty-document contract).
    has_postings = bool(account_ccy)
    if anchor is not None and has_postings and commodities:
        for sym in commodities:
            lines.append(f"{anchor.isoformat()} commodity {sym}")
        lines.append("")

    if anchor is not None and has_postings:
        # Open every account that receives a posting (derived from postings, not
        # just the declared selection) plus the equity contra for openings.
        opens: list[str] = sorted(account_ccy.keys())
        if any(ob.amount for ob in doc.openings):
            opens.append(normalize_default_account(EQUITY_OPENING))
            opens = sorted(set(opens))
        # Sort by (open_date, account) so the block stays date-ordered even when
        # a lot account opens earlier than the cutover.
        opens = sorted(opens, key=lambda a: (lot_dates.get(a, anchor), a))
        for acct in opens:
            ccys = sorted(
                account_ccy.get(acct, set())
                | (
                    {base.INR}
                    if acct == normalize_default_account(EQUITY_OPENING)
                    else set()
                )
            )
            # A lot account whose earliest lot predates the cutover is opened at
            # that lot's date; otherwise the open is cutover-dated.
            open_date = lot_dates.get(acct, anchor)
            if ccys:
                # A comma-separated currency list renders bare
                # (``open Account INR,USD``): beancount parses an unquoted
                # comma list as the currencies arg, but a *quoted* list is
                # read as the booking-method slot, so quoting would be wrong.
                lines.append(f"{open_date.isoformat()} open {acct} {','.join(ccys)}")
            else:
                lines.append(f"{open_date.isoformat()} open {acct}")
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


def _meta_lines(meta, indent_level: int = 1) -> list[str]:
    """Render ``((key, value), ...)`` as beancount metadata: ``key: "value"``
    one per line, lowercase keys, quoted string values. ``indent_level`` 1 =
    entry-level (4 spaces, before postings), 2 = posting-level (8 spaces)."""
    indent = _POSTING_INDENT * indent_level
    return [f"{indent}{k}: {quote_string(v)}" for k, v in meta]


def _txn_meta_or_comment(entry: ProjectedEntry) -> list[str]:
    """Entry-level metadata as beancount ``key: "value"`` lines.

    The backward-compatible ``txn`` key carries the same id as the ledger
    ``; txn:<id>`` comment so drill-through queries work across backends.
    """
    out: list[str] = []
    txn_comment = ", ".join(str(t) for t in entry.txn_ids)
    if txn_comment:
        out.append(f"{_POSTING_INDENT}txn: {quote_string(txn_comment)}")
    if entry.meta:
        out.extend(_meta_lines(entry.meta, indent_level=1))
    return out


def _render_openings(doc: LedgerDocument) -> list[str]:
    nonzero = [ob for ob in doc.openings if ob.amount]
    if not nonzero or doc.cutover_date is None:
        return []
    equity = normalize_default_account(EQUITY_OPENING)
    lines: list[str] = [f'{doc.cutover_date.isoformat()} * "Opening Balances"']
    # Entry-level kind metadata.
    lines.append(f'{_POSTING_INDENT}dashboard_kind: "opening"')
    equity_total = Decimal("0.00")
    for ob in nonzero:
        equity_total += ob.amount
        lines.append(_format_posting(ob.account_name, ob.amount, base.INR))
        if ob.meta:
            lines.extend(_meta_lines(ob.meta, indent_level=2))
    lines.append(_format_posting(equity, -equity_total, base.INR))
    lines.append("")
    return lines


def _render_entry(entry: ProjectedEntry) -> list[str]:
    check_balanced(entry)
    payee = sanitize_text(entry.payee)
    note = sanitize_text(entry.note)[:160] if entry.note else ""
    header = f"{entry.date.isoformat()} *"
    if payee:
        header += f" {quote_string(payee)}"
    if note:
        header += f" {quote_string(note)}"
    lines: list[str] = [header]
    # Entry-level metadata (txn + dashboard_* keys) before postings.
    lines.extend(_txn_meta_or_comment(entry))
    for posting in entry.postings:
        lines.append(
            _format_posting(posting.account, posting.amount, posting.commodity)
        )
        if posting.meta:
            lines.extend(_meta_lines(posting.meta, indent_level=2))
    lines.append("")
    return lines


def _render_prices(doc: LedgerDocument) -> list[str]:
    lines: list[str] = []
    for price in sorted(doc.price_directives, key=lambda p: (p.date, p.currency)):
        rate_text = fmt_price_rate(price.rate, price.unit)
        lines.append(f"{price.date.isoformat()} price {price.currency} {rate_text}")
    return lines


def _render_lot(lot: InvestmentLotEntry) -> list[str]:
    """One conservative investment-lot opening as a beancount cost-basis entry.

    Uses beancount's valid cost syntax ``{unit_cost CURRENCY, acquired_on}`` so
    the lot carries both its per-unit cost and its acquisition date for capital
    gains. The equity contra is :data:`base.INVESTMENT_EQUITY_OPENING`
    (PascalCased to ``Equity:OpeningBalances:Investment``); no cash leg is
    inferred. beancount values the asset leg's cost as ``quantity * unit_cost``
    at full precision, so the equity leg emits that same product (via
    :func:`base.fmt_lot_money`) and the entry balances to the penny *and* to
    full precision for every lot — including one whose product is not 2-dp
    exact. The stored 2-dp ``cost_basis`` is reporting / source-agreement
    metadata only (still validated by check_lot_consistent).
    """
    check_lot_consistent(lot)
    commodity = base.sanitize_commodity(lot.instrument)
    asset_account = validate(f"{base.INVESTMENT_ASSET_ROOT}:{commodity}")
    equity_account = normalize_default_account(base.INVESTMENT_EQUITY_OPENING)
    payee = (
        sanitize_text(f"Investment Lot - {lot.instrument_name}")[:160]
        or "Investment Lot"
    )
    qty_text = base.fmt_lot_decimal(lot.quantity)
    unit_text = base.fmt_lot_decimal(lot.unit_cost)
    header = f"{lot.acquired_on.isoformat()} * {quote_string(payee)}"
    # The equity leg is the exact full-precision negative product so it cancels
    # the asset leg's cost (``quantity × unit_cost``) to the penny AND to full
    # precision — bean-check rejects a sub-cent imbalance. The stored 2-dp
    # ``cost_basis`` is reporting / source-agreement metadata only (still
    # validated by check_lot_consistent); emitting it here would leave a
    # sub-cent imbalance for a non-2dp lot.
    equity_amount = -(lot.quantity * lot.unit_cost)
    lines: list[str] = [header]
    # Lot metadata as entry-level beancount meta (non-sensitive CAS provenance).
    if lot.meta:
        lines.extend(_meta_lines(lot.meta, indent_level=1))
    lines.extend(
        [
            (
                f"{_POSTING_INDENT}{asset_account}"
                f"  {qty_text} {commodity}"
                f" {{{unit_text} {lot.currency}, {lot.acquired_on.isoformat()}}}"
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


__all__ = ["normalize_default_account", "quote_string", "render_document", "validate"]
