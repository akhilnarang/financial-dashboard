"""Shared types, constants and helpers for the Paisa renderer strategies.

Each backend (ledger, hledger, beancount) lives in its own module and exposes
the same small surface the :mod:`renderers` registry dispatches to:

* ``validate_account_name(name)`` — raise :class:`InvalidAccountName` on a name
  that breaks that backend's grammar, else return it cleaned.
* ``normalize_default_account(name)`` — deterministically transform a
  projection-generated *default* name into the backend's legal form. Applied to
  defaults only; operator overrides are validated strictly, never silently
  rewritten.
* ``render_document(doc)`` — pure ``LedgerDocument -> str``.

Everything here is pure and deterministic: the same document always renders to
byte-identical text, which is what makes the publisher's "skip on unchanged
bytes" idempotent.
"""

import datetime
from collections.abc import Callable
from decimal import Decimal
from typing import NamedTuple

# Indentation so postings line up under their date header. The amount column
# (52) matches Paisa's own default ``amount_alignment_column`` config, so the
# rendered file lines up with hand-edited entries. beancount ignores the column
# (it splits postings on any whitespace) but inherits it for free.
_POSTING_INDENT = "    "
AMOUNT_COLUMN = 52

#: Characters that must never appear in a payee/note: ``;`` starts a comment,
#: newlines/tabs break the line, and braces are ledger/beancount expression
#: syntax.
_UNSAFE_TEXT_CHARS = (";", "\n", "\r", "\t", "{", "}")

#: The balancing account every opening-balance entry posts against. Never
#: derived from data — opening balances are an equity plug by definition.
EQUITY_OPENING = "Equity:Opening Balances"

#: The contra liability a card payment posts to when the specific card the
#: payment settled cannot be determined from the source row. Documented in the
#: projection report so the operator knows the pairing was inferred, not known.
CARD_PAYMENT_CLEARING = "Liabilities:Credit Card"

#: The base commodity every amount is implicitly denominated in unless a source
#: row carries an explicit foreign currency. Opening balances are always INR.
INR = "INR"

#: The dedicated asset hierarchy investment-lot openings post to. Each lot
#: lands under ``Assets:Investments:<instrument>`` so it can never overlap the
#: bank/cash hierarchy a transaction projection writes — the two projections
#: stay disjoint and a lot cannot double-count a bank balance.
INVESTMENT_ASSET_ROOT = "Assets:Investments"

#: The equity contra investment-lot openings post against. Distinct from
#: :data:`EQUITY_OPENING` (the bank/cash opening contra) so investment openings
#: never net against bank openings in the same entry; the requirement's
#: "Equity:Opening/Investment" maps to this backend-valid (slash-free) name.
INVESTMENT_EQUITY_OPENING = "Equity:Opening Balances:Investment"

#: The asset root the *valuation-only* CAS fallback posts to. Deliberately a
#: sibling of :data:`INVESTMENT_ASSET_ROOT` rather than a child: a balance under
#: this root is an authoritative portfolio *market value* carrying no cost basis,
#: no acquisition date and no commodity, so it must never be mistaken for — or
#: aggregated with — a cost-annotated lot. A portfolio is represented by exactly
#: one of the two roots, never both, so the two can be summed without double
#: counting.
INVESTMENT_VALUATION_ROOT = "Assets:Investments:Valuation"

#: The equity contra a valuation-only balance posts against. Distinct from every
#: opening-balances contra because these postings are *revaluations*: the delta
#: between two CAS statement values is a change in market value, not a capital
#: contribution and not income. Naming it explicitly keeps an operator from
#: reading the movement as a realized gain.
EQUITY_REVALUATION = "Equity:Revaluation:Investments"

#: The portfolio-less valuation account used when no installation secret exists
#: to derive a non-reversible portfolio token. Merging portfolios is the safe
#: degradation; emitting a raw PAN into the journal is not.
INVESTMENT_VALUATION_SHARED = INVESTMENT_VALUATION_ROOT

#: The contra account an investment-category transaction (purchase or
#: redemption) posts against. This is an **asset movement**, never an expense
#: or income: money moves between the bank and an undifferentiated investments
#: bucket. A lot projection (when enabled) carries the detailed holding;
#: remapping a provably-funding bank leg to :data:`INVESTMENT_EQUITY_OPENING`
#: prevents double-counting the investment asset.
INVESTMENT_UNALLOCATED_ACCOUNT = "Assets:Investments:Unallocated"

#: The contra account a repayment-category transaction posts against. This is a
#: non-income clearing root — money arriving from outside the tracked scope
#: (somebody paying you back) is not earned income, so it lands in equity
#: rather than :data:`INCOME_ROOT`.
REPAYMENT_CLEARING_ACCOUNT = "Equity:Transfers In"


class InvalidAccountName(Exception):
    """Raised when an account name would break the backend's grammar."""


class UnbalancedEntry(Exception):
    """Raised when an entry's postings do not sum to zero (per commodity)."""


class LedgerPosting(NamedTuple):
    """One side of a balanced entry. The renderer balance-checks the entry it
    belongs to; the signed amount is emitted verbatim so the file is
    well-formed.

    ``commodity`` defaults to INR; a posting in an explicit foreign currency
    carries that currency here so the renderer emits ``10.00 USD`` (not a
    silently relabelled ``10.00 INR``).

    ``meta`` is an ordered ``((key, value), ...)`` tuple of sanitized posting-
    level metadata (e.g. an opening posting's account-id/source/as-of). Empty
    by default so a posting constructed without metadata renders unchanged.
    """

    account: str
    amount: Decimal
    commodity: str = INR
    meta: tuple[tuple[str, str], ...] = ()


class LedgerAccount(NamedTuple):
    """Resolved ledger identity of a dashboard account."""

    account_id: int
    name: str
    kind: str  # "asset" | "liability"


class ProjectedEntry(NamedTuple):
    """A single projected entry (one date/payee, one or more postings).

    ``txn_ids`` carries every dashboard transaction id that contributed — a
    self-transfer pair has two; a single spend has one. ``currency`` is the
    commodity the entry's postings are denominated in (INR for native rows, the
    explicit foreign currency for a priced foreign entry).

    ``kind`` is the closed-taxonomy dashboard entry kind (``expense``,
    ``income``, ``contra_expense``, ``investment``, ``repayment``,
    ``self_transfer``, ``card_payment``, ``opening``, ``investment_lot``,
    ``unknown``) — the accounting meaning of the entry, distinct from
    ``dashboard_category`` (the raw slug).

    ``meta`` is an ordered ``((key, value), ...)`` tuple of sanitized entry-
    level metadata rendered as backend tags (ledger/hledger ``; key: value``
    lines) or beancount ``key: "value"`` metadata. Keys are the canonical
    ``dashboard_*`` fields (txn_ids/kind/category/source/channel/email_type/
    account_ids/card_ids/reference). Values are pre-sanitized — no secrets, raw
    bodies, or full masks reach the journal.
    """

    date: datetime.date
    payee: str
    txn_ids: tuple[int, ...]
    postings: tuple[LedgerPosting, ...]
    note: str | None = None
    currency: str = INR
    kind: str = "unknown"
    meta: tuple[tuple[str, str], ...] = ()


class OpeningBalance(NamedTuple):
    """An opening-balance line for one account, struck at the cutover date.

    Opening balances are always INR — a reliable original commodity is not
    fabricated (see projection policy).

    ``meta`` is ordered ``((key, value), ...)`` rendered as posting-level
    metadata on the account posting (account-id/source/as-of), so an operator
    can trace each opening line to its source snapshot or running balance.
    """

    account_id: int
    account_name: str
    amount: Decimal
    source: str  # "snapshot" | "transaction_balance"
    as_of: datetime.date | None
    meta: tuple[tuple[str, str], ...] = ()


class PriceDirective(NamedTuple):
    """A backend price directive: ``<currency>`` is worth ``<rate>`` ``<unit>``
    (INR per unit) as of ``<date>``. Emitted only when an explicit FX rate was
    configured for the date; never synthesized from a network call.

    The projection deduplicates these (one per ``(currency, date)``) before
    handing them to the renderer, so a backend file never carries duplicate
    price lines for the same currency/date.
    """

    date: datetime.date
    currency: str
    rate: Decimal
    unit: str = INR


class InvestmentLotEntry(NamedTuple):
    """A complete investment lot projected as a conservative cost-basis opening.

    Built only from a :class:`InvestmentLot` whose every field is an explicit
    source fact. The renderer posts ``quantity`` of the ``instrument``
    commodity into :data:`INVESTMENT_ASSET_ROOT` with a per-unit ``unit_cost``
    cost annotation and the ``acquired_on`` lot date, balanced against
    :data:`INVESTMENT_EQUITY_OPENING` for the exact full-precision negative
    product ``-(quantity * unit_cost)``. No bank/cash leg is inferred, and
    nothing here is derived from a current market value.

    ``cost_basis`` is the 2-dp-quantized product the service stores as
    reporting / source-agreement metadata; the renderer emits the *full*
    product on the equity leg (via :func:`fmt_lot_money`) so the entry balances
    to the penny *and* to full precision even when ``quantity * unit_cost`` is
    not itself 2-dp exact (e.g. ``3.000300 × 33.33``). ``cost_basis`` is still
    required to equal ``quantity * unit_cost`` at 2 dp (the service enforces
    this before building an entry, and :func:`check_lot_consistent` re-checks
    it at render time) so the stored metadata stays an honest rounding.
    """

    instrument: str
    instrument_name: str
    quantity: Decimal
    unit_cost: Decimal
    cost_basis: Decimal
    currency: str
    acquired_on: datetime.date
    #: Non-sensitive CAS provenance carried as entry-level metadata when the
    #: source lot explicitly states them (instrument is already a field). The
    #: projection populates these from the :class:`InvestmentLot` ORM row; they
    #: never carry credentials or raw CAS payloads.
    cas_upload_id: int | None = None
    source_ref: str | None = None
    reference: str | None = None
    meta: tuple[tuple[str, str], ...] = ()


class LedgerDocument(NamedTuple):
    """Everything the renderer needs to emit a complete journal body.

    The publisher wraps this with the do-not-edit header; the renderer is only
    responsible for the date-ordered entries. ``price_directives`` is the
    deduplicated set of FX prices the projection resolved (empty unless the
    ``priced`` non-INR policy emitted a foreign entry with a configured rate).
    ``lot_postings`` is the generic renderer capability for cost-basis lot
    openings. **The Paisa projection never populates it** — CAS is projected as
    an authoritative aggregate valuation with no cost basis. The capability is
    retained for a future, separately reviewed cost-basis feature and is still
    covered by renderer-syntax tests.
    """

    cutover_date: datetime.date | None
    openings: tuple[OpeningBalance, ...]
    entries: tuple[ProjectedEntry, ...]
    accounts_declared: tuple[str, ...]
    price_directives: tuple[PriceDirective, ...] = ()
    lot_postings: tuple[InvestmentLotEntry, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.openings and not self.entries and not self.lot_postings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def sanitize_text(raw: str | None) -> str:
    """Collapse whitespace and strip comment/directive characters from a free
    text field (payee or note). Returns empty for ``None``/empty input."""
    if not raw:
        return ""
    text = str(raw)
    for ch in _UNSAFE_TEXT_CHARS:
        text = text.replace(ch, " ")
    return " ".join(text.split())


#: Characters stripped from every metadata value regardless of backend: ``;``
#: starts a comment (ledger/hledger), newline/tab break the line, braces are
#: ledger/beancount expression syntax, and ``"``/``\`` would break a beancount
#: quoted string. A comma is stripped too so a value can never be misread as a
#: beancount currency-list separator in an ``open`` directive position.
_META_UNSAFE_CHARS = (";", "\n", "\r", "\t", "{", "}", ",", '"', "\\")


def sanitize_meta_value(raw: str | None) -> str:
    """Sanitize a metadata value for both ledger-family tags and beancount
    quoted strings.

    Strips every char that could break a tag line (``;`` comment, newline/tab)
    or a beancount quoted value (``"`` / ``\\``), collapses whitespace, and
    truncates to a safe length. Returns empty for ``None``/empty. A non-empty
    result is safe to emit verbatim in both backends without further escaping.

    Metadata values never carry secrets, raw email/SMS bodies, or full card/
    account masks — the projection builds them only from non-sensitive fields
    (ids, slugs, channels, references) and this function is the last guard.
    """
    if not raw:
        return ""
    text = str(raw)
    for ch in _META_UNSAFE_CHARS:
        text = text.replace(ch, " ")
    return " ".join(text.split())[:200]


def validate(name: str) -> str:
    """Ledger/hledger-family account-name validation.

    Account names must be a ``:``-joined hierarchy with no empty segments and
    no character ledger/hledger treat specially (newline, tab, ``;``, ``{``,
    ``}`). Spaces are legal in both backends, so a default name with internal
    spaces survives verbatim. This is the validator shared by the ``ledger`` and
    ``hledger`` backend ids; ``beancount`` overrides it with a stricter rule.
    """
    if not isinstance(name, str) or not name.strip():
        raise InvalidAccountName("account name is empty")
    for ch in _UNSAFE_TEXT_CHARS:
        if ch in name:
            raise InvalidAccountName(f"account name contains {ch!r}: {name!r}")
    cleaned = " ".join(name.split())
    if cleaned != name:
        # Internal whitespace runs are fine after collapsing, but the stored
        # form is what gets emitted — collapse it for a stable render.
        name = cleaned
    segments = name.split(":")
    if len(segments) < 2:
        raise InvalidAccountName(
            f"account name must be a ':', separated hierarchy: {name!r}"
        )
    if any(seg.strip() == "" for seg in segments):
        raise InvalidAccountName(
            f"account name has an empty hierarchy segment: {name!r}"
        )
    return name


def check_balanced(entry: ProjectedEntry) -> None:
    """Assert an entry's postings sum to zero, per commodity.

    The projection signs postings to balance by construction; this is the guard
    that a future accounting bug surfaces as a hard failure rather than emitting
    a file the backend would reject or silently equity-balance. Each commodity
    must independently sum to zero — a priced foreign entry balances in its own
    currency, never by mixing currencies.
    """
    if not entry.postings:
        raise UnbalancedEntry(f"entry on {entry.date} has no postings")
    totals: dict[str, Decimal] = {}
    for posting in entry.postings:
        totals[posting.commodity] = (
            totals.get(posting.commodity, Decimal("0.00")) + posting.amount
        )
    for commodity, total in totals.items():
        if total.quantize(Decimal("0.01")) != 0:
            raise UnbalancedEntry(
                f"entry on {entry.date} ({entry.payee!r}) does not balance in "
                f"{commodity}: sum={total}"
            )


def quantize(value: Decimal) -> Decimal:
    """Two-decimal quantization. Renders/diffs cleanly and parses unambiguously."""
    return value.quantize(Decimal("0.01"))


def fmt_amount(value: Decimal, commodity: str = INR) -> str:
    """``<sign><abs>.2f <COMMODITY>`` — the commodity always trails the amount
    by a single space, which ledger/hledger/beancount all parse. INR is the
    default so existing INR rows render identically to v1."""
    q = quantize(value)
    sign = "-" if q < 0 else ""
    return f"{sign}{abs(q):.2f} {commodity}"


def fmt_price_rate(rate: Decimal, unit: str = INR) -> str:
    """``<rate> <UNIT>`` for a price directive, at the rate's stored precision
    (4 dp). Money amounts use :func:`fmt_amount` (2 dp); FX rates carry more
    significant digits and must not be silently rounded to 2 dp — a configured
    ``0.0123`` must render ``0.0123``, not ``0.01``."""
    q = rate.quantize(Decimal("0.0001"))
    sign = "-" if q < 0 else ""
    return f"{sign}{abs(q):.4f} {unit}"


def format_posting_line(
    account: str,
    amount: Decimal,
    *,
    indent: bool,
    commodity: str = INR,
    validate,
    amount_formatter: Callable[[Decimal, str], str] = fmt_amount,
) -> str:
    """One posting, with the amount right-aligned to a fixed column.

    Right-alignment keeps the rendered file diff-stable: a wider account name
    pushes the amount past the column rather than reflowing neighbors, and a
    narrower one pads with spaces. The column (52) matches Paisa's default
    ``amount_alignment_column`` so a hand-edited entry lines up.

    The account name and the amount are always separated by at least two
    spaces: ledger/hledger's parser needs that gap (or a tab) to tell where the
    name ends and the amount begins. A single space would let the amount be
    absorbed into the account name when a long name overruns the alignment
    column, so the overflow case is padded with two spaces rather than one.
    beancount splits on any whitespace so the two-space gap is harmless there.

    ``validate`` is the backend's account-name validator (passed in to avoid a
    circular import on the registry). ``amount_formatter`` defaults to
    :func:`fmt_amount` (2 dp); the investment-lot equity leg passes
    :func:`fmt_lot_money` so a sub-cent product renders at full precision.
    """
    prefix = _POSTING_INDENT if indent else ""
    name = validate(account)
    amount_text = amount_formatter(amount, commodity)
    target = AMOUNT_COLUMN - len(prefix)
    gap_len = max(target - len(name), 2)
    return f"{prefix}{name}{' ' * gap_len}{amount_text}"


def entry_posting_commodities(entry: ProjectedEntry) -> set[str]:
    """The set of commodities an entry's postings touch."""
    return {p.commodity for p in entry.postings}


def sanitize_commodity(raw: str | None) -> str:
    """Reduce an instrument id (ISIN) to a backend-safe commodity symbol.

    Commodities must be a single token with no whitespace and no character a
    backend parser treats specially. ISINs are already uppercase alphanumeric
    (two leading letters + nine alphanumeric + a check digit), which all three
    backends accept; this keeps only ``[A-Za-z0-9]`` and uppercases the result
    so a stray punctuation mark in an instrument id cannot break a posting.
    A falsy input is rejected so an unnamed instrument is never emitted.
    """
    if not raw:
        raise InvalidAccountName("investment instrument (commodity) is empty")
    symbol = "".join(ch for ch in str(raw) if ch.isalnum()).upper()
    if not symbol or not symbol[0].isalpha():
        raise InvalidAccountName(
            f"investment instrument must be alphanumeric and start with a "
            f"letter: {raw!r}"
        )
    return symbol


def commodity_token(raw: str | None) -> str:
    """The single ledger-family commodity token: sanitize, reject empty, and
    quote any symbol that is not pure ASCII letters.

    Used in every ledger-family text position — posting amounts, ``commodity``
    declarations, and ``P`` directives — so the symbol renders identically
    everywhere and parses under both Ledger 3.3.2 and real hledger. Pure-ASCII-
    letter currency codes (INR, USD) stay bare and existing goldens are
    unchanged; anything else (a digit-bearing ISIN, mixed alphanumerics) is
    wrapped in double quotes.

    hledger rejects a bare alphanumeric commodity in a ``commodity`` declaration
    or a ``P`` directive (``unexpected '0'`` / ``unexpected 'A'``), and Ledger
    3.3.2 rejects one in a posting amount — so the *same* quoted token must be
    used in all three positions, not only in amounts. The quotes are a parser
    hint only: ``"INE000A01020"`` and ``INE000A01020`` denote the one commodity
    in both backends (verified against ``ananthakumaran/paisa:v0.7.4``'s Ledger
    3.3.2 and ``dastapov/hledger:1.52.1``).

    The input is sanitized first (alnum-only, uppercased) and rejected if empty,
    so a malformed FX code or instrument id can never create an invalid
    directive. Newlines, tabs, quotes and backslashes cannot survive that
    sanitization; the quote-branch escape is defense-in-depth.
    """
    symbol = sanitize_commodity(raw)
    if symbol.isascii() and symbol.isalpha():
        return symbol
    escaped = symbol.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def all_posting_accounts(doc: LedgerDocument) -> list[str]:
    """Every account named in a posting or opening (sorted, deduped).

    Used by backends (beancount) that must ``open`` every account that receives
    a posting, not just the projection's declared selection. Investment-lot
    postings are included so their asset/equity accounts are opened too.
    """
    seen: dict[str, None] = {}
    for ob in doc.openings:
        if ob.account_name and ob.account_name not in seen:
            seen[ob.account_name] = None
    for entry in doc.entries:
        for posting in entry.postings:
            if posting.account and posting.account not in seen:
                seen[posting.account] = None
    for lot in doc.lot_postings:
        asset_account = investment_asset_account(lot.instrument)
        if asset_account not in seen:
            seen[asset_account] = None
        equity_account = normalize_investment_equity()
        if equity_account not in seen:
            seen[equity_account] = None
    return list(seen)


def investment_asset_account(instrument: str) -> str:
    """The ``Assets:Investments:<commodity>`` account a lot posts to."""
    return f"{INVESTMENT_ASSET_ROOT}:{sanitize_commodity(instrument)}"


def normalize_investment_equity() -> str:
    """The investment opening equity contra, validated (identity for ledger)."""
    return validate(INVESTMENT_EQUITY_OPENING)


def investment_valuation_account(token: str | None) -> str:
    """The valuation-only asset account for an opaque portfolio *token*.

    ``token`` is the keyed, non-reversible handle from
    :mod:`financial_dashboard.services.paisa.portfolio_identity` — never a raw
    portfolio key. ``None`` (no installation secret) collapses to the shared
    portfolio-less account rather than leaking the source identifier.
    """
    if not token:
        return INVESTMENT_VALUATION_SHARED
    segment = "".join(ch for ch in str(token) if ch.isalnum() or ch == "-")
    if not segment:
        return INVESTMENT_VALUATION_SHARED
    return f"{INVESTMENT_VALUATION_ROOT}:{segment}"


def check_lot_consistent(lot: InvestmentLotEntry) -> None:
    """Assert ``quantity * unit_cost`` agrees with ``cost_basis`` to the penny.

    A lot entry that fails this cannot balance in any backend (the cost leg and
    the equity leg disagree), so it surfaces as a hard error rather than a
    corrupt file. The service enforces this before building an entry; this is
    the renderer-side guard against a future caller that bypasses it.
    """
    product = (lot.quantity * lot.unit_cost).quantize(Decimal("0.01"))
    if abs(product - lot.cost_basis.quantize(Decimal("0.01"))) > Decimal("0.01"):
        raise UnbalancedEntry(
            f"investment lot {lot.instrument!r}: cost_basis {lot.cost_basis} "
            f"!= quantity*unit_cost {product}"
        )


def fmt_lot_decimal(value: Decimal) -> str:
    """Render a lot quantity/unit-cost as fixed-point text (no scientific
    notation), preserving the value's significant digits. Money amounts still
    use :func:`fmt_amount` (2 dp); lot quantities/unit costs carry more
    significant digits and must not be rounded or rendered as ``1E+3``."""
    return format(value, "f")


def fmt_lot_money(value: Decimal, commodity: str = INR) -> str:
    """Exact full-precision ``<value> <COMMODITY>`` for an investment-lot equity
    leg — the *only* money position that is not 2-dp quantized.

    Every backend recomputes the asset leg's cost from ``quantity × unit_cost``
    at full precision, so the equity contra must emit the exact negative
    product ``-(quantity * unit_cost)``. A 2-dp-rounded equity (the stored
    :class:`InvestmentLotEntry.cost_basis`) only balances a lot whose product is
    itself 2-dp exact; for a sub-cent product such as ``3.000300 × 33.33 =
    99.999999`` a 2-dp equity (``-100.00``) would leave a ``-0.000001`` imbalance
    that Ledger 3.3.2, hledger 1.52.1, and ``bean-check`` all reject. The stored
    2-dp ``cost_basis`` is retained only as reporting / source-agreement
    metadata (still validated by :func:`check_lot_consistent`); this formatter
    is what the equity leg is actually emitted with.

    For a lot whose product *is* 2-dp exact this renders byte-identically to
    :func:`fmt_amount` (``format(Decimal('-50000.00'), 'f') == '-50000.00'``),
    so clean-lot golden bytes are unchanged; only the extra digits of a
    sub-cent product appear. Fixed-point (never scientific notation), and a
    zero product renders ``0`` (never ``-0``).
    """
    if value == 0:
        value = abs(value)
    return f"{format(value, 'f')} {commodity}"
