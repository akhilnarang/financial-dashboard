"""Offline Paisa 0.7.4 corpus generator (pure, no production imports).

Produces a self-contained, hand-reviewable Paisa corpus from a
:class:`Scenario`:

* ``main.ledger`` — the entry point, ``include``-ing the user-authored and
  dashboard-generated journals and the local ``prices.ledger``.
* ``user-authored.ledger`` — operator-shaped directives: commodity and account
  declarations, recurring (``; Recurring:``/``; Period:``) auto-transaction
  metadata, a budget periodic (``~``) entry, and an opening-balances entry.
* ``dashboard-generated.ledger`` — machine-generated dated postings, one
  ``; txn:<id>`` comment per entry, mirroring the production renderer's
  conventions (``Assets:Bank`` / ``Liabilities:Card`` / ``Expenses`` /
  ``Income`` / ``Equity:Opening Balances``; signed amounts so every entry
  balances to zero; amount column 52).
* ``paisa.yaml`` — a Paisa config using the **real 0.7.4 schema**
  (``journal_path``/``db_path`` required, ``ledger_cli: ledger``,
  ``default_currency: INR``) with non-empty, schema-valid report features
  (goals, allocation_targets, schedule_al, commodities, credit_cards). All
  network fetching stays disabled in the journal-sync path that
  ``scripts/paisa_contract.py`` exercises.
* ``prices.ledger`` — local ``P`` price directives (no network fetch).

Structural invariants (asserted by :func:`check_balanced` and the test
suite): every entry's postings sum to zero, every account name is a
``:``-joined hierarchy, and no free-text field can inject a directive.

The corpus intentionally caps the number of projected entries
(:data:`MAX_LEDGER_ENTRIES`) so even a stress scenario yields a
hand-reviewable journal; the high-volume rows stay in the SQLite DB, not the
ledger file. This is a documented fidelity limit.

The generated ``paisa.yaml`` and ledger directives were verified against the
``ananthakumaran/paisa:0.7.4`` image's published config schema
(``internal/config/schema.json``: ``required=[journal_path, db_path]``,
``additionalProperties: false``) and Cobra command surface
(``paisa --config <p> --now <d> update --journal`` runs the real
``ledger``-CLI ``balance``/``pricesdb``/``csv`` parse path — there is no
``paisa balance`` subcommand).
"""

import datetime
import re
from collections.abc import Callable
from decimal import Decimal
from typing import NamedTuple

from scripts.synth import constants as C
from scripts.synth.models import Scenario, SynthTransaction

MAX_LEDGER_ENTRIES = 500
AMOUNT_COLUMN = 52
_POSTING_INDENT = "    "
_UNSAFE = (";", "\n", "\r", "\t", "{", "}")
EQUITY_OPENING = "Equity:Opening Balances"
#: The dedicated investment hierarchy + equity contra a lot posts to, mirroring
#: the production renderer (``renderers.base.INVESTMENT_ASSET_ROOT`` /
#: ``INVESTMENT_EQUITY_OPENING``). Distinct from the bank equity so a lot never
#: nets against a bank opening in the same entry.
INVESTMENT_ASSET_ROOT = "Assets:Investments"
INVESTMENT_EQUITY_OPENING = "Equity:Opening Balances:Investment"

#: Names of every artefact this module produces, in a stable order.
ARTEFACT_NAMES = (
    "main.ledger",
    "user-authored.ledger",
    "dashboard-generated.ledger",
    "paisa.yaml",
    "prices.ledger",
)


class UnbalancedEntry(ValueError):
    """A generated entry's postings do not sum to zero."""


class InvalidAccountName(ValueError):
    """A generated account name would break the ledger grammar."""


class LedgerPosting(NamedTuple):
    account: str
    amount: Decimal


class LedgerEntry(NamedTuple):
    date: datetime.date
    payee: str
    txn_id: str
    postings: tuple[LedgerPosting, ...]
    note: str | None = None


class LotFact(NamedTuple):
    """A complete MF acquisition lot extracted from a CAS payload — the pure
    mirror of the production :class:`InvestmentLotEntry`. The contract corpus
    renders this with the *exact* ledger-family syntax the production renderer
    emits (quoted amount commodity, bare declaration/price), so a
    ``scripts/paisa_contract.py`` run verifies that syntax against real
    Paisa 0.7.4 + Ledger 3.3.2."""

    symbol: str
    instrument_name: str
    units: Decimal
    nav: Decimal
    amount: Decimal
    acquired_on: datetime.date
    asset_account: str


class LedgerCorpus(NamedTuple):
    """The full Paisa corpus as in-memory bytes, ready to write."""

    artefacts: dict[str, bytes]
    entries: int


def build_corpus(
    scenario: Scenario, *, max_entries: int = MAX_LEDGER_ENTRIES
) -> LedgerCorpus:
    """Build the in-memory Paisa corpus for ``scenario``."""
    entries, declared_accounts, commodities = _project_entries(scenario, max_entries)
    openings = _opening_balances(scenario)
    lots = _lot_facts(scenario)

    # A lot contributes its instrument ``commodity`` declaration and its
    # asset/equity accounts; its acquisition-date ``P`` directive is rendered
    # separately in prices.ledger so the contract corpus carries the quoted-lot
    # syntax end-to-end.
    lot_symbols = {lot.symbol for lot in lots}
    lot_accounts = {
        acct for lot in lots for acct in (lot.asset_account, INVESTMENT_EQUITY_OPENING)
    }
    declared_accounts = sorted(set(declared_accounts) | lot_accounts)
    commodities = sorted(set(commodities) | lot_symbols)

    generated_text = _render_generated(entries, lots)
    user_text = _render_user_authored(
        declared_accounts=declared_accounts,
        commodities=commodities,
        openings=openings,
        cutover=_cutover(scenario),
    )
    main_text = _render_main()
    config_text = _render_paisa_yaml()
    prices_text = _render_prices(scenario, commodities, lots)

    artefacts = {
        "main.ledger": main_text.encode("utf-8"),
        "user-authored.ledger": user_text.encode("utf-8"),
        "dashboard-generated.ledger": generated_text.encode("utf-8"),
        "paisa.yaml": config_text.encode("utf-8"),
        "prices.ledger": prices_text.encode("utf-8"),
    }
    return LedgerCorpus(artefacts=artefacts, entries=len(entries))


# ---------------------------------------------------------------------------
# Projection (scenario -> balanced ledger entries)
# ---------------------------------------------------------------------------


def _project_entries(
    scenario: Scenario, max_entries: int
) -> tuple[list[LedgerEntry], list[str], list[str]]:
    """Pick a representative, date-ordered slice of transactions and turn each
    into a balanced two-posting entry. Returns (entries, accounts, commodities)."""
    eligible = [
        t
        for t in scenario.transactions
        if t.ledger_account
        and t.ledger_counterpart
        and t.currency == "INR"
        and t.transaction_date is not None
    ]
    eligible.sort(key=lambda t: (t.transaction_date, t.stable_id))
    chosen = eligible[:max_entries]

    entries: list[LedgerEntry] = []
    accounts: set[str] = set()
    for t in chosen:
        accounts.update([t.ledger_account, t.ledger_counterpart])
        entries.append(_entry_from_transaction(t))

    declared = sorted(accounts | {EQUITY_OPENING})
    commodities = sorted(_commodities_for(scenario))
    return entries, declared, commodities


def _entry_from_transaction(t: SynthTransaction) -> LedgerEntry:
    """Sign postings so the entry balances to zero.

    Debits (expenses, asset increases, liability decreases) are positive on the
    named account; the counterpart takes the negation. Income is negative on
    the Income account (credit-normal).
    """
    debit = t.direction == C.DIRECTION_DEBIT
    if t.ledger_account.startswith("Income"):
        # Credit-normal: a credit (incoming) posts negative to Income.
        signed = -t.amount if not debit else t.amount
    else:
        signed = t.amount if debit else -t.amount
    postings = (
        LedgerPosting(account=t.ledger_account, amount=_q(signed)),
        LedgerPosting(account=t.ledger_counterpart, amount=_q(-signed)),
    )
    check_balanced(postings, t.transaction_date, _sanitize(t.counterparty))
    return LedgerEntry(
        date=t.transaction_date,
        payee=_sanitize(t.counterparty) or "Synthetic",
        txn_id=t.stable_id[:12],
        postings=postings,
        note=f"synthetic {t.email_type}",
    )


def _opening_balances(scenario: Scenario) -> list[tuple[str, Decimal]]:
    """One opening-balance line per active bank account, struck at the cutover."""
    out: list[tuple[str, Decimal]] = []
    for acct in scenario.accounts:
        if acct.type != C.BANK_ACCOUNT or not acct.active:
            continue
        name = f"Assets:Bank:{_ledger_leaf(acct)}"
        out.append((name, _q(Decimal("5000.00"))))
    return out


def _cutover(scenario: Scenario) -> datetime.date:
    """The opening-balance date: one day before the earliest projected txn."""
    dates = [t.transaction_date for t in scenario.transactions if t.transaction_date]
    if not dates:
        return scenario.as_of
    return min(dates) - datetime.timedelta(days=1)


def _sanitize_commodity(raw: str) -> str:
    """Reduce an ISIN to a backend-safe commodity symbol (pure mirror of the
    production :func:`renderers.base.sanitize_commodity`)."""
    symbol = "".join(ch for ch in str(raw) if ch.isalnum()).upper()
    if not symbol or not symbol[0].isalpha():
        raise InvalidAccountName(f"bad instrument commodity: {raw!r}")
    return symbol


def _commodity_token(raw: str | None) -> str:
    """The ledger-family commodity token (pure mirror of the production
    :func:`renderers.base.commodity_token`): sanitize, reject empty, quote any
    symbol that is not pure ASCII letters.

    Used in every text position — posting amounts, ``commodity`` declarations,
    and ``P`` directives — so the symbol parses under both Ledger 3.3.2 and real
    hledger. Pure-letter codes (INR, USD) stay bare; a digit-bearing symbol (an
    ISIN, a ``SYN...`` holding id) is quoted. hledger rejects a bare
    alphanumeric commodity in a declaration or P directive, so the mirror must
    quote there too — exactly as the production renderer does. The mirror stays
    byte-equivalent to the production token; that equivalence is pinned by a
    strict test (``tests/test_synth_paisa.py``).
    """
    symbol = _sanitize_commodity(raw)
    if symbol.isascii() and symbol.isalpha():
        return symbol
    escaped = symbol.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _fmt_lot_decimal(value: Decimal) -> str:
    """Fixed-point text for a lot quantity/unit-cost (no scientific notation),
    preserving significant digits — mirrors :func:`renderers.base.fmt_lot_decimal`."""
    return format(value, "f")


def _fmt_price_rate(rate: Decimal, unit: str = "INR") -> str:
    """``<rate> <UNIT>`` at 4 dp for a P directive (mirrors
    :func:`renderers.base.fmt_price_rate`). Lot navs carry more precision than
    a 2-dp money amount and must not be rounded."""
    q = rate.quantize(Decimal("0.0001"))
    sign = "-" if q < 0 else ""
    return f"{sign}{abs(q):.4f} {unit}"


def _lot_facts(scenario: Scenario) -> tuple[LotFact, ...]:
    """The complete MF acquisition lots in the scenario's CAS payloads.

    Pure mirror of the production investment classifier's "complete lot"
    shape: an ``mf`` ``purchase`` carrying ``isin``/``units``/``nav``/
    ``amount``/``date`` (the exact CAS fact the service turns into an
    InvestmentLot). Sorted by ``(acquired_on, symbol)`` — the same order the
    production projection emits — so the corpus is deterministic.
    """
    facts: list[LotFact] = []
    for cas in scenario.cas_uploads:
        for tx in cas.raw_payload.get("transactions", ()):
            if tx.get("scope") != "mf" or tx.get("transaction_type") != "purchase":
                continue
            if not all(tx.get(k) for k in ("isin", "units", "nav", "amount", "date")):
                continue
            symbol = _sanitize_commodity(tx["isin"])
            facts.append(
                LotFact(
                    symbol=symbol,
                    instrument_name=_sanitize(tx.get("description")) or "Investment",
                    units=Decimal(str(tx["units"])),
                    nav=Decimal(str(tx["nav"])),
                    amount=Decimal(str(tx["amount"])),
                    acquired_on=datetime.date.fromisoformat(tx["date"]),
                    asset_account=f"{INVESTMENT_ASSET_ROOT}:{symbol}",
                )
            )
    facts.sort(key=lambda f: (f.acquired_on, f.symbol))
    return tuple(facts)


def _commodities_for(scenario: Scenario) -> set[str]:
    """Commodity directives for the currencies and CAS holding names used."""
    out = {"INR", "USD"}
    for cas in scenario.cas_uploads:
        for name, _value in cas.holdings:
            out.add(_commodity_symbol(name))
    return out


def _commodity_symbol(name: str) -> str:
    """Derive a SYN... commodity symbol from a holding name."""
    base = re.sub(r"[^A-Z]", "", name.upper())
    symbol = base[:4] or "XXXX"
    return f"SYN{symbol}"


def _ledger_leaf(account) -> str:
    return account.label.split("(")[0].strip().replace(" ", "")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def check_balanced(
    postings: tuple[LedgerPosting, ...], date: datetime.date, payee: str
) -> None:
    if not postings:
        raise UnbalancedEntry(f"entry on {date} has no postings")
    total = sum((p.amount for p in postings), Decimal("0.00"))
    if _q(total) != 0:
        raise UnbalancedEntry(
            f"entry on {date} ({payee!r}) does not balance: sum={total}"
        )


def _validate_account(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise InvalidAccountName("empty account name")
    for ch in _UNSAFE:
        if ch in name:
            raise InvalidAccountName(f"account name contains {ch!r}: {name!r}")
    segments = name.split(":")
    if len(segments) < 2:
        raise InvalidAccountName(f"not a hierarchy: {name!r}")
    if any(seg.strip() == "" for seg in segments):
        raise InvalidAccountName(f"empty segment: {name!r}")
    return name


def _sanitize(raw: str | None) -> str:
    if not raw:
        return ""
    text = str(raw)
    for ch in _UNSAFE:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _q(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _fmt_amount(value: Decimal) -> str:
    q = _q(value)
    sign = "-" if q < 0 else ""
    return f"{sign}{abs(q):.2f} INR"


def _fmt_lot_money(value: Decimal, unit: str = "INR") -> str:
    """Exact full-precision ``<value> <UNIT>`` for an investment-lot equity leg
    (pure mirror of :func:`renderers.base.fmt_lot_money`) — the *only* money
    position that is not 2-dp quantized.

    The equity contra must emit the exact negative product ``-(units * nav)``
    so it cancels the asset leg's cost (``units × nav``) to the penny *and* to
    full precision in every backend: a sub-cent product (e.g.
    ``3.000300 × 33.33 = 99.99999900``) is not 2-dp exact, and a 2-dp equity
    (``-100.00``) would leave a ``-0.000001`` imbalance that Ledger 3.3.2,
    hledger 1.52.1 and ``bean-check`` all reject. For a lot whose product *is*
    2-dp exact this renders byte-identically to :func:`_fmt_amount`
    (``format(Decimal('-50000.00'), 'f') == '-50000.00'``), so clean-lot golden
    bytes are unchanged. Fixed-point (never scientific notation); a zero
    product renders ``0`` (never ``-0``)."""
    if value == 0:
        value = abs(value)
    return f"{format(value, 'f')} {unit}"


def _posting_line(
    account: str,
    amount: Decimal,
    *,
    amount_formatter: Callable[[Decimal], str] = _fmt_amount,
) -> str:
    name = _validate_account(account)
    amount_text = amount_formatter(amount)
    target = AMOUNT_COLUMN - len(_POSTING_INDENT)
    # Ledger requires at least two spaces (or a tab) between the account name
    # and the amount so it can split them; a single space would let the amount
    # be absorbed into the name when a long name overruns the alignment column.
    gap_len = max(target - len(name), 2)
    return f"{_POSTING_INDENT}{name}{' ' * gap_len}{amount_text}"


def _render_entry(entry: LedgerEntry) -> list[str]:
    check_balanced(entry.postings, entry.date, entry.payee)
    header = f"{entry.date.isoformat()} * {_sanitize(entry.payee)}".rstrip()
    header = f"{header}    ; txn:{entry.txn_id}"
    lines = [header]
    for p in entry.postings:
        lines.append(_posting_line(p.account, p.amount))
    if entry.note:
        cleaned = _sanitize(entry.note)[:160]
        if cleaned:
            lines.append(f"{_POSTING_INDENT}; note: {cleaned}")
    lines.append("")
    return lines


def _lot_entry_lines(lot: LotFact) -> list[str]:
    """One conservative investment-lot opening, mirroring the production
    ledger-family renderer's ``_render_lot`` output byte-for-byte: quantity of
    the (token-quoted) instrument commodity into ``Assets:Investments:<SYMBOL>``
    with a per-unit cost annotation and the acquisition-date lot note, balanced
    against the dedicated investment equity. The amount commodity is the
    ledger-family token (quoted when not pure ASCII letters, so both Ledger 3.3.2
    and hledger parse it); the cost-annotation currency and lot-date bracket stay
    bare. No bank/cash leg is inferred. The ``commodity``/``P`` directives render
    this same symbol via :func:`_commodity_token` elsewhere.

    The equity leg is the exact full-precision negative product ``-(units *
    nav)`` (via :func:`_fmt_lot_money`), exactly as the production renderer —
    never the 2-dp ``amount`` — so the entry balances to the penny *and* to full
    precision even for a sub-cent/zero-cost product."""
    payee = (
        _sanitize(f"Investment Lot - {lot.instrument_name}")[:160] or "Investment Lot"
    )
    amount_commodity = _commodity_token(lot.symbol)
    return [
        f"{lot.acquired_on.isoformat()} * {payee}",
        (
            f"{_POSTING_INDENT}{_validate_account(lot.asset_account)}"
            f"  {_fmt_lot_decimal(lot.units)} {amount_commodity}"
            f" {{{_fmt_lot_decimal(lot.nav)} INR}}"
            f" [{lot.acquired_on.isoformat()}]"
        ),
        _posting_line(
            INVESTMENT_EQUITY_OPENING,
            -(lot.units * lot.nav),
            amount_formatter=_fmt_lot_money,
        ),
        "",
    ]


def _render_generated(
    entries: list[LedgerEntry], lots: tuple[LotFact, ...] = ()
) -> str:
    lines: list[str] = [
        "; AUTO-GENERATED by scripts.synth — do not edit by hand.",
        "; Regenerate with: uv run python -m scripts.synth generate",
        "; Every entry below is balance-checked at generation time.",
        "",
    ]
    for entry in entries:
        lines.extend(_render_entry(entry))
    for lot in lots:
        lines.extend(_lot_entry_lines(lot))
    text = "\n".join(lines)
    return text + "\n" if text else ""


def _render_user_authored(
    *,
    declared_accounts: list[str],
    commodities: list[str],
    openings: list[tuple[str, Decimal]],
    cutover: datetime.date,
) -> str:
    lines: list[str] = [
        "; user-authored.ledger — operator-shaped directives.",
        "; Hand-reviewable. Safe to extend with real account aliases.",
        "",
    ]
    for sym in commodities:
        lines.append(f"commodity {_commodity_token(sym)}")
    if commodities:
        lines.append("")
    for name in declared_accounts:
        lines.append(f"account {_validate_account(name)}")
    if declared_accounts:
        lines.append("")

    # --- Recurring metadata + budget periodic entry -----------------------
    # These mirror Paisa's own demo journal (internal/generator/config.go):
    # ``=`` automated-transaction rules tag matching postings with the
    # ``Recurring`` / ``Period`` metadata Paisa's reports read, and a ``~``
    # periodic (budget) transaction projects forward-looking budget lines.
    # Both are ledger-native directives; ``paisa update --journal`` validates
    # them via the real ``ledger`` CLI. They are emitted *before* the dated
    # opening transaction so the journal parses top-to-bottom.
    lines.append("= Expenses:Rent")
    lines.append("    ; Recurring: Rent")
    lines.append("    ; Period: 1 * ?")
    lines.append("")
    lines.append("= Expenses:Utilities")
    lines.append("    ; Recurring: Utilities")
    lines.append("    ; Period: 1 * ?")
    lines.append("")
    # Budget window: a 12-month periodic envelope starting at the cutover.
    budget_start = cutover.replace(day=1)
    budget_end = _month_shift(budget_start, 12)
    lines.append(
        f"~ Monthly from {budget_start.isoformat()} to {budget_end.isoformat()}"
    )
    lines.append(_posting_line("Expenses:Rent", Decimal("15000.00")))
    lines.append(_posting_line("Expenses:Utilities", Decimal("2000.00")))
    # The balancer: no amount → ledger auto-balances the entry.
    lines.append(f"{_POSTING_INDENT}Assets:Bank:HDFCSavings")
    lines.append("")

    if openings:
        equity_total = Decimal("0.00")
        lines.append(f"{cutover.isoformat()} * Opening Balances")
        for name, amount in openings:
            if amount:
                equity_total += amount
                lines.append(_posting_line(name, amount))
        lines.append(_posting_line(EQUITY_OPENING, _q(-equity_total)))
        lines.append("")
    text = "\n".join(lines)
    return text + "\n" if text else ""


def _month_shift(start: datetime.date, months: int) -> datetime.date:
    """``start`` shifted forward by ``months`` months (first of month)."""
    from dateutil.relativedelta import relativedelta

    return (start + relativedelta(months=months)).replace(day=1)


def _render_main() -> str:
    return (
        "; main.ledger — Paisa entry point.\n"
        "; Includes the operator journal, the dashboard-generated journal, and\n"
        "; the local price directives.\n"
        "; Tested against ananthakumaran/paisa:0.7.4 via scripts/paisa_contract.py\n"
        "; (``paisa --config /data/paisa.yaml --now <d> update --journal``).\n"
        "\n"
        "include user-authored.ledger\n"
        "include dashboard-generated.ledger\n"
        "include prices.ledger\n"
    )


def _render_paisa_yaml() -> str:
    # The exact, schema-valid Paisa 0.7.4 config. ``journal_path`` and
    # ``db_path`` are REQUIRED by internal/config/schema.json; the file also
    # sets ``additionalProperties: false``, so the invented ``ledger_file`` /
    # ``price_db`` / ``data_dir`` / ``binary`` keys the upstream schema does
    # not define would be rejected at load time. These keys mirror
    # ``internal/generator/config.go``'s demo config (the canonical example),
    # trimmed to a self-consistent, non-empty, offline-safe subset:
    #
    # * ``commodities`` are declared so the schema's required ``type`` +
    #   ``price`` fields are populated. Network price fetch only happens on
    #   ``paisa update --commodity`` / ``--portfolio``, which the contract
    #   probe never runs, so the listed providers stay offline in the
    #   journal-sync path.
    # * ``schedule_al`` / ``allocation_targets`` / ``goals`` / ``credit_cards``
    #   are populated with the accounts this corpus actually emits, so every
    #   report feature has real data to render against.
    return (
        "# paisa.yaml — Paisa 0.7.4 config (schema-valid, offline-safe).\n"
        "# Generated by scripts.synth; regenerate with\n"
        "#   uv run python -m scripts.synth generate\n"
        "# Verified against the ananthakumaran/paisa:0.7.4 config schema:\n"
        "#   required=[journal_path, db_path], additionalProperties=false.\n"
        "\n"
        "journal_path: main.ledger\n"
        "db_path: paisa.db\n"
        "ledger_cli: ledger\n"
        "default_currency: INR\n"
        "amount_alignment_column: 52\n"
        "locale: en-IN\n"
        "time_zone: Asia/Kolkata\n"
        "\n"
        "budget:\n"
        "  rollover: yes\n"
        "\n"
        "schedule_al:\n"
        "  - code: bank\n"
        "    accounts:\n"
        "      - Assets:Bank:*\n"
        "  - code: liability\n"
        "    accounts:\n"
        "      - Liabilities:Card:*\n"
        "  - code: share\n"
        "    accounts:\n"
        "      - Equity:Opening Balances\n"
        "\n"
        "allocation_targets:\n"
        "  - name: Bank\n"
        "    target: 100\n"
        "    accounts:\n"
        "      - Assets:Bank:*\n"
        "\n"
        "goals:\n"
        "  retirement:\n"
        "    - name: Emergency\n"
        "      icon: mdi:lifebuoy\n"
        "      swr: 4\n"
        "      savings:\n"
        "        - Assets:Bank:*\n"
        "      expenses:\n"
        "        - Expenses:*\n"
        "  savings:\n"
        "    - name: Savings\n"
        "      icon: mdi:piggy\n"
        "      target: 1000000\n"
        "      accounts:\n"
        "        - Assets:Bank:*\n"
        "\n"
        "commodities:\n"
        "  - name: USD\n"
        "    type: unknown\n"
        "    price:\n"
        "      provider: co-alphavantage\n"
        "      code: USD\n"
        "\n"
        "credit_cards:\n"
        "  - account: Liabilities:Card:HDFCMillenniaCC\n"
        "    credit_limit: 150000\n"
        "    statement_end_day: 8\n"
        "    due_day: 20\n"
        "    network: visa\n"
        '    number: "4242"\n'
        '    expiration_date: "2029-05-01"\n'
    )


def _render_prices(
    scenario: Scenario,
    commodities: list[str],
    lots: tuple[LotFact, ...] = (),
) -> str:
    """Local P price directives, one per commodity per cutover date, plus a
    per-lot acquisition-date price at its explicit CAS nav.

    Emitted as a plain ledger file (``prices.ledger``) and pulled in by
    ``include`` from ``main.ledger`` — a text price file is clearer than a
    binary-looking ``prices.db`` name, and ``include`` is the only way the
    ``ledger`` CLI reads local price directives for ``paisa update --journal``.

    Lot instrument symbols are emitted only at their acquisition date (the
    truthful per-unit CAS nav) — not the cutover. The commodity is the
    ledger-family token (quoted when not pure ASCII letters), mirroring the
    production renderer so a quoted directive parses under hledger too.
    """
    lines = [
        "; prices.ledger — local price directives (no network).",
        "; Included from main.ledger; re-generated deterministically from the",
        "; synthetic scenario.",
        "",
    ]
    lot_symbols = {lot.symbol for lot in lots}
    anchor = _cutover(scenario)
    for sym in commodities:
        if sym == "INR" or sym in lot_symbols:
            continue
        price = Decimal("1.00") if sym == "USD" else Decimal("100.00")
        lines.append(
            f"P {anchor.isoformat()} {_commodity_token(sym)} {_fmt_amount(price)}"
        )
    for lot in lots:
        lines.append(
            f"P {lot.acquired_on.isoformat()} {_commodity_token(lot.symbol)} {_fmt_price_rate(lot.nav)}"
        )
    text = "\n".join(lines)
    return text + "\n" if text else ""
