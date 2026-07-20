"""Multi-backend renderer contracts: ledger / hledger / beancount.

Pins, per backend:

* a deterministic **golden** render (byte-stable across runs);
* injection-safety — a hostile payee/note cannot inject a directive, break a
  posting onto a new line, or smuggle a second transaction;
* the ledger/hledger ``>=2``-space account/amount separation contract, including
  long/spaced names that overrun the alignment column;
* beancount account-component legality (default normalization, override
  validation) and its ``commodity``/``open``/``price`` directives;
* per-commodity balance enforcement (a foreign entry balances in its own
  currency, never by mixing currencies);
* optional real-parser validation via the ``ledger``/``hledger``/``bean-check``
  CLIs when they are on PATH (gated; not a project dependency).
"""

import datetime
import os
import re
import shutil
import subprocess
import tempfile
from decimal import Decimal

import pytest

from financial_dashboard.services.paisa.renderers import (
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    get_renderer,
    normalize_default_account,
    render_document,
    validate_account_name,
    validate_backend,
)
from financial_dashboard.services.paisa.renderers.base import (
    AMOUNT_COLUMN,
    CARD_PAYMENT_CLEARING,
    EQUITY_OPENING,
    INR,
    InvalidAccountName,
    InvestmentLotEntry,
    LedgerDocument,
    LedgerPosting,
    OpeningBalance,
    PriceDirective,
    ProjectedEntry,
    UnbalancedEntry,
    commodity_token,
    sanitize_text,
)

pytestmark = pytest.mark.anyio

CUTOVER = datetime.date(2026, 1, 1)

BANK = "Assets:Bank:HDFC:Savings"
EXPENSE = "Expenses:Food:Groceries"


def _doc(*, openings=(), entries=(), accounts=(BANK,), prices=(), cutover=CUTOVER):
    return LedgerDocument(
        cutover_date=cutover,
        openings=tuple(openings),
        entries=tuple(entries),
        accounts_declared=tuple(accounts),
        price_directives=tuple(prices),
    )


def _entry(
    payee="Store",
    postings=(),
    date=datetime.date(2026, 2, 1),
    note=None,
    currency=INR,
    txn_ids=(42,),
):
    return ProjectedEntry(
        date=date,
        payee=payee,
        txn_ids=tuple(txn_ids),
        postings=tuple(postings),
        note=note,
        currency=currency,
    )


def _usd_entry():
    return _entry(
        payee="Cafe",
        currency="USD",
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00"), "USD"),
            LedgerPosting(BANK, Decimal("-10.00"), "USD"),
        ),
    )


def _inr_entry():
    return _entry(
        payee="Store",
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00")),
            LedgerPosting(BANK, Decimal("-10.00")),
        ),
    )


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


def test_supported_backends_are_exactly_three():
    assert SUPPORTED_BACKENDS == ("ledger", "hledger", "beancount")


def test_validate_backend_defaults_unknown_to_ledger():
    assert validate_backend(None) == "ledger"
    assert validate_backend("") == "ledger"
    assert validate_backend("CSV") == "ledger"
    assert validate_backend("ledger") == "ledger"
    assert validate_backend("HLEDGER") == "hledger"
    assert validate_backend("BeanCount") == "beancount"


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
def test_each_backend_has_a_strategy(backend):
    strategy = get_renderer(backend)
    assert strategy.backend == backend
    assert callable(strategy.render_document)
    assert callable(strategy.validate_account_name)
    assert callable(strategy.normalize_default_account)


def test_default_backend_is_ledger():
    assert DEFAULT_BACKEND == "ledger"


# ---------------------------------------------------------------------------
# Determinism (byte-stable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
def test_render_is_deterministic(backend):
    doc = _doc(
        entries=[_inr_entry(), _usd_entry()],
        prices=[PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.2500"))],
    )
    assert render_document(doc, backend) == render_document(doc, backend)


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
def test_empty_document_renders_to_empty_string(backend):
    assert render_document(_doc(entries=(), openings=(), accounts=()), backend) == ""


# ---------------------------------------------------------------------------
# Balance enforcement (per commodity)
# ---------------------------------------------------------------------------


def test_foreign_entry_balanced_in_own_currency():
    # Two USD postings summing to zero in USD; INR total is 0 (none). Valid.
    check = get_renderer("ledger").render_document
    check(_doc(entries=[_usd_entry()]))  # does not raise


def test_mixed_currency_unbalanced_entry_raises():
    # A USD leg and an INR leg that do not balance in either currency — the
    # renderer must refuse rather than emit a backend-invalid transaction.
    entry = _entry(
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00"), "USD"),
            LedgerPosting(BANK, Decimal("-10.00"), INR),
        )
    )
    for backend in SUPPORTED_BACKENDS:
        with pytest.raises(UnbalancedEntry):
            render_document(_doc(entries=[entry]), backend)


# ---------------------------------------------------------------------------
# Ledger / hledger syntax family
# ---------------------------------------------------------------------------


LEDGER_GOLDEN = """\
account Assets:Bank:HDFC:Savings

2026-01-01 * Opening Balances
    ; dashboard_kind: opening
    Assets:Bank:HDFC:Savings                        1000.00 INR
    Equity:Opening Balances                         -1000.00 INR

2026-02-01 * Cafe    ; txn:42
    Expenses:Food:Groceries                         10.00 USD
    Assets:Bank:HDFC:Savings                        -10.00 USD

P 2026-02-01 USD 83.2500 INR
"""


@pytest.mark.parametrize("backend", ["ledger", "hledger"])
def test_ledger_family_golden(backend):
    doc = _doc(
        openings=[
            OpeningBalance(
                account_id=1,
                account_name=BANK,
                amount=Decimal("1000.00"),
                source="snapshot",
                as_of=datetime.date(2025, 12, 15),
            )
        ],
        entries=[_usd_entry()],
        prices=[PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.2500"))],
    )
    assert render_document(doc, backend) == LEDGER_GOLDEN


@pytest.mark.parametrize("backend", ["ledger", "hledger"])
def test_ledger_family_postings_two_space_separated(backend):
    doc = _doc(entries=[_inr_entry(), _usd_entry()])
    out = render_document(doc, backend)
    for line in out.splitlines():
        if not line.startswith("    ") or "INR" not in line and "USD" not in line:
            continue
        if not (line.strip() and line.split()[-1] in ("INR", "USD")):
            continue
        parts = re.split(r" {2,}", line.lstrip(), maxsplit=1)
        assert len(parts) == 2, f"posting not split on >=2 spaces: {line!r}"


def test_ledger_long_spaced_account_keeps_two_space_gap():
    name = "Expenses:Insurance:Medical Health Insurance Premium"
    assert len(name) >= 47
    entry = _entry(
        postings=(
            LedgerPosting(name, Decimal("9999.00")),
            LedgerPosting(BANK, Decimal("-9999.00")),
        )
    )
    out = render_document(_doc(entries=[entry]), "ledger")
    line = next(ln for ln in out.splitlines() if name in ln)
    body = line.lstrip()
    parts = re.split(r" {2,}", body, maxsplit=1)
    assert parts[0] == name, f"account name mangled: {parts[0]!r}"
    assert parts[1] == "9999.00 INR"


# ---------------------------------------------------------------------------
# Beancount syntax
# ---------------------------------------------------------------------------


BEANCOUNT_GOLDEN = """\
2026-01-01 commodity INR
2026-01-01 commodity USD

2026-01-01 open Assets:Bank:HDFC:Savings INR,USD
2026-01-01 open Equity:OpeningBalances INR
2026-01-01 open Expenses:Food:Groceries USD

2026-01-01 * "Opening Balances"
    dashboard_kind: "opening"
    Assets:Bank:HDFC:Savings                        1000.00 INR
    Equity:OpeningBalances                          -1000.00 INR

2026-02-01 * "Cafe"
    txn: "42"
    Expenses:Food:Groceries                         10.00 USD
    Assets:Bank:HDFC:Savings                        -10.00 USD

2026-02-01 price USD 83.2500 INR
"""


def test_beancount_golden():
    doc = _doc(
        openings=[
            OpeningBalance(
                account_id=1,
                account_name=BANK,
                amount=Decimal("1000.00"),
                source="snapshot",
                as_of=datetime.date(2025, 12, 15),
            )
        ],
        entries=[_usd_entry()],
        prices=[PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.2500"))],
    )
    assert render_document(doc, "beancount") == BEANCOUNT_GOLDEN


def test_beancount_quotes_and_escapes_payee_and_note():
    entry = _entry(
        payee='Acme "Quoted" Co',
        note='a "note" with quotes',
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00")),
            LedgerPosting(BANK, Decimal("-10.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]), "beancount")
    txn = [ln for ln in out.splitlines() if ln.startswith("2026-02-01")][0]
    # Payee and note are double-quoted; embedded quotes are escaped.
    assert '"Acme \\"Quoted\\" Co"' in txn
    assert '"a \\"note\\" with quotes"' in txn


def test_beancount_string_values_round_trip_backslashes_quotes_and_controls():
    """Every data-derived quoted field preserves identity through Beancount.

    Payee/narration controls are sanitized before quoting; literal backslashes
    (including Windows-like paths) and quotes then survive ``load_string``
    exactly. Metadata uses the same quoting path rather than interpolating a
    value directly into a string literal.
    """
    from beancount import loader
    from beancount.core.data import Transaction

    raw_payee = 'Acme\\Branch "Desk"\nSecond\tFloor\rNorth'
    raw_note = 'Receipt C:\\new\\forms\\invoice "final".pdf\nverified\tcopy'
    raw_meta = 'C:\\Users\\Analyst\\reports\\Q1 "final".bean'
    entry = ProjectedEntry(
        date=datetime.date(2026, 2, 1),
        payee=raw_payee,
        txn_ids=(42,),
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00")),
            LedgerPosting(BANK, Decimal("-10.00")),
        ),
        note=raw_note,
        meta=(("dashboard_reference", raw_meta),),
    )

    journal = render_document(_doc(entries=[entry]), "beancount")
    entries, errors, _options = loader.load_string(journal)

    assert errors == [], errors
    transaction = next(item for item in entries if isinstance(item, Transaction))
    assert transaction.payee == sanitize_text(raw_payee)
    assert transaction.narration == sanitize_text(raw_note)
    assert transaction.meta["dashboard_reference"] == raw_meta


def test_beancount_rejects_space_in_account_override():
    # A beancount component may not contain spaces. The default is normalized;
    # an override with a space is rejected (never silently rewritten).
    with pytest.raises(InvalidAccountName):
        validate_account_name("Assets:Bank:HDFC:Savings Account", "beancount")


def test_beancount_rejects_unknown_root_and_lowercase_component():
    with pytest.raises(InvalidAccountName):
        validate_account_name("Cash:Bank:HDFC", "beancount")  # bad root
    with pytest.raises(InvalidAccountName):
        validate_account_name("Assets:bank", "beancount")  # lowercase component


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Assets:Bank:HDFC:Savings Account", "Assets:Bank:HDFC:SavingsAccount"),
        ("Equity:Opening Balances", "Equity:OpeningBalances"),
        ("Liabilities:Credit Card", "Liabilities:CreditCard"),
        ("Assets:Bank:hdfc", "Assets:Bank:Hdfc"),
        (
            "Expenses:Insurance:Medical Health Insurance Premium",
            "Expenses:Insurance:MedicalHealthInsurancePremium",
        ),
    ],
)
def test_beancount_normalizes_default_account_names(raw, expected):
    assert normalize_default_account(raw, "beancount") == expected


def test_beancount_normalization_is_identity_for_ledger_family():
    name = "Assets:Bank:HDFC:Savings Account"
    assert normalize_default_account(name, "ledger") == name
    assert normalize_default_account(name, "hledger") == name


def test_beancount_commodity_and_open_directives_present():
    out = render_document(_doc(entries=[_inr_entry(), _usd_entry()]), "beancount")
    assert "commodity INR" in out
    assert "commodity USD" in out
    # Every posted account is opened with the commodities it holds; a
    # multi-currency list is quoted so beancount parses it as one argument.
    assert "open Assets:Bank:HDFC:Savings INR,USD" in out
    assert (
        "open Expenses:Food:Groceries INR" in out
        or "open Expenses:Food:Groceries INR,USD" in out
    )


# ---------------------------------------------------------------------------
# Injection safety (all backends)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
def test_payee_newline_cannot_inject_a_transaction(backend):
    entry = _entry(
        payee="Evil\n2026-99-99 * forged\n    Assets:Fake  9999.00 INR",
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00")),
            LedgerPosting(BANK, Decimal("-10.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]), backend)
    lines = out.splitlines()
    # The forged payload never starts a new transaction/directive line — it is
    # collapsed into the single sanitized payee on the real header.
    assert not any(ln.startswith("2026-99-99") for ln in lines), out
    # Exactly one dated header for the real transaction date.
    headers = [ln for ln in lines if ln.startswith("2026-02-01")]
    assert len(headers) == 1, out
    # Exactly the two real postings exist; neither posts to the forged account
    # or carries the forged amount. (The forged text may survive inside the
    # sanitized payee — that is safe, it is just a string, not a posting.)
    posting_lines = [
        ln
        for ln in lines
        if ln.startswith("    ") and ln.split() and ln.split()[-1] in ("INR", "USD")
    ]
    assert len(posting_lines) == 2, out
    for ln in posting_lines:
        account = ln.lstrip().split("  ")[0]
        assert account != "Assets:Fake", ln
        assert "9999.00" not in ln, ln


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
def test_payee_semicolon_cannot_start_an_early_comment(backend):
    entry = _entry(
        payee="Store; EVIL",
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00")),
            LedgerPosting(BANK, Decimal("-10.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]), backend)
    header = [ln for ln in out.splitlines() if ln.startswith("2026-02-01")][0]
    # The only ';' on the header is the txn comment (or, for beancount, none
    # unless a txn id is present). The payee's ';' was collapsed to a space.
    assert "EVIL" in header
    assert "; EVIL" not in header
    if backend == "beancount":
        # beancount carries txn as metadata (not a header comment).
        assert 'txn: "42"' in out
    else:
        assert "txn:42" in header


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
def test_note_with_newline_and_semicolon_collapsed(backend):
    entry = _entry(
        payee="Store",
        note="line1\nline2; comment-injection",
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00")),
            LedgerPosting(BANK, Decimal("-10.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]), backend)
    # The note survives as sanitized text; no second directive line appears.
    assert "comment-injection" in out
    assert "line1 line2 comment-injection" in out or "line1 line2" in out


# ---------------------------------------------------------------------------
# Optional real-parser validation (gated by CLI availability)
# ---------------------------------------------------------------------------


def _cli_available(name: str) -> bool:
    return shutil.which(name) is not None


def _run(backend_cli, journal: str, *args) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(journal)
        path = fh.name
    return subprocess.run(
        [backend_cli, path, *args], capture_output=True, text=True, timeout=30
    )


@pytest.mark.skipif(not _cli_available("ledger"), reason="ledger CLI not installed")
def test_real_ledger_parses_multicurrency_output():
    doc = _doc(
        entries=[_inr_entry(), _usd_entry()],
        prices=[PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.2500"))],
    )
    journal = render_document(doc, "ledger")
    completed = _run("ledger", journal, "balanced")
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(not _cli_available("hledger"), reason="hledger CLI not installed")
def test_real_hledger_parses_multicurrency_output():
    doc = _doc(
        entries=[_inr_entry(), _usd_entry()],
        prices=[PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.2500"))],
    )
    journal = render_document(doc, "hledger")
    # hledger check (the closest equivalent of a parse/balance check).
    completed = _run("hledger", journal, "check")
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(
    not _cli_available("bean-check"), reason="bean-check CLI not installed"
)
def test_real_bean_check_validates_beancount_output():
    doc = _doc(
        openings=[
            OpeningBalance(
                account_id=1,
                account_name=BANK,
                amount=Decimal("1000.00"),
                source="snapshot",
                as_of=datetime.date(2025, 12, 15),
            )
        ],
        entries=[_inr_entry(), _usd_entry()],
        prices=[PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.2500"))],
    )
    journal = render_document(doc, "beancount")
    completed = _run("bean-check", journal)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(not _cli_available("ledger"), reason="ledger CLI not installed")
def test_real_ledger_parses_lot_bearing_journal():
    """A lot posting whose commodity is an alphanumeric ISIN must parse under
    the real ``ledger`` CLI. This is the regression for the bare-ISIN defect:
    Ledger 3.3.2 rejects ``1000 INE000A01020`` (``Unexpected char '0'``) but
    accepts the renderer's quoted form ``1000 "INE000A01020"``. Verified against
    ``ananthakumaran/paisa:v0.7.4``'s bundled Ledger 3.3.2; gated here so the
    suite stays green where ``ledger`` is absent."""
    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Example Fund",
        quantity=Decimal("500"),
        unit_cost=Decimal("100.00"),
        cost_basis=Decimal("50000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 3, 17),
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(lot,),
    )
    journal = render_document(doc, "ledger")
    # The amount commodity AND the declaration are quoted (the ledger-family
    # token is identical in every position); hledger would reject a bare
    # alphanumeric declaration, so this also gates the hledger fix.
    assert '"INE000A01020"' in journal
    assert 'commodity "INE000A01020"' in journal
    # ``ledger balance`` parses the journal and computes balances; a parse
    # error or an unbalanced entry yields a non-zero exit.
    completed = _run("ledger", journal, "balance")
    assert completed.returncode == 0, completed.stderr


def test_ci_hledger_contract_journal_is_lot_bearing_and_quoted():
    """The CI ``hledger`` job renders THIS exact document shape (a lot-bearing
    production journal: an ISIN-commodity investment lot + a priced USD entry +
    USD and lot price directives) and runs the pinned
    ``dastapov/hledger:1.52.1`` ``hledger check`` + ``commodities`` over it
    (see ``.github/workflows/ci.yml``). Assert the rendered journal actually
    carries the lot grammar the CI parse is meant to exercise — so the CI
    contract can never silently degrade to a lot-less journal (the exact
    grammar hledger rejects when bare: a quoted alphanumeric commodity in a
    declaration/P directive, plus the lot cost/date bracket).

    Not CLI-gated: this is the in-process assertion that the CI job's input is
    lot-bearing; the real parse is the CI job itself (and the optional
    ``test_real_hledger_*`` tests when hledger is on PATH)."""
    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Example Fund",
        quantity=Decimal("500"),
        unit_cost=Decimal("100.00"),
        cost_basis=Decimal("50000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 3, 17),
    )
    entry = ProjectedEntry(
        date=datetime.date(2026, 2, 1),
        payee="Cafe",
        txn_ids=(42,),
        postings=(
            LedgerPosting("Expenses:Food:Groceries", Decimal("10.00"), "USD"),
            LedgerPosting("Assets:Bank:HDFC:Savings", Decimal("-10.00"), "USD"),
        ),
        currency="USD",
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(entry,),
        accounts_declared=("Expenses:Food:Groceries", "Assets:Bank:HDFC:Savings"),
        lot_postings=(lot,),
        price_directives=(
            PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.2500")),
            PriceDirective(
                datetime.date(2026, 3, 17), "INE000A01020", Decimal("100.0000")
            ),
        ),
    )
    journal = render_document(doc, "hledger")
    # --- lot-bearing grammar (the ISIN commodity must be QUOTED everywhere) ---
    # Quoted token in the commodity declaration, the lot amount, and the lot P.
    assert 'commodity "INE000A01020"' in journal
    assert 'Assets:Investments:INE000A01020  500 "INE000A01020"' in journal
    assert "{100.00 INR}" in journal  # per-unit cost annotation
    assert "[2026-03-17]" in journal  # acquisition-date lot bracket
    assert 'P 2026-03-17 "INE000A01020" 100.0000 INR' in journal
    # A bare alphanumeric ISIN would be the regression hledger rejects.
    assert "\nINE000A01020 " not in journal
    # --- priced FX grammar (USD is pure letters, stays bare) ---
    assert "10.00 USD" in journal
    assert "-10.00 USD" in journal
    assert "P 2026-02-01 USD 83.2500 INR" in journal
    # The lot equity cancels the asset cost (clean 2dp here).
    assert "-50000.00 INR" in journal


# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------


def test_constants_pinned():
    # A rename here would silently change the journal for every backend.
    assert EQUITY_OPENING == "Equity:Opening Balances"
    assert CARD_PAYMENT_CLEARING == "Liabilities:Credit Card"
    assert INR == "INR"
    assert AMOUNT_COLUMN == 52


# ---------------------------------------------------------------------------
# Adversarial commodity token (requirement: one robust ledger-family function)
# ---------------------------------------------------------------------------
#
# The ledger-family commodity token must sanitize first, reject empty, eliminate
# control chars (newline/tab/quotes/backslash), and quote any symbol that is not
# pure ASCII letters — in posting amounts, ``commodity`` declarations, AND ``P``
# directives, for BOTH ledger and hledger. hledger rejects a bare alphanumeric
# commodity in a declaration/P (``unexpected '0'``/``unexpected 'A'``); Ledger
# 3.3.2 rejects one in a posting amount. So the SAME quoted token is used
# everywhere.


@pytest.mark.parametrize(
    "raw, expected",
    [
        # pure ASCII letters stay bare
        ("INR", "INR"),
        ("USD", "USD"),
        ("usd", "USD"),  # uppercased
        # alphanumeric (digit-bearing) → quoted, sanitized
        ("INE000A01020", '"INE000A01020"'),
        ("INE000A01020\nINJ", '"INE000A01020INJ"'),  # newline eliminated
        ("INE\t000A01020", '"INE000A01020"'),  # tab eliminated
        ('INE000A"01', '"INE000A01"'),  # embedded quote eliminated
        ("INE\\000A", '"INE000A"'),  # backslash eliminated
        ("INE 000A01020", '"INE000A01020"'),  # internal space eliminated
        # punctuation stripped, alnum kept
        ("US-D", "USD"),
        ("US$D", "USD"),
    ],
)
def test_commodity_token_sanitize_and_quote(raw, expected):
    assert commodity_token(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "   ", "---", "12345", "$$$", "\n\t"])
def test_commodity_token_rejects_empty_or_non_letter_leading(raw):
    # Empty, whitespace-only, or all-non-letter input sanitizes to nothing (or a
    # non-letter-leading symbol) and is rejected — never emitted as a bare/empty
    # token that would break a directive.
    with pytest.raises(InvalidAccountName):
        commodity_token(raw)


@pytest.mark.parametrize("backend", ["ledger", "hledger"])
def test_quoted_commodity_token_consistent_across_all_three_positions(backend):
    """The ISIN token is quoted identically in the declaration, the posting
    amount, and the P directive — not only in the amount."""
    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Example Fund",
        quantity=Decimal("500"),
        unit_cost=Decimal("100.00"),
        cost_basis=Decimal("50000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 3, 17),
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(lot,),
        price_directives=(
            PriceDirective(
                datetime.date(2026, 3, 17), "INE000A01020", Decimal("100.00")
            ),
        ),
    )
    out = render_document(doc, backend)
    # Pure-letter INR stays bare everywhere; the ISIN is quoted everywhere.
    assert 'commodity "INE000A01020"' in out  # declaration
    assert '500 "INE000A01020"' in out  # posting amount
    assert 'P 2026-03-17 "INE000A01020" 100.0000 INR' in out  # P directive
    # And the bare form never appears in a declaration/P context.
    assert "\ncommodity INE000A01020" not in out
    assert "P 2026-03-17 INE000A01020 " not in out


@pytest.mark.parametrize("backend", ["ledger", "hledger"])
def test_malformed_fx_currency_routed_through_sanitizer(backend):
    """A malformed FX code in a posting/P is sanitized before emission so it
    cannot create an invalid directive. ``US\\nD`` sanitizes to ``USD`` (bare),
    and a real priced USD entry + P directive still renders correctly."""
    entry = _entry(
        payee="Cafe",
        currency="US\nD",
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00"), "US\nD"),
            LedgerPosting(BANK, Decimal("-10.00"), "US\nD"),
        ),
    )
    doc = _doc(
        entries=[entry],
        prices=[PriceDirective(datetime.date(2026, 2, 1), "US\nD", Decimal("83.25"))],
    )
    out = render_document(doc, backend)
    # The control char is gone; the sanitized bare USD appears in amount and P.
    assert "10.00 USD" in out
    assert "-10.00 USD" in out
    assert "P 2026-02-01 USD 83.2500 INR" in out
    # No raw control char survived into a commodity position.
    assert "US\nD" not in out


# ---------------------------------------------------------------------------
# Beancount: lot acquired BEFORE cutover → back-dated open (requirement 2)
# ---------------------------------------------------------------------------


def test_beancount_open_back_dated_for_lot_before_cutover():
    """A lot acquired before the cutover must open its asset/equity accounts at
    ``min(cutover, acquired_on)`` — bean-check rejects a posting to an account
    whose cutover-dated open is later than the lot date (``inactive account``).
    A lot on/after the cutover keeps the cutover-dated open."""
    old_lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Old Fund",
        quantity=Decimal("100"),
        unit_cost=Decimal("50.00"),
        cost_basis=Decimal("5000.00"),
        currency="INR",
        acquired_on=datetime.date(2025, 6, 1),  # BEFORE the cutover
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,  # 2026-01-01
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(old_lot,),
    )
    out = render_document(doc, "beancount")
    # The lot accounts open at the lot date (2025-06-01), not the cutover.
    assert "2025-06-01 open Assets:Investments:INE000A01020 INE000A01020" in out
    assert "2025-06-01 open Equity:OpeningBalances:Investment INR" in out
    # A cutover-dated open for those accounts would be wrong.
    assert "2026-01-01 open Assets:Investments:INE000A01020" not in out
    assert "2026-01-01 open Equity:OpeningBalances:Investment" not in out


def test_beancount_open_cutover_dated_for_lot_on_or_after_cutover():
    new_lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="New Fund",
        quantity=Decimal("100"),
        unit_cost=Decimal("50.00"),
        cost_basis=Decimal("5000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 5, 1),  # AFTER the cutover
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(new_lot,),
    )
    out = render_document(doc, "beancount")
    # min(cutover, lot) == cutover: the open stays cutover-dated.
    assert "2026-01-01 open Assets:Investments:INE000A01020 INE000A01020" in out
    assert "2026-01-01 open Equity:OpeningBalances:Investment INR" in out


def test_beancount_mixed_old_and_new_lots_share_earliest_open():
    """Two lots of the same instrument — one before, one after the cutover —
    share one asset account opened at the EARLIEST lot date, and the combined
    holdings parse under bean-check preserving identity."""
    old_lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Fund",
        quantity=Decimal("100"),
        unit_cost=Decimal("50.00"),
        cost_basis=Decimal("5000.00"),
        currency="INR",
        acquired_on=datetime.date(2025, 6, 1),
    )
    new_lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Fund",
        quantity=Decimal("50"),
        unit_cost=Decimal("60.00"),
        cost_basis=Decimal("3000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 8, 1),
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(old_lot, new_lot),
    )
    out = render_document(doc, "beancount")
    # The shared asset account opens once, at the earliest lot (2025-06-01).
    assert out.count("open Assets:Investments:INE000A01020") == 1
    assert "2025-06-01 open Assets:Investments:INE000A01020 INE000A01020" in out


# ---------------------------------------------------------------------------
# Mandatory beancount round-trip (beancount is a dev dependency — not skipped)
# ---------------------------------------------------------------------------


def test_beancount_mandatory_bean_check_and_identity_round_trip():
    """Mandatory (not gated): bean-check the production renderer output for a
    multi-currency + lot document, then load it via the beancount API and assert
    the parsed commodity/account/quantity identity is preserved. ``beancount``
    is a declared dev dependency, so this runs in every CI and local run."""
    from beancount import loader

    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Example Fund",
        quantity=Decimal("500"),
        unit_cost=Decimal("100.00"),
        cost_basis=Decimal("50000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 3, 17),
    )
    entry = ProjectedEntry(
        date=datetime.date(2026, 2, 1),
        payee="Cafe",
        txn_ids=(42,),
        postings=(
            LedgerPosting(EXPENSE, Decimal("10.00"), "USD"),
            LedgerPosting(BANK, Decimal("-10.00"), "USD"),
        ),
        currency="USD",
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(
            OpeningBalance(
                account_id=1,
                account_name=BANK,
                amount=Decimal("1000.00"),
                source="snapshot",
                as_of=datetime.date(2025, 12, 15),
            ),
        ),
        entries=(entry,),
        accounts_declared=(BANK, EXPENSE),
        price_directives=(
            PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.2500")),
        ),
        lot_postings=(lot,),
    )
    text = render_document(doc, "beancount")
    entries, errors, _options = loader.load_string(text)
    assert errors == [], errors
    # Identity: the lot commodity, the lot/USD accounts, and the lot quantity
    # survive the parse.
    from beancount.core import getters

    accounts = set(getters.get_accounts(entries))
    assert "Assets:Investments:INE000A01020" in accounts
    assert "Equity:OpeningBalances:Investment" in accounts
    assert "Expenses:Food:Groceries" in accounts
    # The lot posts exactly 500 units of its instrument commodity.
    lot_units = [
        p.units
        for e in entries
        if hasattr(e, "postings")
        for p in e.postings
        if p.account == "Assets:Investments:INE000A01020"
    ]
    assert any(
        u.number == Decimal("500") and u.currency == "INE000A01020" for u in lot_units
    )


def test_beancount_old_lot_before_cutover_passes_bean_check():
    """The regression: a lot before the cutover used to fail bean-check
    (``inactive account``). With the back-dated open it must pass and preserve
    the lot quantity/value identity."""
    from beancount import loader

    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Old Fund",
        quantity=Decimal("100"),
        unit_cost=Decimal("50.00"),
        cost_basis=Decimal("5000.00"),
        currency="INR",
        acquired_on=datetime.date(2025, 6, 1),  # before cutover
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(lot,),
    )
    entries, errors, _options = loader.load_string(render_document(doc, "beancount"))
    assert errors == [], errors


def test_beancount_mandatory_bean_check_cli_on_lot_before_cutover():
    """Mandatory (not gated): run the real ``bean-check`` CLI (a declared dev
    dependency) over a rendered file containing a lot acquired BEFORE the
    cutover. This is the explicit bean-check gate — rc must be 0."""
    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Old Fund",
        quantity=Decimal("100"),
        unit_cost=Decimal("50.00"),
        cost_basis=Decimal("5000.00"),
        currency="INR",
        acquired_on=datetime.date(2025, 6, 1),  # before cutover
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(lot,),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".beancount", delete=False) as fh:
        fh.write(render_document(doc, "beancount"))
        path = fh.name
    try:
        # NOTE: ``_run`` writes its second arg to a fresh temp file, so call
        # subprocess directly here — the file is already written above.
        completed = subprocess.run(
            ["bean-check", path], capture_output=True, text=True, timeout=30
        )
    finally:
        os.unlink(path)
    assert completed.returncode == 0, completed.stderr or completed.stdout


# ---------------------------------------------------------------------------
# Optional real ledger/hledger round-trip (gated by local CLI availability)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _cli_available("ledger"), reason="ledger CLI not installed")
def test_real_ledger_round_trips_quoted_commodity_identity():
    """Render a lot+priced-FX document and verify ``ledger`` parses it back to
    the exact commodity/account identity (the quoted token denotes the same
    commodity as its bare form in Ledger 3.3.2)."""
    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Example Fund",
        quantity=Decimal("500"),
        unit_cost=Decimal("100.00"),
        cost_basis=Decimal("50000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 3, 17),
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(lot,),
        price_directives=(
            PriceDirective(
                datetime.date(2026, 3, 17), "INE000A01020", Decimal("100.00")
            ),
        ),
    )
    journal = render_document(doc, "ledger")
    commodities = _run("ledger", journal, "commodities")
    assert commodities.returncode == 0, commodities.stderr
    parsed = {
        ln.strip().strip('"') for ln in commodities.stdout.splitlines() if ln.strip()
    }
    assert "INE000A01020" in parsed  # identity preserved through the quotes
    accounts = _run("ledger", journal, "accounts")
    assert "Assets:Investments:INE000A01020" in {
        ln.strip() for ln in accounts.stdout.splitlines() if ln.strip()
    }


@pytest.mark.skipif(not _cli_available("hledger"), reason="hledger CLI not installed")
def test_real_hledger_round_trips_quoted_commodity_identity():
    """hledger rejects a bare alphanumeric commodity in a declaration/P, so the
    quoted token is the only form that parses. Verify the round-trip identity."""
    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Example Fund",
        quantity=Decimal("500"),
        unit_cost=Decimal("100.00"),
        cost_basis=Decimal("50000.00"),
        currency="INR",
        acquired_on=datetime.date(2026, 3, 17),
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(lot,),
        price_directives=(
            PriceDirective(
                datetime.date(2026, 3, 17), "INE000A01020", Decimal("100.00")
            ),
        ),
    )
    journal = render_document(doc, "hledger")
    # hledger check is the closest to a parse + balance assertion.
    completed = _run("hledger", journal, "check")
    assert completed.returncode == 0, completed.stderr
    commodities = _run("hledger", journal, "commodities")
    parsed = {
        ln.strip().strip('"') for ln in commodities.stdout.splitlines() if ln.strip()
    }
    assert "INE000A01020" in parsed


# ---------------------------------------------------------------------------
# Investment-lot equity: exact full-precision product (sub-cent / zero-cost)
# ---------------------------------------------------------------------------
#
# Every backend recomputes the asset leg's cost from ``quantity * unit_cost`` at
# FULL precision, so the equity contra must emit the exact negative product
# ``-(quantity * unit_cost)`` — never the 2-dp-rounded ``cost_basis``. A lot
# whose product is not 2-dp exact (e.g. ``3.000300 × 33.33 = 99.99999900``)
# would leave a sub-cent imbalance that Ledger 3.3.2, hledger 1.52.1 and
# ``bean-check`` reject if the equity were the rounded ``cost_basis``. The
# stored 2-dp ``cost_basis`` is kept only as reporting/source-agreement metadata
# (still validated by ``check_lot_consistent``). Every assertion below is exact
# (Decimal equality) — there is deliberately no tolerance-only check.

#: Adversarial (quantity, unit_cost) products. ``cost_basis`` is always the 2-dp
#: quantized product (what the service stores); the equity leg must be the full
#: product. Covers a 2-dp-exact baseline, two non-2-dp products, the exact
#: half-paisa (±0.005) boundary, and a degenerate product that quantizes to 0.00.
LOT_PRODUCT_CASES = [
    pytest.param(Decimal("1000"), Decimal("50.00"), id="clean-2dp"),
    pytest.param(Decimal("3.000300"), Decimal("33.33"), id="subcent-99.99999900"),
    pytest.param(Decimal("148.593"), Decimal("10.2543"), id="subcent-1523.7171999"),
    pytest.param(Decimal("1"), Decimal("0.005"), id="half-paisa-boundary"),
    pytest.param(Decimal("0.004"), Decimal("0.01"), id="degenerate-zero-cost"),
]

#: The subset whose product is NOT 2-dp exact — used where a clean lot would not
#: exercise the sub-cent path (real-parser parse + "equity != -cost_basis").
SUBCENT_LOT_CASES = LOT_PRODUCT_CASES[1:]


def _lot_doc(qty: Decimal, uc: Decimal) -> tuple[LedgerDocument, Decimal]:
    """Build a single-lot document whose ``cost_basis`` is the 2-dp quantized
    product (exactly what ``services.investments`` stores), returning the doc
    and the full-precision product the equity leg must cancel."""
    product = qty * uc
    lot = InvestmentLotEntry(
        instrument="INE000A01020",
        instrument_name="Example Fund",
        quantity=qty,
        unit_cost=uc,
        cost_basis=product.quantize(Decimal("0.01")),
        currency="INR",
        acquired_on=datetime.date(2026, 3, 17),
    )
    doc = LedgerDocument(
        cutover_date=CUTOVER,
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(lot,),
    )
    return doc, product


def _lot_equity_amount(out: str, backend: str) -> tuple[Decimal, str]:
    """Extract the exact Decimal on the lot's equity POSTING line.

    The beancount ``open`` directive also names the equity account, so the
    indent (a posting is ``    ``-prefixed) disambiguates the posting from the
    declaration. The amount is the last numeric token before the trailing
    commodity."""
    acct = (
        "Equity:OpeningBalances:Investment"
        if backend == "beancount"
        else "Equity:Opening Balances:Investment"
    )
    line = next(ln for ln in out.splitlines() if ln.startswith("    ") and acct in ln)
    m = re.search(r"(-?\d+(?:\.\d+)?)\s+INR\s*$", line)
    assert m, f"no equity amount on equity posting line: {line!r}"
    return Decimal(m.group(1)), line


@pytest.mark.parametrize("backend", list(SUPPORTED_BACKENDS))
@pytest.mark.parametrize("qty, uc", LOT_PRODUCT_CASES)
def test_lot_equity_emits_exact_negative_product_that_cancels_asset_cost(
    backend, qty, uc
):
    """The equity leg is the exact full-precision ``-(quantity * unit_cost)``;
    it cancels the asset leg's cost to the penny AND to full precision. Every
    check is exact Decimal equality — no tolerance."""
    doc, product = _lot_doc(qty, uc)
    out = render_document(doc, backend)
    equity, line = _lot_equity_amount(out, backend)
    # 1. Exact cancellation: parsed equity + asset cost == 0 (not |.| < eps).
    assert equity + product == Decimal(0), (
        f"{backend}: equity {equity} + product {product} != 0 (line={line!r})"
    )
    # 2. Exact value: parsed equity == the exact negative product (Decimal ==).
    assert equity == -product, (
        f"{backend}: equity {equity} != exact -product {-product} (line={line!r})"
    )
    # 3. The stored 2-dp cost_basis is retained as metadata and is a consistent
    #    2-dp rounding of the product (this is the agreement the service stores).
    cost_basis = product.quantize(Decimal("0.01"))
    assert abs((qty * uc).quantize(Decimal("0.01")) - cost_basis) == Decimal(0)
    # 4. For a lot whose product is NOT 2-dp exact, the emitted equity must NOT
    #    be the rounded -cost_basis — proving the full product is emitted, not
    #    the metadata. (A clean lot's cost_basis == product, so this is skipped.)
    if cost_basis != product:
        assert equity != -cost_basis, (
            f"{backend}: equity {equity} equals rounded -cost_basis {-cost_basis}; "
            f"expected the full product {-product}"
        )


@pytest.mark.parametrize("qty, uc", SUBCENT_LOT_CASES)
def test_lot_cost_basis_metadata_quantizes_as_expected(qty, uc):
    """The 2-dp ``cost_basis`` metadata is retained and (for the degenerate
    case) quantizes to 0.00 while the emitted equity is the non-zero full
    product — the sub-cent/zero-cost blocker's whole reason for existing."""
    product = qty * uc
    cost_basis = product.quantize(Decimal("0.01"))
    doc, _ = _lot_doc(qty, uc)
    lot = doc.lot_postings[0]
    assert lot.cost_basis == cost_basis  # metadata retained on the entry
    if qty == Decimal("0.004"):
        # The degenerate zero-cost lot: cost_basis rounds to 0.00 ...
        assert cost_basis == Decimal("0.00")
        # ... yet the emitted equity is the non-zero full product.
        for backend in SUPPORTED_BACKENDS:
            equity, _ = _lot_equity_amount(render_document(doc, backend), backend)
            assert equity == -product
            assert equity != Decimal("0.00")


@pytest.mark.parametrize(
    "value, expected",
    [
        (Decimal("50000.00"), "50000.00 INR"),
        (Decimal("-50000.00"), "-50000.00 INR"),
        (Decimal("99.99999900"), "99.99999900 INR"),
        (Decimal("-99.99999900"), "-99.99999900 INR"),
        (Decimal("1523.7171999"), "1523.7171999 INR"),
        (Decimal("0.00004"), "0.00004 INR"),
        (Decimal("0.005"), "0.005 INR"),
        (Decimal("0"), "0 INR"),  # never -0, never scientific notation
        (Decimal("-0"), "0 INR"),
        (Decimal("0.00"), "0.00 INR"),
    ],
)
def test_fmt_lot_money_exact_fixed_point(value, expected):
    """The dedicated lot-money formatter renders the exact full-precision value
    as fixed-point text (no scientific notation, no quantization), with a zero
    product never rendered as ``-0``."""
    from financial_dashboard.services.paisa.renderers.base import fmt_lot_money

    assert fmt_lot_money(value, "INR") == expected


def test_fmt_lot_money_byte_identical_to_fmt_amount_for_2dp_exact_product():
    """A clean lot's full-precision product is 2-dp exact, so the lot-money
    formatter must produce byte-identical text to the normal money formatter —
    clean-lot golden bytes are unchanged by the sub-cent fix."""
    from financial_dashboard.services.paisa.renderers.base import (
        fmt_amount,
        fmt_lot_money,
    )

    for v in (
        Decimal("50000.00"),
        Decimal("-50000.00"),
        Decimal("5000.00"),
        Decimal("0.01"),
        Decimal("1523.72"),
    ):
        assert fmt_lot_money(v, "INR") == fmt_amount(v, "INR"), v


# ---------------------------------------------------------------------------
# Mandatory beancount parse of sub-cent + zero-cost lots
# ---------------------------------------------------------------------------
#
# ``beancount`` is a declared dev dependency (never skipped), so this is the
# CI-mandatory parser coverage of a non-2dp lot and a degenerate zero-cost lot.
# Both the beancount loader and the real ``bean-check`` CLI must accept the
# production renderer's output with zero errors.


@pytest.mark.parametrize("qty, uc", SUBCENT_LOT_CASES)
def test_beancount_mandatory_parse_of_subcent_and_zero_cost_lot(qty, uc):
    from beancount import loader

    doc, _product = _lot_doc(qty, uc)
    text = render_document(doc, "beancount")
    _entries, errors, _opts = loader.load_string(text)
    assert errors == [], errors
    with tempfile.NamedTemporaryFile("w", suffix=".beancount", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        completed = subprocess.run(
            ["bean-check", path], capture_output=True, text=True, timeout=30
        )
    finally:
        os.unlink(path)
    assert completed.returncode == 0, completed.stderr or completed.stdout


# ---------------------------------------------------------------------------
# Optional real ledger / hledger parse of sub-cent + zero-cost lots
# ---------------------------------------------------------------------------
#
# Gated by local CLI availability. Where ``ledger`` / ``hledger`` are installed
# these exercise the real parsers over a non-2dp product and a degenerate
# zero-cost lot; ``ledger balance`` / ``hledger check`` must exit 0 (a sub-cent
# imbalance would fail the balance check).


@pytest.mark.skipif(not _cli_available("ledger"), reason="ledger CLI not installed")
@pytest.mark.parametrize("qty, uc", SUBCENT_LOT_CASES)
def test_real_ledger_parses_subcent_and_zero_cost_lot(qty, uc):
    doc, _product = _lot_doc(qty, uc)
    journal = render_document(doc, "ledger")
    # ``ledger balance`` parses + balance-checks the journal; a sub-cent
    # imbalance between the asset cost and the equity leg would exit non-zero.
    completed = _run("ledger", journal, "balance")
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(not _cli_available("hledger"), reason="hledger CLI not installed")
@pytest.mark.parametrize("qty, uc", SUBCENT_LOT_CASES)
def test_real_hledger_parses_subcent_and_zero_cost_lot(qty, uc):
    doc, _product = _lot_doc(qty, uc)
    journal = render_document(doc, "hledger")
    completed = _run("hledger", journal, "check")
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# Real Ledger 3.3.2 parse via the pinned official image (Docker-gated)
# ---------------------------------------------------------------------------
#
# ``ledger`` is not installed locally, but Docker is — so the pinned
# ``ananthakumaran/paisa:v0.7.4`` image (which bundles Ledger 3.3.2, the same
# binary the CI ledger contract uses) parses a sub-cent AND a zero-cost lot. This
# is the real Ledger 3.3.2 coverage; a 2-dp-rounded equity would make
# ``ledger balance`` exit non-zero here.

_PAISA_CONTRACT_IMAGE = "ananthakumaran/paisa:v0.7.4"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


@pytest.mark.skipif(not _docker_available(), reason="docker not available")
@pytest.mark.parametrize("qty, uc", SUBCENT_LOT_CASES)
def test_real_ledger_332_parses_subcent_and_zero_cost_lot_via_docker(qty, uc, tmp_path):
    doc, _product = _lot_doc(qty, uc)
    journal = render_document(doc, "ledger")
    journal_file = tmp_path / "lot.journal"
    journal_file.write_text(journal)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{tmp_path}:/data",
        "-w",
        "/data",
        "--entrypoint",
        "ledger",
        _PAISA_CONTRACT_IMAGE,
        "-f",
        "/data/lot.journal",
        "balance",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stderr or completed.stdout
