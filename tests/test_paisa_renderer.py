"""Ledger-CLI renderer: determinism, syntax-safety sanitization, account-name
validation, and balance enforcement.

Adversarial cases prove a malicious/garbage counterparty, note or account
override cannot inject a ledger directive, start an early comment, or break a
posting onto a new line — and that an unbalanced entry raises rather than
emitting a corrupt file.
"""

import datetime
import re
from decimal import Decimal

import pytest

from financial_dashboard.services.paisa.renderer import (
    AMOUNT_COLUMN,
    CARD_PAYMENT_CLEARING,
    EQUITY_OPENING,
    InvalidAccountName,
    LedgerDocument,
    LedgerPosting,
    OpeningBalance,
    ProjectedEntry,
    UnbalancedEntry,
    _format_posting_line,
    render_document,
    validate_account_name,
)

pytestmark = pytest.mark.anyio


def _doc(entries=(), openings=(), accounts=(), cutover=datetime.date(2026, 1, 1)):
    return LedgerDocument(
        cutover_date=cutover,
        openings=tuple(openings),
        entries=tuple(entries),
        accounts_declared=tuple(accounts),
    )


def _entry(payee="Store", postings=(), date=datetime.date(2026, 2, 1), note=None):
    return ProjectedEntry(
        date=date,
        payee=payee,
        txn_ids=(42,),
        postings=tuple(postings),
        note=note,
    )


# ---------------------------------------------------------------------------
# Account-name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Assets:Bank",
        "Liabilities:Card:ICICI:Platinum",
        "Expenses:Food:Groceries",
        "Equity:Opening Balances",
    ],
)
def test_valid_account_names(name):
    assert validate_account_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",  # empty
        "   ",  # whitespace only
        "Assets",  # no hierarchy
        "Assets:",  # empty trailing segment
        ":Assets",  # empty leading segment
        "Assets::Bank",  # empty middle segment
        "Assets\n:Bank",  # newline
        "Assets\t:Bank",  # tab
        "Assets;Bank:X",  # comment char
        "Assets{Expr}:Bank",  # expression braces
    ],
)
def test_invalid_account_names(name):
    with pytest.raises(InvalidAccountName):
        validate_account_name(name)


def test_render_rejects_invalid_declared_account():
    doc = _doc(accounts=["Assets; Inject"])
    with pytest.raises(InvalidAccountName):
        render_document(doc)


# ---------------------------------------------------------------------------
# Payee / note injection
# ---------------------------------------------------------------------------


def test_payee_newline_collapsed():
    entry = _entry(
        payee="Evil\n2026-99-99 * forged entry\n    Assets:Fake  9999.00 INR",
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00")),
            LedgerPosting("Assets:Bank", Decimal("-10.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]))
    lines = out.splitlines()
    # The header is one line; the forged entry text did not start a new line
    # or a forged transaction — newlines were collapsed into the single payee.
    header_lines = [ln for ln in lines if "2026-02-01" in ln]
    assert len(header_lines) == 1
    assert "\n2026-99-99" not in out
    # Exactly the two real postings exist — no forged posting leaked onto its
    # own line. A posting line is an indented "<account> <amount> INR".
    posting_lines = [ln for ln in lines if ln.startswith("    ") and "INR" in ln]
    assert len(posting_lines) == 2
    assert all("Assets:Fake" not in ln for ln in posting_lines)
    assert all("9999.00" not in ln for ln in posting_lines)


def test_payee_semicolon_stripped():
    # A ';' would start an early comment on the date line, hiding the txn id.
    entry = _entry(
        payee="Store; EVIL",
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00")),
            LedgerPosting("Assets:Bank", Decimal("-10.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]))
    header = [ln for ln in out.splitlines() if "2026-02-01" in ln][0]
    # The txn comment is the only ';' on the header line.
    assert header.count(";") == 1
    assert "txn:42" in header


def test_note_newline_and_semicolon_sanitized():
    entry = _entry(
        payee="Store",
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00")),
            LedgerPosting("Assets:Bank", Decimal("-10.00")),
        ),
        note="line1\nline2; comment-injection",
    )
    out = render_document(_doc(entries=[entry]))
    note_lines = [ln for ln in out.splitlines() if "note:" in ln]
    assert len(note_lines) == 1  # collapsed to a single line
    assert "comment-injection" in note_lines[0]  # ';' removed, kept as text


# ---------------------------------------------------------------------------
# Balance enforcement
# ---------------------------------------------------------------------------


def test_unbalanced_entry_raises():
    entry = _entry(
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00")),
            LedgerPosting("Assets:Bank", Decimal("-9.00")),  # off by one
        ),
    )
    with pytest.raises(UnbalancedEntry):
        render_document(_doc(entries=[entry]))


def test_balanced_entry_renders():
    entry = _entry(
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00")),
            LedgerPosting("Assets:Bank", Decimal("-10.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]))
    assert "10.00 INR" in out
    assert "-10.00 INR" in out


def test_entry_with_no_postings_raises():
    entry = _entry(postings=())
    with pytest.raises(UnbalancedEntry):
        render_document(_doc(entries=[entry]))


# ---------------------------------------------------------------------------
# Opening balances + determinism
# ---------------------------------------------------------------------------


def test_openings_balance_via_equity():
    openings = (
        OpeningBalance(
            account_id=1,
            account_name="Assets:Bank:HDFC",
            amount=Decimal("100000.00"),
            source="snapshot",
            as_of=datetime.date(2025, 12, 15),
        ),
        OpeningBalance(
            account_id=2,
            account_name="Liabilities:Card:ICICI",
            amount=Decimal("-12000.00"),
            source="snapshot",
            as_of=datetime.date(2025, 12, 15),
        ),
    )
    out = render_document(_doc(openings=openings))
    assert EQUITY_OPENING in out
    # Sum of openings is 88000; equity takes -88000 so the entry balances.
    assert "-88000.00 INR" in out


def test_zero_openings_not_emitted():
    openings = (
        OpeningBalance(
            account_id=1,
            account_name="Assets:Bank:HDFC",
            amount=Decimal("0.00"),
            source="snapshot",
            as_of=datetime.date(2025, 12, 15),
        ),
    )
    out = render_document(_doc(openings=openings))
    assert "Opening Balances" not in out


def test_render_is_deterministic():
    entry = _entry(
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00")),
            LedgerPosting("Assets:Bank", Decimal("-10.00")),
        ),
    )
    doc = _doc(entries=[entry])
    assert render_document(doc) == render_document(doc)


def test_amount_column_matches_paisa_default():
    # Paisa's default amount_alignment_column is 52; our column must match so
    # generated entries line up with hand-edited ones.
    assert AMOUNT_COLUMN == 52


def test_card_payment_clearing_and_equity_constants():
    # Pinned constants — a rename here would silently change the journal.
    assert CARD_PAYMENT_CLEARING == "Liabilities:Credit Card"
    assert EQUITY_OPENING == "Equity:Opening Balances"


# ---------------------------------------------------------------------------
# Long / spaced account names: ledger amount-separation contract
# ---------------------------------------------------------------------------
#
# Ledger's parser requires the account name and the amount on a posting line
# to be separated by at least two spaces (or a tab). With only one space,
# ledger cannot tell where the name ends and absorbs the amount into the
# account name — silently producing a bogus account like
# ``Assets:...:Name 10.00 INR`` instead of posting to ``Assets:...:Name``.
# These cases pin the renderer's guarantee that a long/spaced name overruns
# the alignment column with a two-space gap, never a single space.


@pytest.mark.parametrize(
    "name",
    [
        # 52 chars — overruns the 48-char name budget (column 52 - 4 indent).
        "Liabilities:Card:HDFC Millennia Credit Card Platinum",
        # 57 chars — well past the column.
        "Assets:Bank:HDFC:Salary Plus Premium Savings Account Tier",
        # 51 chars with internal spaces — the exact name must survive intact.
        "Expenses:Insurance:Medical Health Insurance Premium",
    ],
)
def test_long_account_name_retained_and_two_space_separated(name):
    assert len(name) >= 50
    entry = _entry(
        postings=(
            LedgerPosting(name, Decimal("10.00")),
            LedgerPosting("Assets:Bank", Decimal("-10.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]))
    line = next(ln for ln in out.splitlines() if name in ln)

    # 1. The exact account name appears verbatim — not truncated, and not
    #    merged with the amount (which a single-space gap would cause).
    assert name in line
    # 2. >=2 spaces separate the name from the amount so ledger can split them.
    tail = line.split(name, 1)[1]
    assert re.match(r"  +\d", tail), (
        f"expected >=2 spaces before the amount in {line!r}"
    )
    # 3. The amount is still emitted on the same line.
    assert "10.00 INR" in tail


def test_posting_line_never_uses_single_space_gap():
    # Directly exercise the boundary: a name exactly one char too long for the
    # alignment column (len == target - 1 == 47) is where the old renderer
    # fell back to a single space. The fix must keep two spaces here.
    prefix_len = 4  # _POSTING_INDENT
    boundary_len = (AMOUNT_COLUMN - prefix_len) - 1  # 47
    # A valid ':'-joined hierarchy that is exactly boundary_len chars long.
    name = ("Expenses:Insurance:" + "A" * 28)[:boundary_len]
    assert len(name) == boundary_len
    validate_account_name(name)  # does not raise

    line = _format_posting_line(name, Decimal("5.00"), indent=True)
    # Strip the indent then find the name; what follows must be >=2 spaces.
    body = line[len("    ") :]
    assert body.startswith(name)
    gap_and_amount = body[len(name) :]
    assert re.match(r"  +5\.00 INR", gap_and_amount), repr(line)


def test_overflow_posting_parses_back_to_exact_account_name():
    # Ledger's own split rule: a posting line is "<indent?><account>  <amount>"
    # where the separator is >=2 spaces or a tab. Simulate that parse and
    # confirm the long account name round-trips exactly.
    name = "Liabilities:Card:HDFC Millennia Credit Card Platinum"
    entry = _entry(
        postings=(
            LedgerPosting(name, Decimal("9999.00")),
            LedgerPosting("Assets:Bank", Decimal("-9999.00")),
        ),
    )
    out = render_document(_doc(entries=[entry]))
    line = next(ln for ln in out.splitlines() if name in ln)
    # Mimic ledger: split on a run of >=2 spaces (after stripping indent).
    body = line.lstrip()
    parts = re.split(r" {2,}", body, maxsplit=1)
    assert len(parts) == 2, f"line did not split into account + amount: {line!r}"
    assert parts[0] == name, (
        f"account name absorbed/mangled: got {parts[0]!r}, want {name!r}"
    )
    assert parts[1] == "9999.00 INR"


# ---------------------------------------------------------------------------
# Multi-currency (ledger facade default): explicit commodity + price directive
# ---------------------------------------------------------------------------


def test_foreign_commodity_rendered_explicitly():
    entry = _entry(
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00"), commodity="USD"),
            LedgerPosting("Assets:Bank", Decimal("-10.00"), commodity="USD"),
        )
    )
    out = render_document(_doc(entries=[entry]))
    # The foreign amount is labelled USD, never silently relabelled INR.
    assert "10.00 USD" in out
    assert "-10.00 USD" in out
    assert "10.00 INR" not in out


def test_price_directive_emitted_when_present():
    from financial_dashboard.services.paisa.renderer import PriceDirective

    entry = _entry(
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00"), commodity="USD"),
            LedgerPosting("Assets:Bank", Decimal("-10.00"), commodity="USD"),
        )
    )
    doc = LedgerDocument(
        cutover_date=datetime.date(2026, 1, 1),
        openings=(),
        entries=(entry,),
        accounts_declared=("Assets:Bank",),
        price_directives=(
            PriceDirective(datetime.date(2026, 2, 1), "USD", Decimal("83.25")),
        ),
    )
    out = render_document(doc)
    assert "P 2026-02-01 USD 83.2500 INR" in out


def test_price_directive_absent_when_not_configured():
    entry = _entry(
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00"), commodity="USD"),
            LedgerPosting("Assets:Bank", Decimal("-10.00"), commodity="USD"),
        )
    )
    out = render_document(_doc(entries=[entry]))
    # No P directive is fabricated when no rate backs the foreign entry.
    assert "\nP " not in out
    assert not any(ln.startswith("P ") for ln in out.splitlines())


def test_foreign_entry_must_balance_in_its_commodity():
    # A USD leg paired with an INR leg cannot balance in either currency; the
    # renderer refuses rather than emit a backend-invalid transaction.
    entry = _entry(
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00"), commodity="USD"),
            LedgerPosting("Assets:Bank", Decimal("-10.00"), commodity="INR"),
        )
    )
    with pytest.raises(UnbalancedEntry):
        render_document(_doc(entries=[entry]))


def test_render_document_for_backend_dispatches():
    from financial_dashboard.services.paisa.renderer import render_document_for_backend

    entry = _entry(
        postings=(
            LedgerPosting("Expenses:Food", Decimal("10.00")),
            LedgerPosting("Assets:Bank", Decimal("-10.00")),
        )
    )
    doc = _doc(entries=[entry], accounts=["Assets:Bank"])
    # Ledger and beancount produce structurally different output for the same
    # document; the dispatch must select the right strategy.
    assert "account Assets:Bank" in render_document_for_backend(doc, "ledger")
    assert "open Assets:Bank" in render_document_for_backend(doc, "beancount")
