"""Manifest tamper detection, ledger structural invariants, and golden corpus
byte-stability tests."""

import json
import re
from pathlib import Path

import pytest

from scripts.synth import build_corpus, build_manifest, build_scenario, verify_manifest
from scripts.synth.manifest import TamperError, sha256_bytes
from scripts.synth.paisa import (
    ARTEFACT_NAMES,
    LedgerPosting,
    UnbalancedEntry,
    _entry_from_transaction,
    _posting_line,
    check_balanced,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "paisa"


# ---------------------------------------------------------------------------
# Manifest tamper detection
# ---------------------------------------------------------------------------


def _write_corpus(out_dir: Path, scenario) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(scenario)
    for name, data in corpus.artefacts.items():
        (out_dir / name).write_bytes(data)
    manifest = build_manifest(
        scenario_counts=scenario.counts(),
        invariants={"journal_balanced": True},
        artefacts=corpus.artefacts,
        seed=scenario.seed,
        as_of=scenario.as_of,
        profile=scenario.profile,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_verify_passes_on_fresh_generation(tmp_path):
    scenario = build_scenario(profile="golden")
    _write_corpus(tmp_path, scenario)
    verify_manifest(tmp_path)  # does not raise


def test_verify_detects_artefact_byte_tamper(tmp_path):
    scenario = build_scenario(profile="golden")
    _write_corpus(tmp_path, scenario)
    ledger = tmp_path / "dashboard-generated.ledger"
    ledger.write_bytes(ledger.read_bytes() + b"\n; tampered\n")
    with pytest.raises(TamperError, match="checksum mismatch"):
        verify_manifest(tmp_path)


def test_verify_detects_missing_artefact(tmp_path):
    scenario = build_scenario(profile="golden")
    _write_corpus(tmp_path, scenario)
    (tmp_path / "paisa.yaml").unlink()
    with pytest.raises(TamperError, match="artefact missing"):
        verify_manifest(tmp_path)


def test_verify_detects_db_count_drift(tmp_path):
    scenario = build_scenario(profile="golden")
    _write_corpus(tmp_path, scenario)
    db_counts = {
        **scenario.counts(),
        "transactions": scenario.counts()["transactions"] + 1,
    }
    with pytest.raises(TamperError, match="transactions"):
        verify_manifest(tmp_path, db_counts=db_counts)


def test_verify_detects_generator_version_mismatch(tmp_path):
    scenario = build_scenario(profile="golden")
    _write_corpus(tmp_path, scenario)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generator_version"] = "0.0.0"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(TamperError, match="generator_version"):
        verify_manifest(tmp_path)


# ---------------------------------------------------------------------------
# Balanced journal structural invariants
# ---------------------------------------------------------------------------


def test_check_balanced_rejects_unbalanced_postings():
    from datetime import date as _date

    with pytest.raises(UnbalancedEntry):
        check_balanced(
            (
                LedgerPosting("Expenses:X", __import__("decimal").Decimal("10.00")),
                LedgerPosting("Assets:Y", __import__("decimal").Decimal("-9.00")),
            ),
            _date(2026, 1, 1),
            "x",
        )


def test_every_generated_entry_balances_to_zero():
    scenario = build_scenario(profile="ci")
    eligible = [
        t
        for t in scenario.transactions
        if t.ledger_account and t.ledger_counterpart and t.currency == "INR"
    ]
    assert len(eligible) > 100
    for t in eligible:
        entry = _entry_from_transaction(t)
        # _entry_from_transaction raises on imbalance, so reaching here is the
        # assertion — but re-check explicitly for clarity.
        total = sum(
            (p.amount for p in entry.postings), __import__("decimal").Decimal("0.00")
        )
        assert total == 0


def test_no_payee_can_inject_a_directive():
    """A crafted counterparty must never start a new dated entry or an account
    directive — it has to stay inside the dated header line it belongs to."""
    from decimal import Decimal

    from scripts.synth.models import SynthTransaction

    scenario = build_scenario(profile="smoke")
    # Splice a malicious counterparty into a transaction and re-project it.
    evil = SynthTransaction(
        **{
            **scenario.transactions[0]._asdict(),
            "counterparty": "Evil\n2020-01-01 * Hijack\n    Assets:X    1.00 INR",
            "ledger_account": "Expenses:Misc",
            "ledger_counterpart": "Assets:Bank:Foo",
            "currency": "INR",
            "direction": "debit",
            "amount": Decimal("10.00"),
        }
    )
    entry = _entry_from_transaction(evil)
    # The evil payload collapses to spaces; the rendered header is one line.
    assert len(entry.payee.splitlines()) == 1
    rendered = (
        build_corpus(scenario).artefacts["dashboard-generated.ledger"].decode("utf-8")
    )
    assert "Hijack" not in rendered


def test_ledger_postings_have_aligned_amounts():
    scenario = build_scenario(profile="golden")
    corpus = build_corpus(scenario)
    generated = corpus.artefacts["dashboard-generated.ledger"].decode("utf-8")
    for line in generated.splitlines():
        if "INR" in line and line.startswith("    "):
            # Amounts land in column 52 (the renderer's fixed alignment).
            idx = line.index("INR")
            assert idx >= 40


def test_ledger_postings_always_two_space_separated():
    # Ledger requires >=2 spaces (or a tab) between account name and amount.
    # A single space would let the amount be absorbed into the account name;
    # this guard catches a regression for short AND long names alike.
    scenario = build_scenario(profile="smoke")
    corpus = build_corpus(scenario)
    generated = corpus.artefacts["dashboard-generated.ledger"].decode("utf-8")
    # A standard posting ends in ``<amount> INR``; the lot cost-annotation line
    # ends in ``[date]`` so it is excluded from this contract (the lot syntax is
    # pinned separately by the renderer/contract tests).
    amount_line = re.compile(r"^\s+\S.*\s-?\d+\.\d{2}\s+INR$")
    posting_lines = [
        ln
        for ln in generated.splitlines()
        if ln.startswith("    ") and amount_line.match(ln)
    ]
    assert posting_lines, "expected at least one standard posting line"
    for line in posting_lines:
        body = line.lstrip()
        parts = re.split(r" {2,}", body, maxsplit=1)
        assert len(parts) == 2, (
            f"posting not split by >=2 spaces into account+amount: {line!r}"
        )
        assert parts[1].endswith("INR")


def test_synth_posting_line_keeps_two_space_gap_on_overflow():
    # A name that overruns the alignment column must still leave >=2 spaces
    # before the amount (the bug this fixes: the old fallback used one space).
    name = "Expenses:Insurance:Medical Health Insurance Premium"
    assert len(name) >= 47
    line = _posting_line(name, __import__("decimal").Decimal("10.00"))
    body = line.lstrip()
    assert body.startswith(name)
    parts = re.split(r" {2,}", body, maxsplit=1)
    assert parts[0] == name, f"account name mangled: {parts[0]!r}"
    assert parts[1] == "10.00 INR"
    # The exact name (with its spaces) survives into the rendered line.
    assert name in line


# ---------------------------------------------------------------------------
# Golden corpus (committed, hand-reviewable) byte-stability
# ---------------------------------------------------------------------------


def test_golden_corpus_files_present():
    assert GOLDEN_DIR.is_dir()
    for name in ARTEFACT_NAMES:
        assert (GOLDEN_DIR / name).exists(), name
    assert (GOLDEN_DIR / "manifest.json").exists()


def test_golden_corpus_is_byte_identical_to_generator_output():
    scenario = build_scenario(seed=4242, profile="golden")
    corpus = build_corpus(scenario)
    for name in ARTEFACT_NAMES:
        committed = (GOLDEN_DIR / name).read_bytes()
        assert committed == corpus.artefacts[name], f"golden drift in {name}"


@pytest.mark.parametrize("profile", ["golden", "smoke"])
def test_corpus_contains_realistic_long_spaced_account(profile):
    # Coverage requirement: golden and smoke each carry at least one realistic
    # account whose rendered path is >=47 chars (the boundary at which a name
    # overruns the alignment column) and which contains internal spaces. The
    # spaces must survive verbatim into the journal — the renderer must not
    # strip them merely to shorten the name.
    scenario = build_scenario(profile=profile)
    corpus = build_corpus(scenario)
    generated = corpus.artefacts["dashboard-generated.ledger"].decode("utf-8")
    user = corpus.artefacts["user-authored.ledger"].decode("utf-8")

    # The long insurance-premium account is declared (account directive) and
    # posted (posting line) with its spaces intact.
    long_name = "Expenses:Insurance:Medical Health Insurance Premium"
    assert len(long_name) >= 47
    assert " " in long_name

    assert f"account {long_name}" in user, (
        f"{profile}: long spaced account not declared in user-authored.ledger"
    )
    posting_lines = [
        ln for ln in generated.splitlines() if ln.startswith("    ") and long_name in ln
    ]
    assert posting_lines, (
        f"{profile}: long spaced account never posted in dashboard-generated.ledger"
    )
    # The exact name (spaces included) is present on the posting line, and the
    # amount is separated from it by >=2 spaces so ledger can split them.
    for line in posting_lines:
        assert long_name in line
        parts = re.split(r" {2,}", line.lstrip(), maxsplit=1)
        assert parts[0] == long_name, (
            f"{profile}: account name mangled in posting: {parts[0]!r}"
        )


def test_golden_manifest_matches_committed_artefacts():
    scenario = build_scenario(seed=4242, profile="golden")
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    for name, meta in manifest["artefacts"].items():
        committed = (GOLDEN_DIR / name).read_bytes()
        assert meta["sha256"] == sha256_bytes(committed)
    assert manifest["expected"] == scenario.counts()


def test_golden_corpus_hand_reviewable_size():
    # The whole committed corpus stays small enough to read by hand. The cap
    # accommodates the full category-vocabulary + named-family coverage the
    # golden scenario seeds (refund, reversal, unmatched self-transfer,
    # card-side payment, one row per seed category) plus the 1.3.0 expansion
    # (distinct refund/cashback/fee-reversal/CC-refund/investment-redemption
    # edges, FX pairs, statement/CAS breadth).
    total = sum((GOLDEN_DIR / name).stat().st_size for name in ARTEFACT_NAMES)
    total += (GOLDEN_DIR / "manifest.json").stat().st_size
    assert total < 24_000


@pytest.mark.parametrize(
    "name",
    sorted(ARTEFACT_NAMES),
)
def test_artefact_is_utf8_decodable(name):
    scenario = build_scenario(profile="smoke")
    corpus = build_corpus(scenario)
    # Must decode cleanly (no stray binary) so a diff tool can review it.
    corpus.artefacts[name].decode("utf-8")


# ---------------------------------------------------------------------------
# Strict equivalence: the synth mirror's feature block == production renderer
# ---------------------------------------------------------------------------
#
# ``scripts.synth.paisa`` is intentionally a pure, hand-reviewable reimplementation
# (no production imports). To keep it from silently drifting from the production
# renderer, the lot/commodity/P feature block it emits must be byte-identical to
# the production ``render_document`` output for the same lot. The token helper
# itself is pinned first, then the full lot block + commodity declaration + P.


def test_synth_commodity_token_equals_production():
    from scripts.synth.paisa import _commodity_token

    from financial_dashboard.services.paisa.renderers.base import commodity_token

    cases = [
        "INR",
        "USD",
        "INE000A01020",
        "INE000A01020\nX",
        "US-D",
        "INE 000A01020",
    ]
    for raw in cases:
        assert _commodity_token(raw) == commodity_token(raw), raw


def test_synth_lot_block_byte_identical_to_production_renderer():
    """The mirror's ``_lot_entry_lines`` (+ commodity declaration + P directive)
    must render byte-for-byte like the production ledger-family renderer for the
    same lot — the strict equivalence that keeps the pure mirror honest."""
    import datetime
    from decimal import Decimal

    from financial_dashboard.services.paisa.renderers import render_document
    from financial_dashboard.services.paisa.renderers.base import (
        InvestmentLotEntry,
        LedgerDocument,
        PriceDirective,
    )
    from scripts.synth.paisa import LotFact, _commodity_token, _lot_entry_lines

    instrument = "INE000A01020"
    name = "Synthetic Liquid Fund"
    qty = Decimal("500")
    nav = Decimal("100.00")
    amount = Decimal("50000.00")
    acquired = datetime.date(2026, 3, 17)

    # Production renderer: a doc with just this lot + its declaration + P.
    prod_doc = LedgerDocument(
        cutover_date=datetime.date(2026, 1, 1),
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(
            InvestmentLotEntry(
                instrument=instrument,
                instrument_name=name,
                quantity=qty,
                unit_cost=nav,
                cost_basis=amount,
                currency="INR",
                acquired_on=acquired,
            ),
        ),
        price_directives=(PriceDirective(acquired, instrument, nav, "INR"),),
    )
    prod_text = render_document(prod_doc, "ledger")

    # Mirror: assemble the same feature block with the pure helpers.
    lot = LotFact(
        symbol=instrument,
        instrument_name=name,
        units=qty,
        nav=nav,
        amount=amount,
        acquired_on=acquired,
        asset_account=f"Assets:Investments:{instrument}",
    )
    mirror_lines = (
        [f"commodity {_commodity_token(instrument)}", ""]
        + _lot_entry_lines(lot)
        + [f"P {acquired.isoformat()} {_commodity_token(instrument)} 100.0000 INR"]
    )
    mirror_text = "\n".join(mirror_lines) + "\n"

    assert mirror_text == prod_text, (
        f"synth mirror drifted from production renderer:\n"
        f"--- mirror ---\n{mirror_text}\n--- production ---\n{prod_text}"
    )


# Adversarial (quantity, unit_cost) products. The mirror's lot equity must emit
# the FULL-PRECISION negative product ``-(units * nav)`` (via ``_fmt_lot_money``),
# exactly like the production renderer — never the 2-dp ``amount``/``cost_basis``.
# Covers a 2-dp-exact baseline, two non-2-dp products, the exact half-paisa
# boundary, and a degenerate product that quantizes to 0.00.
import pytest  # noqa: E402

from decimal import Decimal as _Decimal  # noqa: E402


@pytest.mark.parametrize(
    "units, nav",
    [
        pytest.param(_Decimal("500"), _Decimal("100.00"), id="clean-2dp"),
        pytest.param(_Decimal("3.000300"), _Decimal("33.33"), id="subcent-99.99999900"),
        pytest.param(
            _Decimal("148.593"), _Decimal("10.2543"), id="subcent-1523.7171999"
        ),
        pytest.param(_Decimal("1"), _Decimal("0.005"), id="half-paisa-boundary"),
        pytest.param(_Decimal("0.004"), _Decimal("0.01"), id="degenerate-zero-cost"),
    ],
)
def test_synth_lot_block_byte_identical_to_production_for_adversarial_lots(units, nav):
    """Strict mirror-vs-production equivalence for adversarial lots: the pure
    synth mirror's lot block (+ commodity declaration + P directive) must be
    byte-identical to the production ledger-family renderer for the same lot,
    including sub-cent and zero-cost products where the equity leg is the exact
    full-precision negative product (not the 2-dp ``cost_basis``/``amount``)."""
    import datetime

    from financial_dashboard.services.paisa.renderers import render_document
    from financial_dashboard.services.paisa.renderers.base import (
        InvestmentLotEntry,
        LedgerDocument,
        PriceDirective,
    )
    from scripts.synth.paisa import (
        LotFact,
        _commodity_token,
        _fmt_price_rate,
        _lot_entry_lines,
    )

    instrument = "INE000A01020"
    name = "Adversarial Fund"
    product = units * nav
    cost_basis = product.quantize(_Decimal("0.01"))
    acquired = datetime.date(2026, 3, 17)

    # Production renderer: a doc with just this lot + its declaration + P.
    prod_doc = LedgerDocument(
        cutover_date=datetime.date(2026, 1, 1),
        openings=(),
        entries=(),
        accounts_declared=(),
        lot_postings=(
            InvestmentLotEntry(
                instrument=instrument,
                instrument_name=name,
                quantity=units,
                unit_cost=nav,
                cost_basis=cost_basis,
                currency="INR",
                acquired_on=acquired,
            ),
        ),
        price_directives=(PriceDirective(acquired, instrument, nav, "INR"),),
    )
    prod_text = render_document(prod_doc, "ledger")

    # Mirror: assemble the same feature block with the pure helpers. The equity
    # is computed inside _lot_entry_lines as -(units * nav) at full precision.
    lot = LotFact(
        symbol=instrument,
        instrument_name=name,
        units=units,
        nav=nav,
        amount=cost_basis,
        acquired_on=acquired,
        asset_account=f"Assets:Investments:{instrument}",
    )
    mirror_lines = (
        [f"commodity {_commodity_token(instrument)}", ""]
        + _lot_entry_lines(lot)
        + [
            f"P {acquired.isoformat()} {_commodity_token(instrument)} "
            f"{_fmt_price_rate(nav)}"
        ]
    )
    mirror_text = "\n".join(mirror_lines) + "\n"

    assert mirror_text == prod_text, (
        f"synth mirror drifted from production renderer for units={units} "
        f"nav={nav}:\n--- mirror ---\n{mirror_text}\n"
        f"--- production ---\n{prod_text}"
    )
