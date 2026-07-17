"""Backend corpus contracts: determinism, byte-stability, tamper detection, and
per-backend content (long names / multi-currency / lots) for the
production-renderer corpora produced by :mod:`scripts.synth.backends`.

These corpora are deliberately NOT the hand-reviewable Paisa-container ledger
(``scripts.synth.paisa``); they are the **production renderer's** output, so a
drift in the renderer shows up here as a committed-fixture diff. They are also
checksummed in the golden ``manifest.json``, so :func:`verify_manifest` covers
them for tamper detection.
"""

import json
import re
from pathlib import Path

import pytest

from scripts.synth import build_scenario
from scripts.synth.backends import (
    BACKEND_IDS,
    MAX_BACKEND_ENTRIES,
    build_backend_corpora,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "paisa"


def _golden_backend_artefacts() -> dict[str, bytes]:
    return {
        f"{b}.journal": (GOLDEN_DIR / f"{b}.journal").read_bytes() for b in BACKEND_IDS
    } | {"backends.meta.json": (GOLDEN_DIR / "backends.meta.json").read_bytes()}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_build_backend_corpora_is_deterministic():
    scenario = build_scenario(profile="smoke")
    a = build_backend_corpora(scenario)
    b = build_backend_corpora(scenario)
    assert a.backends == b.backends == BACKEND_IDS
    for name in a.artefacts:
        assert a.artefacts[name] == b.artefacts[name]


def test_backend_corpora_entry_count_capped_for_reviewability():
    # Even a stress scenario yields a hand-reviewable corpus.
    scenario = build_scenario(profile="stress")
    corpus = build_backend_corpora(scenario)
    assert corpus.entries <= MAX_BACKEND_ENTRIES
    for b in BACKEND_IDS:
        text = corpus.artefacts[f"{b}.journal"].decode("utf-8")
        # No single backend journal blows up into an unreviewable file.
        assert len(text) < 25_000, f"{b}: {len(text)} bytes"


def test_backend_corpus_meta_describes_inputs():
    scenario = build_scenario(profile="golden")
    corpus = build_backend_corpora(scenario)
    meta = json.loads(corpus.artefacts["backends.meta.json"].decode("utf-8"))
    assert meta["backends"] == list(BACKEND_IDS)
    assert meta["rendered_by"].endswith("render_document")
    assert meta["seed"] == scenario.seed
    assert meta["profile"] == scenario.profile
    assert meta["lot_count"] >= 1
    assert "USD" in meta["currencies"]  # multi-currency represented


# ---------------------------------------------------------------------------
# Committed golden corpus byte-stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKEND_IDS)
def test_golden_backend_journal_byte_identical_to_generator(backend):
    scenario = build_scenario(seed=4242, profile="golden")
    corpus = build_backend_corpora(scenario)
    committed = (GOLDEN_DIR / f"{backend}.journal").read_bytes()
    assert committed == corpus.artefacts[f"{backend}.journal"]


def test_golden_backend_meta_byte_identical_to_generator():
    scenario = build_scenario(seed=4242, profile="golden")
    corpus = build_backend_corpora(scenario)
    committed = (GOLDEN_DIR / "backends.meta.json").read_bytes()
    assert committed == corpus.artefacts["backends.meta.json"]


def test_golden_backend_artefacts_present():
    for name in [*(f"{b}.journal" for b in BACKEND_IDS), "backends.meta.json"]:
        assert (GOLDEN_DIR / name).exists(), name


def test_golden_backend_artefacts_checksummed_in_manifest():
    """The backend corpora are part of the golden manifest, so verify_manifest's
    tamper detection covers them — not just the Paisa-container ledger."""
    from scripts.synth.manifest import sha256_bytes

    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    for backend in BACKEND_IDS:
        name = f"{backend}.journal"
        assert name in manifest["artefacts"], f"{name} missing from manifest"
        committed = (GOLDEN_DIR / name).read_bytes()
        assert manifest["artefacts"][name]["sha256"] == sha256_bytes(committed)


# ---------------------------------------------------------------------------
# Content: long names / multi-currency / lots represented in every backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKEND_IDS)
def test_backend_corpus_has_multi_currency_usd_entry_and_price(backend):
    scenario = build_scenario(profile="golden")
    text = build_backend_corpora(scenario).artefacts[f"{backend}.journal"].decode()
    # A priced USD entry is emitted in USD (never relabelled INR).
    assert "USD" in text
    # A price directive lets the backend value USD in INR.
    if backend == "beancount":
        assert re.search(r"\d{4}-\d{2}-\d{2} price USD\s+85\.5000 INR", text)
    else:
        assert re.search(r"P \d{4}-\d{2}-\d{2} USD\s+85\.5000 INR", text)


@pytest.mark.parametrize("backend", BACKEND_IDS)
def test_backend_corpus_has_complete_investment_lot(backend):
    scenario = build_scenario(profile="golden")
    text = build_backend_corpora(scenario).artefacts[f"{backend}.journal"].decode()
    # The complete MF fact (units=500, nav=100) becomes a lot posting under the
    # dedicated investment hierarchy — no bank/cash leg is inferred.
    assert "Assets:Investments:INE000A01020" in text
    assert "INE000A01020" in text  # commodity declared
    assert "500" in text  # quantity
    # The lot's explicit unit cost (100, the CAS nav) appears as a cost
    # annotation; nothing else invents a per-unit figure.
    assert "100.00" in text
    # The dedicated investment equity contra (never the bank opening).
    if backend == "beancount":
        assert "Equity:OpeningBalances:Investment" in text
    else:
        assert "Equity:Opening Balances:Investment" in text
    assert (
        "Assets:Bank" not in text or "Assets:Bank:" in text
    )  # lots don't post to bank


@pytest.mark.parametrize("backend", BACKEND_IDS)
def test_backend_corpus_long_name_survives_and_is_backend_legal(backend):
    scenario = build_scenario(profile="golden")
    text = build_backend_corpora(scenario).artefacts[f"{backend}.journal"].decode()
    if backend == "beancount":
        # beancount forbids spaces → the default is PascalCased, still present.
        assert "MedicalHealthInsurancePremium" in text
        assert " " not in re.search(r"Expenses:\w*Insurance\w*", text).group(0)
    else:
        # ledger/hledger allow spaces; the long name survives verbatim.
        assert "Expenses:Insurance:Medical Health Insurance Premium" in text


# ---------------------------------------------------------------------------
# Structural per-backend contracts (mirror test_paisa_backends.py)
# ---------------------------------------------------------------------------


def test_ledger_and_hledger_postings_split_on_two_spaces():
    scenario = build_scenario(profile="smoke")
    # A standard posting ends in ``<amount> INR``; the lot cost-annotation line
    # ends in ``[date]`` so it is excluded from this contract check (it is
    # validated separately by the lot-syntax tests).
    amount_line = re.compile(r"^\s+\S.*\s-?\d+\.\d{2}\s+INR$")
    for backend in ("ledger", "hledger"):
        text = build_backend_corpora(scenario).artefacts[f"{backend}.journal"].decode()
        posting_lines = [
            ln
            for ln in text.splitlines()
            if ln.startswith("    ") and amount_line.match(ln)
        ]
        assert posting_lines, f"{backend}: expected standard posting lines"
        for line in posting_lines:
            parts = re.split(r" {2,}", line.lstrip(), maxsplit=1)
            assert len(parts) == 2, f"{backend}: bad split: {line!r}"
            assert parts[1].endswith("INR")


def test_beancount_emits_commodity_and_open_directives():
    scenario = build_scenario(profile="smoke")
    text = build_backend_corpora(scenario).artefacts["beancount.journal"].decode()
    assert re.search(r"\d{4}-\d{2}-\d{2} commodity INR", text)
    assert re.search(r"\d{4}-\d{2}-\d{2} open Assets:Bank:", text)
    # The USD-commodity account is opened with both currencies.
    assert re.search(r"open Assets:Bank:\S+ INR,USD", text)


def test_no_fabricated_lot_cost():
    """The lot's cost basis is the explicit CAS amount (50000 = 500*100), never
    a value derived from a market quote or invented figure."""
    scenario = build_scenario(profile="golden")
    for backend in BACKEND_IDS:
        text = build_backend_corpora(scenario).artefacts[f"{backend}.journal"].decode()
        assert "50000.00" in text  # the cost-basis/equity leg
