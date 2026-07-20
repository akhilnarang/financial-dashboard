"""Contract probe behavior (``scripts/paisa_contract.py``).

Tests the exit-code contract and the honest distinction between:

* a real ledger parse success (exit 0);
* docker unavailable + opt-in skip (exit 2);
* docker unavailable without opt-in (exit 1, never a false success);
* a backend the official image cannot verify (exit 3, a distinct honest skip,
  never conflated with a parse failure or a success).

The ledger contract is exercised live only when Docker is available — the same
gate ``scripts/paisa_contract.py`` itself uses — so this test is hermetic in a
Dockerless environment and authoritative when Docker is present.
"""

import shutil
import subprocess
import sys

import pytest

from scripts import paisa_contract

SCRIPT = "scripts/paisa_contract.py"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


# ---------------------------------------------------------------------------
# Pure unit-level behavior (no Docker required)
# ---------------------------------------------------------------------------


def test_exit_code_constants_are_distinct_and_ordered():
    # 0 success, 1 failure, 2 docker-skip, 3 backend-unsupported — four distinct
    # outcomes so a caller never confuses a skip with success or a parse failure.
    codes = {
        0,
        paisa_contract.EXIT_FAILED,
        paisa_contract.EXIT_SKIP_DOCKER_UNAVAILABLE,
        paisa_contract.EXIT_BACKEND_UNSUPPORTED_BY_IMAGE,
    }
    assert codes == {0, 1, 2, 3}


def test_default_image_is_exact_pinned_tag():
    # Never 'latest' — a re-run must reproduce the same Paisa/ledger binaries.
    assert paisa_contract.DEFAULT_IMAGE == "ananthakumaran/paisa:v0.7.4"


def test_supported_backends_match_dashboard_renderers():
    from financial_dashboard.services.paisa.renderers import SUPPORTED_BACKENDS

    assert set(paisa_contract.SUPPORTED_BACKENDS) == set(SUPPORTED_BACKENDS)


def test_unknown_backend_rejected_without_docker(monkeypatch):
    monkeypatch.setattr(paisa_contract, "_docker_available", lambda: True)
    rc = paisa_contract.run(
        paisa_contract.DEFAULT_IMAGE,
        backend="plaintext",
        skip_if_unavailable=False,
    )
    assert rc == paisa_contract.EXIT_FAILED


def test_docker_unavailable_with_skip_returns_skip_code(monkeypatch):
    monkeypatch.setattr(paisa_contract, "_docker_available", lambda: False)
    rc = paisa_contract.run(
        paisa_contract.DEFAULT_IMAGE,
        backend="ledger",
        skip_if_unavailable=True,
    )
    assert rc == paisa_contract.EXIT_SKIP_DOCKER_UNAVAILABLE


def test_docker_unavailable_without_skip_is_hard_error_never_false_success(
    monkeypatch,
):
    # Without the opt-in skip, missing Docker is a hard failure (1), NOT 0, so
    # the probe can never report a parse-success it did not perform.
    monkeypatch.setattr(paisa_contract, "_docker_available", lambda: False)
    rc = paisa_contract.run(
        paisa_contract.DEFAULT_IMAGE,
        backend="ledger",
        skip_if_unavailable=False,
    )
    assert rc == paisa_contract.EXIT_FAILED


def test_unsupported_backend_reports_honest_skip_not_success(monkeypatch):
    """hledger/beancount are not bundled in the official image. The probe must
    report that distinctly (exit 3), never as success (0) and never conflated
    with a parse failure (1). Docker is faked-available; the image-CLI probe is
    stubbed to report the binary absent — exactly the real v0.7.4 behavior."""
    monkeypatch.setattr(paisa_contract, "_docker_available", lambda: True)

    calls = []

    def fake_probe(image, binary):
        calls.append((image, binary))
        # ledger is present; hledger/beancount are not (the real image state).
        if binary == "ledger":
            return True, "/usr/bin/ledger"
        return False, f"{binary!r} not found in image"

    monkeypatch.setattr(paisa_contract, "_image_cli_available", fake_probe)
    monkeypatch.setattr(
        paisa_contract, "_run_ledger_contract", lambda image, root: (True, "ok")
    )

    rc_h = paisa_contract.run(
        paisa_contract.DEFAULT_IMAGE, backend="hledger", skip_if_unavailable=False
    )
    rc_b = paisa_contract.run(
        paisa_contract.DEFAULT_IMAGE, backend="beancount", skip_if_unavailable=False
    )
    assert rc_h == paisa_contract.EXIT_BACKEND_UNSUPPORTED_BY_IMAGE
    assert rc_b == paisa_contract.EXIT_BACKEND_UNSUPPORTED_BY_IMAGE
    # The probe actually checked the image (not a hardcoded assumption).
    assert ("ananthakumaran/paisa:v0.7.4", "hledger") in calls
    assert ("ananthakumaran/paisa:v0.7.4", "beancount") in calls


def test_all_backend_succeeds_when_ledger_passes_others_unsupported(monkeypatch):
    monkeypatch.setattr(paisa_contract, "_docker_available", lambda: True)
    monkeypatch.setattr(
        paisa_contract,
        "_image_cli_available",
        lambda image, binary: (
            (True, "/usr/bin/ledger") if binary == "ledger" else (False, "absent")
        ),
    )
    monkeypatch.setattr(
        paisa_contract, "_run_ledger_contract", lambda image, root: (True, "ok")
    )
    rc = paisa_contract.run(
        paisa_contract.DEFAULT_IMAGE, backend="all", skip_if_unavailable=False
    )
    # ledger passed; hledger/beancount were reported as skipped (not failure).
    assert rc == 0


def test_all_backend_fails_when_ledger_fails(monkeypatch):
    monkeypatch.setattr(paisa_contract, "_docker_available", lambda: True)
    monkeypatch.setattr(
        paisa_contract,
        "_image_cli_available",
        lambda image, binary: (
            (True, "/usr/bin/ledger") if binary == "ledger" else (False, "absent")
        ),
    )
    monkeypatch.setattr(
        paisa_contract,
        "_run_ledger_contract",
        lambda image, root: (False, "ledger rejected the journal"),
    )
    rc = paisa_contract.run(
        paisa_contract.DEFAULT_IMAGE, backend="all", skip_if_unavailable=False
    )
    assert rc == paisa_contract.EXIT_FAILED


# ---------------------------------------------------------------------------
# Live ledger contract (only when Docker is present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _docker_available(), reason="docker not available")
def test_live_ledger_contract_parses_synthetic_journal():
    """End-to-end: the pinned official image parses the generated journal via
    the real ``paisa update --journal`` path AND parses the PRODUCTION
    ``render_document`` corpus with commodity/account/quantity identity
    assertions. This is the authoritative contract — the same one CI's ledger
    job runs."""
    rc = subprocess.run(
        [sys.executable, SCRIPT, "--backend", "ledger"],
        capture_output=True,
        text=True,
        timeout=240,
    ).returncode
    assert rc == 0, "ledger contract failed; see scripts/paisa_contract.py"


# ---------------------------------------------------------------------------
# Production ``render_document`` contract identity assertions (unit-level)
# ---------------------------------------------------------------------------
#
# The contract must parse the PRODUCTION renderer corpus (not only the
# hand-mirrored synth bytes) and assert parsed commodity/account/quantity
# identity, not just ``rc=0``. These monkeypatch the docker subprocess so the
# identity-assertion logic is covered hermetically (no Docker required).


def _fake_ledger_response(args):
    """Return a canned CompletedProcess for a ``ledger`` subcommand based on the
    trailing args of the docker command."""
    # The ledger subcommand is the last positional arg(s) after ``-f <file>``.
    cmd = args
    sub = cmd[cmd.index("-f") + 2 :]  # everything after ``-f <journal>``
    if sub[:1] == ["balance"] and len(sub) == 1:
        out = "    500 INE000A01020  Assets:Investments:INE000A01020\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    if sub[:1] == ["commodities"]:
        return subprocess.CompletedProcess(cmd, 0, '"INE000A01020"\nINR\nUSD\n', "")
    if sub[:1] == ["accounts"]:
        return subprocess.CompletedProcess(
            cmd,
            0,
            "Assets:Investments:INE000A01020\n"
            "Equity:Opening Balances:Investment\n"
            "Expenses:Insurance:Medical Health Insurance Premium\n",
            "",
        )
    if sub[:2] == ["balance", "Assets:Investments:INE000A01020"]:
        return subprocess.CompletedProcess(
            cmd, 0, "    500 INE000A01020  Assets:Investments:INE000A01020\n", ""
        )
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_production_ledger_contract_passes_on_full_identity(monkeypatch, tmp_path):
    """With the parser returning the expected commodities/accounts/quantity, the
    production contract reports OK (the identity assertions all hold)."""
    monkeypatch.setattr(
        paisa_contract.subprocess,
        "run",
        lambda cmd, **k: _fake_ledger_response(cmd),
    )
    ok, detail = paisa_contract._run_production_ledger_contract(
        paisa_contract.DEFAULT_IMAGE, tmp_path
    )
    assert ok, detail
    # The journal was rendered from the production renderer.
    assert (tmp_path / "ledger.journal").exists()


def test_production_ledger_contract_fails_on_missing_commodity(monkeypatch, tmp_path):
    """A missing lot commodity in the parsed set is a contract failure (not a
    silent rc=0 success) — the identity assertion must catch it."""

    def missing(cmd):
        resp = _fake_ledger_response(cmd)
        if "commodities" in cmd:
            # The ISIN commodity is absent from the parsed output.
            resp = subprocess.CompletedProcess(cmd, 0, "INR\nUSD\n", "")
        return resp

    monkeypatch.setattr(paisa_contract.subprocess, "run", lambda cmd, **k: missing(cmd))
    ok, detail = paisa_contract._run_production_ledger_contract(
        paisa_contract.DEFAULT_IMAGE, tmp_path
    )
    assert not ok
    assert "INE000A01020" in detail and "commodities" in detail


def test_production_ledger_contract_fails_on_parse_error(monkeypatch, tmp_path):
    """A non-zero ``ledger balance`` rc is a hard contract failure."""

    def failing(cmd):
        if "balance" in cmd and "commodities" not in cmd and "accounts" not in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "parse error: bad token")
        return _fake_ledger_response(cmd)

    monkeypatch.setattr(paisa_contract.subprocess, "run", lambda cmd, **k: failing(cmd))
    ok, detail = paisa_contract._run_production_ledger_contract(
        paisa_contract.DEFAULT_IMAGE, tmp_path
    )
    assert not ok
    assert "balance failed" in detail


@pytest.mark.skipif(not _docker_available(), reason="docker not available")
def test_live_hledger_contract_reports_unsupported_by_image():
    """The official image bundles only ledger, so hledger must report the honest
    unsupported-by-image skip (3), distinct from a parse failure."""
    rc = subprocess.run(
        [sys.executable, SCRIPT, "--backend", "hledger"],
        capture_output=True,
        text=True,
        timeout=180,
    ).returncode
    assert rc == paisa_contract.EXIT_BACKEND_UNSUPPORTED_BY_IMAGE
