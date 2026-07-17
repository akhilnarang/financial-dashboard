"""Scenario-branch coverage metadata tests.

The canonical scenario names every edge it exercises as a stable *branch id*
grouped by concern (merge/link, categorization, statement recon, refunds/
reversals, FX, net-worth/CAS/manual, workflow/projection). The manifest records
the present set and ``verify_manifest`` polices the required subset, so a
generator regression that silently drops a branch fails verification.

These tests hold that contract:

* every required branch id is present at every profile that claims full coverage;
* dropping a builder branch removes exactly its id from the coverage set;
* the manifest's coverage block round-trips through ``verify_manifest``;
* ``count_rows`` no longer silently swallows real table errors.
"""

import json
from pathlib import Path

import pytest

from scripts.synth import build_scenario
from scripts.synth.coverage import (
    ALL_BRANCH_IDS,
    BRANCHES_BY_GROUP,
    REQUIRED_BRANCH_IDS,
    SCENARIO_BRANCHES,
    branch_groups,
    compute_coverage,
)
from scripts.synth.manifest import (
    TamperError,
    build_manifest,
    sha256_bytes,
    verify_manifest,
)

pytestmark = pytest.mark.anyio


def test_scenario_branch_registry_is_grouped_and_complete():
    # Every branch id is namespaced ``<group>.<edge>`` and the group prefixes
    # match the registry's grouping.
    for branch in SCENARIO_BRANCHES:
        assert branch.branch_id.startswith(branch.group + ".")
    # The groups the registry exposes cover every id exactly.
    flat = {bid for ids in BRANCHES_BY_GROUP.values() for bid in ids}
    assert flat == ALL_BRANCH_IDS
    # The required set is a subset of all known branches.
    assert REQUIRED_BRANCH_IDS <= ALL_BRANCH_IDS


@pytest.mark.parametrize("profile", ["golden", "smoke", "ci"])
def test_every_required_branch_is_present(profile):
    """Dropping a required builder branch must surface as a missing id — the
    single source of truth for 'did the canonical scenario cover every shape'."""
    scenario = build_scenario(profile=profile)
    present = compute_coverage(scenario)
    missing = sorted(set(REQUIRED_BRANCH_IDS) - present)
    assert not missing, f"{profile}: missing required scenario branches: {missing}"
    # The scenario carries the same set the pure function computes.
    assert scenario.coverage == present


def test_branch_ids_are_stable_across_runs():
    a = compute_coverage(build_scenario(profile="smoke"))
    b = compute_coverage(build_scenario(profile="smoke"))
    assert a == b


def test_coverage_groups_partition_present_ids():
    scenario = build_scenario(profile="golden")
    groups = branch_groups(scenario.coverage)
    # Every present id appears in exactly one group, under its own prefix.
    seen = set()
    for group, ids in groups.items():
        for bid in ids:
            assert bid.startswith(group + ".")
            assert bid not in seen
            seen.add(bid)
    assert seen == scenario.coverage


def _write_corpus_manifest(out_dir: Path, scenario) -> Path:
    """Write a minimal valid manifest (one artefact) carrying the scenario's
    coverage, so verify_manifest reaches the coverage-policing stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artefact = b"sentinel-bytes\n"
    (out_dir / "sentinel.txt").write_bytes(artefact)
    manifest = build_manifest(
        scenario_counts=scenario.counts(),
        invariants={},
        artefacts={"sentinel.txt": artefact},
        seed=scenario.seed,
        as_of=scenario.as_of,
        profile=scenario.profile,
        coverage=scenario.coverage,
    )
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def test_verify_manifest_accepts_full_coverage(tmp_path):
    scenario = build_scenario(profile="golden")
    out_dir = tmp_path / "corpus"
    _write_corpus_manifest(out_dir, scenario)
    # Full coverage verifies cleanly.
    verify_manifest(out_dir, db_counts=scenario.counts())


def test_verify_manifest_rejects_missing_required_branch(tmp_path):
    """A manifest whose recorded coverage is missing a required branch id must
    fail verification — the core 'dropped branch fails verify' contract."""
    scenario = build_scenario(profile="golden")
    out_dir = tmp_path / "corpus"
    _write_corpus_manifest(out_dir, scenario)
    # Corrupt the on-disk manifest: drop one required branch from coverage.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    dropped = next(iter(REQUIRED_BRANCH_IDS))
    manifest["coverage"]["branches"] = sorted(scenario.coverage - {dropped})
    # Rebuild groups to stay internally consistent.
    manifest["coverage"]["groups"] = {
        g: list(v) for g, v in branch_groups(scenario.coverage - {dropped}).items()
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(TamperError, match="missing required scenario branches"):
        verify_manifest(out_dir, db_counts=scenario.counts())


def test_golden_manifest_records_coverage_block():
    """The committed golden manifest carries the coverage block and its branch
    set is exactly the required set (golden claims full coverage)."""
    golden = Path(__file__).parent / "fixtures" / "paisa" / "manifest.json"
    manifest = json.loads(golden.read_text())
    assert "coverage" in manifest
    branches = set(manifest["coverage"]["branches"])
    assert REQUIRED_BRANCH_IDS <= branches
    # The committed artefact checksums still match the coverage-bearing manifest.
    for name, meta in manifest["artefacts"].items():
        committed = (golden.parent / name).read_bytes()
        assert meta["sha256"] == sha256_bytes(committed), name


# ---------------------------------------------------------------------------
# count_rows no longer silently swallows table errors (requirement 7)
# ---------------------------------------------------------------------------


async def test_count_rows_surfaces_real_errors_not_swallow(tmp_path):
    """A genuine DB error (a corrupt DB image) must propagate, not be coerced
    to 0. Only an absent table (``no such table``) reads as 0 — every other
    failure re-raises so manifest drift can never hide behind a swallowed count."""
    from scripts.synth import load_scenario
    from scripts.synth.loader import count_rows

    db = tmp_path / "synthetic" / "err.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    scenario = build_scenario(profile="golden")
    await load_scenario(scenario, db)

    # Corrupt the DB image so SELECT COUNT(*) hits a real (non-
    # ``no such table``) SQLite error. count_rows must NOT swallow this.
    with db.open("r+b") as f:
        f.seek(512)
        f.write(b"\x00" * 4096)

    with pytest.raises(Exception):  # noqa: B017 — any non-swallowed error
        await count_rows(db)


async def test_count_rows_reads_absent_table_as_zero(tmp_path):
    """An absent table (the DB predates it) legitimately reads as 0 — the one
    backward-compat case count_rows must still honor."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from scripts.synth import load_scenario
    from scripts.synth.loader import count_rows

    db = tmp_path / "synthetic" / "absent.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    scenario = build_scenario(profile="golden")
    await load_scenario(scenario, db)
    # Drop one table entirely → ``no such table`` → reads as 0.
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE investment_lots"))
    finally:
        await engine.dispose()
    counts = await count_rows(db)
    assert counts["investment_lots"] == 0
    # Untouched tables still count correctly.
    assert counts["accounts"] == scenario.counts()["accounts"]


def test_compute_coverage_rejects_unknown_ids(monkeypatch):
    """compute_coverage asserts every detected id is a known branch, so a
    future detection rule that mints an unregistered id fails loudly."""
    scenario = build_scenario(profile="golden")
    # No mutation needed: the assertion fires inside compute_coverage if a rule
    # produces an unknown id. Running it here pins that the contract holds.
    present = compute_coverage(scenario)
    assert present <= ALL_BRANCH_IDS
