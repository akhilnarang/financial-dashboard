"""Manifest write / read / verify with tamper detection.

The manifest is the trust anchor for ``verify``: it records the generator +
schema version, the ``(seed, as_of, profile)`` inputs, the expected DB counts
and structural invariants, and a SHA-256 per generated artefact. ``verify``
recomputes every checksum and (when a DB is present) every count, and reports
any divergence as :class:`TamperError`.

Artefact checksums are computed over the exact bytes written to disk, so any
post-generation edit — even a single byte — is detected.
"""

import datetime
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from scripts.synth import constants as C


class TamperError(ValueError):
    """Raised by :func:`verify_manifest` when on-disk state diverges."""


class FileHash(NamedTuple):
    path: str
    sha256: str
    bytes: int


def sha256_file(path: Path) -> FileHash:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            size += len(chunk)
    return FileHash(path=str(path), sha256=h.hexdigest(), bytes=size)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest(
    *,
    scenario_counts: dict[str, int],
    invariants: dict[str, object],
    artefacts: dict[str, bytes],
    seed: int,
    as_of: datetime.date,
    profile: str,
    coverage: frozenset[str] | None = None,
) -> dict:
    """Build the in-memory manifest dict. Pure — no I/O.

    ``coverage`` is the set of scenario-branch ids the generator exercised (see
    :mod:`scripts.synth.coverage`). It is recorded grouped-by-concern so a
    reader can see which shapes the corpus covers, and policed by
    :func:`verify_manifest` against the required set."""
    artefact_hashes = {
        name: {"sha256": sha256_bytes(data), "bytes": len(data)}
        for name, data in sorted(artefacts.items())
    }
    manifest = {
        "schema_version": C.SCHEMA_VERSION,
        "generator_version": C.GENERATOR_VERSION,
        "seed": seed,
        "as_of": as_of.isoformat(),
        "profile": profile,
        "expected": dict(sorted(scenario_counts.items())),
        "invariants": dict(sorted(invariants.items())),
        "artefacts": artefact_hashes,
    }
    if coverage is not None:
        from scripts.synth.coverage import branch_groups

        manifest["coverage"] = {
            "branches": sorted(coverage),
            "groups": {g: list(v) for g, v in branch_groups(coverage).items()},
        }
    return manifest


def manifest_filename() -> str:
    return "manifest.json"


def write_manifest(manifest: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / manifest_filename()
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_manifest(out_dir: Path) -> dict:
    path = out_dir / manifest_filename()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_manifest(
    out_dir: Path,
    *,
    db_counts: dict[str, int] | None = None,
) -> None:
    """Recompute artefact checksums and (optional) DB counts against the
    manifest on disk. Raise :class:`TamperError` on any divergence."""
    out_dir = Path(out_dir)
    manifest = read_manifest(out_dir)

    expected_version = manifest.get("generator_version")
    if expected_version != C.GENERATOR_VERSION:
        raise TamperError(
            f"manifest generator_version {expected_version!r} != running "
            f"{C.GENERATOR_VERSION!r}; regenerate the corpus."
        )
    schema_version = manifest.get("schema_version")
    if schema_version != C.SCHEMA_VERSION:
        raise TamperError(
            f"manifest schema_version {schema_version!r} != running "
            f"{C.SCHEMA_VERSION!r}"
        )

    # --- artefact checksums (file tamper detection) ---
    artefacts = manifest.get("artefacts", {})
    if not artefacts:
        raise TamperError("manifest lists no artefacts")
    for name, meta in sorted(artefacts.items()):
        path = out_dir / name
        if not path.exists():
            raise TamperError(f"artefact missing: {name}")
        digest = sha256_file(path)
        if digest.sha256 != meta["sha256"]:
            raise TamperError(
                f"artefact checksum mismatch for {name}: "
                f"expected {meta['sha256']}, got {digest.sha256}"
            )
        if digest.bytes != meta["bytes"]:
            raise TamperError(
                f"artefact size mismatch for {name}: "
                f"expected {meta['bytes']}, got {digest.bytes}"
            )

    # --- DB counts (load tamper / drift detection) ---
    if db_counts is not None:
        expected_counts = manifest.get("expected", {})
        for key, expected in sorted(expected_counts.items()):
            actual = db_counts.get(key)
            if actual != expected:
                raise TamperError(
                    f"DB count mismatch for {key!r}: expected {expected}, got {actual}"
                )

    # --- scenario-branch coverage (required-edge detection) ---
    # The manifest's recorded coverage must include every required branch id,
    # so a generator regression that silently drops a canonical edge fails
    # verification. Older manifests without a ``coverage`` block are accepted
    # only when they predate coverage support (their generator version would
    # already have failed the version check above for a current run).
    coverage_block = manifest.get("coverage")
    if coverage_block is not None:
        from scripts.synth.coverage import REQUIRED_BRANCH_IDS

        present = set(coverage_block.get("branches", []))
        missing = sorted(set(REQUIRED_BRANCH_IDS) - present)
        if missing:
            raise TamperError(
                f"manifest coverage missing required scenario branches: {missing}"
            )
