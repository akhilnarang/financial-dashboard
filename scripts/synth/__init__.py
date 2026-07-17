"""Deterministic synthetic seed generator + safe loader for financial-dashboard.

Public surface:

* :func:`build_scenario` — pure, deterministic scenario-graph generator.
* :func:`build_corpus` — pure, offline Paisa 0.7.4 corpus generator.
* :func:`load_scenario` — two-lane (fidelity + bulk) loader into a dedicated
  synthetic SQLite DB.
* :func:`run_cli` — the ``uv run python -m scripts.synth`` entrypoint.

See ``AGENTS.md`` and the module docstrings for the safety model.
"""

from scripts.synth.constants import (
    DEFAULT_PROFILE,
    DEFAULT_SEED,
    DEFAULT_SYNTHETIC_ROOT,
    GENERATOR_VERSION,
    INVALID_CURRENCY_TOKEN,
    PROJECTION_CUTOVER,
    RESET_CONFIRMATION_FLAG,
    SCHEMA_VERSION,
)
from scripts.synth.backends import build_backend_corpora
from scripts.synth.identity import (
    fingerprint_brief,
    load_identity,
    scenario_fingerprint,
)
from scripts.synth.loader import load_scenario
from scripts.synth.manifest import (
    build_manifest,
    read_manifest,
    verify_manifest,
    write_manifest,
)
from scripts.synth.paisa import build_corpus
from scripts.synth.scenario import PROFILES, build_scenario

__all__ = [
    "DEFAULT_PROFILE",
    "DEFAULT_SEED",
    "DEFAULT_SYNTHETIC_ROOT",
    "GENERATOR_VERSION",
    "INVALID_CURRENCY_TOKEN",
    "PROFILES",
    "PROJECTION_CUTOVER",
    "RESET_CONFIRMATION_FLAG",
    "SCHEMA_VERSION",
    "build_backend_corpora",
    "build_corpus",
    "build_manifest",
    "build_scenario",
    "fingerprint_brief",
    "load_identity",
    "load_scenario",
    "read_manifest",
    "scenario_fingerprint",
    "verify_manifest",
    "write_manifest",
]
