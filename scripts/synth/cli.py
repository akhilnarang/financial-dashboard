"""``uv run python -m scripts.synth`` — argparse CLI.

Commands
--------
* ``generate`` — build the deterministic scenario + offline Paisa corpus and
  write them under ``data/synthetic/<profile>/`` with a checksummed manifest.
  Never touches a database. Run this for stress without loading.
* ``load`` — generate (if needed) and load the scenario into the synthetic
  SQLite DB via the two-lane loader. Default target is always under
  ``data/synthetic``.
* ``verify`` — recompute artefact checksums and DB counts against the manifest
  and report any tampering.
* ``reset`` — delete the synthetic DB. Requires ``--reset`` AND
  ``--confirm-reset yes-delete-the-synthetic-db``. Never targets production.
"""

import argparse
import asyncio
import datetime
import json
import logging
import sys
from pathlib import Path

from scripts.synth import constants as C
from scripts.synth.backends import BACKEND_IDS, build_backend_corpora
from scripts.synth.identity import fingerprint_brief
from scripts.synth.loader import count_rows, drop_synthetic_db, load_scenario
from scripts.synth.manifest import (
    TamperError,
    build_manifest,
    read_manifest,
    verify_manifest,
    write_manifest,
)
from scripts.synth.paisa import build_corpus
from scripts.synth.safety import (
    UnsafeTargetError,
    assert_synthetic_db_path,
    confirm_reset,
)
from scripts.synth.scenario import DEFAULT_AS_OF, PROFILES, build_scenario

logger = logging.getLogger("scripts.synth")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _profile_dir(root: str, profile: str) -> Path:
    return Path(root) / profile


def _default_db_path(root: str, profile: str) -> Path:
    return _profile_dir(root, profile) / "synthetic.db"


def _parse_as_of(value: str | None) -> datetime.date:
    if not value:
        return DEFAULT_AS_OF
    return datetime.date.fromisoformat(value)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_generate(args: argparse.Namespace) -> int:
    as_of = _parse_as_of(args.as_of)
    scenario = build_scenario(seed=args.seed, as_of=as_of, profile=args.profile)
    out_dir = _profile_dir(args.root, args.profile)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(scenario)
    for name, data in corpus.artefacts.items():
        (out_dir / name).write_bytes(data)

    # Backend-specific projection corpora (ledger/hledger/beancount), rendered
    # by the production renderer. Capped so even a stress profile stays
    # hand-reviewable; tracked in the same manifest so ``verify`` covers them.
    backend_corpus = build_backend_corpora(scenario, backends=BACKEND_IDS)
    for name, data in backend_corpus.artefacts.items():
        (out_dir / name).write_bytes(data)

    all_artefacts = {**corpus.artefacts, **backend_corpus.artefacts}
    invariants: dict[str, object] = {
        "journal_balanced": True,
        "all_amounts_decimal": True,
        "stable_ids": True,
        "bulk_txn_id_base": C.BULK_TXN_ID_BASE,
        "backend_corpora": list(BACKEND_IDS),
        "fx_rate_count": len(scenario.fx_rates),
        # The scenario fingerprint lets the loader detect a shape change when
        # loading over an existing synthetic DB (see scripts.synth.identity).
        "scenario_fingerprint": fingerprint_brief(scenario),
    }
    manifest = build_manifest(
        scenario_counts=scenario.counts(),
        invariants=invariants,
        artefacts=all_artefacts,
        seed=args.seed,
        as_of=as_of,
        profile=args.profile,
        coverage=scenario.coverage,
    )
    write_manifest(manifest, out_dir)

    logger.info(
        "generated profile=%s seed=%s as_of=%s -> %s (%d txns, %d ledger entries, "
        "%d backend corpora: %s)",
        args.profile,
        args.seed,
        as_of,
        out_dir,
        len(scenario.transactions),
        corpus.entries,
        len(backend_corpus.backends),
        ",".join(backend_corpus.backends),
    )
    return 0


async def cmd_load(args: argparse.Namespace) -> int:
    as_of = _parse_as_of(args.as_of)
    scenario = build_scenario(seed=args.seed, as_of=as_of, profile=args.profile)
    db_path = assert_synthetic_db_path(
        args.db_path or _default_db_path(args.root, args.profile)
    )

    stats = await load_scenario(
        scenario, db_path, fidelity_txn_count=args.fidelity_txns
    )
    logger.info("loaded into %s — %s", db_path, json.dumps(stats, sort_keys=True))
    return 0


async def cmd_verify(args: argparse.Namespace) -> int:
    out_dir = _profile_dir(args.root, args.profile)
    db_path = args.db_path or _default_db_path(args.root, args.profile)

    db_counts = None
    if Path(db_path).exists():
        try:
            db_counts = await count_rows(db_path)
        except UnsafeTargetError as exc:
            logger.error("refused to read DB: %s", exc)
            return 2

    try:
        verify_manifest(out_dir, db_counts=db_counts)
    except TamperError as exc:
        logger.error("TAMPER DETECTED: %s", exc)
        return 1

    manifest = read_manifest(out_dir)
    logger.info(
        "verified profile=%s seed=%s as_of=%s — manifest intact",
        manifest.get("profile"),
        manifest.get("seed"),
        manifest.get("as_of"),
    )
    if db_counts is not None:
        logger.info("db_counts=%s", json.dumps(db_counts, sort_keys=True))
    else:
        logger.info("no synthetic DB present; verified artefact checksums only")
    return 0


async def cmd_reset(args: argparse.Namespace) -> int:
    if not args.reset:
        logger.error("reset requires --reset (and --confirm-reset); refusing.")
        return 2
    confirm_reset(args.confirm_reset)
    db_path = assert_synthetic_db_path(
        args.db_path or _default_db_path(args.root, args.profile)
    )
    await drop_synthetic_db(db_path)
    logger.info("deleted synthetic DB: %s", db_path)
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", choices=sorted(PROFILES), default=C.DEFAULT_PROFILE)
    p.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to 2026-07-15")
    p.add_argument("--root", default=C.DEFAULT_SYNTHETIC_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.synth",
        description="Deterministic synthetic seed generator + safe loader.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate", help="write the scenario + Paisa corpus + manifest"
    )
    _add_common(gen)
    gen.set_defaults(func=cmd_generate)

    load = sub.add_parser("load", help="load the scenario into the synthetic DB")
    _add_common(load)
    load.add_argument(
        "--db-path", default=None, help="override DB path (must be synthetic)"
    )
    load.add_argument(
        "--fidelity-txns",
        type=int,
        default=None,
        help="override the fidelity-lane transaction count",
    )
    load.set_defaults(func=cmd_load)

    verify = sub.add_parser(
        "verify", help="recompute checksums/counts; detect tampering"
    )
    _add_common(verify)
    verify.add_argument("--db-path", default=None)
    verify.set_defaults(func=cmd_verify)

    reset = sub.add_parser(
        "reset", help="delete the synthetic DB (destructive, guarded)"
    )
    _add_common(reset)
    reset.add_argument("--db-path", default=None)
    reset.add_argument("--reset", action="store_true", help="required to enable reset")
    reset.add_argument(
        "--confirm-reset",
        default=None,
        help=f"must be exactly {C.RESET_CONFIRMATION_FLAG!r}",
    )
    reset.set_defaults(func=cmd_reset)

    return parser


async def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return await args.func(args)
    except UnsafeTargetError as exc:
        logger.error("refused: %s", exc)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
