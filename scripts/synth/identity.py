"""Deterministic scenario identity for safe reload-over-existing-DB.

The loader is designed for **same-shape idempotency**: structural and bulk rows
use explicit primary keys with ``ON CONFLICT DO NOTHING``, and fidelity emails
are gated on their ``message_id`` natural key, so a rerun with the same
``(seed, as_of, profile)`` adds zero rows. That contract breaks the moment the
scenario *shape* changes — a new generator version, a different seed/profile,
or a code change that alters the graph for the same inputs. The new scenario's
emails carry fresh message ids but the same low explicit PKs (100+), so the
fidelity lane's raw INSERT collides with an existing email's PK
(``UNIQUE constraint failed: emails.id``); bulk/structural rows silently retain
stale data via ``ON CONFLICT DO NOTHING``.

The fix is a **transactional corpus reset on shape mismatch**: the loader stamps
a deterministic identity into the DB on every successful full load, and on the
next load it compares the stored identity to the current scenario's. When they
differ (or the stamp is missing), the loader wipes every loader-owned table in
one transaction and rebuilds from scratch — so an upgrade over an existing
synthetic DB is always clean. The stamp is written *only* on full success, so a
failed or partial load leaves a stale/missing stamp and the next run resets and
rebuilds (recoverable).

The identity folds the generator version, the ``(seed, as_of, profile)`` inputs
and a content fingerprint of the graph, so a silent code change (one that
alters the output without bumping the version) is caught too. It is a pure
function of the scenario — deterministic, no I/O.
"""

import hashlib
import json
from typing import Iterable

from scripts.synth import constants as C
from scripts.synth.models import Scenario

#: Setting key under which the loader stores the last successfully-loaded
#: scenario identity. Read on the next load to decide whether a reset is needed.
IDENTITY_SETTING_KEY = "synthetic.identity"
GENERATOR_VERSION_SETTING_KEY = "synthetic.generator_version"


def _fold_transaction(h: hashlib._Hash, t) -> None:  # noqa: SLF001
    h.update(b"T|")
    h.update(t.stable_id.encode())
    h.update(b"|")
    h.update(str(t.amount).encode())
    h.update(b"|")
    h.update(t.direction.encode())
    h.update(b"|")
    h.update((t.category or "").encode())
    h.update(b"|")
    h.update((t.currency or "").encode())
    h.update(b"|")
    h.update(
        (t.transaction_date.isoformat() if t.transaction_date else "None").encode()
    )
    h.update(b"|")
    h.update((t.category_method or "").encode())
    h.update(b"|")
    h.update((t.review_status or "").encode())


def _fold_stable_ids(h: hashlib._Hash, kind: str, items: Iterable) -> None:  # noqa: SLF001
    for x in items:
        sid = getattr(x, "stable_id", None)
        if sid is not None:
            h.update(kind.encode())
            h.update(sid.encode())


def scenario_fingerprint(scenario: Scenario) -> str:
    """A deterministic SHA-256 fingerprint of the scenario graph's content.

    Folds the generator version, the ``(seed, as_of, profile)`` inputs, every
    per-table count, the transactional content (stable_id + amount + direction
    + category + currency + date + category_method + review_status), and the
    structural entities' stable ids. Two scenarios with the same fingerprint
    are byte-identical, so the loader can safely skip the reset for them.
    """
    h = hashlib.sha256()
    h.update(C.GENERATOR_VERSION.encode())
    h.update(b"|seed=")
    h.update(str(scenario.seed).encode())
    h.update(b"|as_of=")
    h.update(scenario.as_of.isoformat().encode())
    h.update(b"|profile=")
    h.update(scenario.profile.encode())
    # Per-table counts capture sizing/structural changes cheaply.
    for key, value in sorted(scenario.counts().items()):
        h.update(f"|{key}={value}".encode())
    # Transactional content — the rows whose PK collision broke the old loader.
    for t in scenario.transactions:
        _fold_transaction(h, t)
    # Structural stable ids — accounts/cards/sources/rules/manual/cas/statements.
    _fold_stable_ids(h, "a", scenario.accounts)
    _fold_stable_ids(h, "c", scenario.cards)
    _fold_stable_ids(h, "s", scenario.email_sources)
    _fold_stable_ids(h, "r", scenario.fetch_rules)
    _fold_stable_ids(h, "m", scenario.manual_items)
    _fold_stable_ids(h, "cas", scenario.cas_uploads)
    _fold_stable_ids(h, "st", scenario.statement_uploads)
    _fold_stable_ids(h, "e", scenario.emails)
    _fold_stable_ids(h, "sm", scenario.sms)
    _fold_stable_ids(h, "sn", scenario.account_snapshots)
    _fold_stable_ids(h, "oe", scenario.orphan_emails)
    _fold_stable_ids(h, "fx", scenario.fx_rates)
    return h.hexdigest()


def load_identity(scenario: Scenario) -> str:
    """The identity string the loader stamps and compares: generator version
    plus the content fingerprint. Distinct scenarios never share it."""
    return f"{C.GENERATOR_VERSION}:{scenario_fingerprint(scenario)}"


def identity_matches(stored: str | None, scenario: Scenario) -> bool:
    """True iff ``stored`` is the current scenario's identity (same version +
    same content fingerprint). A missing stamp is always a mismatch so a
    pre-stamp (old or partially-loaded) DB is reset."""
    if not stored:
        return False
    return stored == load_identity(scenario)


def fingerprint_brief(scenario: Scenario) -> str:
    """A short (16-char) fingerprint for manifest/invariant visibility."""
    return scenario_fingerprint(scenario)[:16]


__all__ = [
    "GENERATOR_VERSION_SETTING_KEY",
    "IDENTITY_SETTING_KEY",
    "fingerprint_brief",
    "identity_matches",
    "load_identity",
    "scenario_fingerprint",
]

# Silence the json import linter (kept for future manifest serialization).
_ = json
