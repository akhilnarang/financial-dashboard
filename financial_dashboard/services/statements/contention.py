"""Contention analysis shared by the bank- and CC-statement reconcilers.

Both reconcilers pair statement rows with DB transactions greedily, and the
rows they fail to pair are what the import path inserts. A row that *could*
have claimed a DB candidate but ended up with none did not go unmatched
because its transaction is absent — it lost a race, or the matcher refused
the pairing — and importing it would store a second copy of a transaction
the DB already holds.

So each reconciler computes, per statement row, the set of DB rows that
could be its transaction. The set must be **conservative**: it may include
a row the matcher would go on to reject, but it must never omit one the DB
might really be holding. An over-wide set only costs a human a look; a set
that wrongly empties reads as "nothing can already hold this, so import
it", and the duplicate lands silently. For the same reason, rules about
which pairing is *correct* (rather than possible) belong in the picker,
never here — and the sets are evaluated as though nothing had been
consumed, so a rival stays visible after it loses.

Contention is read off these sets, not off date proximity: rows two days
apart contend if they both reach the DB row between them, and same-day rows
do not contend if nothing nearby could be either of them.
"""

CANDIDATE_EVIDENCE_LIMIT = 20


def candidate_evidence(
    candidates: set[int],
    *,
    reason: str,
    gates: tuple[str, ...],
) -> dict:
    """Return bounded, deterministic candidate evidence for operators."""
    ordered = sorted(candidates)
    return {
        "candidate_transaction_ids": ordered[:CANDIDATE_EVIDENCE_LIMIT],
        "candidate_count": len(ordered),
        "candidate_ids_truncated": len(ordered) > CANDIDATE_EVIDENCE_LIMIT,
        "decision_reason": reason,
        "gates": list(gates),
    }


def contended_miss(candidates: set[int]) -> bool:
    """Whether an unmatched statement row is unresolved rather than new.

    A row with any candidate went unmatched because of a refusal, not a
    finding that its transaction is absent, so it is held back from
    auto-import. A row with an *empty* set must still import — that is what
    keeps a genuinely new transaction from being dropped merely because an
    unrelated DB row shares its date and amount.
    """
    return bool(candidates)
