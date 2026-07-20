"""Classification of Paisa v0.7.4 diagnosis dangers for sync.

Paisa's doctor (``internal/server/doctor.go``) emits a ``Debit Entry`` danger
for *every* posting under ``Expenses:%`` whose amount is negative. Our canonical
taxonomy intentionally posts refunds/cashback and expense reversals (fee
reversals, reversed groceries, etc.) as *negative* Expenses (contra-expense) so
reversals net against the original spend rather than being relabelled as Income.
Those negative postings are correct accounting, so the resulting ``Debit Entry``
dangers are expected and must not fail a sync — but **only** the ones our
projection generated, never operator-authored journal content, and never a
balance/parse problem.

Paisa v0.7.4 exposes **no config flag** to disable or allowlist these checks:
the doctor rules are hardcoded and run unconditionally over the entire journal
(every account that matches ``Expenses:%``, no exclusion list). The config
schema (``paisa.yaml``) carries no diagnosis overrides either. So the safest
production contract is exact, multiplicity-aware per-posting fingerprint
matching (option (b)):

1. :func:`expected_debit_entry_fingerprints` derives a *multiset* of expected
   ``(account, date, amount)`` fingerprints from the
   :class:`~financial_dashboard.services.paisa.projection.ProjectionReport` —
   one entry per negative ``Expenses:`` posting the projection generated.
2. :func:`classify_diagnosis` walks each upstream ``danger`` issue. A
   ``Debit Entry`` whose parsed ``(account, date, amount)`` consumes an expected
   fingerprint is **accepted** (downgraded). Anything else — an extra posting
   beyond the expected multiplicity, an operator-authored negative Expenses
   posting, an unparseable issue, or any non-``Debit Entry`` danger
   (``Negative Balance``, ``Credit Entry``, ``Exchange Price Missing``) — stays
   **fatal**.

Matching is exact and multiplicity-aware: two identical generated postings
consume two fingerprints, so a third same-account/date/amount danger stays
fatal. The probe path is **never** touched — :func:`probe` still surfaces the
raw upstream diagnosis so an operator can see everything Paisa reports.

This module is pure over its arguments (no I/O, no session) so it is trivially
testable. It places no raw journal text and no credentials in anything — the
orchestrator surfaces only integer counts (expected/accepted/fatal) into audit
details, and a sanitized first-fatal summary into the sync ``reason``.
"""

import datetime
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from financial_dashboard.integrations.paisa import PaisaDiagnosis
from financial_dashboard.services.paisa.renderers.base import ProjectedEntry

#: Paisa v0.7.4 ``doctor.go`` summary for the negative-Expenses rule. Only this
#: one danger kind is ever classifiable as expected — every other danger summary
#: (``Negative Balance``, ``Credit Entry``, ``Exchange Price Missing``, …) is
#: always fatal.
DEBIT_ENTRY_SUMMARY = "Debit Entry"

#: Four-decimal quantization for the fingerprint amount. Paisa formats the
#: posting amount as ``%.4f`` in the diagnosis ``details``; our postings are
#: 2-dp in the journal, so quantizing both sides to 4 dp makes them compare
#: equal regardless of trailing-zero formatting.
_Q4 = Decimal("0.0001")

#: Paisa's ``DATE_FORMAT`` (Go ``"02 Jan 2006"``) renders as e.g. ``15 Jan
#: 2026`` — always English month abbreviations (Go's ``time.Format`` is
#: locale-independent, always English). Python's ``strptime("%b")`` is
#: locale-sensitive, so this explicit map parses the month without depending on
#: the server's locale — a non-English locale can never turn an expected danger
#: fatal by failing to parse the date.
_PAISA_MONTHS: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class DebitEntryFingerprint(NamedTuple):
    """The exact identity of one negative Expenses posting Paisa's ``Debit
    Entry`` rule will flag.

    ``(account, date, amount)`` is the smallest tuple that uniquely identifies a
    single flagged posting: Paisa's rule iterates postings and reports each
    negative one with exactly these three fields. ``amount`` is the *negative*
    posting amount (what Paisa reports), quantized to 4 dp.
    """

    account: str
    date: datetime.date
    amount: Decimal


class ClassifiedDiagnosis(NamedTuple):
    """The result of classifying a raw :class:`PaisaDiagnosis` against the
    projection's expected contra-expense postings.

    * ``expected_count`` — how many ``Debit Entry`` dangers the projection
      *should* produce (the total multiplicity of expected fingerprints).
    * ``accepted_count`` — how many upstream ``Debit Entry`` dangers matched an
      expected fingerprint and were downgraded.
    * ``fatal_count`` — how many dangers remain fatal (unmatched ``Debit Entry``,
      any non-``Debit Entry`` danger, or an unparseable issue). Sync fails when
      this is nonzero.
    * ``first_fatal_message`` — a sanitized (HTML-stripped) summary of the first
      fatal danger, for the sync ``reason``. ``None`` when nothing is fatal.
    """

    expected_count: int
    accepted_count: int
    fatal_count: int
    first_fatal_message: str | None
    #: Whether the *raw* upstream diagnosis carried any danger. Lets a caller
    #: distinguish "clean" from "all dangers were expected contra-expense".
    raw_danger_count: int


# ---------------------------------------------------------------------------
# Expected-fingerprint derivation (from the typed ProjectionReport)
# ---------------------------------------------------------------------------


def expected_debit_entry_fingerprints(
    entries: "tuple[ProjectedEntry, ...] | list[ProjectedEntry]",
) -> Counter[DebitEntryFingerprint]:
    """Build the multiset of ``Debit Entry`` fingerprints the projection
    *should* produce.

    One entry per posting the projection emitted under ``Expenses:`` with a
    negative amount — i.e. exactly what Paisa's ``ruleNonDebitAccount`` flags.
    Refunds/cashback (``dashboard_kind=contra_expense``) and every expense
    reversal (a credit on any expense slug, e.g. a fee reversal) land here,
    because our canonical semantics post them as negative Expenses so they net
    against the original spend.

    The date is the entry's date (the posting date Paisa will report) and the
    amount is the negative posting amount quantized to 4 dp (Paisa's ``%.4f``
    format). Derived from the typed report, never from re-parsing the journal,
    so a renderer formatting change cannot silently drift the fingerprint.
    """
    expected: Counter[DebitEntryFingerprint] = Counter()
    for entry in entries:
        date = entry.date
        for posting in entry.postings:
            if posting.account.startswith("Expenses:") and posting.amount < 0:
                expected[
                    DebitEntryFingerprint(
                        account=posting.account,
                        date=date,
                        amount=posting.amount.quantize(_Q4),
                    )
                ] += 1
    return expected


# ---------------------------------------------------------------------------
# Upstream details-string parsing
# ---------------------------------------------------------------------------


#: Anchored on Paisa v0.7.4's exact ``ruleNonDebitAccount`` template
#: ``"<b>%.4f</b> got debited from <b>%s</b> on %s"``. The ``<b>`` tags are the
#: stable anchors (they wrap exactly the amount and the account); the account
#: permits anything but ``<`` (account names never contain a literal ``<``);
#: the date is ``DD Mon YYYY``. A details string that does not match this shape
#: (a future upstream change, or a non-``Debit Entry`` issue routed here by
#: mistake) yields ``None`` and the caller leaves the danger fatal — we never
#: accept something we cannot prove.
_DEBIT_ENTRY_RE = re.compile(
    r"<b>(?P<amount>-?\d+(?:\.\d+)?)</b>\s+got\s+debited\s+from\s+"
    r"<b>(?P<account>[^<]+)</b>\s+on\s+"
    r"(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3,})\s+(?P<year>\d{4})",
    re.IGNORECASE,
)


def parse_debit_entry_details(details: str) -> DebitEntryFingerprint | None:
    """Parse a Paisa ``Debit Entry`` ``details`` string into a fingerprint.

    Returns ``None`` when the string does not match Paisa v0.7.4's exact
    template, or the amount/date cannot be parsed — the caller treats that as
    "not provably expected" and leaves the danger fatal. Robust to the
    ``<b>``/``</b>`` HTML Paisa wraps the fields in; no blanket tag-stripping is
    applied (the regex anchors on the tags so a subtly-different upstream
    message does not half-match). The month is parsed from a fixed English map
    so a non-English server locale cannot mis-parse Paisa's always-English
    ``time.Format`` output.
    """
    if not details:
        return None
    match = _DEBIT_ENTRY_RE.search(details)
    if match is None:
        return None
    account = match.group("account").strip()
    try:
        amount = Decimal(match.group("amount")).quantize(_Q4)
    except InvalidOperation, ValueError:
        return None
    month = _PAISA_MONTHS.get(match.group("mon").lower())
    if month is None:
        return None
    try:
        parsed_date = datetime.date(
            int(match.group("year")), month, int(match.group("day"))
        )
    except ValueError:
        return None
    return DebitEntryFingerprint(account=account, date=parsed_date, amount=amount)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace for a safe one-line message.

    Paisa wraps diagnosis fields in ``<b>``/``<a>`` tags. This produces a plain
    readable string for the sync ``reason``; it is *not* used for matching
    (matching is the anchored regex above). Never returns secrets — diagnosis
    details carry only account/date/amount.
    """
    return " ".join(_HTML_TAG_RE.sub(" ", text or "").split())


def _fatal_message(summary: str, details: str) -> str:
    """A bounded, single-line message for one fatal danger: summary + sanitized
    detail. The summary alone (e.g. ``Negative Balance``) is the stable machine
    label; the detail adds the account/date an operator needs to act on it."""
    cleaned = _strip_html(details)
    if cleaned:
        return f"{summary}: {cleaned}"[:300]
    return summary[:300]


def classify_diagnosis(
    diagnosis: PaisaDiagnosis,
    entries: "tuple[ProjectedEntry, ...] | list[ProjectedEntry]",
) -> ClassifiedDiagnosis:
    """Classify a raw upstream diagnosis against the projection's expected
    contra-expense postings.

    Walks every ``danger``-level issue. A ``Debit Entry`` danger whose parsed
    ``(account, date, amount)`` fingerprint is still in the expected multiset
    consumes one fingerprint and is **accepted** (downgraded). Every other
    danger — an unmatched/extra ``Debit Entry``, a non-``Debit Entry`` danger
    (``Negative Balance``, …), or an unparseable issue — is **fatal**. Warning
    issues are ignored (they never affected sync and still do not).

    Multiplicity is exact: the expected multiset is mutated as fingerprints are
    consumed, so a third danger for a fingerprint expected twice stays fatal.
    """
    expected = expected_debit_entry_fingerprints(entries)
    expected_total = sum(expected.values())
    accepted = 0
    fatal = 0
    first_fatal_message: str | None = None
    for issue in diagnosis.issues:
        if issue.level != "danger":
            continue
        if issue.summary == DEBIT_ENTRY_SUMMARY:
            fingerprint = parse_debit_entry_details(issue.details)
            if fingerprint is not None and expected.get(fingerprint, 0) > 0:
                expected[fingerprint] -= 1
                accepted += 1
                continue
        # Any danger that did not consume an expected fingerprint is fatal:
        # an extra/unknown Debit Entry, a Negative Balance, an unparseable
        # issue, etc. Never accepted, never silently dropped.
        fatal += 1
        if first_fatal_message is None:
            first_fatal_message = _fatal_message(issue.summary, issue.details)
    return ClassifiedDiagnosis(
        expected_count=expected_total,
        accepted_count=accepted,
        fatal_count=fatal,
        first_fatal_message=first_fatal_message,
        raw_danger_count=diagnosis.danger_count,
    )


__all__ = [
    "DEBIT_ENTRY_SUMMARY",
    "ClassifiedDiagnosis",
    "DebitEntryFingerprint",
    "classify_diagnosis",
    "expected_debit_entry_fingerprints",
    "parse_debit_entry_details",
]
