"""Input normalization and light PII redaction for categorization."""

import re
from collections.abc import Sequence

# Any run of characters that aren't a lowercase letter, digit, or space.
_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9 ]+")
# 7+ consecutive digits (card/account/long refs) or phone-like numbers.
_LONG_DIGITS = re.compile(r"\d{7,}")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,13}(?!\d)")


def normalize_text(s: str | None) -> str:
    """Lowercase narration and reduce punctuation to spaces for comparison.

    Replaces every run of non-alphanumeric characters with a single space, then
    collapses/strips whitespace — so the result is just space-separated
    alphanumeric tokens, used for substring matching (merchant rules, few-shot).
    Regex handles the char-class replacement; the whitespace collapse is the
    idiomatic ``" ".join(... .split())`` rather than a second regex. Returns ""
    for None/empty.
    """
    if not s:
        return ""
    return " ".join(_NON_ALNUM_SPACE.sub(" ", s.lower()).split())


def normalize_counterparty(s: str | None) -> str:
    """Reduce a counterparty to an alphanumeric comparison key.

    Lowercase, keep only alphanumerics (dropping spaces, dots, ``@``, slashes,
    etc.) — e.g. ``"Foo Bar @upi" -> "foobarupi"`` — so the same party compares
    equal across narration formats. Pure ``str`` methods (``isalnum``); no regex
    needed. Returns "" for None/empty.
    """
    if not s:
        return ""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def redact_pii(s: str | None) -> str:
    """Mask numeric PII before text is sent to the LLM.

    Phone-like sequences become ``[redacted-phone]`` and any run of 7+ digits
    (card / account / long reference numbers) becomes ``[redacted-num]``. These
    are positional digit patterns that ``str`` methods can't express, so regex
    is the right tool. Returns "" for None/empty.
    """
    if not s:
        return ""
    redacted = _PHONE.sub("[redacted-phone]", s)
    redacted = _LONG_DIGITS.sub("[redacted-num]", redacted)
    return redacted


_REDACTED_NAME = "[redacted-name]"
_MARKER = re.escape(_REDACTED_NAME)
# A name-like word: Title-case (Alex/Mr) or ALL-CAPS (DOE). Used to absorb
# the rest of the name around a matched token — unlisted parts (a middle name)
# and honorifics — so the whole name collapses to a single marker.
_NAME_WORD = r"[A-Z][A-Za-z]+|[A-Z]{2,}"
# Marker followed by more name-words / markers (forward absorb, whitespace).
_ABSORB_AFTER = re.compile(rf"{_MARKER}(?:\s+(?:{_NAME_WORD}|{_MARKER}))+")
# Name-words preceding a marker (backward absorb). The name-word alternation is
# wrapped so the trailing \s+ applies to BOTH branches, not just the last.
_ABSORB_BEFORE = re.compile(rf"(?:(?:{_NAME_WORD})\s+)+{_MARKER}")
# A fragment flanked by two markers, with ANY non-letter separators (including
# none / punctuation): collapse the whole span. Safe because both ends are
# already-identified name markers, so a fragment between them is part of the
# name even when there's no whitespace ('ALEXQUINSHDOE') or it's
# punctuation-separated ('ALEX/QUINN/DOE'). The [^A-Za-z\[\]] separator
# never swallows a marker's own characters.
_MARKER_RUN = re.compile(
    rf"{_MARKER}(?:[^A-Za-z\[\]]*(?:[A-Za-z]+[^A-Za-z\[\]]*)?{_MARKER})+"
)


def redact_names(s: str | None, tokens: Sequence[str]) -> str:
    """Remove your/family names before sending text to the LLM.

    Step 1: replace each configured token (case-insensitive substring, so
    truncated forms like 'ALEXQUIN' are caught) with [redacted-name].
    Step 2: absorb the rest of the name — adjacent capitalized name-words and
    neighbouring markers (separated by whitespace only, so punctuation is never
    consumed) collapse into ONE [redacted-name]. So 'ALEX QUINN DOE' (with
    only alex+doe listed) and 'Bob Quinn Doe' (only doe listed)
    each become a single [redacted-name]. Tokens under 3 chars are ignored.

    This is deliberately aggressive: a token appearing inside an unrelated
    merchant name (e.g. a shop literally named 'Roe') is also masked. That
    matches the intent — remove the name wherever it appears — at the cost of an
    occasional merchant false-positive.
    """
    if not s:
        return ""
    result = s
    for tok in tokens:
        tok = tok.strip()
        if len(tok) < 3:
            continue
        result = re.sub(re.escape(tok), _REDACTED_NAME, result, flags=re.IGNORECASE)

    prev = None
    while prev != result:
        prev = result
        result = _MARKER_RUN.sub(_REDACTED_NAME, result)
        result = _ABSORB_AFTER.sub(_REDACTED_NAME, result)
        result = _ABSORB_BEFORE.sub(_REDACTED_NAME, result)
    return result
