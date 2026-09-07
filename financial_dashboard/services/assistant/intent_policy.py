"""Pure, versioned checks for high-impact assistant intent."""

import re
from difflib import SequenceMatcher
from typing import NamedTuple

from financial_dashboard.services.categorization.normalize import normalize_text

POLICY_VERSION = "v1"
_CATEGORY_CREATE_CUE = re.compile(
    r"\b(?:create|add|make|new)\b.{0,32}\b(?:category|classification)\b"
    r"|\b(?:category|classification)\b.{0,32}\b(?:create|add|make|new)\b",
    re.IGNORECASE,
)
_DURABLE_RULE_CUE = re.compile(
    r"\b(?:always|every\s+time|from\s+now\s+on|going\s+forward)\b"
    r"|\b(?:create|add|make)\b.{0,32}\b(?:merchant\s+)?rule\b",
    re.IGNORECASE,
)
_AFFIRMATIVE = re.compile(
    r"^(?:yes|yep|yeah|confirm|confirmed|do it|go ahead|please do|create it)"
    r"(?:\s*,?\s*(?:create|do)\s+it)?[.! ]*$",
    re.IGNORECASE,
)
_REFERENCE_DIGITS = re.compile(r"\d{7,}")
_GENERIC_PATTERNS = frozenset(
    {
        "ach",
        "atm",
        "auto",
        "autopay",
        "bank",
        "bill",
        "card",
        "cash",
        "credit",
        "debit",
        "imps",
        "instruction",
        "merchant",
        "nach",
        "neft",
        "offline",
        "online",
        "paid",
        "pay",
        "payment",
        "pos",
        "purchase",
        "qr",
        "rtgs",
        "standing",
        "transaction",
        "transfer",
        "txn",
        "upi",
        "vpa",
        "wallet",
        "withdrawal",
        "to",
        "from",
        "via",
        "at",
        "for",
        "the",
        "a",
        "an",
        "on",
        "of",
        "by",
        "in",
        "and",
        "or",
        "but",
        "with",
        "through",
        "per",
        "toward",
        "towards",
        "into",
        "onto",
        "under",
        "about",
        "over",
        "thru",
    }
)
_NEGATION_CUE = re.compile(
    r"\b(?:don['’]?t|don\s+t|do\s+not|never|no\s+longer|stop|not|"
    r"shouldn['’]?t|shouldn\s+t|should\s+not|can['’]?t|can\s+t|cannot|"
    r"won['’]?t|won\s+t|will\s+not|wouldn['’]?t|wouldn\s+t|would\s+not|"
    r"mustn['’]?t|mustn\s+t|must\s+not|couldn['’]?t|couldn\s+t|could\s+not|"
    r"may\s+not|might\s+not)\b",
    re.IGNORECASE,
)
_GLOBAL_NO_CHANGE = re.compile(
    r"^(?:(?:but|and|also|however|just)\s+)?(?:please\s+)?(?:"
    r"(?:don\s+t|do\s+not|never|shouldn\s+t|should\s+not|can\s+t|cannot|"
    r"won\s+t|will\s+not|wouldn\s+t|would\s+not|mustn\s+t|must\s+not|"
    r"couldn\s+t|could\s+not|may\s+not|might\s+not)"
    r"\s+(?:make|apply)\s+(?:any\s+)?changes?"
    r"(?:\s+(?:to|for)\s+(?:anything|everything|this|(?:this|the)\s+transaction|"
    r"(?:this|the)\s+record|(?:the\s+)?data))?"
    r"|(?:don\s+t|do\s+not|never|shouldn\s+t|should\s+not|can\s+t|cannot|"
    r"won\s+t|will\s+not|wouldn\s+t|would\s+not|mustn\s+t|must\s+not|"
    r"couldn\s+t|could\s+not|may\s+not|might\s+not)"
    r"\s+(?:change|modify|update)\s+"
    r"(?:anything|everything|this|(?:this|the)\s+transaction|"
    r"(?:this|the)\s+record|(?:the\s+)?data)"
    r"|(?:no|without)\s+changes?"
    r")(?:\s+(?:please|at\s+all|for\s+now|right\s+now|yet|anymore|though))?$",
    re.IGNORECASE,
)
_GLOBAL_NO_CHANGE_CUE = re.compile(
    r"\b(?:don\s+t|do\s+not|never|shouldn\s+t|should\s+not|can\s+t|cannot|"
    r"won\s+t|will\s+not|wouldn\s+t|would\s+not|mustn\s+t|must\s+not|"
    r"couldn\s+t|could\s+not|may\s+not|might\s+not)"
    r"\s+(?:make|apply)\s+(?:any\s+)?changes?\b"
    r"|\b(?:don\s+t|do\s+not|never|shouldn\s+t|should\s+not|can\s+t|cannot|"
    r"won\s+t|will\s+not|wouldn\s+t|would\s+not|mustn\s+t|must\s+not|"
    r"couldn\s+t|could\s+not|may\s+not|might\s+not)"
    r"\s+change\s+anything\b"
    r"|\b(?:no|without)\s+changes?\b",
    re.IGNORECASE,
)
_AMBIGUOUS_INTENT = re.compile(
    r"\b(?:maybe|likely|perhaps|possibly|probably|unsure)\b"
    r"|\b(?:may|might|could)\b(?!\s+not\b)(?:\s+\w+){0,2}\s+"
    r"(?:be|belong|fit|go|fall|categorize|categorise|classify)\b"
    r"|\b(?:either\b[^,.;!?]*\bor\b)"
    r"|\b[\w-]+\s+or\s+[\w-]+\b|\bbetween\b[^,.;!?]*\band\b"
    r"|\bvs\.?\b|\b[A-Za-z][\w-]*\s*/\s*[A-Za-z][\w-]*\b",
    re.IGNORECASE,
)
_NON_MUTATING_START = re.compile(
    r"^(?:please\s+)?(?:why|what|when|where|who|how|is|are|was|were|do|does|"
    r"did|can|could|would|should|will|"
    r"may\s+(?:i|we|you|this|that|it|the|there)\b|might|"
    r"explain|describe|show|list|"
    r"find|search|tell|give|check|inspect|review|look\s+up|"
    r"take\s+a\s+look(?:\s+at)?|leave\b.{0,48}\bunchanged|"
    r"keep\b.{0,48}\b(?:unchanged|as\s+is)|preserve|"
    r"help\s+me\s+understand)\b"
    r"|^i\s+(?:want|need|would\s+like|'d\s+like)\s+to\s+"
    r"(?:understand|know|see|review|inspect)\b",
    re.IGNORECASE,
)


def _find_closing_quote(text: str, start: int, closing: str) -> int | None:
    """Find a quoted note terminator without treating contractions as quotes."""
    index = start + 1
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            index += 2
            continue
        if text[index] == closing:
            if (
                closing in {"'", "’"}
                and index > start + 1
                and index + 1 < len(text)
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            ):
                index += 1
                continue
            return index
        index += 1
    return None


def intent_is_negated(text: str) -> bool:
    """Return whether the turn explicitly denies or withdraws an action."""
    return bool(_NEGATION_CUE.search(text))


def has_global_no_change(text: str) -> bool:
    """Return whether the turn explicitly forbids all changes."""
    for clause in re.split(r"[,.;:!?\n]", text):
        normalized_clause = normalize_text(clause).strip()
        if re.search(
            r"\b(?:make|apply) changes?\s+to\s+(?:the\s+)?(?:category|note)\b",
            normalized_clause,
        ):
            continue
        if _GLOBAL_NO_CHANGE.fullmatch(
            normalized_clause
        ) or _GLOBAL_NO_CHANGE_CUE.search(normalized_clause):
            return True
    return False


def has_ambiguous_intent(text: str) -> bool:
    return bool(_AMBIGUOUS_INTENT.search(text))


def has_non_mutating_intent(text: str) -> bool:
    """Return whether the turn begins as a read or preservation request."""
    return bool(_NON_MUTATING_START.search(normalize_text(text).strip()))


class InstructionParts(NamedTuple):
    text: str
    note_payload: str | None
    note_requires_category: bool
    note_shorthand: bool


def _safe_shorthand_note(payload: str) -> bool:
    """Accept plain note text only; directive-like shorthand must clarify."""
    normalized = normalize_text(payload).strip()
    return bool(
        normalized
        and "?" not in payload
        and not _NEGATION_CUE.search(payload)
        and not _CATEGORY_CREATE_CUE.search(payload)
        and not _DURABLE_RULE_CUE.search(payload)
        and not _AMBIGUOUS_INTENT.search(payload)
        and not re.search(
            r"\b(?:category|categorize|categorise|classify|cashflow|cash\s+flow|"
            r"exclude|include|rule|set|change|update|clear|remove|delete)\b",
            normalized,
        )
        and not has_non_mutating_intent(payload)
    )


def _parse_note_shorthand(text: str) -> InstructionParts | None:
    contextual = re.fullmatch(
        r"(?P<prefix>\s*this\s+was\s+)(?P<payload>[^,.;:!?\n]+?)"
        r"(?P<suffix>,\s*note\s+that\b.*)",
        text,
        re.IGNORECASE,
    )
    if contextual is not None:
        payload = contextual.group("payload").strip()
        if _safe_shorthand_note(payload):
            return InstructionParts(
                contextual.group("prefix") + contextual.group("suffix"),
                payload,
                True,
                True,
            )

    bare = re.fullmatch(
        r"\s*(?P<payload>[^,.;:!?\n]+?)\s*,\s*"
        r"(?P<category>[A-Za-z][\w -]*?)\s*",
        text,
    )
    if bare is None:
        return None
    payload = bare.group("payload").strip()
    if not _safe_shorthand_note(payload):
        return None
    return InstructionParts(", " + bare.group("category").strip(), payload, True, True)


def parse_instruction(text: str) -> InstructionParts:
    """Split application-bounded note content from the remaining instructions."""
    match = re.search(
        r"(?:\b(?:set|change|update|add|write)\s+(?:the\s+)?note\s*"
        r"(?:(?:to|as|is)\b|[:=])?\s*|\bnote\s*:\s*)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return _parse_note_shorthand(text) or InstructionParts(text, None, False, False)
    start = match.end()
    if start < len(text) and text[start] in "\"'“‘":
        closing = {"“": "”", "‘": "’"}.get(text[start], text[start])
        end = _find_closing_quote(text, start, closing)
        if end is not None:
            suffix = text[end + 1 :]
            return InstructionParts(
                text[:start] + suffix,
                text[start + 1 : end].strip(),
                bool(
                    re.match(
                        r"\s+and\s+(?:(?:set|change|update|assign)\s+)?"
                        r"(?:the\s+)?category\b",
                        suffix,
                        re.IGNORECASE,
                    )
                ),
                False,
            )
    delimiter = re.search(
        r"[,.;:!?\n]|\s+and\s+(?=(?:(?:set|change|update|assign)\s+)?"
        r"(?:the\s+)?category\b)",
        text[start:],
        re.IGNORECASE,
    )
    end = start + delimiter.start() if delimiter else len(text)
    requires_category = bool(
        delimiter is not None
        and re.fullmatch(r"\s+and\s+", delimiter.group(0), re.IGNORECASE)
    )
    return InstructionParts(
        text[:start] + text[end:],
        text[start:end].strip(),
        requires_category,
        False,
    )


def instruction_view(text: str) -> str:
    """Return the application-derived text used for privileged intent checks."""
    return parse_instruction(text).text


def negates_target(text: str, target: str) -> bool:
    """Return whether a negation cue governs this specific target phrase."""
    target_text = normalize_text(target.replace("_", " ")).strip()
    if not target_text:
        return False
    # Keep the scope local to a clause: a negation followed by at most a few
    # words and the requested target. This avoids treating note payload text
    # or a negation in a separate clause as a denied mutation.
    for clause in re.split(r"[,.;:!?\n]", text):
        normalized_clause = normalize_text(clause)
        if re.search(
            rf"{_NEGATION_CUE.pattern}(?:\s+\w+){{0,7}}\s+"
            rf"{re.escape(target_text)}(?!\w)",
            normalized_clause,
            re.IGNORECASE,
        ):
            return True
    return False


def evidence_is_current(user_text: str, evidence: str) -> bool:
    """Require the model's evidence to be an exact span of this user turn."""
    cleaned = evidence.strip()
    return bool(cleaned) and cleaned.casefold() in user_text.casefold()


def _mentions_category(text: str, category_slug: str) -> bool:
    words = normalize_text(category_slug.replace("_", " ")).strip().casefold()
    normalized = normalize_text(text).casefold()
    if not words:
        return False
    if re.search(rf"(?<!\w){re.escape(words)}(?!\w)", normalized):
        return True
    width = len(words.split())
    tokens = normalized.split()
    return any(
        SequenceMatcher(None, words, " ".join(tokens[index : index + width])).ratio()
        >= 0.82
        for index in range(max(0, len(tokens) - width + 1))
    )


def category_creation_is_explicit(
    user_text: str,
    evidence: str,
    category_slug: str,
) -> bool:
    """Authorize vocabulary growth only from a direct current-turn request."""
    denied_targets = (category_slug, "category", "classification")
    if any(
        negates_target(text, target)
        for text in (user_text, evidence)
        for target in denied_targets
    ):
        return False
    if not evidence_is_current(user_text, evidence):
        return False
    if not _CATEGORY_CREATE_CUE.search(evidence):
        return False
    return _mentions_category(evidence, category_slug)


def merchant_rule_is_explicit(
    user_text: str,
    evidence: str,
    category_slug: str,
) -> bool:
    """Authorize a durable rule only from a current-turn durable cue."""
    # An explicit denial vetoes any positive durable cue in another clause.
    if any(
        negates_target(text, target)
        for text in (user_text, evidence)
        for target in ("rule", "permanent")
    ):
        return False
    if not evidence_is_current(user_text, evidence):
        return False

    def clause_authorizes(clause: str) -> bool:
        cue = _DURABLE_RULE_CUE.search(clause)
        return bool(
            cue
            and _mentions_category(clause, category_slug)
            and not negates_target(clause, category_slug)
            and not negates_target(clause, cue.group(0))
        )

    # Check both strings. Evidence may be an exact positive-looking substring
    # that omits a governing negation from the surrounding user clause.
    return all(
        any(clause_authorizes(clause) for clause in re.split(r"[,.;:!?\n]", text))
        for text in (user_text, evidence)
    )


def is_direct_affirmative(user_text: str) -> bool:
    """Recognize a terse reply to an application-authored confirmation."""
    return bool(_AFFIRMATIVE.fullmatch(user_text.strip()))


def derive_merchant_pattern(counterparty: str | None) -> str:
    """Derive and validate the V1 rule pattern from trusted transaction data."""
    pattern = normalize_text(counterparty or "").strip()
    meaningful_tokens = [
        token
        for token in pattern.split()
        if token not in _GENERIC_PATTERNS
        and any(character.isalpha() for character in token)
    ]
    if (
        len(pattern) < 4
        or not any(character.isalpha() for character in pattern)
        or not meaningful_tokens
        or _REFERENCE_DIGITS.search(pattern)
    ):
        raise ValueError("Transaction counterparty is too generic for a merchant rule")
    return pattern
