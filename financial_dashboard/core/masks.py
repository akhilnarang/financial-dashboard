"""Shared helpers for card/account mask strings.

Banks emit the same card or account in many masking styles — "XX4321",
"x4321", "4000 XXXX XXXX 4321", "0000 XXXX XXXX 4321". These helpers
reduce a mask to its digits and to its last-4 so the rest of the app can
compare "same card/account" regardless of cosmetic format.

Reducing a mask to its digits is lossy, and unsafe for identity matching:
"73XXXXXX3942" and "730055113942" flatten to "733942" and "730055113942",
which share no common suffix even though they are the same account — the
leading "73" is a prefix, but concatenation makes it part of the suffix.
``normalize_mask`` / ``mask_matches`` keep the wildcard *positions* and
compare right-aligned, so a mask with visible digits at both ends still
matches. Use those to answer "same account?"; ``mask_digits`` and
``mask_last4`` remain for display and last-4 bookkeeping.
"""

WILDCARD_CHARS = frozenset("Xx*#")
"""Characters a bank may use to hide a digit. Each stands for exactly one digit."""


def _is_separator(ch: str) -> bool:
    """Whether *ch* is cosmetic grouping, occupying no digit position.

    Any whitespace, plus the group dash banks use in "4000-XXXX-XXXX-1234".
    """
    return ch.isspace() or ch == "-"


WILDCARD = "X"
"""The single character every wildcard normalizes to."""


def normalize_mask(s: str | None) -> str:
    """Reduce a mask to digits and wildcards, preserving their positions.

    Separators are dropped, every wildcard becomes ``WILDCARD``, and digits keep
    their place. Position is what makes the result matchable, so — unlike
    ``mask_digits`` — nothing is allowed to collapse the string.

    A character that is neither digit, wildcard nor separator means this is not
    a mask we understand, and the whole string is **rejected** (""). Dropping the
    stray character instead would silently change which account the mask denotes:
    "12A3456" would become "123456" and match an account ending in that run. An
    unreadable mask must fail to match, not match something else.

    "XX1234"              -> "XX1234"
    "0000 XXXX XXXX 4321" -> "0000XXXXXXXX4321"
    "73XXXXXX3942"        -> "73XXXXXX3942"
    "730***113942"        -> "730XXX113942"
    "730055113942"        -> "730055113942"   (a bare number is its own mask)
    "12A3456"             -> ""               (rejected, not silently "123456")
    None                  -> ""
    """
    if not s:
        return ""
    out = []
    for ch in s:
        if "0" <= ch <= "9":
            out.append(ch)
        elif ch in WILDCARD_CHARS:
            out.append(WILDCARD)
        elif not _is_separator(ch):
            return ""
    return "".join(out)


def trailing_visible_digits(normalized: str) -> int:
    """Length of the run of literal digits at the right edge of a normalized mask.

    This is the part of a mask that actually identifies an account. A mask whose
    visible digits are all at the *left* ("123XXXXXXXXX") identifies nothing but
    a bank/branch prefix, which accounts share — so callers gate on this rather
    than on the total digit count.

    "XX1234"       -> 4
    "73XXXXXX3942" -> 4
    "123XXXXXXXXX" -> 0
    "1234"         -> 4
    """
    n = 0
    for ch in reversed(normalized):
        if not ("0" <= ch <= "9"):
            break
        n += 1
    return n


def mask_matches(pattern: str, stored: str) -> bool:
    """Whether normalized mask *pattern* can denote normalized *stored*.

    Both sides are right-aligned and compared over their overlap: two literal
    digits must be equal, and a wildcard on *either* side matches any digit.
    Overhang beyond the overlap is ignored, so a mask legitimately shorter than
    the value it masks still matches ("XX234" denotes "000000001234").

    Both arguments must already be normalized — the stored side is a mask too
    (a card_mask carries wildcards of its own), and flattening either side to
    its digits would reintroduce the very bug this function exists to fix.
    """
    if not pattern or not stored:
        return False
    for i in range(1, min(len(pattern), len(stored)) + 1):
        p, s = pattern[-i], stored[-i]
        if p == WILDCARD or s == WILDCARD:
            continue
        if p != s:
            return False
    return True


def mask_digits(s: str | None) -> str:
    """Every ASCII 0-9 in *s*, in order, with everything else stripped.
    "" if empty.

    "XX4321" → "4321"; "0000 XXXX XXXX 4321" → "00004321"; None → "".

    Restricted to ASCII on purpose: str.isdigit() also accepts unicode
    digits (Arabic-Indic, fullwidth, superscripts) that a card mask
    should never contain — treating those as significant would silently
    corrupt last-4 / suffix matching.
    """
    if not s:
        return ""
    return "".join(ch for ch in s if "0" <= ch <= "9")


def display_mask(value: str | None) -> str | None:
    """Return a redacted display value containing at most four trailing digits."""
    last_digits = mask_last4(value, partial=True)
    return f"XXXX{last_digits}" if last_digits else None


def mask_last4(s: str | None, *, partial: bool = False) -> str | None:
    """Last 4 digits of mask *s*.

    With fewer than 4 digits: return None by default, or the digits we do
    have (possibly "") when ``partial=True`` — used where a statement
    shows a short suffix like "XX34" and a 2-digit match is still useful.
    """
    digits = mask_digits(s)
    if len(digits) >= 4:
        return digits[-4:]
    if partial:
        return digits or None
    return None
