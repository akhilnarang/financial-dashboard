"""Shared helpers for card/account mask strings.

Banks emit the same card or account in many masking styles — "XX4321",
"x4321", "4000 XXXX XXXX 4321", "0000 XXXX XXXX 4321". These helpers
reduce a mask to its digits and to its last-4 so the rest of the app can
compare "same card/account" regardless of cosmetic format.
"""


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
