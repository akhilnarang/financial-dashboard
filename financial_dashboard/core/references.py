"""Payment references that name exactly one payment."""

import re

# UPI and IMPS share the 12-digit RRN. A NEFT or RTGS UTR starts with a letter.
RRN_PATTERN = r"\d{12}"
BANK_UTR_PATTERN = r"[A-Za-z][A-Za-z0-9]{15,21}"
_FORMS = (re.compile(RRN_PATTERN), re.compile(BANK_UTR_PATTERN))


def reference_form(ref: str | None) -> int | None:
    """Return which payment reference form a reference has.

    Args:
        ref: The reference as stored or read.

    Returns:
        The index of the form it fully matches, or None when it is not a
        payment reference.
    """
    text = (ref or "").strip()
    return next((i for i, form in enumerate(_FORMS) if form.fullmatch(text)), None)


def payment_references_differ(first: str | None, second: str | None) -> bool:
    """Tell if two references name two different payments.

    Both must have the same form and differ. References of two forms are
    never compared, because a bank can show its own number in place of the
    UTR.

    Args:
        first: One reference.
        second: The other reference.

    Returns:
        True when both are payment references of one form and differ.
    """
    form = reference_form(first)
    return (
        form is not None
        and form == reference_form(second)
        and (first or "").strip().upper() != (second or "").strip().upper()
    )
