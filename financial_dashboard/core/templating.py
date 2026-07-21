"""Jinja templating helpers."""

from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

from fastapi.templating import Jinja2Templates


def currency_symbol(code: str | None) -> str:
    """Return a safe amount prefix without mislabelling foreign currencies."""
    normalized = (code or "").strip().upper() or "INR"
    if normalized == "INR":
        return "₹"
    if normalized == "USD":
        return "$"
    return f"{normalized} "


def format_inr_compact(value) -> str:
    amount = value or 0
    abs_amount = abs(float(amount))
    if abs_amount >= 1_00_00_000:
        scaled, suffix = float(amount) / 1_00_00_000, "Cr"
    elif abs_amount >= 1_00_000:
        scaled, suffix = float(amount) / 1_00_000, "L"
    elif abs_amount >= 1_000:
        scaled, suffix = float(amount) / 1_000, "K"
    else:
        return f"₹{float(amount):,.2f}"
    decimals = 1 if abs(scaled) >= 10 else 2
    formatted = f"{scaled:.{decimals}f}".rstrip("0").rstrip(".")
    return f"₹{formatted}{suffix}"


def format_inr_exact(value: Decimal | int | None) -> str:
    """Render an exact rupee amount with Indian digit grouping: ``₹12,34,567.89``.

    The last three integer digits form one group and everything above them is
    grouped in twos, which is how rupee figures are read here — Python's own
    ``{:,}`` would render the same number as ``1,234,567.89``.

    ``float`` is deliberately not accepted, and money must not be one. The
    rendering is exact and rounds half-up, and a float cannot hold most amounts
    exactly: 2.675 is really 2.67499999999999982..., so the "half" the rounding
    is supposed to round up is not there by the time this function sees it, and
    the rendered paisa comes out a unit low. Passing a float would make the
    output silently wrong rather than merely imprecise, so it is a type error.
    """
    if isinstance(value, float):
        raise TypeError(
            f"format_inr_exact takes Decimal or int, not float ({value!r}); "
            "a float cannot hold most rupee amounts exactly and would render "
            "a paisa low. Convert at the source, not here."
        )
    amount = Decimal(value or 0).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    sign = "-" if amount < 0 else ""
    digits, _, fraction = format(abs(amount), "f").partition(".")
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        groups.insert(0, head)
        groups.append(tail)
        digits = ",".join(groups)
    return f"{sign}₹{digits}.{fraction}"


@lru_cache
def get_templates() -> Jinja2Templates:
    templates = Jinja2Templates(
        directory=Path(__file__).resolve().parent.parent / "templates"
    )
    templates.env.filters["inr_compact"] = format_inr_compact
    templates.env.filters["inr_exact"] = format_inr_exact
    templates.env.filters["currency_symbol"] = currency_symbol
    return templates
