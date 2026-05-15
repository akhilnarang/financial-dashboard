"""Jinja templating helpers."""

from functools import lru_cache
from pathlib import Path

from fastapi.templating import Jinja2Templates


def currency_symbol(code: str | None) -> str:
    if code and code.upper() == "USD":
        return "$"
    return "₹"


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


@lru_cache
def get_templates() -> Jinja2Templates:
    templates = Jinja2Templates(
        directory=Path(__file__).resolve().parent.parent / "templates"
    )
    templates.env.filters["inr_compact"] = format_inr_compact
    templates.env.filters["currency_symbol"] = currency_symbol
    return templates
