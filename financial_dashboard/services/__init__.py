"""Domain services."""

from __future__ import annotations

from typing import Any

__all__ = ["FetchService"]


def __getattr__(name: str) -> Any:
    if name == "FetchService":
        from .fetch import FetchService

        return FetchService
    raise AttributeError(name)
