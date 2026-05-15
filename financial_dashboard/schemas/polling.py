from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PollStatusResponse(BaseModel):
    state: str
    started_at: str | None = None
    finished_at: str | None = None
    last_stats: dict[str, Any] | None = None
    last_error: str | None = None
    progress: dict[str, Any] | None = None
