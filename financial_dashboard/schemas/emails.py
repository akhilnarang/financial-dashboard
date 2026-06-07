"""Email endpoint response schemas."""

from typing import Literal

from pydantic import BaseModel


class ReparseEmailResponse(BaseModel):
    message: str
    new_status: Literal["parsed", "skipped"]
    txn_id: int | None = None


class ReparseAllFailedResponse(BaseModel):
    succeeded: int
    skipped: int
    failed: int
