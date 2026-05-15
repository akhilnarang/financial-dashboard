"""Email endpoint response schemas."""

from pydantic import BaseModel


class ReparseEmailResponse(BaseModel):
    message: str
    new_status: str
    txn_id: int | None = None


class ReparseAllFailedResponse(BaseModel):
    succeeded: int
    skipped: int
    failed: int
