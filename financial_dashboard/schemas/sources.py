from pydantic import BaseModel


class SourceTestResponse(BaseModel):
    ok: bool
    message: str | None = None
    error: str | None = None
