from pydantic import BaseModel


class TransactionNoteUpdate(BaseModel):
    note: str = ""


class TransactionNoteResponse(BaseModel):
    ok: bool
    note: str | None = None


class TransactionCategoryUpdate(BaseModel):
    category: str = ""


class TransactionCategoryResponse(BaseModel):
    ok: bool
    category: str | None = None
