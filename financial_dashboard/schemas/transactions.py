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


class TransactionRelinkUpdate(BaseModel):
    """Request body for manual relink. Either field may be null to clear
    the corresponding link. If only ``card_id`` is given, the service
    derives ``account_id`` from the card's owning account."""

    account_id: int | None = None
    card_id: int | None = None


class TransactionRelinkResponse(BaseModel):
    ok: bool
    account_id: int | None = None
    card_id: int | None = None
    account_label: str | None = None
    card_label: str | None = None
    statement_marked_paid: bool = False
