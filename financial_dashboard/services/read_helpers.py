"""Small reusable helpers for bounded read services."""

from typing import Generic, NamedTuple, TypeVar


class BoundedText(NamedTuple):
    """A possibly truncated text value and its truncation flag."""

    value: str | None
    truncated: bool


T = TypeVar("T")


class OrderedBatchResult(NamedTuple, Generic[T]):
    """Items in requested order plus IDs absent from the database."""

    items: list[T]
    missing_ids: list[int]


def bound_text(value: str | None, limit: int) -> BoundedText:
    """Truncate an optional string to a fixed character limit."""
    if value is None or len(value) <= limit:
        return BoundedText(value, False)
    return BoundedText(value[:limit], True)


def order_batch(ids: list[int], items_by_id: dict[int, T]) -> OrderedBatchResult[T]:
    """Order found items like the request and retain missing IDs in that order."""
    return OrderedBatchResult(
        items=[items_by_id[item_id] for item_id in ids if item_id in items_by_id],
        missing_ids=[item_id for item_id in ids if item_id not in items_by_id],
    )
